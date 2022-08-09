//===- InlineArcInputs.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-lookup-tables"

using namespace circt;
using namespace arc;
using namespace hw;

namespace {
struct InlineArcInputsPass : public InlineArcInputsBase<InlineArcInputsPass> {
  void runOnOperation() override;
  void runOnModule();
};
} // namespace

struct ConstArg {
  unsigned i;
  ConstantOp op;

  bool operator==(const ConstArg &a) const {
    return a.i != i;
  }
};

void InlineArcInputsPass::runOnOperation() {
  // For each arc, this map tracks which arguments are constant.
  DenseMap<StringRef, SmallVector<ConstArg>> arcConstArgs;
  // All stateOps that use a particular arc.
  DenseMap<StringRef, SmallVector<StateOp>> arcUses;

  auto module = getOperation();
  for (auto moduleOp : llvm::make_early_inc_range(module.getOps<HWModuleOp>())) {
    for (auto op : moduleOp.getBody().getOps<StateOp>()) {
      // Find all stateOps that use constant operands.
      SmallVector<ConstArg> constArgs;
      auto ops = op.operands();
      for (unsigned i = 0; i < ops.size(); i++) {
        auto arg = ops[i];
        if (auto c = dyn_cast_or_null<ConstantOp>(arg.getDefiningOp()))
          constArgs.push_back({i, c});
      }
      auto arc = op.arc();
      arcUses[arc].push_back(op);
      if (arcConstArgs.count(arc) == 0)
        arcConstArgs[arc] = constArgs; // make an entry for this arc
      else if (arcConstArgs[arc] != constArgs)
        arcConstArgs.erase(arc); // remove this arc if the constants are not equal to a previous use
    }
  }

  // Now we go through all the defines and move the constant ops into the
  // bodies and rewrite the function types.
  for (auto defOp : llvm::make_early_inc_range(module.getOps<DefineOp>())) {
    if (arcConstArgs.count(defOp.getName()) >= 1) {
      OpBuilder builder = OpBuilder::atBlockBegin(&defOp.bodyBlock());
      auto args = arcConstArgs[defOp.getName()];
      DenseSet<unsigned> deletedArgs;
      SmallVector<unsigned> toDelete;
      for (auto arg : args) {
        // Put the constant into the define's body.
        auto* newOp = arg.op->clone();
        builder.insert(newOp);
        auto blockArg = defOp.getArgument(arg.i);
        blockArg.replaceAllUsesWith(newOp->getResult(0));
        // Register the arg for deletion from the function signature.
        deletedArgs.insert(arg.i);
        toDelete.push_back(arg.i);
      }
      defOp.bodyBlock().eraseArguments(toDelete);

      // Rewrite the FunctionType
      SmallVector<Type> inputTypes;
      SmallVector<Type> outputTypes;
      auto types = defOp.getArgumentTypes();
      for (unsigned i = 0; i < defOp.getNumArguments(); i++) {
        // Skip deleted inputs.
        if (deletedArgs.contains(i))
          continue;
        inputTypes.push_back(types[i]);
      }
      for (auto type : defOp.getResultTypes())
        outputTypes.push_back(type);

      defOp.setType(builder.getFunctionType(inputTypes, outputTypes));

      // Rewrite all stateOp arc uses to not pass in the constant anymore.
      auto uses = arcUses[defOp.getName()];
      for (auto stateOp : uses) {
        SmallVector<Value> inputs;
        auto ops = stateOp.operands();
        for (unsigned i = 0; i < ops.size(); i++) {
          if (deletedArgs.contains(i))
            continue;
          inputs.push_back(ops[i]);
        }

        builder.setInsertionPoint(stateOp);
        // Create a new stateOp with the reduced input vector.
        auto newStateOp = builder.create<StateOp>(stateOp.getLoc(), stateOp.arcAttr(), stateOp.getResultTypes(), stateOp.clock(), stateOp.enable(), stateOp.latency(), inputs);
        ArrayRef<Attribute> oldNames;
        if (auto names = stateOp->getAttrOfType<ArrayAttr>("names"))
          oldNames = names.getValue();
        if (!oldNames.empty()) {
          newStateOp->setAttr("names", builder.getArrayAttr(oldNames));
        }
        stateOp.replaceAllUsesWith(newStateOp);
        stateOp.erase();
      }
    }
  }
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createInlineArcInputsPass() {
  return std::make_unique<InlineArcInputsPass>();
}
} // namespace arc
} // namespace circt
