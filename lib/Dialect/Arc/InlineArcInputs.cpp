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
  DenseMap<StringRef, SmallVector<ConstArg>> arcConstArgs;
  DenseMap<StringRef, SmallVector<StateOp>> arcUses;

  auto module = getOperation();
  for (auto moduleOp : llvm::make_early_inc_range(module.getOps<HWModuleOp>())) {
    for (auto op : moduleOp.getBody().getOps<StateOp>()) {
      SmallVector<ConstArg> constArgs;
      auto ops = op.getOperands();
      for (unsigned i = 1; i < op.getNumOperands(); i++) {
        auto arg = ops[i];
        if (auto c = dyn_cast_or_null<ConstantOp>(arg.getDefiningOp())) {
          constArgs.push_back({i, c});
        }
      }
      auto arc = op.arc();
      arcUses[arc].push_back(op);
      if (arcConstArgs.count(arc) == 0) {
        arcConstArgs[arc] = constArgs;
      } else if (arcConstArgs[arc] != constArgs) {
        arcConstArgs.erase(arc);
      }
    }
  }

  for (auto defOp : llvm::make_early_inc_range(module.getOps<DefineOp>())) {
    if (arcConstArgs.count(defOp.getName()) >= 1) {
      OpBuilder builder = OpBuilder::atBlockBegin(&defOp.bodyBlock());
      auto args = arcConstArgs[defOp.getName()];
      DenseSet<unsigned> deletedArgs;
      SmallVector<unsigned> toDelete;
      for (auto arg : args) {
        auto* newOp = arg.op->clone();
        builder.insert(newOp);
        auto blockArg = defOp.getArgument(arg.i-1);
        blockArg.replaceAllUsesWith(newOp->getResult(0));
        deletedArgs.insert(arg.i);
        toDelete.push_back(arg.i-1);
      }
      defOp.bodyBlock().eraseArguments(toDelete);

      SmallVector<Type> inputTypes;
      SmallVector<Type> outputTypes;
      auto types = defOp.getArgumentTypes();
      for (unsigned i = 0; i < defOp.getNumArguments(); i++) {
        if (deletedArgs.contains(i+1))
          continue;
        inputTypes.push_back(types[i]);
      }
      for (auto type : defOp.getResultTypes())
        outputTypes.push_back(type);

      defOp.setType(builder.getFunctionType(inputTypes, outputTypes));

      auto uses = arcUses[defOp.getName()];
      for (auto stateOp : uses) {
        SmallVector<Value> inputs;
        auto ops = stateOp.getOperands();
        for (unsigned i = 1; i < stateOp.getNumOperands(); i++) {
          if (deletedArgs.contains(i))
            continue;
          inputs.push_back(ops[i]);
        }

        builder.setInsertionPoint(stateOp);
        auto newStateOp = builder.create<StateOp>(stateOp->getLoc(), defOp, stateOp.clock(), stateOp.enable(), stateOp.latency(), inputs);
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
