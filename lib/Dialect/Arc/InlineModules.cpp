//===- InlineModules.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-inline-modules"

using namespace circt;
using namespace arc;
using namespace hw;
using llvm::SetVector;
using llvm::SmallDenseSet;

namespace {
struct InlineModulesPass : public InlineModulesBase<InlineModulesPass> {
  void runOnOperation() override;
  void inlineModule(HWModuleOp module, InstanceOp inst, bool removeModule);
};
} // namespace

void InlineModulesPass::runOnOperation() {
  auto &instanceGraph = getAnalysis<InstanceGraph>();
  DenseSet<Operation *> handled;

  for (auto *startNode : instanceGraph) {
    if (handled.count(startNode->getModule().getOperation()))
      continue;
    for (InstanceGraphNode *node : llvm::post_order(startNode)) {
      if (!handled.insert(node->getModule().getOperation()).second)
        continue;
      unsigned numUsesLeft = node->getNumUses();
      if (numUsesLeft == 0)
        continue;
      for (auto *instRecord : node->uses()) {
        if (auto module =
                dyn_cast_or_null<HWModuleOp>(node->getModule().getOperation()))
          if (auto inst = dyn_cast_or_null<InstanceOp>(
                  instRecord->getInstance().getOperation()))
            inlineModule(module, inst, --numUsesLeft == 0);
      }
    }
  }
}

void InlineModulesPass::inlineModule(HWModuleOp module, InstanceOp inst,
                                     bool removeModule) {
  LLVM_DEBUG(
      llvm::dbgs() << "Inlining " << module.moduleNameAttr() << " into "
                   << inst->getParentOfType<HWModuleLike>().moduleNameAttr()
                   << "\n");

  auto updateName = [&](StringAttr attr) {
    return StringAttr::get(attr.getContext(),
                           inst.instanceName() + "/" + attr.getValue());
  };
  auto updateNames = [&](Operation *op) {
    if (auto name = op->getAttrOfType<StringAttr>("name")) {
      op->setAttr("name", updateName(name));
      return;
    }
    if (auto namesAttr = op->getAttrOfType<ArrayAttr>("names")) {
      SmallVector<Attribute> names(namesAttr.getValue().begin(),
                                   namesAttr.getValue().end());
      for (auto &name : names)
        if (auto nameStr = name.dyn_cast<StringAttr>())
          name = updateName(nameStr);
      op->setAttr("names", ArrayAttr::get(namesAttr.getContext(), names));
    }
  };

  // Inline the operations.
  OpBuilder builder(inst);
  if (removeModule) {
    // Simple implementation where we can just move the operations since the
    // module will be removed after inlining anyway.
    for (auto &op :
         llvm::make_early_inc_range(module.getBodyBlock()->getOperations())) {
      op.remove();
      for (auto &operand : op.getOpOperands()) {
        if (auto arg = operand.get().dyn_cast<BlockArgument>())
          operand.set(inst.getOperands()[arg.getArgNumber()]);
      }
      if (auto outputOp = dyn_cast<hw::OutputOp>(&op)) {
        inst.replaceAllUsesWith(outputOp.getOperands());
        op.erase();
        continue;
      }
      builder.insert(&op);
      updateNames(&op);
    }
  } else {
    // General implementation where we leave the original module in tact and
    // build up a clone of each operation.
    DenseMap<Value, Value> mapping; // mapping from outer to inner values

    for (auto [oldValue, newValue] :
         llvm::zip(module.getArguments(), inst.getOperands()))
      mapping.insert({oldValue, newValue});

    BackedgeBuilder backedgeBuilder(builder, module.getLoc());
    SmallDenseMap<Value, Backedge, 8> backedges;

    for (auto &oldOp : module.getBodyBlock()->getOperations()) {
      if (auto outputOp = dyn_cast<hw::OutputOp>(&oldOp)) {
        for (auto [result, output] :
             llvm::zip(inst.getResults(), outputOp.getOperands())) {
          auto mapped = mapping.lookup(output);
          assert(mapped);
          result.replaceAllUsesWith(mapped);
        }
        continue;
      }
      auto *newOp = oldOp.clone();
      builder.insert(newOp);
      updateNames(newOp);
      for (auto &operand : newOp->getOpOperands()) {
        auto &mapped = mapping[operand.get()];
        if (!mapped) {
          auto backedge = backedgeBuilder.get(operand.get().getType());
          backedges.insert({operand.get(), backedge});
          mapped = backedge;
        }
        operand.set(mapped);
      }
      for (auto [newResult, oldResult] :
           llvm::zip(newOp->getResults(), oldOp.getResults())) {
        mapping[oldResult] = newResult;
        auto it = backedges.find(oldResult);
        if (it != backedges.end()) {
          it->second.setValue(newResult);
          backedges.erase(it);
        }
      }
    }
  }

  // Remove the instance.
  inst->erase();

  // Remove the module.
  if (removeModule)
    module->erase();
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createInlineModulesPass() {
  return std::make_unique<InlineModulesPass>();
}
} // namespace arc
} // namespace circt
