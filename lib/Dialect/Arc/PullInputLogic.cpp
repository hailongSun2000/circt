//===- PullInputLogic.cpp
//-------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/Namespace.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include <variant>

#define DEBUG_TYPE "arc-pull-input-logic"

using namespace circt;
using namespace arc;
using namespace hw;
using llvm::SetVector;
using llvm::SmallDenseSet;

namespace {
struct PullInputLogicPass : public PullInputLogicBase<PullInputLogicPass> {
  void runOnOperation() override;
  void runOnModule(HWModuleOp module, InstanceGraphNode *instanceGraphNode);

  InstanceGraph *instanceGraph;
  Namespace globalNamespace;
};
} // namespace

void PullInputLogicPass::runOnOperation() {
  instanceGraph = &getAnalysis<InstanceGraph>();
  for (auto module : getOperation().getOps<HWModuleLike>())
    globalNamespace.newName(module.moduleName());

  DenseSet<Operation *> handledModules;

  for (auto *startNode : llvm::make_early_inc_range(*instanceGraph)) {
    if (handledModules.count(startNode->getModule().getOperation()))
      continue;
    for (InstanceGraphNode *node : llvm::inverse_post_order(startNode)) {
      auto module =
          dyn_cast_or_null<HWModuleOp>(node->getModule().getOperation());
      if (!handledModules.insert(module).second)
        continue;
      if (!module || module.getNumArguments() == 0 || node->noUses())
        continue;
      runOnModule(module, node);
    }
  }

  markAnalysesPreserved<InstanceGraph>();
}

namespace {
struct LogicCone {
  InstanceOp inst;
  SmallVector<Value, 0> final; // values that make up input of the fanin cone
  DenseMap<Value, unsigned> finalIndices;
  SmallVector<Operation *, 0> inlined; // operations to be inlined

  LogicCone(InstanceOp inst) : inst(inst) {}
};
} // namespace

using OutliningGroup = SmallVector<LogicCone, 1>;

void PullInputLogicPass::runOnModule(HWModuleOp module,
                                     InstanceGraphNode *instanceGraphNode) {
  auto *context = &getContext();

  // Trace the fan-in cone of logic operations in front of each instance of the
  // module.
  SmallVector<LogicCone, 1> cones;
  for (auto *record : instanceGraphNode->uses()) {
    auto inst =
        dyn_cast_or_null<InstanceOp>(record->getInstance().getOperation());
    if (!inst)
      return;
    cones.emplace_back(inst);
  }
  LLVM_DEBUG(llvm::dbgs() << "\nAnalyzing " << module.moduleNameAttr() << " ("
                          << cones.size() << " instances)\n");

  for (auto &cone : cones) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Tracing fan-in cone of " << cone.inst.instanceNameAttr()
               << " in "
               << cone.inst->getParentOfType<HWModuleLike>().moduleNameAttr()
               << "\n");

    // Populate a worklist with the values to trace.
    SmallDenseSet<Value, 4> handled;
    SmallVector<Value> worklist;
    for (auto value : cone.inst.getOperands()) {
      worklist.push_back(value);
    }
    std::reverse(worklist.begin(), worklist.end());

    while (!worklist.empty()) {
      Value value = worklist.back();
      if (handled.contains(value)) {
        worklist.pop_back();
        continue;
      }

      auto *op = value.getDefiningOp();
      if (!op || isArcBreakingOp(op)) {
        handled.insert(value);
        worklist.pop_back();
        if (cone.finalIndices.insert({value, cone.final.size()}).second)
          cone.final.push_back(value);
        continue;
      }

      bool allHandled = true;
      for (auto operand : op->getOperands()) {
        if (!handled.count(operand)) {
          worklist.push_back(operand);
          allHandled = false;
        }
      }
      if (!allHandled)
        continue;

      assert(handled.insert(value).second);
      cone.inlined.push_back(op);
      worklist.pop_back();
    }
  }

  // Separate the instance logic cones into groups with identical cones.
  // TODO: Actually do the separating. This just creates separate groups
  // everywhere.
  using LogicConeGroup = SmallVector<LogicCone *, 1>;
  SmallVector<LogicConeGroup, 1> coneGroups;
  for (auto &cone : cones)
    coneGroups.push_back({&cone});

  // At this point each cone group represents a set of instances with an
  // identical fan-in logic cone. Process each group separately, inlining a copy
  // of the logic cone into the module and updating all instances to match.
  std::reverse(coneGroups.begin(), coneGroups.end());
  for (unsigned groupIdx = 0, groupEnd = coneGroups.size();
       groupIdx != groupEnd; ++groupIdx) {
    auto &coneGroup = coneGroups[groupIdx];
    auto *firstCone = coneGroup[0];

    // Create a clone of the module or reuse the existing module if this is the
    // last group.
    HWModuleOp newModule;
    if (groupIdx != groupEnd - 1) {
      newModule = module.clone();
      newModule->setAttr(SymbolTable::getSymbolAttrName(),
                         StringAttr::get(context, globalNamespace.newName(
                                                      module.moduleName())));
      OpBuilder builder(module);
      builder.insert(newModule);
      instanceGraph->addModule(newModule);
    } else {
      newModule = module;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "- Inlining " << firstCone->inlined.size() << " ops, "
               << firstCone->final.size() << " inputs into "
               << newModule.moduleNameAttr() << " (from " << coneGroup.size()
               << " instances)\n");

    // Update the module ports.
    SmallVector<Attribute> newInputNames;
    SmallVector<Type> newInputTypes;

    for (auto outerValue : firstCone->final) {
      newInputTypes.push_back(outerValue.getType());
      StringAttr name;
      if (auto reg = outerValue.getDefiningOp<seq::CompRegOp>())
        name = reg.nameAttr();
      else if (auto inst = outerValue.getDefiningOp<InstanceOp>())
        name = StringAttr::get(
            context, inst.instanceName() + "." +
                         inst.getResultName(
                                 outerValue.cast<OpResult>().getResultNumber())
                             .getValue());
      else
        name = StringAttr::get(context, "arcIn" + Twine(newInputNames.size()));
      newInputNames.push_back(name);
    }

    SmallVector<std::pair<unsigned, PortInfo>> inputsToInsert;
    SmallVector<unsigned> inputsToErase;
    for (unsigned i = 0, e = newModule.getNumArguments(); i != e; ++i)
      inputsToErase.push_back(i);
    for (auto [name, type] : llvm::zip(newInputNames, newInputTypes)) {
      PortInfo port;
      port.name = name.cast<StringAttr>();
      port.direction = PortDirection::INPUT;
      port.type = type;
      inputsToInsert.emplace_back(0, port);
    }
    newModule.modifyPorts(inputsToInsert, {}, inputsToErase, {});

    auto builder = OpBuilder::atBlockBegin(newModule.getBodyBlock());
    DenseMap<Value, Value> mapping; // mapping from outer to inner values

    // Create block arguments for the new module inputs.
    for (auto outerValue : firstCone->final) {
      Value innerValue = newModule.getBodyBlock()->addArgument(
          outerValue.getType(), outerValue.getLoc());
      mapping.insert({outerValue, innerValue});
    }

    // Pull the marked operations from the instantiation site into the
    // module.
    BackedgeBuilder backedgeBuilder(builder, module.getLoc());
    SmallDenseMap<Value, Backedge, 8> backedges;
    unsigned backedgesUsed = 0;

    for (auto *outerOp : firstCone->inlined) {
      // LLVM_DEBUG(llvm::dbgs() << "  - Inlining " << *outerOp << "\n");
      auto *innerOp = outerOp->clone();
      builder.insert(innerOp);
      for (auto &operand : innerOp->getOpOperands()) {
        Value mapped = mapping.lookup(operand.get());
        if (!mapped) {
          auto it = backedges.find(operand.get());
          if (it != backedges.end()) {
            mapped = it->second;
          } else {
            auto backedge = backedgeBuilder.get(operand.get().getType());
            backedges.insert({operand.get(), backedge});
            mapped = backedge;
            ++backedgesUsed;
          }
        }
        operand.set(mapped);
      }
      for (auto [outerValue, innerValue] :
           llvm::zip(outerOp->getResults(), innerOp->getResults())) {
        mapping.insert({outerValue, innerValue});
        auto it = backedges.find(outerValue);
        if (it != backedges.end()) {
          it->second.setValue(innerValue);
          backedges.erase(it);
        }
      }
    }

    assert(backedges.empty());

    // Replace uses of block arguments with their newly-inlined
    // definitions.
    for (auto [outerValue, innerValue] :
         llvm::zip(firstCone->inst.getOperands(), newModule.getArguments())) {
      auto mapped = mapping.lookup(outerValue);
      assert(mapped);
      innerValue.replaceAllUsesWith(mapped);
    }

    // Remove old block arguments.
    newModule.getBodyBlock()->eraseArguments(inputsToErase);

    // Update each of the instantiations.
    for (auto *cone : coneGroup) {
      auto oldInst = cone->inst;
      builder.setInsertionPoint(oldInst);
      LLVM_DEBUG(llvm::dbgs()
                 << "  - Updating instance " << oldInst.instanceNameAttr()
                 << " in "
                 << oldInst->getParentOfType<HWModuleLike>().moduleNameAttr()
                 << "\n");
      auto newInst = builder.create<InstanceOp>(
          oldInst.getLoc(), oldInst.getResultTypes(),
          oldInst.instanceNameAttr(),
          FlatSymbolRefAttr::get(newModule.moduleNameAttr()), cone->final,
          ArrayAttr::get(context, newInputNames), oldInst.resultNamesAttr(),
          oldInst.parametersAttr(), oldInst.inner_symAttr());
      oldInst.replaceAllUsesWith(newInst);

      SmallVector<Value> inputsToPrune;
      for (auto operand : oldInst.getOperands())
        if (operand.getDefiningOp() != oldInst)
          inputsToPrune.push_back(operand);
      instanceGraph->replaceInstance(oldInst, newInst);
      oldInst->erase();
      pruneUnusedValues(inputsToPrune);
    }
  }
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createPullInputLogicPass() {
  return std::make_unique<PullInputLogicPass>();
}
} // namespace arc
} // namespace circt
