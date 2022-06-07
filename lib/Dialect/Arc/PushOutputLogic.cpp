//===- PushOutputLogic.cpp ------------------------------------------------===//
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

#define DEBUG_TYPE "arc-push-output-logic"

using namespace circt;
using namespace arc;
using namespace hw;
using llvm::SetVector;
using llvm::SmallDenseSet;

namespace {
struct PushOutputLogicPass : public PushOutputLogicBase<PushOutputLogicPass> {
  void runOnOperation() override;
  void runOnModule(HWModuleOp module, InstanceGraphNode *instanceGraphNode);

  InstanceGraph *instanceGraph;
};
} // namespace

void PushOutputLogicPass::runOnOperation() {
  DenseSet<Operation *> handledModules;
  instanceGraph = &getAnalysis<InstanceGraph>();

  for (auto *startNode : llvm::make_early_inc_range(*instanceGraph)) {
    if (handledModules.count(startNode->getModule().getOperation()))
      continue;
    for (InstanceGraphNode *node : llvm::post_order(startNode)) {
      auto module =
          dyn_cast_or_null<HWModuleOp>(node->getModule().getOperation());
      handledModules.insert(module);
      if (!module || module.getNumResults() == 0 || node->noUses() ||
          llvm::all_of(node->uses(), [](InstanceRecord *record) {
            return !record->getInstance();
          }))
        continue;
      runOnModule(module, node);
    }
  }

  markAnalysesPreserved<InstanceGraph>();
}

void PushOutputLogicPass::runOnModule(HWModuleOp module,
                                      InstanceGraphNode *instanceGraphNode) {
  auto *context = &getContext();

  // Trace all output values through combinational operations up to either an
  // input or a register.
  SetVector<Operation *> opsToExtract;
  SmallVector<Value> newModuleOutputs;
  SmallDenseSet<Value, 4> handled;

  SmallVector<Value> oldModuleOutputs;
  SmallVector<Value> worklist;
  for (auto value : module.getBodyBlock()->getTerminator()->getOperands()) {
    worklist.push_back(value);
    oldModuleOutputs.push_back(value);
  }
  std::reverse(worklist.begin(), worklist.end());

  while (!worklist.empty()) {
    Value value = worklist.back();
    if (handled.contains(value)) {
      worklist.pop_back();
      continue;
    }

    if (auto blockArg = value.dyn_cast<BlockArgument>()) {
      handled.insert(value);
      worklist.pop_back();
      continue;
    }

    auto result = value.cast<OpResult>();
    auto *op = result.getOwner();

    if (isArcBreakingOp(op)) {
      newModuleOutputs.push_back(value);
      handled.insert(value);
      worklist.pop_back();
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

    assert(handled.insert(result).second);
    assert(opsToExtract.insert(op));
  }

  // For the extracted ops, find the ones that will no longer have any use
  // inside the module.
  SmallPtrSet<Operation *, 4> fullyExtractedOps;
  auto isFullyExtracted = [&](Operation *userOp) {
    return isa<hw::OutputOp>(userOp) || fullyExtractedOps.contains(userOp);
  };
  for (auto *op : llvm::reverse(opsToExtract)) {
    auto onlyExtractedUses = llvm::all_of(op->getUsers(), isFullyExtracted);
    if (onlyExtractedUses)
      fullyExtractedOps.insert(op);
  }

  // Find the module inputs that are no longer needed after the fully extracted
  // ops are gone.
  SmallVector<unsigned> inputsToErase;
  SmallVector<unsigned> inputsToKeep;
  for (auto arg : module.getArguments()) {
    if (llvm::all_of(arg.getUsers(), isFullyExtracted))
      inputsToErase.push_back(arg.getArgNumber());
    else
      inputsToKeep.push_back(arg.getArgNumber());
  }

  // Early exit if there's nothing to do.
  if (newModuleOutputs == oldModuleOutputs && inputsToErase.empty() &&
      opsToExtract.empty())
    return;

  LLVM_DEBUG(llvm::dbgs() << "Updating " << module.moduleNameAttr() << " ("
                          << newModuleOutputs.size() << " outputs, "
                          << inputsToErase.size() << " removed inputs, "
                          << opsToExtract.size() << " extracted ops, "
                          << fullyExtractedOps.size() << " removed ops)\n");

  // Update the module's output terminator op.
  auto *outputOp = module.getBodyBlock()->getTerminator();
  OpBuilder builder(outputOp);
  builder.create<hw::OutputOp>(outputOp->getLoc(), newModuleOutputs);
  outputOp->erase();

  // Prepare the new result types and result names for updated instances.
  SmallVector<Type> newResultTypes;
  SmallVector<Attribute> newResultNames;
  for (auto output : newModuleOutputs) {
    newResultTypes.push_back(output.getType());
    StringAttr name;
    if (auto reg = output.getDefiningOp<seq::CompRegOp>())
      name = reg.nameAttr();
    else if (auto inst = output.getDefiningOp<InstanceOp>())
      name = StringAttr::get(
          context,
          inst.instanceName() + "." +
              inst.getResultName(output.cast<OpResult>().getResultNumber())
                  .getValue());
    else
      name = StringAttr::get(context, "anon");
    newResultNames.push_back(name);
  }
  auto newResultNamesAttr = ArrayAttr::get(context, newResultNames);

  // Update the module ports.
  SmallVector<std::pair<unsigned, PortInfo>> outputsToInsert;
  SmallVector<unsigned> outputsToErase;
  for (unsigned i = 0, e = module.getNumResults(); i != e; ++i)
    outputsToErase.push_back(i);
  for (auto [name, type] : llvm::zip(newResultNames, newResultTypes)) {
    PortInfo port;
    port.name = name.cast<StringAttr>();
    port.direction = PortDirection::OUTPUT;
    port.type = type;
    outputsToInsert.emplace_back(0, port);
  }
  module.modifyPorts({}, outputsToInsert, inputsToErase, outputsToErase);

  // Outline the marked operations at all instantiation sites.
  for (InstanceRecord *instRecord : instanceGraphNode->uses()) {
    auto oldInst =
        dyn_cast<InstanceOp>(instRecord->getInstance().getOperation());
    if (!oldInst) {
      instRecord->getInstance()->emitOpError("failed to extract outputs");
      return signalPassFailure();
    }
    LLVM_DEBUG(llvm::dbgs()
               << "- Inlining ops at " << oldInst.instanceNameAttr() << " in "
               << instRecord->getParent()->getModule().moduleNameAttr()
               << "\n");

    // Map module inputs to the actual values provided at the instantiation
    // site.
    DenseMap<Value, Value> mapping;
    for (auto [moduleInput, instInput] :
         llvm::zip(module.getArguments(), oldInst.getOperands())) {
      mapping.insert({moduleInput, instInput});
    }

    // Update the inputs and outputs of the instance.
    OpBuilder builder(oldInst);
    SmallVector<Value> newInputs;
    SmallVector<Attribute> newArgNames;
    for (auto idx : inputsToKeep) {
      newInputs.push_back(oldInst.inputs()[idx]);
      newArgNames.push_back(oldInst.argNames()[idx]);
    }
    auto newInst = builder.create<InstanceOp>(
        oldInst.getLoc(), newResultTypes, oldInst.instanceNameAttr(),
        oldInst.moduleNameAttr(), newInputs,
        ArrayAttr::get(context, newArgNames), newResultNamesAttr,
        oldInst.parametersAttr(), oldInst.inner_symAttr());
    for (auto [newModuleOutput, newInstOutput] :
         llvm::zip(newModuleOutputs, newInst.getResults()))
      mapping.insert({newModuleOutput, newInstOutput});

    // Clone the operations to extract and remap their operands from the values
    // within the module to the corresponding values at the instantiation site.
    for (auto *oldOp : opsToExtract) {
      auto *newOp = oldOp->cloneWithoutRegions();
      for (auto &operand : newOp->getOpOperands()) {
        auto mapped = mapping.lookup(operand.get());
        assert(mapped);
        operand.set(mapped);
      }
      builder.insert(newOp);
      for (auto [oldResult, newResult] :
           llvm::zip(oldOp->getResults(), newOp->getResults()))
        mapping.insert({oldResult, newResult});
    }

    // Replace the outputs of the original instance with the remapped values.
    assert(oldModuleOutputs.size() == oldInst.getNumResults());
    for (auto [oldModuleOutput, oldInstOutput] :
         llvm::zip(oldModuleOutputs, oldInst.getResults())) {
      auto newValue = mapping.lookup(oldModuleOutput);
      assert(newValue);

      // Emit a warning if this is a circular port connection that is used
      // anywhere.
      auto newResult = newValue.dyn_cast<OpResult>();
      if (newResult && newResult.getOwner() == oldInst &&
          !oldInstOutput.use_empty()) {
        auto d =
            oldInst.emitWarning("cyclical instance output `")
            << oldInst.getResultName(newResult.getResultNumber()).getValue()
            << "`";

        auto &note = d.attachNote(oldModuleOutput.getLoc());
        if (auto oldModuleInput = oldModuleOutput.dyn_cast<BlockArgument>())
          note << "connected to itself through input `"
               << oldInst.getArgumentName(oldModuleInput.getArgNumber())
                      .getValue()
               << "` here:";
        else
          note << "connected to itself through here:";

        for (auto *op : oldInstOutput.getUsers())
          d.attachNote(op->getLoc()) << "assuming value zero for use here:";

        newValue = builder.create<ConstantOp>(oldInstOutput.getLoc(),
                                              oldInstOutput.getType(), 0);
      }
      oldInstOutput.replaceAllUsesWith(newValue);
    }

    SmallVector<Value> inputsToPrune;
    for (auto operand : oldInst.getOperands())
      if (operand.getDefiningOp() != oldInst)
        inputsToPrune.push_back(operand);
    instanceGraph->replaceInstance(oldInst, newInst);
    oldInst->erase();
    pruneUnusedValues(inputsToPrune);
  }

  // Remove the ops in the module that have lost all users due to the
  // extraction.
  pruneUnusedValues(oldModuleOutputs);

  // Remove the block arguments that correspond to removed inputs.
  module.getBodyBlock()->eraseArguments(inputsToErase);
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createPushOutputLogicPass() {
  return std::make_unique<PushOutputLogicPass>();
}
} // namespace arc
} // namespace circt
