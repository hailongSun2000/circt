//===- RegToArc.cpp -------------------------------------------------------===//
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

#define DEBUG_TYPE "reg-to-arc"

using namespace circt;
using namespace arc;
using namespace hw;
using llvm::SetVector;
using llvm::SmallDenseSet;
using llvm::SmallSetVector;

namespace {
struct Cone {
  Value clock = {};
  SetVector<Value> inputs;
  SetVector<Operation *> ops;
  SmallSetVector<seq::CompRegOp, 1> regs;
  StateOp state = {}; // replacement for the registers
};

struct RegToArcPass : public RegToArcBase<RegToArcPass> {
  void runOnOperation() override;
  void runOnModule(HWModuleOp module);
  void analyzeCones();
  void mergeCones();
  void createArc(HWModuleOp module, Cone &cone);

  /// The global namespace used to create unique definition names.
  Namespace globalNamespace;
  /// All registers in the current module.
  SmallVector<seq::CompRegOp> regsInModule;
  /// Logic cones in the current module.
  SmallVector<Cone> cones;
  /// Mapping from registers to their logic cone.
  DenseMap<seq::CompRegOp, unsigned> coneIdxForReg;
  /// Mapping from operations in the module to the set of registers in whose
  /// fan-in cone the operation is.
  DenseMap<Operation *, SmallDenseSet<seq::CompRegOp, 1>> coneOpsToRegs;
};
} // namespace

void RegToArcPass::runOnOperation() {
  globalNamespace.clear();
  for (auto &op : getOperation().getOps())
    if (auto sym =
            op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      globalNamespace.newName(sym.getValue());
  for (auto module : getOperation().getOps<HWModuleOp>()) {
    runOnModule(module);
  }
}

void RegToArcPass::runOnModule(HWModuleOp module) {
  // Find all registers in the module.
  auto ops = module.getBodyBlock()->getOps<seq::CompRegOp>();
  regsInModule.assign(ops.begin(), ops.end());
  if (regsInModule.empty())
    return;
  LLVM_DEBUG(llvm::dbgs() << "Analyzing " << module.moduleNameAttr() << "\n");

  // Find all register fan-in cones, merge cones that have sufficient overlap,
  // and outline the cone's logic ops into an arc definition.
  analyzeCones();
  mergeCones();
  for (auto &cone : cones)
    if (!cone.regs.empty())
      createArc(module, cone);

  // Replace the registers with the created arcs.
  SmallVector<Value> valuesToPrune;
  for (auto &cone : cones) {
    if (cone.regs.empty() || !cone.state)
      continue;
    for (auto regAndResult : llvm::zip(cone.regs, cone.state.getResults())) {
      auto regOp = std::get<0>(regAndResult);
      if (!regOp.input().getDefiningOp<seq::CompRegOp>())
        valuesToPrune.push_back(regOp.input());
      regOp.replaceAllUsesWith(std::get<1>(regAndResult));
      regOp->erase();
    }
  }
  pruneUnusedValues(valuesToPrune);
}

void RegToArcPass::analyzeCones() {
  // For each register gather all operations in its fan-in cone, and the input
  // values to the cone.
  // TODO: Add per-op cone cache.
  SmallVector<Value> worklist;
  SmallPtrSet<Value, 8> handled;
  cones.clear();
  coneIdxForReg.clear();
  coneOpsToRegs.clear();
  for (auto regOp : regsInModule) {
    unsigned coneIdx = cones.size();
    Cone &cone = cones.emplace_back();
    cone.clock = regOp.clk();
    cone.regs.insert(regOp);
    handled.clear();
    worklist = {regOp.input()};

    while (!worklist.empty()) {
      Value value = worklist.back();
      if (handled.contains(value)) {
        worklist.pop_back();
        continue;
      }
      auto *op = value.getDefiningOp();
      if (!op || isArcBreakingOp(op)) {
        cone.inputs.insert(value);
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
      handled.insert(value);
      cone.ops.insert(op);
      coneOpsToRegs[op].insert(regOp);
      worklist.pop_back();
    }

    coneIdxForReg.insert({regOp, coneIdx});
  }
  LLVM_DEBUG(llvm::dbgs() << "- Analyzed " << coneOpsToRegs.size()
                          << " ops across " << cones.size()
                          << " logic cones\n");
}

void RegToArcPass::mergeCones() {
  // Combine registers if their fan-in cones share a sufficiently high number of
  // operations.
  SmallVector<unsigned> sharedOpHistogram;
  unsigned numDroppedCones = 0;
  for (auto coneAndIdx : llvm::enumerate(cones)) {
    auto &cone = coneAndIdx.value();
    auto coneIdx = coneAndIdx.index();
    if (cone.regs.empty())
      continue;

    // Build a histogram that tracks how many ops this cone has in common with
    // all other cones.
    sharedOpHistogram.assign(cones.size(), 0);
    for (auto *op : cone.ops) {
      for (auto regOp : coneOpsToRegs[op]) {
        if (regOp.clk() == cone.clock) {
          auto idx = coneIdxForReg[regOp];
          if (idx != coneIdx)
            sharedOpHistogram[idx]++;
        }
      }
    }

    // Merge all other cones with a sufficient fraction of shared operations
    // into the current cone.
    for (auto countAndConeIdx : llvm::enumerate(sharedOpHistogram)) {
      auto otherIdx = countAndConeIdx.index();
      if (otherIdx == coneIdx)
        continue;
      auto &otherCone = cones[otherIdx];
      if (otherCone.clock != cone.clock)
        continue;
      unsigned numSharedOps = countAndConeIdx.value();
      unsigned numUniqueOps =
          cone.ops.size() + otherCone.ops.size() - 2 * numSharedOps;
      if (numSharedOps == 0 || numSharedOps / 10 < numUniqueOps)
        continue;

      LLVM_DEBUG(llvm::dbgs()
                 << "- Merge cone " << coneIdx << " with " << otherIdx
                 << ", share "
                 << 100 * numSharedOps / (numSharedOps + numUniqueOps)
                 << "% ops (" << numSharedOps << " shared, " << numUniqueOps
                 << " unique)\n");
      for (auto regOp : otherCone.regs) {
        cone.regs.insert(regOp);
        coneIdxForReg[regOp] = coneIdx;
      }
      for (auto input : otherCone.inputs)
        cone.inputs.insert(input);
      for (auto op : otherCone.ops)
        cone.ops.insert(op);
      otherCone.inputs.clear();
      otherCone.ops.clear();
      otherCone.regs.clear();
      ++numDroppedCones;
    }
  }
  if (numDroppedCones > 0)
    LLVM_DEBUG(llvm::dbgs() << "- " << (cones.size() - numDroppedCones)
                            << " cones remain after merging\n");
}

void RegToArcPass::createArc(HWModuleOp module, Cone &cone) {
  assert(!cone.regs.empty());
  OpBuilder builder(module);

  // Determine the input and output types for the arc.
  SmallVector<Type> inputTypes, outputTypes;
  for (auto input : cone.inputs)
    inputTypes.push_back(input.getType());
  for (auto reg : cone.regs)
    outputTypes.push_back(reg.getType());

  // Create the arc definition.
  auto firstReg = cone.regs[0];
  auto defineOp = builder.create<DefineOp>(
      firstReg.getLoc(),
      builder.getStringAttr(
          globalNamespace.newName(module.moduleName() + "_arc")),
      builder.getFunctionType(inputTypes, outputTypes));
  defineOp.body().push_back(new Block());
  builder.setInsertionPointToStart(&defineOp.bodyBlock());

  DenseMap<Value, Value> mapping; // mapping from outer to inner values

  // Add block arguments.
  for (auto input : cone.inputs) {
    auto arg = defineOp.body().addArgument(input.getType(), input.getLoc());
    mapping.insert({input, arg});
  }

  // Outline operations into the arc.
  BackedgeBuilder backedgeBuilder(builder, module.getLoc());
  SmallDenseMap<Value, Backedge, 8> backedges;

  for (auto *outerOp : cone.ops) {
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

  // Create the output terminator operation.
  SmallVector<Value> outputValues;
  for (auto regOp : cone.regs) {
    auto mapped = mapping.lookup(regOp.input());
    assert(mapped);
    outputValues.push_back(mapped);
  }
  builder.create<arc::OutputOp>(defineOp.getLoc(), outputValues);

  // Instantiate the arc.
  builder.setInsertionPoint(firstReg);
  cone.state = builder.create<StateOp>(firstReg.getLoc(), defineOp, cone.clock,
                                       Value{}, 1, cone.inputs.getArrayRef());
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createRegToArcPass() {
  return std::make_unique<RegToArcPass>();
}
} // namespace arc
} // namespace circt
