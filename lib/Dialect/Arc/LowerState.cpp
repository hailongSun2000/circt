//===- LowerState.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-lower-state"

using namespace circt;
using namespace arc;
using namespace hw;

namespace {
struct LowerStatePass : public LowerStateBase<LowerStatePass> {
  void runOnOperation() override;
  void runOnModule();
  ClockTreeOp getClockTree(Value clock);

  HWModuleOp moduleOp;
  DenseMap<Value, ClockTreeOp> clockTrees;
};
} // namespace

void LowerStatePass::runOnOperation() {
  auto module = getOperation();
  for (auto op : llvm::make_early_inc_range(module.getOps<HWModuleOp>())) {
    moduleOp = op;
    runOnModule();
  }
}

void LowerStatePass::runOnModule() {
  clockTrees.clear();
  LLVM_DEBUG(llvm::dbgs() << "Lowering state in `" << moduleOp.moduleName()
                          << "`\n");

  // Lower states.
  for (auto &op : llvm::make_early_inc_range(*moduleOp.getBodyBlock())) {
    if (auto stateOp = dyn_cast<StateOp>(&op)) {
      if (stateOp.latency() == 0)
        continue;
      if (!stateOp.clock()) {
        stateOp.emitError("state with latency > 0 requires a clock");
        return signalPassFailure();
      }
      if (stateOp.enable()) {
        stateOp.emitError("enable conditions on states not supported");
        return signalPassFailure();
      }
      if (stateOp.latency() > 1) {
        stateOp.emitError("state with latency > 1 not supported");
        return signalPassFailure();
      }
      auto clockTree = getClockTree(stateOp.clock());
      ImplicitLocOpBuilder builder(stateOp.getLoc(), stateOp);
      auto state = builder.create<AllocStateOp>(stateOp.getResultTypes());
      if (auto names = stateOp->getAttr("names"))
        state->setAttr("names", names);
      stateOp.replaceAllUsesWith(state);

      auto newStateOp = builder.create<StateOp>(
          stateOp.getLoc(), stateOp.arcAttr(), stateOp.getResultTypes(),
          Value{}, Value{}, 0, stateOp.operands());

      builder.setInsertionPointToEnd(&clockTree.bodyBlock());
      builder.create<UpdateStateOp>(state.getResults(),
                                    newStateOp.getResults());
      stateOp.erase();
      continue;
    }
    if (auto memWriteOp = dyn_cast<MemoryWriteOp>(&op)) {
      LLVM_DEBUG(llvm::dbgs() << "- Updating write " << memWriteOp << "\n");
      auto clockTree = getClockTree(memWriteOp.clock());
      SmallVector<Value> newReads;
      for (auto read : memWriteOp.reads()) {
        LLVM_DEBUG(llvm::dbgs() << "  - Visiting read " << read << "\n");
        if (auto readOp = read.getDefiningOp<MemoryReadOp>())
          // HACK: This check for a constant clock is ugly. The read ops should
          // instead be replicated for every clock domain that they are used in,
          // and then dependencies should be tracked between reads and writes
          // within that clock domain. Lack of a clock (comb mem) should be
          // handled properly as well. Presence of a clock should group the read
          // under that clock as expected, and write to a "read buffer" that can
          // be read again by actual uses in different clock domains. LLVM
          // lowering already has such a read buffer. Just need to formalize it.
          if (!readOp.clock().getDefiningOp<hw::ConstantOp>() &&
              getClockTree(readOp.clock()).clock() != clockTree.clock())
            continue;
        newReads.push_back(read);
      }
      auto builder = ImplicitLocOpBuilder::atBlockEnd(memWriteOp.getLoc(),
                                                      &clockTree.bodyBlock());
      builder.create<MemoryWriteOp>(memWriteOp.memory(), memWriteOp.address(),
                                    clockTree.clock(), memWriteOp.enable(),
                                    memWriteOp.data(), newReads);
      memWriteOp.erase();
      continue;
    }
  }

  // Lower primary inputs.
  auto builder = OpBuilder::atBlockBegin(moduleOp.getBodyBlock());
  for (auto blockArg : moduleOp.getArguments()) {
    auto value = builder.create<RootInputOp>(
        blockArg.getLoc(), blockArg.getType(),
        moduleOp.argNames()[blockArg.getArgNumber()].cast<StringAttr>());
    blockArg.replaceAllUsesWith(value);
  }
  moduleOp.getBodyBlock()->eraseArguments([](auto) { return true; });

  // Lower primary outputs.
  auto outputOp = cast<hw::OutputOp>(moduleOp.getBodyBlock()->getTerminator());
  if (outputOp.getNumOperands() > 0) {
    ImplicitLocOpBuilder updateBuilder(outputOp.getLoc(), outputOp);
    auto passThrough = updateBuilder.create<PassThroughOp>();
    passThrough.body().emplaceBlock();
    updateBuilder.setInsertionPointToEnd(&passThrough.bodyBlock());
    for (auto [value, name] :
         llvm::zip(outputOp.getOperands(), moduleOp.resultNames())) {
      auto storage = builder.create<RootOutputOp>(
          outputOp.getLoc(), value.getType(), name.cast<StringAttr>());
      updateBuilder.create<UpdateStateOp>(ValueRange{storage},
                                          ValueRange{value});
    }
  }
  outputOp->erase();

  // Replace the `HWModuleOp` with a `ModelOp`.
  builder.setInsertionPoint(moduleOp);
  auto modelOp =
      builder.create<ModelOp>(moduleOp.getLoc(), moduleOp.moduleNameAttr());
  modelOp.body().takeBody(moduleOp.body());
  moduleOp->erase();
}

/// Return the `ClockTreeOp` for a given clock value, creating it if necessary.
/// If the clock is derived from a clock gate, a nested clock tree op is created
/// with the appropriate enable condition.
ClockTreeOp LowerStatePass::getClockTree(Value clock) {
  if (auto op = clockTrees.lookup(clock))
    return op;
  Block *withinBlock = moduleOp.getBodyBlock();
  Value clockRoot = clock;
  Value enable = {};

  // Look through clock gates and absorb their enable condition.
  if (auto clockGate = clock.getDefiningOp<ClockGateOp>()) {
    auto parent = getClockTree(clockGate.input());
    clockRoot = parent.clock();
    enable = clockGate.enable();
    withinBlock = &parent.bodyBlock();
  }

  // Create the new `ClockTreeOp` at the end of the block, just before the
  // terminator.
  auto builder = OpBuilder::atBlockEnd(withinBlock);
  if (!withinBlock->getParentOp()->hasTrait<OpTrait::NoTerminator>())
    builder.setInsertionPoint(withinBlock->getTerminator());

  auto clockTree =
      builder.create<ClockTreeOp>(clock.getLoc(), clockRoot, enable);
  clockTree.body().emplaceBlock();
  clockTrees.insert({clock, clockTree});
  return clockTree;
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createLowerStatePass() {
  return std::make_unique<LowerStatePass>();
}
} // namespace arc
} // namespace circt
