//===- ScheduleModels.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/ControlFlowSinkUtils.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-schedule"

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace hw;

using llvm::MapVector;
using llvm::SmallDenseSet;
using llvm::SmallMapVector;

namespace {
struct ScheduleModelsPass : public ScheduleModelsBase<ScheduleModelsPass> {
  void runOnOperation() override;
  LogicalResult runOnModel();
  void assignOperationMasks();
  LogicalResult replicateOpsIntoClockTrees();

  Namespace globalNamespace;
  /// The current model being processed.
  ModelOp modelOp;
  /// Unique IDs assigned to top-level `ClockTreeOp` and `PassThroughOp`.
  SmallDenseMap<Operation *, unsigned> clockTreeIds;
  /// For every op the set of clock trees it contributes to, as a bit mask of
  /// clock tree IDs;
  MapVector<Operation *, APInt> operationMasks;
};
} // namespace

void ScheduleModelsPass::runOnOperation() {
  globalNamespace.clear();
  for (auto &op : *getOperation().getBody())
    if (auto sym = op.getAttrOfType<StringAttr>("sym_name"))
      globalNamespace.newName(sym.getValue());
  for (auto op : getOperation().getOps<ModelOp>()) {
    modelOp = op;
    if (failed(runOnModel()))
      return signalPassFailure();
  }
}

LogicalResult ScheduleModelsPass::runOnModel() {
  operationMasks.clear();
  clockTreeIds.clear();
  LLVM_DEBUG(llvm::dbgs() << "Scheduling `" << modelOp.name() << "`\n");

  // Establish a dense integer ID assignment for every clock tree and then
  // establish a mask of clock trees every operation contributes to.
  for (auto &clockTree : modelOp.bodyBlock())
    if (isa<ClockTreeOp, PassThroughOp>(&clockTree))
      clockTreeIds.insert({&clockTree, clockTreeIds.size()});
  LLVM_DEBUG(llvm::dbgs() << "- Assigned " << clockTreeIds.size()
                          << " clock tree IDs\n");
  assignOperationMasks();

  // Group operations based on the clock trees they contribute to, excluding
  // operations that contribute only to one such tree. Each group will be
  // isolated into a reusable region afterwards.
  SmallMapVector<APInt, SetVector<Operation *>, 16> operationsByMask;
  for (auto [op, mask] : operationMasks)
    // TODO: Re-enable this once we allow for the result of a `arc.memory` op to
    // be threaded into sub-arcs (i.e. we have an `!arc.memory<T>` type).
    if (false)
      if (op->getParentOp() == modelOp && mask.countPopulation() > 1)
        operationsByMask[mask].insert(op);
  LLVM_DEBUG(llvm::dbgs() << "- Operations form " << operationsByMask.size()
                          << " clock tree contribution groups\n");

  // Move operations into dedicated reusable regions.
  // SmallVector<std::pair<StateOp, APInt>> reusableArcsAndMask;
  for (auto [mask, ops] : operationsByMask) {
    SmallString<16> maskStr;
    mask.toStringUnsigned(maskStr, 2);
    LLVM_DEBUG(llvm::dbgs() << "- Grouping " << ops.size() << " ops with mask "
                            << maskStr << "\n");

    auto block = std::make_unique<Block>();
    OpBuilder builder(modelOp);
    builder.setInsertionPoint(block.get(), block->begin());

    DenseMap<Value, Value> valueMappings;
    SmallVector<Type> inputTypes;
    SetVector<Value> inputs;
    SetVector<Value> outputs;

    for (auto *op : ops) {
      op->remove();
      builder.insert(op);
      for (auto &operand : op->getOpOperands()) {
        if (auto mapping = valueMappings.lookup(operand.get())) {
          operand.set(mapping);
          continue;
        }
        if (auto *op = operand.get().getDefiningOp();
            !op || !ops.contains(op)) {
          auto arg = block->addArgument(operand.get().getType(),
                                        operand.get().getLoc());
          inputs.insert(operand.get());
          inputTypes.push_back(arg.getType());
          valueMappings.insert({operand.get(), arg});
          operand.set(arg);
        }
      }
      for (auto result : op->getResults())
        for (auto *user : result.getUsers())
          if (!ops.contains(user))
            outputs.insert(result);
    }
    auto outputOp =
        builder.create<arc::OutputOp>(modelOp.getLoc(), outputs.getArrayRef());
    sortTopologically(block.get());

    builder.setInsertionPoint(modelOp);
    auto defOp = builder.create<DefineOp>(
        modelOp.getLoc(),
        builder.getStringAttr(
            globalNamespace.newName(modelOp.name() + "_shared_" + maskStr)),
        builder.getFunctionType(inputTypes, outputOp.getOperandTypes()));
    defOp.body().push_back(block.release());

    builder.setInsertionPointToEnd(&modelOp.bodyBlock());
    auto useOp = builder.create<StateOp>(modelOp.getLoc(), defOp, Value{},
                                         Value{}, 0, inputs.getArrayRef());
    auto outputsVector = outputs.takeVector();
    for (auto [oldValue, newValue] :
         llvm::zip(outputsVector, useOp.getResults()))
      oldValue.replaceUsesWithIf(newValue, [&](auto &use) {
        return use.getOwner()->getBlock() != &defOp.bodyBlock();
      });
    // reusableArcsAndMask.push_back({useOp, mask});
  }

  // // Duplicate the arc uses for the arcs we've just created above, such that
  // we
  // // have one use per clock tree which can then be sunk into that clock tree
  // // properly.
  // SmallDenseMap<StateOp, SmallDenseMap<APInt, StateOp>> splitReusableArcs;
  // for (auto [useOp, mask] : reusableArcsAndMask) {
  //   unsigned numLeft = mask.countPopulation();
  //   OpBuilder builder(useOp);
  //   for (unsigned id = 0; id < clockTreeIds.size(); ++id) {
  //     if (!mask[id])
  //       continue;
  //     auto singleMask = APInt::getOneBitSet(clockTreeIds.size(), id);
  //     auto &createdArcs = splitReusableArcs[useOp];
  //     bool isLast = (--numLeft == 0);
  //     if (isLast) {
  //       operationMasks[useOp] = singleMask;
  //       createdArcs.insert({singleMask, useOp});
  //     } else {
  //       auto newUseOp = useOp.cloneWithoutRegions();
  //       builder.insert(newUseOp);
  //       operationMasks[newUseOp] = singleMask;
  //       createdArcs.insert({singleMask, newUseOp});
  //     }
  //   }
  // }
  // for (auto &pair : splitReusableArcs) {
  //   auto originalUseOp = pair.first;
  //   auto &splitArcs = pair.second;
  //   for (auto result : originalUseOp.getResults()) {
  //     for (auto &use : llvm::make_early_inc_range(result.getUses())) {
  //       auto maskIt = operationMasks.find(use.getOwner());
  //       if (maskIt == operationMasks.end()) {
  //         auto d = originalUseOp.emitError("arc used in a strange way");
  //         d.attachNote(use.getOwner()->getLoc()) << "weird user";
  //         // continue;
  //         return failure();
  //       }
  //       auto newUseOp = splitArcs.lookup(maskIt->second);
  //       assert(newUseOp && "should have dedicated use");
  //       use.set(newUseOp.getResult(result.getResultNumber()));
  //     }
  //   }
  // }

  // Move operations down into clock trees, replicating them if they are used in
  // multiple clock trees. This is necessary since we eventually want to have
  // the clock trees be independent pieces of work containing all the necessary
  // computation. Needless replication should be handled by factoring logic
  // common among multiple clock trees into a separate arc.
  if (failed(replicateOpsIntoClockTrees()))
    return failure();

  // Topologically sort operations.
  modelOp.walk([](Block *block) { sortTopologically(block); });

  // Ensure that memory reads dominate all memory writes.
  bool anyFailures = false;
  SmallVector<MemoryReadOp> memReads;
  SmallVector<MemoryWriteOp> memWrites;
  DominanceInfo domInfo;
  modelOp.walk([&](MemoryOp memOp) {
    memReads.clear();
    memWrites.clear();
    for (auto *user : memOp->getUsers()) {
      if (auto readOp = dyn_cast<MemoryReadOp>(user))
        memReads.push_back(readOp);
      if (auto writeOp = dyn_cast<MemoryWriteOp>(user))
        memWrites.push_back(writeOp);
    }
    for (auto writeOp : memWrites) {
      for (auto read : writeOp.reads()) {
        Block *commonParent = read.getDefiningOp()->getBlock();
        while (commonParent && commonParent != writeOp->getBlock())
          commonParent = commonParent->getParentOp()->getBlock();
        if (!commonParent) {
          writeOp->replaceUsesOfWith(read, writeOp.data());
          continue;
        }
        if (domInfo.properlyDominates(read, writeOp))
          continue;
        auto d = mlir::emitError(read.getLoc(),
                                 "memory read does not dominate all writes");
        d.attachNote(writeOp.getLoc()) << "does not dominate this write:";
        anyFailures = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (anyFailures)
    return failure();

  return success();

  // // Gather up all operations that we'll need to move around.

  // // Establish an order among the operations such that defs dominate uses.
  // SmallVector<Operation *> order;
  // SmallVector<std::pair<Operation *, unsigned>, 0> worklist;
  // DenseSet<Operation *> finished;
  // DenseSet<Operation *> seen;

  // for (auto &op : modelOp.bodyBlock()) {
  //   if (finished.contains(&op))
  //     continue;
  //   assert(seen.empty());
  //   worklist.push_back({&op, 0});

  //   while (!worklist.empty()) {
  //     auto [op, idx] = worklist.back();
  //     ++worklist.back().second; // advance to next operand

  //     // If we have reached the end of the operand list, we have visited the
  //     // entire fanin of the operation. Finalize by removing the op from the
  //     // `seen` set (because we're no longer in its fanin) and adding it to
  //     // the `finished` set so we don't needlessly revisit.
  //     if (idx == op->getNumOperands()) {
  //       order.push_back(op);
  //       seen.erase(op);
  //       finished.insert(op);
  //       worklist.pop_back();
  //       continue;
  //     }

  //     auto operand = op->getOperand(idx);
  //     auto *def = operand.getDefiningOp();
  //     if (!def || finished.contains(def))
  //       continue;
  //     if (auto stateOp = dyn_cast<StateOp>(def);
  //         stateOp && stateOp.latency() > 0)
  //       continue;

  //     // Add the operand to the worklist. If it already exists in the `seen`
  //     // set, which indicates that we have already had to go through this
  //     // value and have arrived at it again, we have found a loop.
  //     if (!seen.insert(def).second) {
  //       auto d = def->emitError("loop detected");
  //       for (auto [op, idx] : llvm::reverse(worklist)) {
  //         d.attachNote(op->getLoc())
  //             << "through operand " << (idx - 1) << " here:";
  //         if (op == def)
  //           break;
  //       }
  //       return signalPassFailure();
  //     }
  //     worklist.push_back({def, 0});
  //   }
  // }
}

void ScheduleModelsPass::assignOperationMasks() {
  // Mark all operations already nested in one of the root clock trees with that
  // clock tree.
  unsigned numInitialOpsMarked = 0;
  for (auto pair : clockTreeIds) {
    pair.first->walk([&](Operation *op) {
      if (isa<RootInputOp, RootOutputOp, AllocStateOp, MemoryOp, MemoryReadOp,
              MemoryWriteOp>(op))
        return;
      ++numInitialOpsMarked;
      operationMasks[op] =
          APInt::getOneBitSet(clockTreeIds.size(), pair.second);
    });
  }
  LLVM_DEBUG(llvm::dbgs() << "- Marked " << numInitialOpsMarked
                          << " initial ops already in clock trees\n");

  // Perform a post-order traversal of the operations and establish the union of
  // clock trees in the user fan-out tree for each.
  SmallVector<std::tuple<Operation *, Operation::user_iterator, APInt>, 0>
      worklist;
  DenseSet<Operation *> seen;
  auto emptyMask = APInt::getZero(clockTreeIds.size());

  for (auto &op : modelOp.bodyBlock()) {
    if (isa<RootInputOp, RootOutputOp, AllocStateOp, MemoryOp>(op))
      continue;
    if (operationMasks.count(&op))
      continue;
    assert(seen.empty());
    worklist.push_back({&op, op.user_begin(), emptyMask});

    while (!worklist.empty()) {
      auto &[op, userIt, mask] = worklist.back();

      // If we have reached the end of the user list, finalize the operation by
      // computing its overall mask.
      if (userIt == op->user_end()) {
        operationMasks[op] = mask;
        seen.erase(op);
        worklist.pop_back();
        continue;
      }
      auto *user = *userIt++; // advance to next user

      // If we have reached a user that has already a final mask of clock trees
      // computed, add that mask to the entire worklist, since every entry in
      // the worklist uses it.
      auto userMaskIt = operationMasks.find(user);
      if (userMaskIt != operationMasks.end()) {
        mask |= userMaskIt->second;
        for (auto &[otherOp, otherUserIt, otherMask] : worklist)
          otherMask |= mask;
        continue;
      }

      // Otherwise break loops and add this to the worklist.
      if (!seen.insert(user).second)
        continue;
      worklist.push_back({user, user->user_begin(), emptyMask});
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "- Marked " << operationMasks.size()
                          << " ops overall\n");

  // At this point all operations should have a mask assigned.
  LLVM_DEBUG({
    modelOp->walk([&](Operation *op) {
      if (!isa<RootInputOp, RootOutputOp, AllocStateOp, MemoryOp>(op) &&
          op != modelOp)
        assert(operationMasks.find(op) != operationMasks.end() &&
               "all interesting ops should have a mask assigned");
    });
  });
}

static bool shouldReplicateOp(Operation *op) {
  return !isa<ClockTreeOp, PassThroughOp, RootInputOp, RootOutputOp,
              AllocStateOp, UpdateStateOp, MemoryOp, MemoryWriteOp>(op);
}

LogicalResult ScheduleModelsPass::replicateOpsIntoClockTrees() {
  // TODO: This is a horrible implementation that does a lot of unneccessary
  // work. It would be better to use a post-order traversal of the users of each
  // op, to ensure that all users are sunk before we even attempt to sink an op.
  unsigned numSunk = 0;
  unsigned numCloned = 0;

  SetVector<Operation *> worklist, shadowWorklist;
  for (auto &op : modelOp.bodyBlock())
    if (shouldReplicateOp(&op))
      worklist.insert(&op);
  LLVM_DEBUG(llvm::dbgs() << "- Sinking " << worklist.size()
                          << " ops into clock trees\n");

  DenseSet<Operation *> fullySunkOps;
  SmallPtrSet<Block *, 8> usedInBlocks;
  SmallDenseMap<Operation *, Block *> userBlocks;
  SmallDenseMap<Block *, Operation *> clonesByBlock;

  unsigned numIterations = 0;
  while (!worklist.empty()) {
    ++numIterations;
    std::swap(worklist, shadowWorklist);
    for (auto *op : shadowWorklist) {
      assert(shouldReplicateOp(op));
      if (fullySunkOps.contains(op))
        continue;

      // Find the clock tree blocks this operation is used in.
      usedInBlocks.clear();
      userBlocks.clear();
      for (auto *user : op->getUsers()) {
        Block *block = user->getBlock();
        while (!clockTreeIds.count(block->getParentOp()) &&
               block != &modelOp.bodyBlock())
          block = block->getParentOp()->getBlock();
        assert(block && "should stop at ModelOp body block");
        usedInBlocks.insert(block);
        userBlocks[user] = block;
      }

      // If there are uses in the op's current block, we can't sink it.
      if (usedInBlocks.contains(op->getBlock()))
        continue;

      // Sink a copy of the operation into every block we've found.
      clonesByBlock.clear();
      unsigned numLeft = usedInBlocks.size();
      for (auto *block : usedInBlocks) {
        Operation *clonedOp;
        if (--numLeft == 0) {
          // reuse original op for the last move
          op->remove();
          clonedOp = op;
        } else {
          ++numCloned;
          clonedOp = op->cloneWithoutRegions();
        }
        OpBuilder builder(block, block->begin());
        builder.insert(clonedOp);
        clonesByBlock[block] = clonedOp;
        ++numSunk;
      }

      // Update the users of the original operation to point at the copy that
      // corresponds to the user's parent block.
      for (auto result : op->getResults()) {
        for (auto &use : llvm::make_early_inc_range(result.getUses())) {
          auto *block = userBlocks[use.getOwner()];
          assert(block && "should have seen this user before");
          auto *clonedOp = clonesByBlock[block];
          assert(clonedOp && "should have a cloned op for this block");
          use.set(clonedOp->getResult(result.getResultNumber()));
        }
      }

      // If the operation has been successfully sunk into all blocks, i.e. the
      // original op no longer remains in its block, attempt to sink all its
      // operands as well.
      // if (!originalRemains) {
      fullySunkOps.insert(op);
      for (auto value : op->getOperands())
        if (auto *def = value.getDefiningOp(); def)
          if (def->getParentOp() == modelOp && !fullySunkOps.contains(def))
            if (shouldReplicateOp(def))
              worklist.insert(def);
      // }

      // for (auto value : user->getOperands()) {
      //   Operation *op = value.getDefiningOp();
      //   // Ignore block arguments and ops that are already inside the region.
      //   if (!op || op->getParentRegion() == region)
      //     continue;
      //   LLVM_DEBUG(op->print(llvm::dbgs() << "\nTry to sink:\n"));

      //   // If the op's users are all in the region and it can be moved, then
      //   do
      //   // so.
      //   if (allUsersDominatedBy(op, region) && shouldMoveIntoRegion(op,
      //   region)) {
      //     moveIntoRegion(op, region);
      //     ++numSunk;
      //     // Add the op to the work queue.
      //     worklist.push_back(op);
      //   }
      // }
    }
    shadowWorklist.clear();
  }

  // for (auto &op :
  //      llvm::make_early_inc_range(llvm::reverse(modelOp.bodyBlock()))) {
  //   if (isa<ClockTreeOp, PassThroughOp, RootInputOp, RootOutputOp>(&op))
  //     continue;

  //   // Find the clock tree blocks this operation is used in.
  //   usedInBlocks.clear();
  //   for (auto *user : op.getUsers()) {
  //     Block *block = user->getParent();
  //     while (block && !clockTreeIds.count(block->getParentOp()))
  //       block = block->getParentOp()->getParent();
  //     if (!block)
  //   }
  // }

  // DominanceInfo domInfo;
  // modelOp.walk<WalkOrder::PreOrder>([&](Region *region) {
  //   numSunk += controlFlowSink(
  //       region, domInfo,
  //       [](auto *op, auto *region) {
  //         return !isa<ClockTreeOp, PassThroughOp, RootInputOp, RootOutputOp>(
  //                    op) &&
  //                isa<ClockTreeOp, PassThroughOp>(region->getParentOp());
  //       },
  //       [](auto *op, auto *region) {
  //         auto &block = region->front();
  //         op->moveBefore(&block, block.begin());
  //       });
  // });

  LLVM_DEBUG(llvm::dbgs() << "- Sunk " << numSunk
                          << " ops into clock trees, created " << numCloned
                          << " duplicates, finished in " << numIterations
                          << " iterations\n");

#ifndef NDEBUG
  // There shouldn't really be anything left on the top-level.
  for (auto &op : modelOp.bodyBlock()) {
    if (!shouldReplicateOp(&op))
      continue;
    bool allNotInTop = llvm::all_of(op.getUsers(), [&](auto *user) {
      return user->getParentOp() != modelOp;
    });
    if (allNotInTop) {
      auto d = op.emitOpError("left in top-level despite all uses not in top");
      SmallPtrSet<Operation *, 8> reported;
      for (auto *user : op.getUsers())
        if (auto *parent = user->getParentOp(); reported.insert(parent).second)
          d.attachNote(parent->getLoc()) << "used inside here:";
      return failure();
    }
  }
#endif

  // Now that every clock tree has its own copy of an operation, try to sink
  // operations further into nested clock trees.
  DominanceInfo domInfo;
  unsigned numSunkIntoSubtrees = 0;
  modelOp.walk([&](Region *region) {
    numSunkIntoSubtrees += controlFlowSink(
        region, domInfo,
        [](auto *op, auto *region) {
          return !isa<ClockTreeOp, PassThroughOp, RootInputOp, RootOutputOp>(
                     op) &&
                 isa<ClockTreeOp, PassThroughOp>(region->getParentOp());
        },
        [](auto *op, auto *region) {
          auto &block = region->front();
          op->moveBefore(&block, block.begin());
        });
  });
  LLVM_DEBUG(llvm::dbgs() << "- Sunk " << numSunkIntoSubtrees
                          << " ops further into subtrees\n");

  return success();
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createScheduleModelsPass() {
  return std::make_unique<ScheduleModelsPass>();
}
} // namespace arc
} // namespace circt
