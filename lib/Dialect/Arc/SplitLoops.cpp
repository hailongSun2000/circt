//===- SplitLoops.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-loops"

using namespace circt;
using namespace arc;
using namespace hw;

using llvm::SmallSetVector;

namespace {
struct DefInfo {
  DefineOp defOp;
  /// For each output a mask of inputs that it depends on.
  SmallVector<APInt> outputDeps;
  /// Uses of this arc.
  SmallVector<StateOp> uses;
};

struct SplitLoopsPass : public SplitLoopsBase<SplitLoopsPass> {
  using ConflictSet = SmallSetVector<APInt, 1>;
  using ConflictVector = ConflictSet::vector_type;

  void runOnOperation() override;
  void analyzeArcDef(DefineOp defOp);
  void analyzeModule(HWModuleOp moduleOp);
  LogicalResult splitLoopyArc(DefineOp defOp, ConflictSet &loopyResults);
  APInt findResultsToSplit(DefineOp defOp, ConflictVector &conflicts);
  LogicalResult splitArcDefinition(DefineOp defOp, APInt resultsToSplit);
  LogicalResult ensureNoLoops();

  Namespace arcNamespace;

  /// The key is the name of an arc definition.
  DenseMap<StringAttr, DefInfo> defInfo;

  /// A map of arc results that are involved in a loop. If a loop passes through
  /// the same arc through multiple results, this combination of results
  /// represented as a bit mask is added to the set. The key is the name of an
  /// arc definition.
  DenseMap<StringAttr, ConflictSet> loopyResults;

  /// The operations in the current arc to be split, grouped by the split
  /// results that the operations contribute to.
  DenseMap<Operation *, APInt> splitMasks;

  /// All arc uses in the design.
  DenseSet<StateOp> allArcUses;
};
} // namespace

void SplitLoopsPass::runOnOperation() {
  auto module = getOperation();
  arcNamespace.clear();
  defInfo.clear();
  loopyResults.clear();

  // Establish the output-to-input dependencies of all arc definitions.
  for (auto defOp : module.getOps<DefineOp>()) {
    arcNamespace.newName(defOp.sym_name());
    analyzeArcDef(defOp);
  }

  // Find loops in the modules.
  for (auto moduleOp : module.getOps<HWModuleOp>())
    analyzeModule(moduleOp);

#if 0
  // Split the arcs involved in loops.
  for (auto defOp : llvm::make_early_inc_range(module.getOps<DefineOp>()))
    if (auto it = loopyResults.find(defOp.sym_nameAttr());
        it != loopyResults.end())
      if (failed(splitLoopyArc(defOp, it->second)))
        return signalPassFailure();
#else
  // Split all arcs with more than one result.
  // TODO: This is ugly and we should only split arcs that are truly involved in
  // a loop. But this requires some more work on the loop detection above.
  for (auto defOp : llvm::make_early_inc_range(module.getOps<DefineOp>())) {
    if (auto num = defOp.getNumResults(); num > 1) {
      ConflictSet set;
      set.insert(APInt::getAllOnes(num));
      if (failed(splitLoopyArc(defOp, set)))
        return signalPassFailure();
    }
  }
#endif

  // Ensure that there are no loops through arcs remaining.
  if (failed(ensureNoLoops()))
    return signalPassFailure();
}

/// Populate the `depInfo` data structure with initial output-to-input
/// dependencies of each arc definition.
void SplitLoopsPass::analyzeArcDef(DefineOp defOp) {
  auto &info = defInfo[defOp.sym_nameAttr()];
  info.defOp = defOp;

  unsigned numInputs = defOp.getNumArguments();
  SmallDenseMap<Value, APInt> masks;
  for (auto blockArg : defOp.getArguments())
    masks.insert(
        {blockArg, APInt::getOneBitSet(numInputs, blockArg.getArgNumber())});

  for (auto &op : defOp.bodyBlock()) {
    if (op.getNumResults() == 0)
      continue;
    APInt mask = APInt::getZero(numInputs);
    for (auto operand : op.getOperands())
      mask |= masks.lookup(operand);
    for (auto result : op.getResults())
      masks[result] = mask;
  }

  auto outputOp = cast<arc::OutputOp>(defOp.bodyBlock().getTerminator());
  for (auto output : outputOp->getOperands())
    info.outputDeps.push_back(masks.lookup(output));
}

/// Find latency zero loops within modules.
void SplitLoopsPass::analyzeModule(HWModuleOp moduleOp) {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing " << moduleOp.moduleNameAttr() << "\n");

  SmallVector<std::pair<Value, unsigned>, 0> worklist;
  DenseSet<Value> finished;
  DenseSet<Value> seen;
  DenseMap<StateOp, APInt> seenStateResults;

  auto isInteresting = [&](Value value) {
    if (auto stateOp = value.getDefiningOp<StateOp>())
      if (stateOp.latency() > 0)
        return false;
    return !finished.contains(value);
  };

  for (auto &op : *moduleOp.getBodyBlock()) {
    if (auto stateOp = dyn_cast<StateOp>(&op)) {
      defInfo[stateOp.arcAttr().getAttr()].uses.push_back(stateOp);
      allArcUses.insert(stateOp);
    }
    if (op.getNumOperands() == 0)
      continue;
    for (auto result : op.getResults()) {
      assert(seen.empty());
      if (isInteresting(result) && seen.insert(result).second)
        worklist.push_back({result, 0});

      while (!worklist.empty()) {
        auto [value, idx] = worklist.back();
        ++worklist.back().second; // advance to next index

        auto result = value.cast<OpResult>();
        auto *op = result.getOwner();
        assert(op);

        // If we have just started working on paths through an arc, mark the
        // specific arc result through which we came as visited. This allows us
        // to track if we loop through the same arc multiple times through
        // different results, which is what we want to know.
        if (auto stateOp = dyn_cast<StateOp>(op); stateOp && idx == 0) {
          auto &seenResults =
              seenStateResults.try_emplace(stateOp, stateOp.getNumResults(), 0)
                  .first->second;
          assert(!seenResults[result.getResultNumber()]);
          seenResults.setBit(result.getResultNumber());
          if (seenResults.countPopulation() > 1)
            loopyResults[stateOp.arcAttr().getAttr()].insert(seenResults);
        }

        // If we have reached the end of the operand list, we have visited the
        // entire fanin of the operation. Finalize by removing the op from the
        // `seen` set (because we're no longer in its fanin) and adding it to
        // the `finished` set so we don't needlessly revisit.
        if (idx == op->getNumOperands()) {
          if (auto stateOp = dyn_cast<StateOp>(op)) {
            auto &seenResults = seenStateResults[stateOp];
            assert(seenResults[result.getResultNumber()]);
            seenResults.clearBit(result.getResultNumber());
            if (seenResults.isZero())
              seenStateResults.erase(stateOp);
          }
          seen.erase(value);
          finished.insert(value);
          worklist.pop_back();
          continue;
        }

        auto operand = op->getOperand(idx);
        if (!isInteresting(operand))
          continue;

        // Don't add the operand to the worklist if it's a block argument or the
        // defining operation has no operands that we could trace through.
        auto *def = operand.getDefiningOp();
        if (!def || def->getNumOperands() == 0)
          continue;

        if (auto stateOp = dyn_cast<StateOp>(op)) {
          auto &info = defInfo[stateOp.arcAttr().getAttr()];
          if (!info.outputDeps[result.getResultNumber()][idx])
            continue;
        }

        // Add the operand to the worklist. If it already exists in the `seen`
        // set, which indicates that we have already had to go through this
        // value and have arrived at it again, we have found a loop.
        if (!seen.insert(operand).second) {
          // for (auto [value, idx] : worklist) {
          //   if (auto stateOp = value.getDefiningOp<StateOp>())
          //     loopyResults[stateOp.arcAttr().getAttr()].insert(
          //         seenStateResults.lookup(stateOp));
          //   if (value == operand)
          //     break;
          // }
          continue;
          if (isa<StateOp>(op)) {
            auto d = def->emitRemark("zero latency loop detected in result ")
                     << idx;
            for (auto [value, idx] : llvm::reverse(worklist)) {
              d.attachNote(value.getLoc())
                  << "through operand " << (idx - 1) << " here:";
              if (value == operand)
                break;
            }
          }
          continue;
          return signalPassFailure();
        }
        worklist.push_back({operand, 0});
      }
    }
  }
}

LogicalResult SplitLoopsPass::splitLoopyArc(DefineOp defOp,
                                            ConflictSet &loopyResults) {
  auto conflicts = loopyResults.takeVector();
  LLVM_DEBUG({
    llvm::dbgs() << "Splitting arc @" << defOp.sym_name() << " with "
                 << conflicts.size() << " loops\n";
  });
  auto resultsToSplit = findResultsToSplit(defOp, conflicts);

  // Mark all operations in the arc with the results they will contribute to. We
  // do this through a bit mask, with one bit for every result being split and
  // an additional bit for all other results (the ones that remain with the
  // current arc).
  unsigned numSplits = resultsToSplit.countPopulation() + 1;
  splitMasks.clear();
  for (auto &op : llvm::reverse(defOp.bodyBlock())) {
    if (auto outputOp = dyn_cast<arc::OutputOp>(&op)) {
      unsigned split = 1;
      for (auto &operand : outputOp->getOpOperands()) {
        unsigned bit = resultsToSplit[operand.getOperandNumber()] ? split++ : 0;
        if (auto *definingOp = operand.get().getDefiningOp()) {
          auto &mask =
              splitMasks.try_emplace(definingOp, numSplits, 0).first->second;
          mask.setBit(bit);
        }
      }
    } else {
      auto mask = splitMasks.lookup(&op);
      for (auto operand : op.getOperands())
        if (auto *definingOp = operand.getDefiningOp())
          splitMasks.try_emplace(definingOp, numSplits, 0).first->second |=
              mask;
    }
  }

  // Split the arc definition.
  return splitArcDefinition(defOp, resultsToSplit);
}

/// Figure out which results to split off from an arc. We do this iteratively.
/// In every iteration we find the result that appears most frequently in the
/// masks, mark it as to be split, and remove it from the masks.
APInt SplitLoopsPass::findResultsToSplit(DefineOp defOp,
                                         ConflictVector &conflicts) {
  auto numResults = defOp.getNumResults();
  auto resultsToSplit = APInt::getZero(numResults);
  SmallVector<uint8_t> numAppearances(numResults);

  for (auto conflict : conflicts)
    resultsToSplit |= conflict;
  for (unsigned i = 0; i < numResults; ++i)
    if (resultsToSplit[i])
      LLVM_DEBUG(llvm::dbgs() << "- Split result " << i << "\n");
  return resultsToSplit;

  while (!conflicts.empty()) {
    // Count the number of times each result appears in a mask.
    for (auto &num : numAppearances)
      num = 0;
    for (auto &conflict : conflicts)
      for (unsigned resultIdx = 0; resultIdx < numResults; ++resultIdx)
        if (conflict[resultIdx])
          if (numAppearances[resultIdx] < UINT8_MAX)
            ++numAppearances[resultIdx];

    // Find the result with the highest number of appearances.
    unsigned resultToSplit = 0;
    for (unsigned resultIdx = 1; resultIdx < numResults; ++resultIdx)
      if (numAppearances[resultIdx] > numAppearances[resultToSplit])
        resultToSplit = resultIdx;
    resultsToSplit.setBit(resultToSplit);

    // Remove the result from the conflict sets and remove sets that contain
    // fewer than two results.
    unsigned idxOut = 0;
    unsigned numResolved = 0;
    for (auto &conflict : conflicts) {
      conflict.clearBit(resultToSplit);
      if (conflict.countPopulation() < 2) {
        ++numResolved;
        continue;
      }
      conflicts[idxOut++] = conflict;
    }
    conflicts.truncate(idxOut);
    LLVM_DEBUG(llvm::dbgs() << "- Split result " << resultToSplit
                            << " (resolves " << numResolved << " loops)\n");
  }
  return resultsToSplit;
}

namespace {
struct SplitArc {
  explicit SplitArc(unsigned id, const APInt &mask)
      : id(id), mask(mask), block(new Block) {}
  unsigned id;
  APInt mask;
  std::unique_ptr<Block> block;
  SmallDenseMap<Value, Value, 8> valueMapping;
  SmallSetVector<Value, 8> outputs;
  SmallVector<unsigned> newToOldOutputMapping;
  /// The original value that each block argument of the new arc maps to.
  SmallVector<Value> inputMapping;
  /// The arc definition split off from the original one.
  DefineOp defOp;
};
} // namespace

/// Split the given `DefineOp` according to the `splitMaskGroups`.
LogicalResult SplitLoopsPass::splitArcDefinition(DefineOp defOp,
                                                 APInt resultsToSplit) {
  // A dense set of IDs for each new arc.
  SmallDenseMap<APInt, std::unique_ptr<SplitArc>> arcsByMask;
  SmallVector<SplitArc *> arcs;
  auto getArcForMask = [&](const APInt &mask) -> SplitArc & {
    if (auto it = arcsByMask.find(mask); it != arcsByMask.end())
      return *it->second.get();
    auto *arc =
        arcsByMask
            .insert({mask, std::make_unique<SplitArc>(arcsByMask.size(), mask)})
            .first->second.get();
    arcs.emplace_back(arc);
    LLVM_DEBUG({
      SmallString<16> maskStr;
      mask.toStringUnsigned(maskStr, 2);
      llvm::dbgs() << "- [Arc " << arc->id << "] Allocated for mask " << maskStr
                   << "\n";
    });
    return *arc;
  };

  // Ensure that a `value` can be used from a given `arc`. This creates
  // additional inputs on the arc if necessary, e.g. if the value is defined by
  // an op in a different arc. Returns the value within the arc that represents
  // the original value.
  auto mapValueIntoArc = [&](Value value, SplitArc &arc) -> Value {
    if (auto mapping = arc.valueMapping.lookup(value))
      return mapping;

    if (auto oldArg = value.dyn_cast<BlockArgument>()) {
      auto newArg = arc.block->addArgument(oldArg.getType(), oldArg.getLoc());
      arc.inputMapping.push_back(oldArg);
      arc.valueMapping.insert({oldArg, newArg});
      LLVM_DEBUG(llvm::dbgs()
                 << "- [Arc " << arc.id << "] New input "
                 << newArg.getArgNumber() << " -> old input "
                 << oldArg.getArgNumber() << " (" << newArg.getType() << ")\n");
      return newArg;
    }

    auto *valueDef = value.getDefiningOp();
    assert(valueDef);
    auto valueMask = splitMasks.lookup(valueDef);
    if (arc.mask == valueMask)
      return {};

    auto &valueArc = getArcForMask(valueMask);
    auto valueMapped = valueArc.valueMapping.lookup(value);
    if (!valueMapped)
      valueMapped = value;
    valueArc.outputs.insert(valueMapped);
    auto newArg = arc.block->addArgument(value.getType(), value.getLoc());
    arc.inputMapping.push_back(valueMapped);
    arc.valueMapping.insert({value, newArg});
    LLVM_DEBUG(llvm::dbgs()
               << "- [Arc " << arc.id << "] New input " << newArg.getArgNumber()
               << " -> value from arc " << valueArc.id << " ("
               << newArg.getType() << ")\n");
    return newArg;
  };

  // Distribute the operations in the old arc definition into the new split
  // arcs.
  OpBuilder builder(defOp);
  SmallVector<Value> replacementOutputsAfterSplit;
  for (auto &op : defOp.bodyBlock()) {
    // Output values of the original arc become output values of the respective
    // new arcs where the defining operations are located in.
    if (isa<arc::OutputOp>(&op)) {
      unsigned numSplits = resultsToSplit.countPopulation() + 1;
      unsigned split = 1;
      for (auto &operand : op.getOpOperands()) {
        unsigned bit = resultsToSplit[operand.getOperandNumber()] ? split++ : 0;
        auto mask = APInt::getOneBitSet(numSplits, bit);
        auto &arc = getArcForMask(mask);
        auto mapping = mapValueIntoArc(operand.get(), arc);
        if (!mapping)
          mapping = operand.get();
        arc.outputs.insert(mapping);
        arc.newToOldOutputMapping.push_back(operand.getOperandNumber());
        replacementOutputsAfterSplit.push_back(mapping);
      }
      continue;
    }

    // Determine the arc this operation will go to.
    auto mask = splitMasks.lookup(&op);
    auto &arc = getArcForMask(mask);

    // Clone the operation.
    auto *newOp = op.cloneWithoutRegions();
    builder.setInsertionPointToEnd(arc.block.get());
    builder.insert(newOp);

    // Update the operands of this operation.
    for (auto &operand : newOp->getOpOperands()) {
      auto mapping = mapValueIntoArc(operand.get(), arc);
      if (mapping)
        operand.set(mapping);
    }
    for (auto [oldResult, newResult] :
         llvm::zip(op.getResults(), newOp->getResults()))
      arc.valueMapping.insert({oldResult, newResult});
  }

  // Build the replacement arc definitions.
  for (auto *arc : arcs) {
    LLVM_DEBUG(llvm::dbgs() << "- Finalizing arc " << arc->id << "\n");
    builder.setInsertionPointToEnd(arc->block.get());
    builder.create<arc::OutputOp>(defOp.getLoc(), arc->outputs.getArrayRef());

    SmallVector<Type> inputTypes;
    SmallVector<Type> outputTypes;
    for (auto arg : arc->block->getArguments())
      inputTypes.push_back(arg.getType());
    for (auto result : arc->outputs)
      outputTypes.push_back(result.getType());

    // Create the arc definition.
    builder.setInsertionPoint(defOp);
    arc->defOp = builder.create<DefineOp>(
        defOp.getLoc(),
        builder.getStringAttr(arcNamespace.newName(defOp.sym_name() +
                                                   "_split_" + Twine(arc->id))),
        builder.getFunctionType(inputTypes, outputTypes));
    arc->defOp.body().push_back(arc->block.release());
  }

  // Replace uses of the original arc with the split arcs.
  for (auto oldStateOp : defInfo[defOp.sym_nameAttr()].uses) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Updating use " << oldStateOp.getAsOpaquePointer() << "\n");
    builder.setInsertionPoint(oldStateOp);

    DenseMap<Value, Value> valueMapping;
    for (auto [useOperand, defArg] :
         llvm::zip(oldStateOp.operands(), defOp.getArguments()))
      valueMapping.insert({defArg, useOperand});

    BackedgeBuilder backedgeBuilder(builder, oldStateOp.getLoc());
    SmallDenseMap<Value, Backedge> backedges;

    ArrayRef<Attribute> oldNames;
    if (auto names = oldStateOp->getAttrOfType<ArrayAttr>("names"))
      oldNames = names.getValue();

    for (auto *arc : arcs) {
      // Determine if this is a prefix arc that will have latency 0, or if this
      // represents a portion of the original state op and must inherit the
      // original latency.
      bool isPrefix = arc->mask.countPopulation() > 1;
      LLVM_DEBUG(llvm::dbgs() << "  - Using @" << arc->defOp.sym_name()
                              << (isPrefix ? " (prefix)" : "") << "\n");

      // Determine the operands for the new arc.
      SmallVector<Value> inputs;
      for (auto oldValue : arc->inputMapping) {
        auto &mapping = valueMapping[oldValue];
        if (!mapping) {
          auto backedge =
              backedgeBuilder.get(oldValue.getType(), oldValue.getLoc());
          backedges.insert({oldValue, backedge});
          mapping = backedge;
        }
        inputs.push_back(mapping);
      }

      // Instantiate the new arc.
      auto newStateOp =
          builder.create<StateOp>(oldStateOp.getLoc(), arc->defOp,
                                  isPrefix ? Value{} : oldStateOp.clock(),
                                  isPrefix ? Value{} : oldStateOp.enable(),
                                  isPrefix ? 0 : oldStateOp.latency(), inputs);
      if (!isPrefix && !oldNames.empty()) {
        SmallVector<Attribute> newNames;
        for (auto oldNameIdx : arc->newToOldOutputMapping)
          newNames.push_back(oldNames[oldNameIdx]);
        newStateOp->setAttr("names", builder.getArrayAttr(newNames));
      }
      allArcUses.insert(newStateOp);

      // Track the outputs produced by this arc and resolve backedges.
      for (auto [useOutput, defOutput] :
           llvm::zip(newStateOp.getResults(), arc->outputs)) {
        valueMapping[defOutput] = useOutput;
        auto it = backedges.find(defOutput);
        if (it != backedges.end()) {
          it->second.setValue(useOutput);
          backedges.erase(it);
        }
      }
    }

    // At this point all backedges should have been resolved.
    if (failed(backedgeBuilder.clearOrEmitError()))
      return failure();

    // Replace uses of the old arc results with the new ones.
    for (auto [oldUseResult, newDefOutput] :
         llvm::zip(oldStateOp.getResults(), replacementOutputsAfterSplit)) {
      auto mapping = valueMapping.lookup(newDefOutput);
      assert(
          mapping &&
          "an output of the original arc is no longer produced by any new arc");
      oldUseResult.replaceAllUsesWith(mapping);
    }

    // Remove the original use.
    allArcUses.erase(oldStateOp);
    oldStateOp.erase();
  }

  // Remove the original arc definition.
  defOp.erase();
  return success();
}

/// Check that there are no more zero-latency loops through arcs.
LogicalResult SplitLoopsPass::ensureNoLoops() {
  SmallVector<std::pair<Operation *, unsigned>, 0> worklist;
  DenseSet<Operation *> finished;
  DenseSet<Operation *> seen;
  for (auto op : allArcUses) {
    if (finished.contains(op))
      continue;
    assert(seen.empty());
    worklist.push_back({op, 0});
    while (!worklist.empty()) {
      auto [op, idx] = worklist.back();
      ++worklist.back().second;
      if (idx == op->getNumOperands()) {
        seen.erase(op);
        finished.insert(op);
        worklist.pop_back();
        continue;
      }
      auto operand = op->getOperand(idx);
      auto *def = operand.getDefiningOp();
      if (!def || finished.contains(def))
        continue;
      if (auto stateOp = dyn_cast<StateOp>(def);
          stateOp && stateOp.latency() > 0)
        continue;
      if (!seen.insert(def).second) {
        auto d = def->emitError(
            "loop splitting did not eliminate all loops; loop detected");
        for (auto [op, idx] : llvm::reverse(worklist)) {
          d.attachNote(op->getLoc())
              << "through operand " << (idx - 1) << " here:";
          if (op == def)
            break;
        }
        return failure();
      }
      worklist.push_back({def, 0});
    }
  }

  return success();
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createSplitLoopsPass() {
  return std::make_unique<SplitLoopsPass>();
}
} // namespace arc
} // namespace circt
