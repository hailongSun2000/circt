//===- Dedup.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/SHA256.h"

#define DEBUG_TYPE "arc-dedup"

using namespace circt;
using namespace arc;
using namespace hw;
using llvm::SetVector;
using llvm::SmallDenseSet;
using llvm::SmallMapVector;
using llvm::SmallSetVector;

namespace {
struct StructuralHash {
  using Hash = std::array<uint8_t, 32>;
  Hash hash;
  Hash constInvariant; // a hash that ignores constants
};

struct StructuralHasher {
  explicit StructuralHasher(MLIRContext *context) {}

  StructuralHash hash(DefineOp arc) {
    reset();
    update(arc);
    return StructuralHash{state.final(), stateConstInvariant.final()};
  }

private:
  void reset() {
    currentIndex = 0;
    disableConstInvariant = 0;
    indices.clear();
    state.init();
    stateConstInvariant.init();
  }

  void update(const void *pointer) {
    auto *addr = reinterpret_cast<const uint8_t *>(&pointer);
    state.update(ArrayRef(addr, sizeof pointer));
    if (disableConstInvariant == 0)
      stateConstInvariant.update(ArrayRef(addr, sizeof pointer));
  }

  void update(size_t value) {
    auto *addr = reinterpret_cast<const uint8_t *>(&value);
    state.update(ArrayRef(addr, sizeof value));
    if (disableConstInvariant == 0)
      stateConstInvariant.update(ArrayRef(addr, sizeof value));
  }

  void update(TypeID typeID) { update(typeID.getAsOpaquePointer()); }

  void update(Type type) { update(type.getAsOpaquePointer()); }

  void update(Attribute attr) { update(attr.getAsOpaquePointer()); }

  void update(mlir::OperationName name) { update(name.getAsOpaquePointer()); }

  void update(BlockArgument arg) {
    indices[arg] = currentIndex++;
    update(arg.getType());
  }

  void update(OpResult result) {
    indices[result] = currentIndex++;
    update(result.getType());
  }

  void update(OpOperand &operand) {
    // We hash the value's index as it apears in the block.
    auto it = indices.find(operand.get());
    assert(it != indices.end() && "op should have been previously hashed");
    update(it->second);
  }

  void update(Block &block) {
    for (auto arg : block.getArguments())
      update(arg);
    for (auto &op : block)
      update(&op);
  }

  void update(Operation *op) {
    update(op->getName());

    // Hash the attributes. (Excluded in constant invariant hash.)
    for (auto namedAttr : op->getAttrDictionary()) {
      auto name = namedAttr.getName();
      auto value = namedAttr.getValue();
      // Skip names.
      if (name == "sym_name")
        continue;

      // Hash the interned pointer.
      unsigned skipConstInvariant = isa<ConstantOp>(op) && name == "value";
      disableConstInvariant += skipConstInvariant;
      update(name.getAsOpaquePointer());
      update(value.getAsOpaquePointer());
      disableConstInvariant -= skipConstInvariant;
    }

    // Hash the operands.
    for (auto &operand : op->getOpOperands())
      update(operand);
    // Hash the regions. We need to make sure an empty region doesn't hash the
    // same as no region, so we include the number of regions.
    update(op->getNumRegions());
    for (auto &region : op->getRegions())
      for (auto &block : region.getBlocks())
        update(block);
    // Record any op results.
    for (auto result : op->getResults())
      update(result);
  }

  // Every value is assigned a unique id based on their order of appearance.
  unsigned currentIndex = 0;
  DenseMap<Value, unsigned> indices;

  unsigned disableConstInvariant = 0;

  // This is the actual running hash calculation. This is a stateful element
  // that should be reinitialized after each hash is produced.
  llvm::SHA256 state;
  llvm::SHA256 stateConstInvariant;
};
} // namespace

namespace {
struct StructuralEquivalence {
  using ValuePair = std::pair<Value, Value>;
  explicit StructuralEquivalence(MLIRContext *context) {}

  void check(DefineOp arcA, DefineOp arcB) {
    if (!checkImpl(arcA, arcB)) {
      match = false;
      matchConstInvariant = false;
    }
  }

  SmallSetVector<ValuePair, 1> divergences;
  bool match;
  bool matchConstInvariant;

private:
  bool checkImpl(DefineOp arcA, DefineOp arcB) {
    worklist.clear();
    divergences.clear();
    match = true;
    matchConstInvariant = true;
    handled.clear();

    if (arcA.getFunctionType() != arcB.getFunctionType())
      return false;

    auto outputA = cast<arc::OutputOp>(arcA.bodyBlock().getTerminator());
    auto outputB = cast<arc::OutputOp>(arcB.bodyBlock().getTerminator());
    for (auto [a, b] : llvm::zip(outputA.getOperands(), outputB.getOperands()))
      worklist.emplace_back(a, b);

    while (!worklist.empty()) {
      ValuePair values = worklist.back();
      if (handled.contains(values)) {
        worklist.pop_back();
        continue;
      }
      if (values.first.getType() != values.second.getType())
        return false;

      auto *opA = values.first.getDefiningOp();
      auto *opB = values.second.getDefiningOp();
      if (!opA || !opB) {
        auto argA = values.first.dyn_cast<BlockArgument>();
        auto argB = values.second.dyn_cast<BlockArgument>();
        if (argA && argB) {
          if (argA.getArgNumber() != argB.getArgNumber())
            return false;
        } else {
          assert(argA || argB);
          divergences.insert(values);
        }
        handled.insert(values);
        worklist.pop_back();
        continue;
      }

      bool allHandled = true;
      for (auto [operandA, operandB] :
           llvm::zip(opA->getOperands(), opB->getOperands())) {
        if (!handled.count({operandA, operandB})) {
          worklist.push_back({operandA, operandB});
          allHandled = false;
        }
      }
      if (!allHandled)
        continue;
      handled.insert(values);
      worklist.pop_back();

      if (opA->getName() != opB->getName())
        return false;
      if (opA->getAttrDictionary() != opB->getAttrDictionary()) {
        for (auto [namedAttrA, namedAttrB] :
             llvm::zip(opA->getAttrDictionary(), opB->getAttrDictionary())) {
          if (namedAttrA.getName() != namedAttrB.getName())
            return false;
          auto name = namedAttrA.getName();
          if (namedAttrA.getValue() == namedAttrB.getValue())
            continue;
          bool mayDiverge = isa<ConstantOp>(opA) && name == "value";
          if (!mayDiverge)
            return false;
          divergences.insert(values);
          match = false;
        }
      }
    }

    return true;
  }

  SmallVector<ValuePair, 0> worklist;
  DenseSet<ValuePair> handled;
};
} // namespace

static void addCallSiteOperands(MutableArrayRef<StateOp> callSites,
                                ArrayRef<ConstantOp> operands) {
  SmallVector<Value> newOperands;
  for (auto &stateOp : callSites) {
    OpBuilder builder(stateOp);
    newOperands = stateOp.operands();
    for (auto constOp : operands) {
      auto *newOp = constOp->clone();
      builder.insert(newOp);
      newOperands.push_back(newOp->getResult(0));
    }
    auto newStateOp = builder.create<StateOp>(
        stateOp.getLoc(), stateOp.arcAttr(), stateOp.getResultTypes(),
        stateOp.clock(), stateOp.enable(), stateOp.latency(), newOperands);
    if (auto names = stateOp->getAttr("names"))
      newStateOp->setAttr("names", names);
    stateOp.replaceAllUsesWith(newStateOp);
    stateOp->erase();
    stateOp = newStateOp;
  }
}

namespace {
struct DedupPass : public DedupBase<DedupPass> {
  void runOnOperation() override;
  void replaceArcWith(DefineOp oldArc, DefineOp newArc);

  /// A mapping from arc names to arc definitions.
  DenseMap<StringAttr, DefineOp> arcByName;
  /// A mapping from arc definitions to call sites.
  DenseMap<DefineOp, SmallVector<StateOp, 1>> callSites;
};
} // namespace

void DedupPass::runOnOperation() {
  arcByName.clear();
  callSites.clear();

  // Compute the structural hash for each arc definition.
  SmallVector<std::pair<DefineOp, StructuralHash>> arcHashes;
  StructuralHasher hasher(&getContext());
  for (auto defineOp : getOperation().getOps<DefineOp>()) {
    arcHashes.emplace_back(defineOp, hasher.hash(defineOp));
    arcByName.insert({defineOp.sym_nameAttr(), defineOp});
  }

  // Collect the arc call sites.
  getOperation().walk([&](Operation *op) {
    if (auto defineOp = dyn_cast<DefineOp>(op))
      return WalkResult::skip();
    if (auto stateOp = dyn_cast<StateOp>(op))
      callSites[arcByName.lookup(stateOp.arcAttr().getAttr())].push_back(
          stateOp);
    return WalkResult::advance();
  });

  // Perform deduplications that do not require modification of the arc call
  // sites. (No additional ports.)
  llvm::sort(arcHashes,
             [](auto a, auto b) { return a.second.hash < b.second.hash; });
  LLVM_DEBUG(llvm::dbgs() << "Check for exact merges (" << arcHashes.size()
                          << " arcs)\n");
  StructuralEquivalence equiv(&getContext());
  for (unsigned arcIdx = 0, arcEnd = arcHashes.size(); arcIdx != arcEnd;
       ++arcIdx) {
    auto [defineOp, hash] = arcHashes[arcIdx];
    if (!defineOp)
      continue;
    for (unsigned otherIdx = arcIdx + 1; otherIdx != arcEnd; ++otherIdx) {
      auto [otherDefineOp, otherHash] = arcHashes[otherIdx];
      if (hash.hash != otherHash.hash)
        break;
      if (!otherDefineOp)
        continue;
      equiv.check(defineOp, otherDefineOp);
      if (!equiv.match)
        continue;
      LLVM_DEBUG(llvm::dbgs() << "- Merge " << defineOp.sym_nameAttr() << " <- "
                              << otherDefineOp.sym_nameAttr() << "\n");
      replaceArcWith(otherDefineOp, defineOp);
      arcHashes[otherIdx].first = {};
    }
  }

  // Perform deduplication of arcs that differ only in constant values.
  llvm::sort(arcHashes, [](auto a, auto b) {
    if (!a.first && !b.first)
      return false;
    if (!a.first)
      return false;
    if (!b.first)
      return true;
    return a.second.constInvariant < b.second.constInvariant;
  });
  while (!arcHashes.empty() && !arcHashes.back().first)
    arcHashes.pop_back();
  LLVM_DEBUG(llvm::dbgs() << "Check for constant-agnostic merges ("
                          << arcHashes.size() << " arcs)\n");
  for (unsigned arcIdx = 0, arcEnd = arcHashes.size(); arcIdx != arcEnd;
       ++arcIdx) {
    auto [defineOp, hash] = arcHashes[arcIdx];
    if (!defineOp)
      continue;

    // Perform a prepass to find the constants to be outlined.
    SmallSetVector<ConstantOp, 2> outlinedConsts;
    for (unsigned otherIdx = arcIdx + 1; otherIdx != arcEnd; ++otherIdx) {
      auto [otherDefineOp, otherHash] = arcHashes[otherIdx];
      if (hash.constInvariant != otherHash.constInvariant)
        break;
      if (!otherDefineOp)
        continue;
      equiv.check(defineOp, otherDefineOp);
      if (!equiv.matchConstInvariant)
        continue;
      for (auto [value, otherValue] : equiv.divergences)
        if (auto constOp = value.getDefiningOp<ConstantOp>())
          outlinedConsts.insert(constOp);
    }
    if (outlinedConsts.empty())
      continue;
    LLVM_DEBUG(llvm::dbgs()
               << "- Outlining " << outlinedConsts.size() << " consts from "
               << defineOp.sym_nameAttr() << "\n");

    // Add additional ports to the arc and replace the outlined constants.
    unsigned newArgOffset = defineOp.getNumArguments();
    SmallVector<Type> newInputTypes;
    for (auto type : defineOp.getFunctionType().getInputs())
      newInputTypes.push_back(type);
    SmallVector<ConstantOp> callSiteOps;
    for (auto constOp : outlinedConsts) {
      auto arg =
          defineOp.bodyBlock().addArgument(constOp.getType(), constOp.getLoc());
      constOp.replaceAllUsesWith(arg);
      newInputTypes.push_back(arg.getType());
      callSiteOps.push_back(constOp);
    }
    addCallSiteOperands(callSites[defineOp], callSiteOps);
    auto newArcType = FunctionType::get(
        &getContext(), newInputTypes, defineOp.getFunctionType().getResults());

    // Perform the actual deduplication with other arcs.
    for (unsigned otherIdx = arcIdx + 1; otherIdx != arcEnd; ++otherIdx) {
      auto [otherDefineOp, otherHash] = arcHashes[otherIdx];
      if (hash.constInvariant != otherHash.constInvariant)
        break;
      if (!otherDefineOp)
        continue;

      // Check for structural equivalence between the two arcs.
      equiv.check(defineOp, otherDefineOp);
      if (!equiv.matchConstInvariant)
        continue;

      if (equiv.divergences.size() != outlinedConsts.size())
        continue;

      callSiteOps.clear();
      callSiteOps.resize(outlinedConsts.size());
      if (!llvm::all_of(equiv.divergences, [&](auto div) {
            auto [value, otherValue] = div;
            auto arg = value.template dyn_cast<BlockArgument>();
            if (!arg || arg.getArgNumber() < newArgOffset)
              return false;
            auto otherConstOp = otherValue.template getDefiningOp<ConstantOp>();
            if (!otherConstOp)
              return false;
            callSiteOps[arg.getArgNumber() - newArgOffset] = otherConstOp;
            return true;
          }))
        continue;

      LLVM_DEBUG(llvm::dbgs()
                 << "  - Merge " << defineOp.sym_nameAttr() << " <- "
                 << otherDefineOp.sym_nameAttr() << "\n");
      addCallSiteOperands(callSites[otherDefineOp], callSiteOps);
      replaceArcWith(otherDefineOp, defineOp);
      arcHashes[otherIdx].first = {};
    }

    defineOp.setType(newArcType);
  }
}

void DedupPass::replaceArcWith(DefineOp oldArc, DefineOp newArc) {
  auto &oldUses = callSites[oldArc];
  auto &newUses = callSites[newArc];
  auto newArcName = FlatSymbolRefAttr::get(newArc.sym_nameAttr());
  for (auto stateOp : oldUses) {
    stateOp.arcAttr(newArcName);
    newUses.push_back(stateOp);
  }
  callSites.erase(oldArc);
  arcByName.erase(oldArc.sym_nameAttr());
  oldArc->erase();
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createDedupPass() {
  return std::make_unique<DedupPass>();
}
} // namespace arc
} // namespace circt
