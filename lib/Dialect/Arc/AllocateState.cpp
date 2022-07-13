//===- AllocateState.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-alloc-state"

using namespace mlir;
using namespace circt;
using namespace arc;

using llvm::SmallMapVector;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct AllocateStatePass : public AllocateStateBase<AllocateStatePass> {
  void runOnOperation() override;
  void allocateBlock(Block *block);
  void allocateOps(Value storage, Block *block, ArrayRef<Operation *> ops);
};
} // namespace

void AllocateStatePass::runOnOperation() {
  ModelOp modelOp = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "Allocating state in `" << modelOp.name()
                          << "`\n");

  // Walk the blocks from innermost to outermost and group all state allocations
  // in that block in one larger allocation.
  modelOp.walk([&](Block *block) { allocateBlock(block); });
}

void AllocateStatePass::allocateBlock(Block *block) {
  SmallMapVector<Value, std::vector<Operation *>, 1> opsByStorage;

  // Group operations by their storage. There is generally just one storage,
  // passed into the model as a block argument.
  for (auto &op : *block) {
    if (isa<AllocStateOp, RootInputOp, RootOutputOp, AllocMemoryOp,
            AllocStorageOp>(&op))
      opsByStorage[op.getOperand(0)].push_back(&op);
  }
  LLVM_DEBUG(llvm::dbgs() << "- Visiting block in "
                          << block->getParentOp()->getName() << "\n");

  // Actually allocate each operation.
  for (auto &[storage, ops] : opsByStorage)
    allocateOps(storage, block, ops);
}

void AllocateStatePass::allocateOps(Value storage, Block *block,
                                    ArrayRef<Operation *> ops) {
  // Helper function to allocate storage aligned to its own size, or 8 bytes at
  // most.
  unsigned currentByte = 0;
  auto allocBytes = [&](unsigned numBytes) {
    if (numBytes >= 2)
      currentByte = (currentByte + 1) / 2 * 2;
    if (numBytes >= 4)
      currentByte = (currentByte + 3) / 4 * 4;
    if (numBytes >= 8)
      currentByte = (currentByte + 7) / 8 * 8;
    unsigned offset = currentByte;
    currentByte += numBytes;
    return offset;
  };

  // Allocate storage for the operations.
  OpBuilder builder(block->getParentOp());
  for (auto *op : ops) {
    if (isa<AllocStateOp, RootInputOp, RootOutputOp>(op)) {
      SmallVector<Attribute> offsets;
      for (auto result : op->getResults()) {
        auto intType = result.getType().cast<IntegerType>();
        unsigned numBytes = (intType.getWidth() + 7) / 8;
        unsigned offset = allocBytes(numBytes);
        offsets.push_back(builder.getI32IntegerAttr(offset));
      }
      op->setAttr("offsets", builder.getArrayAttr(offsets));
      continue;
    }

    if (auto memOp = dyn_cast<AllocMemoryOp>(op)) {
      auto memType = memOp.getType();
      auto intType = memType.getWordType();
      unsigned stride = (intType.getWidth() + 7) / 8;
      if (stride > 1)
        stride = (stride + 1) / 2 * 2;
      if (stride > 2)
        stride = (stride + 3) / 4 * 4;
      if (stride > 4)
        stride = (stride + 7) / 8 * 8;
      unsigned numBytes = memType.getNumWords() * stride;
      unsigned offset = allocBytes(numBytes);
      op->setAttr("offset", builder.getI32IntegerAttr(offset));
      op->setAttr("stride", builder.getI32IntegerAttr(stride));
      memOp.getResult().setType(MemoryType::get(memOp.getContext(),
                                                memType.getNumWords(),
                                                memType.getWordType(), stride));
      continue;
    }

    if (auto allocStorageOp = dyn_cast<AllocStorageOp>(op)) {
      unsigned offset = allocBytes(allocStorageOp.getType().getSize());
      allocStorageOp.offsetAttr(builder.getI32IntegerAttr(offset));
      continue;
    }

    assert("unsupported op for allocation" && false);
  }

  // Create the substorage accessor at the beginning of the block.
  Operation *storageOwner = storage.getDefiningOp();
  if (!storageOwner)
    storageOwner = storage.cast<BlockArgument>().getOwner()->getParentOp();

  if (storageOwner->isProperAncestor(block->getParentOp())) {
    auto substorage = builder.create<AllocStorageOp>(
        block->getParentOp()->getLoc(),
        StorageType::get(&getContext(), currentByte), storage);
    for (auto *op : ops)
      op->replaceUsesOfWith(storage, substorage);
  } else {
    storage.setType(StorageType::get(&getContext(), currentByte));
  }
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createAllocateStatePass() {
  return std::make_unique<AllocateStatePass>();
}
} // namespace arc
} // namespace circt
