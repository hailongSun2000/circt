//===- Utils.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

using namespace mlir;
using namespace circt;
using namespace arc;

bool arc::isArcBreakingOp(Operation *op) {
  return isa<hw::InstanceOp, seq::CompRegOp, StateOp, ClockGateOp, MemoryOp,
             MemoryReadOp, MemoryWriteOp, hw::ConstantOp>(op) ||
         op->getNumResults() > 1;
}

void arc::pruneUnusedValues(ArrayRef<Value> initial) {
  llvm::SmallDenseSet<Operation *> worklist;
  for (auto value : initial)
    if (auto *op = value.getDefiningOp())
      worklist.insert(op);
  while (!worklist.empty()) {
    Operation *op = *worklist.begin();
    worklist.erase(op);
    if (isArcBreakingOp(op))
      continue;
    if (op->use_empty()) {
      for (auto operand : op->getOperands())
        if (auto *op = operand.getDefiningOp())
          worklist.insert(op);
      op->erase();
    }
  }
}
