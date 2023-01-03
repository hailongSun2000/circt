//===- LTLOps.cpp - Implement the LTL operations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LTL/LTLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace circt::ltl;

//===----------------------------------------------------------------------===//
// DelayOp
//===----------------------------------------------------------------------===//

OpFoldResult DelayOp::fold(ArrayRef<Attribute> operands) {
  if (getDelay() == 0)
    return getInput();
  return {};
}

LogicalResult DelayOp::canonicalize(DelayOp op, PatternRewriter &rewriter) {
  // delay(delay(x, a), b) -> delay(x, a+b)
  if (auto innerDelayOp = op.getInput().getDefiningOp<DelayOp>()) {
    rewriter.replaceOpWithNewOp<DelayOp>(
        op, innerDelayOp.getInput(), innerDelayOp.getDelay() + op.getDelay());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// TableGen
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/LTL/LTL.cpp.inc"
