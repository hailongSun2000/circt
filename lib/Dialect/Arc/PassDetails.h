//===- PassDetails.h ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_PASSDETAILS_H
#define CIRCT_DIALECT_ARC_PASSDETAILS_H

#include "circt/Dialect/Arc/Ops.h"
#include "circt/Dialect/Arc/Types.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace circt {

namespace hw {
class HWDialect;
} // namespace hw

namespace seq {
class SeqDialect;
} // namespace seq

namespace sv {
class SVDialect;
} // namespace sv

namespace arc {

#define GEN_PASS_CLASSES
#include "circt/Dialect/Arc/Passes.h.inc"

bool isArcBreakingOp(Operation *op);
void pruneUnusedValues(ArrayRef<Value> initial);

} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_PASSDETAILS_H
