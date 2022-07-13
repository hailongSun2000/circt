//===- Ops.h --------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_OPS_H
#define CIRCT_DIALECT_ARC_OPS_H

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVOps.h"

#include "circt/Dialect/Arc/Dialect.h"
#include "circt/Dialect/Arc/Types.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Arc/Arc.h.inc"

#endif // CIRCT_DIALECT_ARC_OPS_H
