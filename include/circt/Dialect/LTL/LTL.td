//===- LTL.td - LTL top-level tablegen file ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LTL_LTL
#define CIRCT_DIALECT_LTL_LTL

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "circt/Dialect/HW/HWOpInterfaces.td"

include "circt/Dialect/LTL/LTLDialect.td"
include "circt/Dialect/LTL/LTLOps.td"
include "circt/Dialect/LTL/LTLTypes.td"

#endif // CIRCT_DIALECT_LTL_LTL
