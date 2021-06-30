//===- Verif.h - Verif dialect ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_VERIF_VERIF_H
#define CIRCT_DIALECT_VERIF_VERIF_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "circt/Dialect/Verif/VerifDialect.h.inc"
#include "circt/Dialect/Verif/VerifEnums.h.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/Verif/Verif.h.inc"

#endif // CIRCT_DIALECT_VERIF_VERIF_H
