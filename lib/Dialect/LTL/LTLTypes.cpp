//===- LTLTypes.cpp - Implement the LTL types -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LTL/LTLTypes.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::ltl;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/LTL/LTLTypes.cpp.inc"

void LTLDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/LTL/LTLTypes.cpp.inc"
      >();
}
