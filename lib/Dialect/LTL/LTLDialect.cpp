//===- LTLDialect.cpp - Implement the LTL dialect ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LTL/LTLOps.h"

using namespace circt;
using namespace circt::ltl;

//===----------------------------------------------------------------------===//
// Dialect Specification
//===----------------------------------------------------------------------===//

void LTLDialect::initialize() {
  // Register types.
  registerTypes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/LTL/LTL.cpp.inc"
      >();
}

#include "circt/Dialect/LTL/LTLDialect.cpp.inc"
