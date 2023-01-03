//===- LTLDialect.td - LTL dialect definition --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LTL_LTLDIALECT
#define CIRCT_DIALECT_LTL_LTLDIALECT

def LTLDialect : Dialect {
  let name = "ltl";
  let cppNamespace = "::circt::ltl";

  let summary = "Types and operations for the LTL dialect";
  let description = [{
    This dialect defines operations and types to represent Linear Temporal Logic
    and Computational Tree Logic properties and sequences.
  }];

  let extraClassDeclaration = [{
    /// Register all LTL types.
    void registerTypes();
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif // CIRCT_DIALECT_LTL_LTLDIALECT
