//===- Dialect.td ------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_DIALECT_TD
#define CIRCT_DIALECT_ARC_DIALECT_TD

def ArcDialect : Dialect {
  let name = "arc";
  let summary = "Canonicalized representation of state transfer in a circuit";
  let cppNamespace = "::circt::arc";
}

#endif // CIRCT_DIALECT_ARC_DIALECT_TD
