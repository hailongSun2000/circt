//===- Verif.td - Verif dialect definition -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VERIF_TD
#define VERIF_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def VerifDialect : Dialect {
  let name = "verif";
  let summary = "A verification dialect.";
  let description = [{
    This is the `verif` dialect, intended to be a generic representation for
    design verification constructs.
  }];
  let cppNamespace = "circt::verif";
}

// Base class for the operation in this dialect.
class VerifOp<string mnemonic, list<OpTrait> traits = []> :
    Op<VerifDialect, mnemonic, traits>;

#endif // VERIF_TD
