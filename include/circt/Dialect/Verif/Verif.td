//===- Verif.td - Verif dialect definition -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VERIF_TD
#define VERIF_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def VerifDialect : Dialect {
  let name = "verif";
  let summary = "A verification dialect";
  let description = [{
    This is the `verif` dialect, intended to be a generic representation for
    design verification constructs.
  }];
  let cppNamespace = "circt::verif";
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

class VerifOp<string mnemonic, list<OpTrait> traits = []> :
    Op<VerifDialect, mnemonic, traits>;

def TestOp : VerifOp<"test", [IsolatedFromAbove, HasOnlyGraphRegion,
                              NoTerminator]> {
  let summary = "A verification test case";
  let description = [{
    The `verif.test` operation represents a verification test case.
  }];
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def CheckOp : VerifOp<"check"> {
  let summary = "A litmus test assertion to check that a condition holds";
  let description = [{
    The `verif.check` operation verifies that a condition holds. It does not
    perform any formal proof, but rather acts as an assertion in a litmus test.
  }];
  let arguments = (ins I1:$cond);
  let assemblyFormat = "$cond attr-dict";
}

#endif // VERIF_TD
