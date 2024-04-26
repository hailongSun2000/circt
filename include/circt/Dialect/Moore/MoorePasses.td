//===--- Passes.td - Moore pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the Moore dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREPASSES_TD
#define CIRCT_DIALECT_MOORE_MOOREPASSES_TD

include "mlir/Pass/PassBase.td"

def InfoCollection : Pass<"info-collection", "mlir::ModuleOp"> {
    let summary = "Collect all nets and variables declarations";
    let description = [{
      This pass will perform a pre-pass over the IR to find all concurrently-
      assigned nets and variables. Basically go through all net/variable 
      declarations and collect the connected value in a hash map (either the
      initial value in net declaration or the assigned value).
    }];
    let constructor = "circt::moore::createInfoCollectionPass()";
}

def DeleteLocalVar : Pass<"delete-local-var", "mlir::ModuleOp"> {
  let summary = "Delete local temporary variables";
  let description = [{
    This pass is aimed to delete local temporary variables and replace all
    uses by directly using their value.
  }];
  let constructor = "circt::moore::createDeleteLocalVarPass()";
}

#endif // CIRCT_DIALECT_MOORE_MOOREPASSES_TD
