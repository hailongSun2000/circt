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

def MooreDeclarations : Pass<"moore-declarations", "mlir::ModuleOp"> {
    let summary = "Collect all nets and variables declarations";
    let description = [{
      This pass will perform a pre-pass over the IR to find all concurrently-
      assigned output ports, nets and variables. Basically go through all output
      port/net/variable declarations and collect the connected value in a hash map
      (either the initial value in nets, or the cassigned value). At the same time
      add all cassigns that has considered here into a set of ops to be deleted.
    }];
    let constructor = "circt::moore::createMooreDeclarationsPass()";
}

#endif // CIRCT_DIALECT_MOORE_MOOREPASSES_TD
