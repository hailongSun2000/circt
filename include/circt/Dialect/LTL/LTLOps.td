//===- LTLOps.td - LTL operation definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LTL_LTLOPS
#define CIRCT_DIALECT_LTL_LTLOPS

include "circt/Dialect/LTL/LTLTypes.td"

// Base class for the operations in this dialect.
class LTLOp<string mnemonic, list<Trait> traits = []> :
  Op<LTLDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Sequences
//===----------------------------------------------------------------------===//

def DelayOp : LTLOp<"delay", [Pure]> {
  let summary = "A temporal delay of a fixed number of cycles.";
  let description = [{
    The `ltl.delay` operation delays a sequence or property by a fixed number of
    cycles. It is equivalent to the `##N` operator in SystemVerilog, and repeat
    application of the unary `N`/`next` operator in LTL.

    ### Example

    The operation `ltl.delay %x, 3` is equivalent to `##3 x` in SystemVerilog or
    `NNN x` in LTL.
  }];

  let arguments = (ins AnySequence:$input, I32Attr:$delay);
  let results = (outs SequenceType:$output);
  let assemblyFormat = "$input `,` $delay attr-dict `:` type($input)";
  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Properties
//===----------------------------------------------------------------------===//

def ImplicationOp : LTLOp<"implication", [Pure]> {
  let summary = "A property where one sequence implies another.";
  let description = [{
    The `ltl.implication` operation represents the property where one sequence
    implies another sequence. It is equivalent to the `|->` operator in
    SystemVerilog ("non-overlapping sequence implication").
  }];

  let arguments = (ins AnySequence:$lhs, AnySequence:$rhs);
  let results = (outs PropertyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)";
}

#endif // CIRCT_DIALECT_LTL_LTLOPS
