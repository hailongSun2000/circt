//===- LTLTypes.td - LTL type definitions ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LTL_LTLTYPES
#define CIRCT_DIALECT_LTL_LTLTYPES

include "circt/Dialect/LTL/LTLDialect.td"

// Base class for the type definitions in this dialect.
class LTLTypeDef<string name> : TypeDef<LTLDialect, name>;

// Base class for the type constraints in this dialect.
class LTLType<Pred pred, string desc, string cppClassName>
  : DialectType<LTLDialect, pred, desc, cppClassName>;

//===----------------------------------------------------------------------===//
// Sequence type
//===----------------------------------------------------------------------===//

def SequenceType : LTLTypeDef<"Sequence"> {
  let summary = "a temporal sequence";
  let description = [{
    This type represents a temporal sequence.
  }];
  let mnemonic = "sequence";
  let assemblyFormat = "";
}

def AnySequence : AnyTypeOf<[SequenceType, I1]>;

//===----------------------------------------------------------------------===//
// Property type
//===----------------------------------------------------------------------===//

def PropertyType : LTLTypeDef<"Property"> {
  let summary = "a property";
  let description = [{
    This type represents a property.
  }];
  let mnemonic = "property";
  let assemblyFormat = "";
}

def AnyProperty : AnyTypeOf<[PropertyType, AnySequence]>;

#endif // CIRCT_DIALECT_LTL_LTLTYPES
