//===- SVVerification.td - SV verification ops -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for SystemVerilog verification statements and
// declarations.
//
//===----------------------------------------------------------------------===//

/// Immediate assertions, like `assert`.
def ImmediateAssert: I32EnumAttrCase<"Immediate", 0, "immediate">;
/// Observed deferred assertions, like `assert #0`.
def ObservedAssert: I32EnumAttrCase<"Observed", 1, "observed">;
/// Final deferred assertions, like `assert final`.
def FinalAssert: I32EnumAttrCase<"Final", 2, "final">;

/// A mode specifying how immediate/deferred assertions operate.
def DeferAssertAttr : I32EnumAttr<
  "DeferAssert", "assertion deferring mode",
  [ImmediateAssert, ObservedAssert, FinalAssert]>
{
  let cppNamespace = "circt::sv";
}

/// A constraint forcing `cover` ops to have no message.
def NoMessageVerif : PredOpTrait<"has no message",
  CPred<"!$message && $substitutions.empty()">>;

/// Immediate verification operations defined by the SV standard.
class ImmediateVerifOp<string mnemonic, list<Trait> traits = []> :
    SVOp<mnemonic, [ProceduralOp] # traits> {
  let arguments = (ins
    I1:$expression,
    DeferAssertAttr:$defer,
    OptionalAttr<StrAttr>:$label,
    OptionalAttr<StrAttr>:$message, Variadic<AnyType>:$substitutions);

  let assemblyFormat = [{
    $expression `,` $defer
    (`label` $label^)?
    (`message` $message^ (`(` $substitutions^ `)` `:`
    qualified(type($substitutions)))?)? attr-dict
  }];

  let builders = [
    // no label, no message
    OpBuilder<(ins "mlir::Value":$expression,
      DeferAssertAttr.storageType:$defer),
      [{ build(odsBuilder, odsState, expression, defer, StringAttr{},
        StringAttr{}, ValueRange{}); }]>,
    // no message
    OpBuilder<(ins "mlir::Value":$expression,
      DeferAssertAttr.storageType:$defer, StrAttr.storageType:$label),
      [{ build(odsBuilder, odsState, expression, defer, label, StringAttr{},
        ValueRange{}); }]>
  ];

  let hasCanonicalizer = 1;
}

def AssertOp : ImmediateVerifOp<"assert"> {
  let summary = "immediate assertion statement";
  let description = [{
    Specify that a Boolean expression is always true.  This can be used to both
    document the behavior of the design and to test that the design behaves as
    expected.  See Section 16.3 of the SystemVerilog 2017 specification.
  }];
}

def AssumeOp : ImmediateVerifOp<"assume"> {
  let summary = "immediate assume statement";
  let description = [{
    Specify that a Boolean expression is assumed to always be true.  This can
    either be used as an assertion-like check that the expression is, in fact,
    always true or to bound legal input values during testing.  See Section 16.3
    of the SystemVerilog 2017 specification.
  }];
}

def CoverOp : ImmediateVerifOp<"cover", [NoMessageVerif]> {
  let summary = "immediate cover statement";
  let description = [{
    Specify that a Boolean expression should be monitored for coverage, i.e., a
    simulator will watch if it occurs and how many times it occurs.  See section
    16.3 of the SystemVerilog 2017 specification.
  }];
}

/// Concurrent verification operations defined by the SV standard.
class ConcurrentVerifOp<string mnemonic, list<Trait> traits = []> :
    SVOp<mnemonic, traits> {
  let arguments = (ins
    EventControlAttr:$event, I1:$clock, I1:$property,
    OptionalAttr<StrAttr>:$label,
    OptionalAttr<StrAttr>:$message, Variadic<AnyType>:$substitutions);

  let assemblyFormat = [{
    $event $clock `,` $property
    (`label` $label^)?
    (`message` $message^ (`(` $substitutions^ `)` `:`
    qualified(type($substitutions)))?)? attr-dict
  }];

  let builders = [
    // no label, no message
    OpBuilder<(ins
      EventControlAttr.storageType:$event,
      "mlir::Value":$clock,
      "mlir::Value":$property),
      [{ build(odsBuilder, odsState, event, clock, property, StringAttr{},
        StringAttr{}, ValueRange{}); }]>,
    // no message
    OpBuilder<(ins
      EventControlAttr.storageType:$event,
      "mlir::Value":$clock,
      "mlir::Value":$property,
      StrAttr.storageType:$label),
      [{ build(odsBuilder, odsState, event, clock, property, label,
        StringAttr{}, ValueRange{}); }]>
  ];

  let hasCanonicalizer = 1;
}

def AssertConcurrentOp : ConcurrentVerifOp<"assert.concurrent"> {
  let summary = "concurrent assertion statement, i.e., assert property";
  let description = [{
    Specify that a property of the hardware design is true whenever the property
    is evaluated.  This can be used to both document the behavior of the design
    and to test that the design behaves as expected.  See section 16.5 of the
    SystemVerilog 2017 specification.
  }];
}

def AssumeConcurrentOp : ConcurrentVerifOp<"assume.concurrent"> {
  let summary = "concurrent assume statement, i.e., assume property";
  let description = [{
    Specify that a property is assumed to be true whenever the property is
    evaluated.  This can be used to both document the behavior of the design and
    to test that the design behaves as expected.  See section 16.5 of the
    SystemVerilog 2017 specification.
  }];
}

def CoverConcurrentOp : ConcurrentVerifOp<"cover.concurrent",
                                          [NoMessageVerif]> {
  let summary = "concurrent cover statement, i.e., cover property";
  let description = [{
    Specify that a specific property should be monitored for coverage, i.e., a
    simulation will watch if it occurrs and how many times it occurs.  See
    section 16.5 of the SystemVerilog 2017 specification.
  }];
}

//===----------------------------------------------------------------------===//
// SVA Properties
//===----------------------------------------------------------------------===//

class PropertyOp<string mnemonic, list<Trait> traits = []> :
    SVOp<"property." # mnemonic, traits>;

def PropNotOp : PropertyOp<"not", [Pure]> {
  let arguments = (ins AnyProperty:$input);
  let results = (outs PropertyType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
  let summary = "A negation property";
  let description = [{
    See IEEE 1800-2017 §16.12.3 Negation property.
  }];
}

class BinaryPropOp<string mnemonic, list<Trait> traits = []> :
    PropertyOp<mnemonic, traits # [Pure]> {
  let arguments = (ins AnyProperty:$lhs, AnyProperty:$rhs);
  let results = (outs PropertyType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
  }];
}

def PropAndOp : BinaryPropOp<"and", [Commutative]> {
  let summary = "A conjunction property";
  let description = [{
    See IEEE 1800-2017 §16.12.5 Conjunction property.
  }];
}

def PropOrOp : BinaryPropOp<"or", [Commutative]> {
  let summary = "A disjunction property";
  let description = [{
    See IEEE 1800-2017 §16.12.4 Disjunction property.
  }];
}

def PropImplicationOp : PropertyOp<"implication", [Pure]> {
  let arguments = (ins AnySequence:$lhs, AnyProperty:$rhs, UnitAttr:$nonoverlap);
  let results = (outs PropertyType:$result);
  let assemblyFormat = [{
    (`nonoverlap` $nonoverlap^)? $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
  }];
  let summary = "A sequence implying a property";
  let description = [{
    This operation represents the overlapping implication operator `|->` as well
    as the non-overlapping `|=>` of SystemVerilog.

    See IEEE 1800-2017 §16.12.7 Implication.
  }];
}
