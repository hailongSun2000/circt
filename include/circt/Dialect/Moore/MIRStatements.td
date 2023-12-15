//===- MIRStatements.td - Moore MIR statements ops ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for Moore MIR statements.
//
//===----------------------------------------------------------------------===//

def VariableDeclOp : MIROp<"vardecl", []> {
  let summary = "A variable declaration";

  let arguments = (ins StrAttr:$name, I32Attr:$init);
  let results = (outs MooreLValueType:$result);
  let assemblyFormat = [{
    $name `=` $init attr-dict `:` custom<LValueType>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// Assignment Statements
//===----------------------------------------------------------------------===//

def AssignOp : MIROp<"assign",
    [TypesMatchWith<"src and dest types have to match", "src", "dest",
     "LValueType::get($_self)">]> {

  let summary = "Continuous assignment";
  let description = [{
    A SystemVerilog assignment statement 'x = y;'.
    These occur in module scope.  See SV Spec 10.3.2.
  }];
  let arguments = (ins MooreLValueType:$dest, MooreRValueType:$src);
  let results = (outs);
  let assemblyFormat = [{
    $dest `,` $src  attr-dict `:` qualified(type($src))
  }];
}

def CAssignOp : MIROp<"cassign",
    [TypesMatchWith<"src and dest types have to match", "src", "dest",
     "$_self">]> {

  let summary = "Continuous assignment";
  let description = [{
    A SystemVerilog assignment statement 'x = y;'.
    These occur in module scope.  See SV Spec 10.3.2.
  }];
  let arguments = (ins AnyType:$dest, AnyType:$src);
  let results = (outs);
  let assemblyFormat = [{
    $dest `,` $src  attr-dict `:` qualified(type($src))
  }];
}

def BPAssignOp : MIROp<"bpassign",
    [TypesMatchWith<"src and dest types have to match", "src", "dest",
     "$_self">]> {

  let summary = "Blocking procedural assignment";
  let description = [{
    A SystemVerilog blocking procedural assignment statement 'x = y;'.
    These occur in a sequential block.  See SV Spec 10.4.1.
  }];
  let arguments = (ins AnyType:$dest, AnyType:$src);
  let results = (outs);
  let assemblyFormat = [{
    $dest `,` $src  attr-dict `:` qualified(type($src))
  }];
}

def PAssignOp : MIROp<"passign",
    [TypesMatchWith<"src and dest types have to match", "src", "dest",
     "$_self">]> {

  let summary = "Nonblocking procedural assignment";
  let description = [{
    A SystemVerilog nonblocking procedural assignment statement 'x <= y;'.
    These occur in a sequential block.  See SV Spec 10.4.12.
  }];
  let arguments = (ins AnyType:$dest, AnyType:$src);
  let results = (outs);
  let assemblyFormat = [{
    $dest `,` $src  attr-dict `:` qualified(type($src))
  }];
}

def PCAssignOp : MIROp<"pcassign",
    [TypesMatchWith<"src and dest types have to match", "src", "dest",
     "$_self">]> {

  let summary = "Procedural continuous assignment";
  let description = [{
    A SystemVerilog assignment statement 'assign x = y;'.
    The procedural continuous assignments (using keywords assign and force) are procedural statements
    that allow expressions to be driven continuously onto variables or nets
    These occur in procedural scope.  See SV Spec 10.6
  }];
  let arguments = (ins AnyType:$dest, AnyType:$src);
  let results = (outs);
  let assemblyFormat = [{
    $dest `,` $src  attr-dict `:` qualified(type($src))
  }];
}
