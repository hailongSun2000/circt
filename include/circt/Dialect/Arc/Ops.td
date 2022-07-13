//===- Ops.td ----------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_OPS_TD
#define CIRCT_DIALECT_ARC_OPS_TD

include "circt/Dialect/Arc/Dialect.td"
include "circt/Dialect/Arc/Types.td"
include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/SV/SVTypes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class ArcOp<string mnemonic, list<Trait> traits = []> :
  Op<ArcDialect, mnemonic, traits>;

// Stolen from SV dialect:

def NonProceduralOp : NativeOpTrait<"NonProceduralOp"> {
  let cppNamespace = "::circt::sv";
}

def RegAssignOp : ArcOp<"regassign", [InOutTypeConstraint<"src", "dest">,
                                      NonProceduralOp]> {
  let summary = "Register update procedural assignment";
  let arguments = (ins InOutType:$dest, InOutElementType:$src, I1:$clock);
  let results = (outs);
  let assemblyFormat = [{
    $dest `,` $src `,` $clock attr-dict `:` qualified(type($src))
  }];
}

def DefineOp : ArcOp<"define", [
  IsolatedFromAbove,
  FunctionOpInterface,
  Symbol,
  RegionKindInterface,
  SingleBlockImplicitTerminator<"arc::OutputOp">,
  HasParent<"mlir::ModuleOp">
]> {
  let summary = "State transfer arc definition";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type
  );
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static RegionKind getRegionKind(unsigned index) {
      return RegionKind::SSACFG;
    }

    Block &bodyBlock() { return body().front(); }

    // Get the arc's symbolic name.
    StringAttr getNameAttr() {
      return (*this)->getAttrOfType<StringAttr>(
        ::mlir::SymbolTable::getSymbolAttrName());
    }

    // Get the arc's symbolic name.
    StringRef getName() {
      return getNameAttr().getValue();
    }

    /// Returns the type of this arc.
    FunctionType getFunctionType() {
      return getFunctionTypeAttr().getValue().cast<FunctionType>();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  }];
}

def OutputOp : ArcOp<"output", [
  Terminator, HasParent<"DefineOp">, NoSideEffect, ReturnLike
]> {
  let summary = "Arc terminator";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = [{
    attr-dict ($operands^ `:` qualified(type($operands)))?
  }];
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, llvm::None);
  }]>];
  let hasVerifier = 1;
}

def StateOp : ArcOp<"state", [
  CallOpInterface, MemRefsNormalizable,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  AttrSizedOperandSegments, NoSideEffect
]> {
  let summary = "State transfer arc";

  let arguments = (ins
    FlatSymbolRefAttr:$arc,
    Optional<I1>:$clock,
    Optional<I1>:$enable,
    I32Attr:$latency,
    Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    $arc `(` $operands `)` (`clock` $clock^)? (`enable` $enable^)? `lat` $latency attr-dict
    `:` functional-type($operands, results)
  }];

  let builders = [
    OpBuilder<(ins "DefineOp":$arc, "Value":$clock, "Value":$enable,
      "unsigned":$latency, CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(arc),
            arc.getFunctionType().getResults(), clock, enable, latency,
            operands);
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$arc, "TypeRange":$results, "Value":$clock,
      "Value":$enable, "unsigned":$latency, CArg<"ValueRange", "{}">:$operands
    ), [{
      if (clock)
        $_state.addOperands(clock);
      if (enable)
        $_state.addOperands(enable);
      $_state.addOperands(operands);
      $_state.addAttribute("arc", arc);
      $_state.addAttribute("latency", $_builder.getI32IntegerAttr(latency));
      $_state.addAttribute(getOperandSegmentSizeAttr(),
        $_builder.getI32VectorAttr({
          clock ? 1 : 0,
          enable ? 1 : 0,
          static_cast<int32_t>(operands.size())}));
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$arc, "TypeRange":$results, "Value":$clock,
      "Value":$enable, "unsigned":$latency, CArg<"ValueRange", "{}">:$operands
    ), [{
      build($_builder, $_state, SymbolRefAttr::get(arc), results, clock, enable,
            latency, operands);
    }]>,
    OpBuilder<(ins "StringRef":$arc, "TypeRange":$results, "Value":$clock,
      "Value":$enable, "unsigned":$latency, CArg<"ValueRange", "{}">:$operands
    ), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), arc),
            results, clock, enable, latency, operands);
    }]>
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return {operand_begin(), operand_end()};
    }

    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("arc");
    }
  }];
}

def ClockGateOp : ArcOp<"clock_gate", [NoSideEffect]> {
  let summary = "Clock Gate";
  let arguments = (ins I1:$input, I1:$enable);
  let results = (outs I1:$output);
  let assemblyFormat = [{
    $input `,` $enable attr-dict
  }];
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$enable), [{
      build($_builder, $_state, input.getType(), input, enable);
    }]>
  ];
}

def MemoryOp : ArcOp<"memory", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Memory";
  let results = (outs MemoryType:$memory);
  let assemblyFormat = [{
    type($memory) attr-dict
  }];
}

class MemoryAndDataTypesMatch<string mem, string data> : TypesMatchWith<
  "memory and data types must match", mem, data,
  "$_self.cast<MemoryType>().getWordType()">;

def MemoryReadOp : ArcOp<"memory_read", [
  MemoryEffects<[MemRead]>,
  MemoryAndDataTypesMatch<"memory", "data">
]> {
  let summary = "Read word from memory";
  let arguments = (ins
    MemoryType:$memory,
    AnyInteger:$address,
    I1:$clock,
    I1:$enable
  );
  let results = (outs AnyInteger:$data);
  let assemblyFormat = [{
    $memory `[` $address `]` `,` $clock `,` $enable
    attr-dict `:` type($memory) `,` type($address)
  }];
}

def MemoryWriteOp : ArcOp<"memory_write", [
  MemoryEffects<[MemWrite]>,
  MemoryAndDataTypesMatch<"memory", "data">
]> {
  let summary = "Write word to memory";
  let arguments = (ins
    MemoryType:$memory,
    AnyInteger:$address,
    I1:$clock,
    I1:$enable,
    AnyInteger:$data,
    Variadic<AnyType>:$reads
  );
  let assemblyFormat = [{
    $memory `[` $address `]` `,` $clock `,` $enable `,` $data
    (` ` `(` `reads` $reads^ `:` type($reads) `)`)?
    attr-dict `:` type($memory) `,` type($address)
  }];
}

def ClockTreeOp : ArcOp<"clock_tree", [NoTerminator]> {
  let summary = "A clock tree";
  let arguments = (ins I1:$clock, Optional<I1>:$enable);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $clock (`if` $enable^)? attr-dict-with-keyword $body
  }];
  let extraClassDeclaration = [{
    Block &bodyBlock() { return body().front(); }
  }];
}

def PassThroughOp : ArcOp<"passthrough", [NoTerminator]> {
  let summary = "Clock-less logic that is on the pass-through path";
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    attr-dict-with-keyword $body
  }];
  let extraClassDeclaration = [{
    Block &bodyBlock() { return body().front(); }
  }];
}

def AllocStateOp : ArcOp<"alloc_state", [MemoryEffects<[MemAlloc, MemRead]>]> {
  let summary = "Allocate internal state";
  let arguments = (ins StorageType:$storage);
  let results = (outs Variadic<AnyInteger>:$states);
  let assemblyFormat = [{
    $storage attr-dict `:` functional-type($storage, $states)
  }];
}

def AllocMemoryOp : ArcOp<"alloc_memory", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a memory";
  let arguments = (ins StorageType:$storage);
  let results = (outs MemoryType:$memory);
  let assemblyFormat = [{
    $storage attr-dict `:` functional-type($storage, $memory)
  }];
}

def AllocStorageOp : ArcOp<"alloc_storage", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate contiguous storage space from a larger storage space";
  let arguments = (ins StorageType:$input, OptionalAttr<I32Attr>:$offset);
  let results = (outs StorageType:$output);
  let assemblyFormat = [{
    $input (`,` $offset^)? attr-dict `:` functional-type($input, $output)
  }];
}

def UpdateStateOp : ArcOp<"update_state", [
  MemoryEffects<[MemWrite]>,
  SameVariadicOperandSize,
  TypesMatchWith<"state and value types must match", "states", "values",
                 "$_self">
]> {
  let summary = "Update internal state or an output value";
  let arguments = (ins Variadic<AnyInteger>:$states,
                       Variadic<AnyInteger>:$values);
  let assemblyFormat = [{
    $states `=` $values attr-dict `:` type($states)
  }];
}

def RootInputOp : ArcOp<"root_input", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "A root input";
  let arguments = (ins StrAttr:$name, StorageType:$storage);
  let results = (outs AnyInteger:$state);
  let assemblyFormat = [{ $name `,` $storage attr-dict `:` functional-type($storage, $state) }];
}

def RootOutputOp : ArcOp<"root_output", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "A root output";
  let arguments = (ins StrAttr:$name, StorageType:$storage);
  let results = (outs AnyInteger:$state);
  let assemblyFormat = [{ $name `,` $storage attr-dict `:` functional-type($storage, $state) }];
}

def ModelOp : ArcOp<"model", [RegionKindInterface, IsolatedFromAbove,
                              NoTerminator]> {
  let summary = "A model with stratified clocks";
  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $name attr-dict-with-keyword $body
  }];

  let extraClassDeclaration = [{
    static RegionKind getRegionKind(unsigned index) {
      return RegionKind::Graph;
    }
    Block &bodyBlock() { return body().front(); }
  }];
}

#endif // CIRCT_DIALECT_ARC_OPS_TD
