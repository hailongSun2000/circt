//===- TypesImpl.td ----------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/Arc/Dialect.td"

class ArcTypeDef<string name> : TypeDef<ArcDialect, name> { }

def MemoryType : ArcTypeDef<"Memory"> {
  let mnemonic = "memory";
  let parameters = (ins "unsigned":$numWords, "::mlir::IntegerType":$wordType,
    OptionalParameter<"unsigned">:$stride);
  let assemblyFormat = "`<` $numWords `x` $wordType (`,` $stride^)? `>`";
}

def StorageType : ArcTypeDef<"Storage"> {
  let mnemonic = "storage";
  let parameters = (ins OptionalParameter<"unsigned">:$size);
  let assemblyFormat = "(`<` $size^ `>`)?";

  // let builders = [
  //   AttrBuilderWithInferredContext<(ins "::mlir::Type":$nestedType), [{
  //     return $_get(nestedType.getContext(), nestedType);
  //   }]>
  // ];
}
