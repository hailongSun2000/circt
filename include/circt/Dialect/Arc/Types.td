//===- Types.td --------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_TYPES_TD
#define CIRCT_DIALECT_ARC_TYPES_TD

include "circt/Dialect/Arc/Dialect.td"

class ArcType<Pred condition, string description, string cppClassName>
  : DialectType<ArcDialect, condition, description, cppClassName>;

// def StateType : ArcType<CPred<"$_self.isa<arc::StateType>()">,
//   "state type", "arc::StateType">;

#endif // CIRCT_DIALECT_ARC_TYPES_TD
