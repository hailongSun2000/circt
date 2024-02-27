//===- MooreTypes.td - Moore type definitions --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SystemVerilog types for the Moore dialect. Includable in other dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOORETYPES
#define CIRCT_DIALECT_MOORE_MOORETYPES

include "circt/Dialect/Moore/MooreDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class MooreType<Pred condition, string description, string cppClassName>
  : DialectType<MooreDialect, condition, description, cppClassName>;

/// A simple bit vector type.
def SimpleBitVectorType : MooreType<CPred<[{
    $_self.isa<moore::UnpackedType>() &&
    $_self.cast<moore::UnpackedType>().isSimpleBitVector()
  }]>, "simple bit vector type", "moore::UnpackedType">;

/// A single bit type (`bit`, `logic`, `reg`).
def AnySingleBitType : MooreType<CPred<[{
    $_self.isa<moore::IntType>() &&
    $_self.cast<moore::IntType>().getBitSize() == 1
  }]>, "single bit type", "moore::IntType">;

/// A `bit` type.
def BitType : MooreType<CPred<[{
    $_self.isa<moore::IntType>() &&
    $_self.cast<moore::IntType>().getBitSize() == 1 &&
    $_self.cast<moore::IntType>().getDomain() == moore::Domain::TwoValued
  }]>, "`bit` type", "moore::IntType"> {
  let builderCall = [{
    $_builder.getType<moore::IntType>(IntType::Kind::Bit)
  }];
}

/// A `logic` type.
def LogicType : MooreType<CPred<[{
    $_self.isa<moore::IntType>() &&
    $_self.cast<moore::IntType>().getBitSize() == 1 &&
    $_self.cast<moore::IntType>().getDomain() == moore::Domain::FourValued &&
    $_self.cast<moore::IntType>().getKeyword() == "logic"
  }]>, "`logic` type", "moore::IntType"> {
  let builderCall = [{
    $_builder.getLogic()
  }];
}

/// A `reg` type.
def RegType : MooreType<CPred<[{
    $_self.isa<moore::IntType>() &&
    $_self.cast<moore::IntType>().getBitSize() == 1 &&
    $_self.cast<moore::IntType>().getDomain() == moore::Domain::FourValued &&
    $_self.cast<moore::IntType>().getKeyword() == "reg"
  }]>, "`reg` type", "moore::RegType"> {
  let builderCall = [{
    $_builder.getType<moore::IntType>(IntType::Kind::Reg)
  }];
}

/// A `byte` type.
def ByteType : MooreType<CPred<[{
    $_self.isa<moore::IntType>() &&
    $_self.cast<moore::IntType>().getBitSize() == 8
  }]>, "`byte` type", "moore::ByteType"> {
  let builderCall = [{
    $_builder.getType<moore::IntType>(IntType::Kind::Byte)
  }];
}

/// A `shortInt` type.
def ShortIntType : MooreType<CPred<[{
    $_self.isa<moore::IntType>() &&
    $_self.cast<moore::IntType>().getBitSize() == 16
  }]>, "`shortInt` type", "moore::ShortIntType"> {
  let builderCall = [{
    $_builder.getType<moore::IntType>(IntType::Kind::ShortInt)
  }];
}

/// A `int` type.
def IntType : MooreType<CPred<[{
    $_self.isa<moore::IntType>() &&
    $_self.cast<moore::IntType>().getBitSize() == 32 &&
    $_self.cast<moore::IntType>().getDomain() == moore::Domain::TwoValued
  }]>, "`int` type", "moore::IntType"> {
  let builderCall = [{
    $_builder.getInt()
  }];
}

/// A `longInt` type.
def LongIntType : MooreType<CPred<[{
    $_self.isa<moore::IntType>() &&
    $_self.cast<moore::IntType>().getBitSize() == 64 &&
    $_self.cast<moore::IntType>().getDefaultSign() == moore::Sign::Signed
  }]>, "`longInt` type", "moore::LongIntType"> {
  let builderCall = [{
    $_builder.getType<moore::IntType>(IntType::Kind::LongInt)
  }];
}

/// A `integer` type.
def IntegerType : MooreType<CPred<[{
    $_self.isa<moore::IntType>() &&
    $_self.cast<moore::IntType>().getBitSize() == 32 &&
    $_self.cast<moore::IntType>().getDomain() == moore::Domain::FourValued
  }]>, "`integer` type", "moore::IntegerType"> {
  let builderCall = [{
    $_builder.getType<moore::IntType>(IntType::Kind::Integer)
  }];
}

/// A `time` type.
def TimeType : MooreType<CPred<[{
    $_self.isa<moore::IntType>() &&
    $_self.cast<moore::IntType>().getBitSize() == 64 &&
    $_self.cast<moore::IntType>().getDefaultSign() == moore::Sign::Unsigned
  }]>, "`time` type", "moore::TimeType"> {
  let builderCall = [{
    $_builder.getTime()
  }];
}

//===----------------------------------------------------------------------===//
// Packed type & Unpacked type
//===----------------------------------------------------------------------===//
def PackedType : MooreType<CPred<"$_self.isa<moore::PackedType>()">,
  "packed type", "moore::PackedType">;

def UnpackedType : MooreType<CPred<"$_self.isa<moore::UnpackedType>()">,
  "unpacked type", "moore::UnpackedType">;

//===----------------------------------------------------------------------===//
// Real types
//===----------------------------------------------------------------------===//
/// A `shortReal` type.
def ShortRealType : MooreType<CPred<[{
    $_self.isa<moore::RealType>() &&
    $_self.cast<moore::RealType>().getBitSize() == 32
  }]>, "`shortReal` type", "moore::ShortRealType"> {
    let builderCall = [{
      $_builder.getType<moore::RealType>(RealType::Kind::ShortReal)
    }];
  }

/// A `real` type.
def RealType : MooreType<CPred<[{
    $_self.isa<moore::RealType>() &&
    $_self.cast<moore::RealType>().getBitSize() == 64 &&
    $_self.cast<moore::RealType>().getKind() == moore::RealType::Real
  }]>, "`real` type", "moore::RealType"> {
    let builderCall = [{
      $_builder.getType<moore::RealType>(RealType::Kind::Real)
    }];
  }

/// A `realTime` type.
def RealTimeType : MooreType<CPred<[{
    $_self.isa<moore::RealType>() &&
    $_self.cast<moore::RealType>().getBitSize() == 64 &&
    $_self.cast<moore::RealType>().getKind() == moore::RealType::RealTime
  }]>, "`realTime` type", "moore::RealTimeType"> {
    let builderCall = [{
      $_builder.getType<moore::RealType>(RealType::Kind::RealTime)
    }];
  }
//===----------------------------------------------------------------------===//
// Enum type
//===----------------------------------------------------------------------===//

def EnumType : MooreType<CPred<"$_self.isa<moore::EnumType>()">,
  "a SystemVerilog Enum type", "moore::EnumType">;


#endif // CIRCT_DIALECT_MOORE_MOORETYPES
