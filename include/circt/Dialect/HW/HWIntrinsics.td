//===- HWIntrinsics.td - HW intrinsic ops ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWINTRINSICS_TD
#define CIRCT_DIALECT_HW_HWINTRINSICS_TD

include "circt/Dialect/HW/HWDialect.td"

class HWIntrinsic<string name, dag inputs, dag outputs> : HWOp<"int." # name> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let arguments = inputs;
  let results = outputs;
}

def IntIsXOp : HWIntrinsic<"is_x",
  (ins HWIntegerType:$input), (outs I1:$result)>;
def IntIsZOp : HWIntrinsic<"is_z",
  (ins HWIntegerType:$input), (outs I1:$result)>;
def IntSizeOfOp : HWIntrinsic<"sizeof",
  (ins HWIntegerType:$input), (outs I32:$result)>;
def IntPlusArgTestOp : HWIntrinsic<"plusarg.test",
  (ins StrAttr:$formatString), (outs I1:$found)>;
def IntPlusArgValueOp : HWIntrinsic<"plusarg.value",
  (ins StrAttr:$formatString), (outs I1:$found, HWIntegerType:$value)>;
def IntClockGateOp : HWIntrinsic<"clock_gate",
  (ins I1:$input, I1:$enable), (outs I1:$output)>;

#endif // CIRCT_DIALECT_HW_HWINTRINSICS_TD
