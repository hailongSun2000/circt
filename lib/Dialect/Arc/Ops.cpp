//===- Ops.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace circt;
using namespace arc;
using namespace mlir;
using hw::InOutType;

//===----------------------------------------------------------------------===//
// DefineOp
//===----------------------------------------------------------------------===//

ParseResult DefineOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false, buildFuncType);
}

void DefineOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false);
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

LogicalResult OutputOp::verify() {
  return success();

  auto parent = cast<DefineOp>((*this)->getParentOp());
  ArrayRef<Type> types = parent.getResultTypes();
  OperandRange values = getOperands();
  if (types.size() != values.size()) {
    emitOpError("must have same number of operands as parent arc has results");
    return failure();
  }

  for (size_t i = 0, e = types.size(); i < e; ++i) {
    if (types[i] != values[i].getType()) {
      emitOpError("output operand ")
          << i << " type mismatch: arc requires " << types[i] << ", operand is "
          << values[i].getType();
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// StateOp
//===----------------------------------------------------------------------===//

LogicalResult StateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the arc attribute was specified.
  auto arcName = (*this)->getAttrOfType<FlatSymbolRefAttr>("arc");
  if (!arcName)
    return emitOpError("requires a `arc` symbol reference attribute");
  DefineOp arc = symbolTable.lookupNearestSymbolFrom<DefineOp>(*this, arcName);
  if (!arc)
    return emitOpError() << "`" << arcName.getValue()
                         << "` does not reference a valid function";

  // Verify that the operand and result types match the arc.
  auto type = arc.getFunctionType();
  if (type.getNumInputs() != operands().size())
    return emitOpError("incorrect number of operands for arc");

  for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i) {
    if (operands()[i].getType() != type.getInput(i)) {
      auto diag = emitOpError("operand type mismatch: operand ") << i;
      diag.attachNote() << "expected type: " << type.getInput(i);
      diag.attachNote() << "  actual type: " << operands()[i].getType();
      return diag;
    }
  }

  if (type.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for arc");

  for (unsigned i = 0, e = type.getNumResults(); i != e; ++i) {
    if (getResult(i).getType() != type.getResult(i)) {
      auto diag = emitOpError("result type mismatch: result ") << i;
      diag.attachNote() << "expected type: " << type.getResult(i);
      diag.attachNote() << "  actual type: " << getResult(i).getType();
      return diag;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RootInputOp
//===----------------------------------------------------------------------===//

void RootInputOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  SmallString<32> buf("in_");
  buf += name();
  setNameFn(state(), buf);
}

//===----------------------------------------------------------------------===//
// RootOutputOp
//===----------------------------------------------------------------------===//

void RootOutputOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  SmallString<32> buf("out_");
  buf += name();
  setNameFn(state(), buf);
}

#define GET_OP_CLASSES
#include "circt/Dialect/Arc/Arc.cpp.inc"
