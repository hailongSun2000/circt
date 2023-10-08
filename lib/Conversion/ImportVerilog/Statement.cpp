//===- Statement.cpp - Slang statement conversion--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/ASTVisitor.h"
#include "slang/ast/Symbol.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/ast/types/AllTypes.h"
#include "slang/ast/types/Type.h"
#include "slang/syntax/SyntaxVisitor.h"

using namespace circt;
using namespace ImportVerilog;

// Detail processing about integer literal.
Value Context::visitIntegerLiteral(
    const slang::ast::IntegerLiteral *integerLiteralExpr) {
  auto srcValue = rootBuilder.getI32IntegerAttr(
      integerLiteralExpr->getValue().as<int32_t>().value());
  return rootBuilder.create<moore::ConstantOp>(
      convertLocation(integerLiteralExpr->sourceRange.start()),
      convertType(*integerLiteralExpr->type), srcValue);
}

// Detail processing about named value.
Value Context::visitNamedValue(
    const slang::ast::NamedValueExpression *namedValueExpr) {
  auto destName = namedValueExpr->getSymbolReference()->name;
  return varSymbolTable.lookup(destName);
}

// Detail processing about assignment.
LogicalResult Context::visitAssignmentExpr(
    const slang::ast::AssignmentExpression *assignmentExpr) {
  auto loc = convertLocation(assignmentExpr->sourceRange.start());
  Value lhs = visitExpression(&assignmentExpr->left());
  if (!lhs)
    return failure();
  Value rhs = visitExpression(&assignmentExpr->right());
  if (!rhs)
    return failure();
  if (assignmentExpr->right().kind == slang::ast::ExpressionKind::NamedValue) {
    mlir::emitError(loc, "unsupported assignment of kind like a = b");
    return failure();
  }
  rootBuilder.create<moore::AssignOp>(loc, lhs, rhs);
  auto lhsName = assignmentExpr->left().getSymbolReference()->name;
  varSymbolTable.insert(lhsName, rhs);
  return success();
}

// Detail processing about conversion
Value Context::visitConversion(
    const slang::ast::ConversionExpression *conversionExpr,
    const slang::ast::Type &type) {
  auto loc = convertLocation(conversionExpr->sourceRange.start());
  switch (conversionExpr->operand().kind) {
  case slang::ast::ExpressionKind::IntegerLiteral:
    return rootBuilder.create<moore::ConstantOp>(
        loc, convertType(type),
        conversionExpr->operand()
            .as<slang::ast::IntegerLiteral>()
            .getValue()
            .as<uint32_t>()
            .value());
  case slang::ast::ExpressionKind::NamedValue:
    return visitNamedValue(
        &conversionExpr->operand().as<slang::ast::NamedValueExpression>());
  case slang::ast::ExpressionKind::BinaryOp:
    mlir::emitError(loc, "unsupported conversion expression: binary operator");
    return nullptr;
  case slang::ast::ExpressionKind::ConditionalOp:
    mlir::emitError(loc,
                    "unsupported conversion expression: conditional operator");
    return nullptr;
  case slang::ast::ExpressionKind::Conversion:
    return visitConversion(
        &conversionExpr->operand().as<slang::ast::ConversionExpression>(),
        *conversionExpr->type);
  case slang::ast::ExpressionKind::LValueReference:
    mlir::emitError(loc, "unsupported conversion expression: lValue reference");
    return nullptr;
  // There is other cases.
  default:
    mlir::emitError(loc, "unsupported conversion expression");
    return nullptr;
  }
  return nullptr;
}

// It can handle the expressions like literal, assignment, conversion, and etc,
// which can be reviewed in slang/include/slang/ast/ASTVisitor.h.
Value Context::visitExpression(const slang::ast::Expression *expression) {
  auto loc = convertLocation(expression->sourceRange.start());
  switch (expression->kind) {
  case slang::ast::ExpressionKind::IntegerLiteral:
    return visitIntegerLiteral(&expression->as<slang::ast::IntegerLiteral>());
  case slang::ast::ExpressionKind::NamedValue:
    return visitNamedValue(&expression->as<slang::ast::NamedValueExpression>());
  case slang::ast::ExpressionKind::Assignment:
    visitAssignmentExpr(&expression->as<slang::ast::AssignmentExpression>());
    break;
  case slang::ast::ExpressionKind::Conversion:
    return visitConversion(&expression->as<slang::ast::ConversionExpression>(),
                           *expression->type);
  // There is other cases.
  default:
    mlir::emitError(loc, "unsupported expression");
    return nullptr;
  }
  return nullptr;
}

// It can handle the statements like case, conditional(if), for loop, and etc.
LogicalResult
Context::convertStatement(const slang::ast::Statement *statement) {
  auto loc = convertLocation(statement->sourceRange.start());
  switch (statement->kind) {
  case slang::ast::StatementKind::List:
    for (auto *stmt : statement->as<slang::ast::StatementList>().list) {
      convertStatement(stmt);
    }
    break;
  case slang::ast::StatementKind::Block:
    convertStatement(&statement->as<slang::ast::BlockStatement>().body);
    break;
  case slang::ast::StatementKind::ExpressionStatement:
    visitExpression(&statement->as<slang::ast::ExpressionStatement>().expr);
    break;
  case slang::ast::StatementKind::VariableDeclaration:
    return mlir::emitError(loc, "unsupported statement: variable declaration");
  case slang::ast::StatementKind::Return:
    return mlir::emitError(loc, "unsupported statement: return");
  case slang::ast::StatementKind::Break:
    return mlir::emitError(loc, "unsupported statement: break");
  case slang::ast::StatementKind::Continue:
    return mlir::emitError(loc, "unsupported statement: continue");
  case slang::ast::StatementKind::Case:
    return mlir::emitError(loc, "unsupported statement: case");
  case slang::ast::StatementKind::PatternCase:
    return mlir::emitError(loc, "unsupported statement: pattern case");
  case slang::ast::StatementKind::ForLoop:
    return mlir::emitError(loc, "unsupported statement: for loop");
  case slang::ast::StatementKind::RepeatLoop:
    return mlir::emitError(loc, "unsupported statement: repeat loop");
  case slang::ast::StatementKind::ForeachLoop:
    return mlir::emitError(loc, "unsupported statement: foreach loop");
  case slang::ast::StatementKind::WhileLoop:
    return mlir::emitError(loc, "unsupported statement: while loop");
  case slang::ast::StatementKind::DoWhileLoop:
    return mlir::emitError(loc, "unsupported statement: do while loop");
  case slang::ast::StatementKind::ForeverLoop:
    return mlir::emitError(loc, "unsupported statement: forever loop");
  case slang::ast::StatementKind::Timed:
    return mlir::emitError(loc, "unsupported statement: timed");
  case slang::ast::StatementKind::ImmediateAssertion:
    return mlir::emitError(loc, "unsupported statement: immediate assertion");
  case slang::ast::StatementKind::ConcurrentAssertion:
    return mlir::emitError(loc, "unsupported statement: concurrent assertion");
  case slang::ast::StatementKind::DisableFork:
    return mlir::emitError(loc, "unsupported statement: diable fork");
  case slang::ast::StatementKind::Wait:
    return mlir::emitError(loc, "unsupported statement: wait");
  case slang::ast::StatementKind::WaitFork:
    return mlir::emitError(loc, "unsupported statement: wait fork");
  case slang::ast::StatementKind::WaitOrder:
    return mlir::emitError(loc, "unsupported statement: wait order");
  case slang::ast::StatementKind::EventTrigger:
    return mlir::emitError(loc, "unsupported statement: event trigger");
  case slang::ast::StatementKind::ProceduralAssign:
    return mlir::emitError(loc, "unsupported statement: procedural assign");
  case slang::ast::StatementKind::ProceduralDeassign:
    return mlir::emitError(loc, "unsupported statement: procedural deassign");
  case slang::ast::StatementKind::RandCase:
    return mlir::emitError(loc, "unsupported statement: rand case");
  case slang::ast::StatementKind::RandSequence:
    return mlir::emitError(loc, "unsupported statement: rand sequence");
  case slang::ast::StatementKind::ProceduralChecker:
    return mlir::emitError(loc, "unsupported statement: procedural checker");
  case slang::ast::StatementKind::Conditional:
    return mlir::emitError(loc, "unsupported statement: conditional");
  default:
    mlir::emitRemark(loc, "unsupported statement");
    return failure();
  }

  return success();
}
