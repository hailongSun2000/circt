//===- Statement.cpp - Slang statement conversion--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "slang/ast/ASTContext.h"
#include "slang/ast/ASTVisitor.h"
#include "slang/ast/Symbol.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/syntax/SyntaxVisitor.h"
#include <cassert>
#include <cstddef>
#include <slang/ast/Statements.h>
#include <slang/ast/expressions/AssignmentExpressions.h>
#include <slang/text/SourceLocation.h>
#include <variant>

using namespace circt;
using namespace ImportVerilog;

decltype(auto)
Context::visitStmt(const slang::ast::ExpressionStatement *exprStmt) {

  auto assignExpr = &exprStmt->expr.as<slang::ast::AssignmentExpression>();
  // Get the name of variable.
  auto destName = assignExpr->left().getSymbolReference()->name;
  // Get the type of variable.
  auto destType = convertType(*assignExpr->left().type);
  // Get the location of the variable.
  auto destLoc =
      convertLocation(assignExpr->left().getSymbolReference()->location);

  auto exprRight = &assignExpr->right().as<slang::ast::ConversionExpression>();
  slang::ast::EvalContext ctx(compilation);
  // Get the outer type, rather than the type of the operand scope.
  IntegerType srcType =
      rootBuilder.getIntegerType(exprRight->operand().type->getBitWidth());
  IntegerAttr srcValue = rootBuilder.getIntegerAttr(
      srcType, *exprRight->eval(ctx).integer().getRawPtr());
  Value dest = rootBuilder.create<moore::VariableDeclOp>(
      destLoc, moore::LValueType::get(destType),
      rootBuilder.getStringAttr(destName),
      *exprRight->eval(ctx).integer().getRawPtr());
  Value src = rootBuilder.create<moore::ConstantOp>(
      destLoc, convertType(*assignExpr->right().type), srcValue);

  rootBuilder.create<moore::AssignOp>(destLoc, dest, src);
}

void Context::convertStatement(const slang::ast::Statement *statement) {
  slang::ast::StatementKind statementKind = statement->kind;
  auto loc = convertLocation(statement->sourceRange.start());
  switch (statementKind) {
  case slang::ast::StatementKind::Invalid:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::Empty:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::List:
    for (auto it = statement->as<slang::ast::StatementList>().list.begin();
         it != statement->as<slang::ast::StatementList>().list.end(); it++) {
      convertStatement(*it);
    }
    break;
  case slang::ast::StatementKind::Block:
    convertStatement(&statement->as<slang::ast::BlockStatement>().body);
    break;
  case slang::ast::StatementKind::ExpressionStatement:
    visitStmt(&statement->as<slang::ast::ExpressionStatement>());
    break;
  case slang::ast::StatementKind::VariableDeclaration:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::Return:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::Break:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::Continue:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::Case:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::PatternCase:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::ForLoop:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::RepeatLoop:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::ForeachLoop:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::WhileLoop:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::DoWhileLoop:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::ForeverLoop:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::Timed:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::ImmediateAssertion:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::ConcurrentAssertion:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::DisableFork:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::Wait:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::WaitFork:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::WaitOrder:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::EventTrigger:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::ProceduralAssign:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::ProceduralDeassign:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::RandCase:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::RandSequence:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::ProceduralChecker:
    assert(0 && "TODO");
    break;
  case slang::ast::StatementKind::Conditional:
    assert(0 && "TODO");
    break;
  default:
    mlir::emitRemark(loc, "unsupported statement");
    break;
  }
}