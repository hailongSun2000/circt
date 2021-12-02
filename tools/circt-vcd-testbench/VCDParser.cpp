//===- VCDParser.cpp - A parser for VCD files -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a parser for Value Change Dump files.
//
//===----------------------------------------------------------------------===//

#include "VCDParser.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "vcd-parser"

using namespace llvm;
using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Parser Implementation
//===----------------------------------------------------------------------===//

struct VCDParser::Impl {
  Impl(llvm::SourceMgr &sourceMgr, unsigned fileID, mlir::MLIRContext *context)
      : sourceMgr(sourceMgr), fileID(fileID), context(context) {
    auto *memBuffer = sourceMgr.getMemoryBuffer(fileID);
    bufferName = StringAttr::get(memBuffer->getBufferIdentifier(), context);
    buffer = memBuffer->getBuffer();
    cur = buffer;
  }

  Location translateLocation();
  Location translateLocation(const char *ptr);
  Location translateLocation(SMLoc loc);

  bool isEOF() { return cur.empty(); }
  void skipWS(bool skipVertical = true);
  ParseResult consumeUntilEndKeyword(StringRef &text);
  ParseResult parseIdentifier(StringRef &text, const Twine &message);
  ParseResult parseKeyword(StringRef keyword);
  ParseResult parseInteger(StringRef &text, const Twine &message);
  ParseResult parseInteger(APInt &result, const Twine &message);
  ParseResult parseInteger(uint64_t &result, const Twine &message);

  ParseResult parseHeader();
  ParseResult parseScope(VCDScope *&scope);
  ParseResult parseNextTimestamp(Optional<uint64_t> &time);
  ParseResult
  parseSignalChanges(llvm::function_ref<void(StringRef, StringRef)> callback);

  llvm::SourceMgr &sourceMgr;
  const unsigned fileID;
  mlir::MLIRContext *context;

  StringAttr bufferName;
  StringRef buffer;
  StringRef cur;

  const char *prevDate = nullptr;
  const char *prevVersion = nullptr;
  const char *prevTimescale = nullptr;

  VCDHeader header;
  llvm::DenseMap<StringRef, SmallVector<VCDVariable *, 1>> variablesByAbbrev;
};

/// Encode the current source location information into a Location object for
/// error reporting.
Location VCDParser::Impl::translateLocation() {
  return translateLocation(cur.begin());
}

/// Encode the current source location information into a Location object for
/// error reporting.
Location VCDParser::Impl::translateLocation(const char *ptr) {
  assert(ptr >= buffer.begin() && ptr <= buffer.end());
  return translateLocation(SMLoc::getFromPointer(ptr));
}

/// Encode the specified source location information into a Location object
/// for error reporting.
Location VCDParser::Impl::translateLocation(SMLoc loc) {
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, fileID);
  return FileLineColLoc::get(bufferName, lineAndColumn.first,
                             lineAndColumn.second);
}

static inline bool isHorizontalWS(char c) { return c == ' ' || c == '\t'; }

static inline bool isVerticalWS(char c) {
  return c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

static inline bool isWS(char c) { return isHorizontalWS(c) || isVerticalWS(c); }

void VCDParser::Impl::skipWS(bool skipVertical) {
  cur = cur.drop_while(skipVertical ? isWS : isHorizontalWS);
}

ParseResult VCDParser::Impl::consumeUntilEndKeyword(StringRef &text) {
  auto end = cur.find("$end");
  if (end == StringRef::npos) {
    mlir::emitError(translateLocation(cur.end()),
                    "expected `$end` keyword before end of input");
    return failure();
  }
  text = cur.slice(0, end).trim();
  cur = cur.drop_front(end + 4);
  return success();
}

ParseResult VCDParser::Impl::parseIdentifier(StringRef &text,
                                             const Twine &message) {
  skipWS();
  text = cur.take_until(isWS);
  if (text.empty()) {
    mlir::emitError(translateLocation(), message);
    return failure();
  }
  cur = cur.drop_front(text.size());
  return success();
}

ParseResult VCDParser::Impl::parseKeyword(StringRef keyword) {
  skipWS();
  if (cur.consume_front(keyword))
    return success();
  mlir::emitError(translateLocation(), "expected keyword `") << keyword << "`";
  return failure();
}

ParseResult VCDParser::Impl::parseInteger(StringRef &text,
                                          const Twine &message) {
  skipWS();
  text = cur.take_while([](char c) { return c >= '0' && c <= '9'; });
  if (text.empty()) {
    mlir::emitError(translateLocation(), message);
    return failure();
  }
  cur = cur.drop_front(text.size());
  return success();
}

ParseResult VCDParser::Impl::parseInteger(APInt &result, const Twine &message) {
  StringRef text;
  if (parseInteger(text, message))
    return failure();
  if (text.getAsInteger(10, result)) {
    mlir::emitError(translateLocation(), message);
    return failure();
  }
  return success();
}

ParseResult VCDParser::Impl::parseInteger(uint64_t &result,
                                          const Twine &message) {
  const char *ptr = cur.begin();
  APInt value;
  if (parseInteger(value, message))
    return failure();
  result = value.getLimitedValue();
  if (result != value) {
    mlir::emitError(translateLocation(ptr), "integer `")
        << result << "` is too big";
    return failure();
  }
  return success();
}

ParseResult VCDParser::Impl::parseHeader() {
  while (!isEOF()) {
    skipWS();
    const char *ptr = cur.begin();

    // `$date` <text> `$end`
    if (cur.consume_front("$date")) {
      if (consumeUntilEndKeyword(header.date))
        return failure();
      if (prevDate) {
        mlir::emitWarning(translateLocation(ptr),
                          "`$date` overrides earlier directive")
                .attachNote(translateLocation(prevDate))
            << "earlier `$date` directive was here:";
      }
      prevDate = ptr;
      continue;
    }

    // `$version` <text> `$end`
    if (cur.consume_front("$version")) {
      if (consumeUntilEndKeyword(header.version))
        return failure();
      if (prevVersion) {
        mlir::emitWarning(translateLocation(ptr),
                          "`$version` overrides earlier directive")
                .attachNote(translateLocation(prevVersion))
            << "earlier `$version` directive was here:";
      }
      prevVersion = ptr;
      continue;
    }

    // `$comment` <text> `$end`
    if (cur.consume_front("$comment")) {
      StringRef text;
      if (consumeUntilEndKeyword(text))
        return failure();
      header.comments.push_back(text);
      continue;
    }

    // `$timescale` <integer> <unit> `$end`
    if (cur.consume_front("$timescale")) {
      if (parseInteger(header.timescale, "expected timescale value") ||
          parseIdentifier(header.timescaleUnit, "expected timescale unit") ||
          parseKeyword("$end"))
        return failure();
      if (prevTimescale) {
        mlir::emitWarning(translateLocation(ptr),
                          "`$timescale` overrides earlier directive")
                .attachNote(translateLocation(prevTimescale))
            << "earlier `$timescale` directive was here:";
      }
      prevTimescale = ptr;
      continue;
    }

    // `$scope` ...
    if (cur.startswith("$scope")) {
      VCDScope *scope;
      if (parseScope(scope))
        return failure();
      header.scopes.push_back(scope);
      continue;
    }

    // `$enddefinitions` `$end`
    if (cur.consume_front("$enddefinitions")) {
      if (parseKeyword("$end"))
        return failure();
      break;
    }

    // `$dumpvars` indicates the end of the header.
    if (cur.startswith("$dumpvars"))
      break;

    // Everything else is a syntax error.
    mlir::emitError(translateLocation(), "expected header directive");
    return failure();
  }

  // `$dumpvars` should follow the header.
  if (parseKeyword("$dumpvars"))
    return failure();

  return success();
}

ParseResult VCDParser::Impl::parseScope(VCDScope *&scope) {
  // `$scope` <scope-type> <scope-name> `$end`
  scope = new (header.allocator) VCDScope();
  if (parseKeyword("$scope") ||
      parseIdentifier(scope->type, "expected scope type") ||
      parseIdentifier(scope->name, "expected scope name") ||
      parseKeyword("$end"))
    return failure();

  // Parse the scope body.
  SmallVector<VCDVariable *> variables;
  SmallVector<VCDScope *> scopes;

  while (!isEOF()) {
    skipWS();

    // `$var` <type> <width> <abbrev> <name> `$end`
    if (cur.consume_front("$var")) {
      VCDVariable *var = new (header.allocator) VCDVariable();
      var->scope = scope;
      if (parseIdentifier(var->type, "expected variable type") ||
          parseInteger(var->width, "expected width") ||
          parseIdentifier(var->abbrev, "expected variable abbreviation") ||
          parseIdentifier(var->name, "expected variable name") ||
          consumeUntilEndKeyword(var->tail))
        return failure();
      variables.push_back(var);
      variablesByAbbrev[var->abbrev].push_back(var);
      continue;
    }

    // `$scope` ...
    if (cur.startswith("$scope")) {
      VCDScope *subscope;
      if (parseScope(subscope))
        return failure();
      subscope->parent = scope;
      scopes.push_back(subscope);
      continue;
    }

    // `$upscope` indicates the end of the scope.
    if (cur.startswith("$upscope"))
      break;

    // Everything else is a syntax error.
    mlir::emitError(translateLocation(), "expected variable or nested scope");
    return failure();
  }

  // `$upscope` `$end` terminates the scope.
  if (parseKeyword("$upscope") || parseKeyword("$end"))
    return failure();

  // Allocate the scope and variable arrays.
  VCDScope **scopesArray = header.allocator.Allocate<VCDScope *>(scopes.size());
  VCDVariable **variablesArray =
      header.allocator.Allocate<VCDVariable *>(variables.size());
  std::copy(scopes.begin(), scopes.end(), scopesArray);
  std::copy(variables.begin(), variables.end(), variablesArray);
  scope->scopes = ArrayRef(scopesArray, scopes.size());
  scope->variables = ArrayRef(variablesArray, variables.size());

  return success();
}

ParseResult VCDParser::Impl::parseNextTimestamp(Optional<uint64_t> &time) {
  size_t pos = cur.find('#');
  if (pos == StringRef::npos) {
    time = {};
    return success();
  }
  cur = cur.drop_front(pos + 1);
  uint64_t timeValue;
  if (parseInteger(timeValue, "expected time value"))
    return failure();
  time = timeValue;
  return success();
}

ParseResult VCDParser::Impl::parseSignalChanges(
    llvm::function_ref<void(StringRef, StringRef)> callback) {
  while (!isEOF()) {
    skipWS();
    if (isEOF())
      break;

    // Multi-bit value change.
    if (cur.consume_front("b")) {
      StringRef value, abbrev;
      if (parseIdentifier(value, "expected signal value") ||
          parseIdentifier(abbrev, "expected signal abbreviation"))
        return failure();
      callback(abbrev, value);
      continue;
    }

    // Next timestamp is introduced with `#`.
    if (cur[0] == '#')
      break;

    // Single-bit value change.
    StringRef value, abbrev;
    value = cur.slice(0, 1);
    cur = cur.drop_front();
    if (parseIdentifier(abbrev, "expected signal abbreviation"))
      return failure();
    callback(abbrev, value);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Parser Interface
//===----------------------------------------------------------------------===//

VCDParser::VCDParser(llvm::SourceMgr &sourceMgr, unsigned fileID,
                     mlir::MLIRContext *context)
    : impl(std::make_unique<Impl>(sourceMgr, fileID, context)) {}

VCDParser::~VCDParser() {}

LogicalResult VCDParser::parseHeader() {
  LLVM_DEBUG(llvm::dbgs() << "Parsing VCD file header\n");
  return impl->parseHeader();
}

LogicalResult VCDParser::parseNextTimestamp(Optional<uint64_t> &time) {
  return impl->parseNextTimestamp(time);
}

LogicalResult VCDParser::parseSignalChanges(
    llvm::function_ref<void(StringRef, StringRef)> callback) {
  return impl->parseSignalChanges(callback);
}

LogicalResult VCDParser::parseSignalChanges(
    llvm::function_ref<void(VCDVariable *, StringRef)> callback) {
  return impl->parseSignalChanges([&](StringRef abbrev, StringRef value) {
    auto it = impl->variablesByAbbrev.find(abbrev);
    if (it == impl->variablesByAbbrev.end())
      return;
    for (auto *var : it->second)
      callback(var, value);
  });
}

const VCDHeader &VCDParser::getHeader() const { return impl->header; }
