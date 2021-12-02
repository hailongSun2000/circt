//===- VCDParser.h - A parser for VCD files -------------------------------===//
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

#ifndef CIRCT_VCD_PARSER_H
#define CIRCT_VCD_PARSER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include <memory>

namespace llvm {
class SourceMgr;
template <typename T>
class function_ref;
template <typename T>
class Optional;
} // namespace llvm

namespace mlir {
struct LogicalResult;
class MLIRContext;
} // namespace mlir

namespace circt {

class VCDVariable;
class VCDScope;
class VCDHeader;

/// A parser for VCD files.
class VCDParser {
public:
  VCDParser(llvm::SourceMgr &sourceMgr, unsigned fileID,
            mlir::MLIRContext *context);
  ~VCDParser();
  mlir::LogicalResult parseHeader();
  const VCDHeader &getHeader() const;

  mlir::LogicalResult parseNextTimestamp(llvm::Optional<uint64_t> &time);
  mlir::LogicalResult parseSignalChanges(
      llvm::function_ref<void(llvm::StringRef, llvm::StringRef)> callback);
  mlir::LogicalResult parseSignalChanges(
      llvm::function_ref<void(VCDVariable *, llvm::StringRef)> callback);

private:
  friend class VCDHeader;
  struct Impl;
  std::unique_ptr<Impl> impl;
};

/// A variable in a VCD file.
class VCDVariable {
public:
  VCDScope *scope = nullptr;
  llvm::StringRef type;
  uint64_t width;
  llvm::StringRef abbrev;
  llvm::StringRef name;
  llvm::StringRef tail;
};

/// A scope in a VCD file.
class VCDScope {
public:
  VCDScope *parent = nullptr;
  llvm::StringRef type;
  llvm::StringRef name;
  llvm::ArrayRef<VCDVariable *> variables;
  llvm::ArrayRef<VCDScope *> scopes;

  const VCDScope *findScope(llvm::StringRef name) const {
    for (auto *scope : scopes)
      if (scope->name == name)
        return scope;
    return nullptr;
  }
};

/// Header information contained in a VCD file.
class VCDHeader {
public:
  llvm::StringRef date;
  llvm::StringRef version;
  llvm::SmallVector<llvm::StringRef, 0> comments;
  uint64_t timescale;
  llvm::StringRef timescaleUnit;
  llvm::SmallVector<VCDScope *, 0> scopes;

  const VCDScope *findScope(llvm::StringRef name) const {
    for (auto *scope : scopes)
      if (scope->name == name)
        return scope;
    return nullptr;
  }

private:
  friend struct VCDParser::Impl;

  /// Allocator for the scope and variable structures.
  llvm::BumpPtrAllocator allocator;
};

} // namespace circt

#endif // CIRCT_VCD_PARSER_H
