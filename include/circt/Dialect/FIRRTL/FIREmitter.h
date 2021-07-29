//===- FIREmitter.h - FIRRTL dialect to .fir emitter ------------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the .fir file emitter.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIREMITTER_H
#define CIRCT_DIALECT_FIRRTL_FIREMITTER_H

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
struct LogicalResult;
class ModuleOp;
} // namespace mlir

namespace circt {
namespace firrtl {

struct FIREmitterOptions {};

mlir::LogicalResult exportFIRRTL(mlir::ModuleOp module, llvm::raw_ostream &os,
                                 const FIREmitterOptions &options = {});

void registerToFIRRTLTranslation();

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIREMITTER_H
