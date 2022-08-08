//===- Passes.h -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_PASSES_H
#define CIRCT_DIALECT_ARC_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace circt {
namespace arc {

std::unique_ptr<mlir::Pass> createStripSVPass();
std::unique_ptr<mlir::Pass> createPushOutputLogicPass();
std::unique_ptr<mlir::Pass> createPullInputLogicPass();
std::unique_ptr<mlir::Pass> createRegToArcPass();
std::unique_ptr<mlir::Pass> createInferMemoriesPass();
std::unique_ptr<mlir::Pass> createConvertToArcsPass();
std::unique_ptr<mlir::Pass> createInlineModulesPass();
std::unique_ptr<mlir::Pass> createInlineArcInputsPass();
std::unique_ptr<mlir::Pass> createDedupPass();
std::unique_ptr<mlir::Pass> createDumpArcGraphPass();
std::unique_ptr<mlir::Pass> createPrintArcInfoPass();
std::unique_ptr<mlir::Pass> createLowerStatePass();
std::unique_ptr<mlir::Pass> createScheduleModelsPass();
std::unique_ptr<mlir::Pass> createSplitLoopsPass();
std::unique_ptr<mlir::Pass> createAllocateStatePass();
std::unique_ptr<mlir::Pass> createPrintStateInfoPass();
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Arc/Passes.h.inc"

} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_PASSES_H
