//===- InlineArcInputs.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-lookup-tables"

using namespace circt;
using namespace arc;
using namespace hw;

namespace {
struct InlineArcInputsPass : public InlineArcInputsBase<InlineArcInputsPass> {
  void runOnOperation() override;
  void runOnModule();
  HWModuleOp moduleOp;
};
} // namespace

void InlineArcInputsPass::runOnOperation() {
  auto module = getOperation();
  for (auto op : llvm::make_early_inc_range(module.getOps<HWModuleOp>())) {
    moduleOp = op;
    runOnModule();
  }
}

void InlineArcInputsPass::runOnModule() {
  LLVM_DEBUG(llvm::dbgs() << "Inline arc inputs in `" << moduleOp.moduleName()
                          << "`\n");
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createInlineArcInputsPass() {
  return std::make_unique<InlineArcInputsPass>();
}
} // namespace arc
} // namespace circt
