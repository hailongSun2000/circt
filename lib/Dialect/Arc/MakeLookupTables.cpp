//===- MakeLookupTables.cpp -----------------------------------------------===//
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
struct MakeLookupTablesPass : public MakeLookupTablesBase<MakeLookupTablesPass> {
  void runOnOperation() override;
  void runOnModule();
  HWModuleOp moduleOp;
};
} // namespace

void MakeLookupTablesPass::runOnOperation() {
  auto module = getOperation();
  for (auto op : llvm::make_early_inc_range(module.getOps<HWModuleOp>())) {
    moduleOp = op;
    runOnModule();
  }
}

void MakeLookupTablesPass::runOnModule() {
  LLVM_DEBUG(llvm::dbgs() << "Making lookup tables in `" << moduleOp.moduleName()
                          << "`\n");
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createMakeLookupTablesPass() {
  return std::make_unique<MakeLookupTablesPass>();
}
} // namespace arc
} // namespace circt
