//===- InfoCollection.cpp - Collect net/variable declarations ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InfoCollection pass.
// Use to collect net/variable declarations and bound a value to them.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/Moore/MoorePasses.h"

using namespace circt;
using namespace moore;

namespace {
struct InfoCollectionPass : public InfoCollectionBase<InfoCollectionPass> {
  void runOnOperation() override;
};
} // namespace

void Declaration::addValue(Operation *op) {
  TypeSwitch<Operation *, void>(op)
      .Case<VariableOp>([&](auto op) {
        auto value = op.getInitial();
        assignmentChains[op] = value;
      })
      .Case<NetOp>([&](auto op) {
        auto value = op.getAssignment();
        assignmentChains[op] = value;
      })
      .Case<ContinuousAssignOp, BlockingAssignOp, NonBlockingAssignOp>(
          [&](auto op) {
            auto destOp = op.getDst().getDefiningOp();
            auto srcValue = op.getSrc();
            assignmentChains[destOp] = srcValue;
          })
      .Case<ProcedureOp>([&](auto op) {
        for (auto &nestOp : op.getOps()) {
          addValue(&nestOp);
        }
      });
}

extern Declaration moore::decl;
void InfoCollectionPass::runOnOperation() {
  getOperation()->walk([&](SVModuleOp moduleOp) {
    for (auto &op : moduleOp.getOps())
      decl.addValue(&op);

    return WalkResult::advance();
  });
}

std::unique_ptr<mlir::Pass> circt::moore::createInfoCollectionPass() {
  return std::make_unique<InfoCollectionPass>();
}
