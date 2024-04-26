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
      // Collect all variables and their initial values.
      .Case<VariableOp>([&](auto op) {
        auto value = op.getInitial();
        assignmentChains[op] = value;
      })
      // Collect all nets and their initial values.
      .Case<NetOp>([&](auto op) {
        auto value = op.getAssignment();
        assignmentChains[op] = value;
      })
      // Update the values of the nets/variables. Just reserve the last
      // assignment.
      .Case<ContinuousAssignOp, NonBlockingAssignOp>([&](auto op) {
        auto destOp = op.getDst().getDefiningOp();
        auto srcValue = op.getSrc();
        assignmentChains[destOp] = srcValue;
      })
      // Simulate the blocking behavior by substituting the variable with its
      // value.
      .Case<BlockingAssignOp>([&](auto op) {
        auto destOp = op.getDst().getDefiningOp();
        auto srcValue = op.getSrc();
        auto srcOp = srcValue.getDefiningOp();
        assignmentChains[destOp] =
            assignmentChains.contains(srcOp)
                ? assignmentChains.lookup(srcValue.getDefiningOp())
                : srcValue;
      })
      .Case<ProcedureOp>([&](auto op) {
        for (auto &nestOp : op) {
          addValue(&nestOp);

          // To simulate the blocking behavior and avoid the cycle/feedback
          // circuit, replace the operands for other operations excluding
          // blocking assignment.
          if (!isa<BlockingAssignOp>(nestOp))
            for (auto operand : nestOp.getOperands())
              if (auto value = assignmentChains.lookup(operand.getDefiningOp()))
                nestOp.replaceUsesOfWith(operand, value);
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
