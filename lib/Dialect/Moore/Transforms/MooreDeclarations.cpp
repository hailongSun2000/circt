//===- MooreDeclarations.cpp - Collect net/variable declarations ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MooreDeclarations pass.
// Use to collect net/variable declarations and bound a value to them.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace circt;
using namespace moore;

namespace {
struct MooreDeclarationsPass
    : public MooreDeclarationsBase<MooreDeclarationsPass> {
  void runOnOperation() override;
};
} // namespace

void Declaration::addValue(Operation *op) {
  TypeSwitch<Operation *, void>(op)
      // TODO: The loop, if, case, and other statements.
      .Case<VariableOp, NetOp>([&](auto op) {
        auto operandIt = op.getOperands();
        auto value = operandIt.empty() ? nullptr : op.getOperand(0);
        assignmentChains[op] = value;
      })
      .Case<CAssignOp, BPAssignOp, PAssignOp, PCAssignOp>([&](auto op) {
        auto destOp = op.getOperand(0).getDefiningOp();
        auto srcValue = op.getOperand(1);
        assignmentChains[destOp] = srcValue;
      })
      .Case<PortOp>([&](auto op) {
        SmallVector<Operation *> assignments;

        if (op.getDirection() == Direction::Out) {
          auto users = op.getResult().getUsers();
          if (!users.empty()) {
            for (auto *user : users)
              assignments.push_back(user);

            portChains[op] = assignments.front();
          }
        }
      })
      .Case<ProcedureOp>([&](auto op) {
        for (auto &nestOp : op.getOps()) {
          addValue(&nestOp);
        }
      });
}

extern Declaration moore::decl;
void MooreDeclarationsPass::runOnOperation() {
  getOperation()->walk([&](SVModuleOp moduleOp) {
    for (auto &op : moduleOp.getOps()) {
      decl.addValue(&op);
    };
    return WalkResult::advance();
  });
}

std::unique_ptr<mlir::Pass> circt::moore::createMooreDeclarationsPass() {
  return std::make_unique<MooreDeclarationsPass>();
}
