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
      .Case<VariableOp>([&](auto op) {
        auto operandIt = op.getOperands();
        auto value = operandIt.empty() ? nullptr : op.getOperand(0);
        assignmentChains[op] = value;
        nameBindings[cast<VariableOp>(op).getName()] = op;
      })
      .Case<NetOp>([&](auto op) {
        auto operandIt = op.getOperands();
        auto value = operandIt.empty() ? nullptr : op.getOperand(0);
        assignmentChains[op] = value;
        nameBindings[cast<NetOp>(op).getName()] = op;
      })
      .Case<ContinuousAssignOp, BlockingAssignOp, NonBlockingAssignOp>(
          [&](auto op) {
            auto destOp = op.getOperand(0).getDefiningOp();
            auto srcValue = op.getOperand(1);
            assignmentChains[destOp] = srcValue;
          })
      .Case<ProcedureOp>([&](auto op) {
        for (auto &nestOp : op.getOps()) {
          addValue(&nestOp);
        }
      });
}

void Declaration::buildPortBending(PortOp op) {
  SmallVector<Operation *> assignment;
  if (op.getDirection() == Direction::Out) {
    auto users = nameBindings[op.getName()]->getUsers();
    for (auto *user : users)
      assignment.push_back(user);

    outPortChains[op] =
        std::make_pair(nameBindings[op.getName()], assignment.front());
  }
}

extern Declaration moore::decl;
void MooreDeclarationsPass::runOnOperation() {
  getOperation()->walk([&](SVModuleOp moduleOp) {
    for (auto &op : moduleOp.getOps())
      decl.addValue(&op);

    for (auto portOp : moduleOp.getOps<PortOp>())
      decl.buildPortBending(portOp);

    return WalkResult::advance();
  });
}

std::unique_ptr<mlir::Pass> circt::moore::createMooreDeclarationsPass() {
  return std::make_unique<MooreDeclarationsPass>();
}
