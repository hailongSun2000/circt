//===- DeleteLocalVar.cpp - Delete local temporary variables --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DeleteLocalVar pass.
// Use to collect net/variable declarations and bound a value to them.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace circt;
using namespace moore;

namespace {
struct DeleteLocalVarPass : public DeleteLocalVarBase<DeleteLocalVarPass> {
  void runOnOperation() override;
};
} // namespace

static void deleteLocalVar(DenseMap<Operation *, Value> &localVars,
                           DenseSet<Operation *> &users, mlir::Region &region) {
  for (auto &op : region.getOps()) {
    // Assume not to assign a new value to the local variable.
    bool isNewValue = false;

    TypeSwitch<Operation *, void>(&op)
        // Collect all local variables and their initial values if existand all
        // users.
        .Case<VariableOp>([&](auto op) {
          localVars[op] = op.getInitial();
          for (auto *user : op->getUsers())
            users.insert(user);
        })
        // Update the values of local variables.
        .Case<BlockingAssignOp>([&](auto op) {
          auto *destOp = op.getDst().getDefiningOp();
          auto srcValue = op.getSrc();
          if (localVars.contains(destOp)) {
            localVars[destOp] = srcValue;
            isNewValue = true;
          }
        })
        // Delete the local variables defined in `if` statements.
        .Case<mlir::scf::IfOp>([&](auto op) {
          auto *thenRegion = &op.getThenRegion();
          auto *elseRegion = &op.getElseRegion();

          // Handle `then` region.
          if (!thenRegion->empty()) {
            DenseMap<Operation *, Value> localVars;
            DenseSet<Operation *> users;
            deleteLocalVar(localVars, users, *thenRegion);
          }

          // Handle `else` region.
          if (!elseRegion->empty()) {
            DenseMap<Operation *, Value> localVars;
            DenseSet<Operation *> users;
            deleteLocalVar(localVars, users, *elseRegion);
          }
        });

    // Assume the `a` is a local variable, which one user is `b = a`.
    // Replace `a` with its value, then erase this user from `users` container.
    // Although `a = 1` is also the user of `a`, don't replace it.
    if (users.contains(&op) && !isNewValue) {
      for (auto operand : op.getOperands()) {
        if (auto value = localVars.lookup(operand.getDefiningOp()))
          op.replaceUsesOfWith(operand, value);
        users.erase(&op);
      }
    }
  }

  // Erase the redundant users of local varaibles like `a = 1`.
  for (auto *user : users)
    user->erase();

  // Erase the local variables.
  for (auto var : localVars)
    var.first->erase();
}

void DeleteLocalVarPass::runOnOperation() {
  getOperation()->walk([&](SVModuleOp moduleOp) {
    moduleOp->walk([&](ProcedureOp procedureOp) {
      // Used to collect the local temporary variables and their values.
      DenseMap<Operation *, Value> localVars;

      // Used to collect all users of local variables
      DenseSet<Operation *> users;

      deleteLocalVar(localVars, users, procedureOp.getBodyRegion());

      return WalkResult::advance();
    });
    return WalkResult::advance();
  });
}

std::unique_ptr<mlir::Pass> circt::moore::createDeleteLocalVarPass() {
  return std::make_unique<DeleteLocalVarPass>();
}
