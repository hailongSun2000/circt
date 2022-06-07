//===- StripSV.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"
#include <variant>

#define DEBUG_TYPE "arc-strip-sv"

using namespace circt;
using namespace arc;

namespace {
struct StripSVPass : public StripSVBase<StripSVPass> {
  void runOnOperation() override;
  SmallVector<Operation *> opsToDelete;
  SmallPtrSet<StringAttr, 4> clockGateModuleNames;
  SmallPtrSet<StringAttr, 4> externModuleNames;
};
} // namespace

void StripSVPass::runOnOperation() {
  auto mlirModule = getOperation();
  opsToDelete.clear();
  clockGateModuleNames.clear();
  externModuleNames.clear();

  auto expectedClockGateInputs =
      ArrayAttr::get(&getContext(), {StringAttr::get(&getContext(), "in"),
                                     StringAttr::get(&getContext(), "test_en"),
                                     StringAttr::get(&getContext(), "en")});
  auto expectedClockGateOutputs =
      ArrayAttr::get(&getContext(), {StringAttr::get(&getContext(), "out")});
  auto i1Type = IntegerType::get(&getContext(), 1);

  for (auto extModOp : mlirModule.getOps<hw::HWModuleExternOp>()) {
    if (extModOp.getVerilogModuleName() == "EICG_wrapper") {
      if (extModOp.argNames() != expectedClockGateInputs ||
          extModOp.resultNames() != expectedClockGateOutputs) {
        extModOp.emitError("clock gate module `")
            << extModOp.moduleName() << "` has incompatible port names "
            << extModOp.argNamesAttr() << " -> " << extModOp.resultNamesAttr();
        return signalPassFailure();
      }
      if (extModOp.getArgumentTypes() !=
              ArrayRef<Type>{i1Type, i1Type, i1Type} ||
          extModOp.getResultTypes() != ArrayRef<Type>{i1Type}) {

        extModOp.emitError("clock gate module `")
            << extModOp.moduleName() << "` has incompatible port types "
            << extModOp.getArgumentTypes() << " -> "
            << extModOp.getResultTypes();
        return signalPassFailure();
      }
      clockGateModuleNames.insert(extModOp.moduleNameAttr());
    } else {
      externModuleNames.insert(extModOp.moduleNameAttr());
    }
    opsToDelete.push_back(extModOp);
  }
  LLVM_DEBUG(llvm::dbgs() << "Found " << clockGateModuleNames.size()
                          << " clock gates, " << externModuleNames.size()
                          << " other extern modules\n");

  for (auto module : mlirModule.getOps<hw::HWModuleOp>()) {
    for (Operation &op : *module.getBodyBlock()) {
      // Remove wires.
      if (auto assign = dyn_cast<sv::AssignOp>(&op)) {
        auto wire = assign.dest().getDefiningOp<sv::WireOp>();
        if (!wire) {
          assign.emitOpError("expected wire lhs");
          return signalPassFailure();
        }
        for (Operation *user : wire->getUsers()) {
          if (user == assign)
            continue;
          auto readInout = dyn_cast<sv::ReadInOutOp>(user);
          if (!readInout) {
            user->emitOpError("has user that is not `sv.read_inout`");
            return signalPassFailure();
          }
          readInout.replaceAllUsesWith(assign.src());
          opsToDelete.push_back(readInout);
        }
        opsToDelete.push_back(assign);
        opsToDelete.push_back(wire);
      }

      // Canonicalize registers.
      if (auto assign = dyn_cast<arc::RegAssignOp>(&op)) {
        auto reg = assign.dest().getDefiningOp<sv::RegOp>();
        if (!reg) {
          assign.emitOpError("expected reg lhs");
          return signalPassFailure();
        }
        OpBuilder builder(assign);
        Value compReg = builder.create<seq::CompRegOp>(
            assign.getLoc(), assign.src().getType(), assign.src(),
            assign.clock(), reg.nameAttr(), Value{}, Value{},
            reg.inner_symAttr());
        for (Operation *user : reg->getUsers()) {
          if (user == assign)
            continue;
          auto readInout = dyn_cast<sv::ReadInOutOp>(user);
          if (!readInout) {
            user->emitOpError("has user that is not `sv.read_inout`");
            return signalPassFailure();
          }
          readInout.replaceAllUsesWith(compReg);
          opsToDelete.push_back(readInout);
        }
        opsToDelete.push_back(assign);
        opsToDelete.push_back(reg);
      }

      // Replace clock gate instances with the dedicated arc op and stub out
      // other external modules.
      if (auto instOp = dyn_cast<hw::InstanceOp>(&op)) {
        auto modName = instOp.moduleNameAttr().getAttr();
        ImplicitLocOpBuilder builder(instOp.getLoc(), instOp);
        if (clockGateModuleNames.contains(modName)) {
          auto enable = builder.createOrFold<comb::OrOp>(instOp.getOperand(1),
                                                         instOp.getOperand(2));
          auto gated =
              builder.create<arc::ClockGateOp>(instOp.getOperand(0), enable);
          instOp.replaceAllUsesWith(gated);
          opsToDelete.push_back(instOp);
        } else if (externModuleNames.contains(modName)) {
          for (auto result : instOp.getResults())
            result.replaceAllUsesWith(
                builder.create<hw::ConstantOp>(result.getType(), 0));
          opsToDelete.push_back(instOp);
        }
      }
    }
  }
  for (auto *op : opsToDelete)
    op->erase();
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createStripSVPass() {
  return std::make_unique<StripSVPass>();
}
} // namespace arc
} // namespace circt
