//===- PrintStateInfo.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Conversion/LLHDToLLVM.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/ControlFlowSinkUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "lower-arc-to-llvm"

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace hw;

using llvm::MapVector;
using llvm::SmallDenseSet;
using llvm::SmallSetVector;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct StateInfo {
  enum Type { Input, Output, Register, Memory } type;
  StringAttr name;
  unsigned offset;
  unsigned numBits;
  unsigned memoryStride = 0; // byte separation between memory words
  unsigned memoryDepth = 0;  // number of words in a memory
};

struct ModelInfo {
  size_t numStateBytes;
  std::vector<StateInfo> states;
};

struct PrintStateInfoPass : public PrintStateInfoBase<PrintStateInfoPass> {
  void runOnOperation() override;
  void collectStates(Value storage, unsigned offset,
                     std::vector<StateInfo> &stateInfos);

  ModelOp modelOp;
  DenseMap<ModelOp, ModelInfo> modelInfos;
};
} // namespace

void PrintStateInfoPass::runOnOperation() {
  if (stateFile.empty())
    return;

  // Open the output file.
  std::error_code ec;
  llvm::ToolOutputFile outputFile(stateFile, ec,
                                  llvm::sys::fs::OpenFlags::OF_None);
  if (ec) {
    mlir::emitError(getOperation().getLoc(), "unable to open state file: ")
        << ec.message();
    return signalPassFailure();
  }

  llvm::json::OStream json(outputFile.os(), 2);
  json.array([&] {
    std::vector<StateInfo> states;
    for (auto modelOp : getOperation().getOps<ModelOp>()) {
      auto storageArg = modelOp.body().getArgument(0);
      auto storageType = storageArg.getType().cast<StorageType>();
      states.clear();
      collectStates(storageArg, 0, states);
      llvm::sort(states, [](auto &a, auto &b) { return a.offset < b.offset; });

      json.object([&] {
        json.attribute("name", modelOp.name());
        json.attribute("numStateBytes", storageType.getSize());
        json.attributeArray("states", [&] {
          for (const auto &state : states) {
            json.object([&] {
              if (state.name && !state.name.getValue().empty())
                json.attribute("name", state.name.getValue());
              json.attribute("offset", state.offset);
              json.attribute("numBits", state.numBits);
              auto typeStr = [](StateInfo::Type type) {
                switch (type) {
                case StateInfo::Input:
                  return "input";
                case StateInfo::Output:
                  return "output";
                case StateInfo::Register:
                  return "register";
                case StateInfo::Memory:
                  return "memory";
                }
              };
              json.attribute("type", typeStr(state.type));
              if (state.type == StateInfo::Memory) {
                json.attribute("stride", state.memoryStride);
                json.attribute("depth", state.memoryDepth);
              }
            });
          }
        });
      });
    }
  });
  outputFile.keep();
}

void PrintStateInfoPass::collectStates(Value storage, unsigned offset,
                                       std::vector<StateInfo> &stateInfos) {
  for (auto *op : storage.getUsers()) {
    if (auto substorage = dyn_cast<AllocStorageOp>(op)) {
      assert(substorage.offset().hasValue());
      collectStates(substorage.output(), *substorage.offset() + offset,
                    stateInfos);
      continue;
    }
    if (isa<AllocStateOp, RootInputOp, RootOutputOp>(op)) {
      ArrayRef<Attribute> names;
      if (auto namesAttr = op->getAttrOfType<ArrayAttr>("names"))
        names = namesAttr.getValue();
      ArrayRef<Attribute> offsets =
          op->getAttrOfType<ArrayAttr>("offsets").getValue();
      for (auto result : op->getResults()) {
        auto &stateInfo = stateInfos.emplace_back();
        stateInfo.type = StateInfo::Register;
        if (isa<RootInputOp>(op))
          stateInfo.type = StateInfo::Input;
        if (isa<RootOutputOp>(op))
          stateInfo.type = StateInfo::Output;
        if (!names.empty())
          stateInfo.name =
              names[result.getResultNumber()].dyn_cast<StringAttr>();
        else
          stateInfo.name = op->getAttrOfType<StringAttr>("name");
        stateInfo.offset = offsets[result.getResultNumber()]
                               .cast<IntegerAttr>()
                               .getValue()
                               .getZExtValue() +
                           offset;
        stateInfo.numBits = result.getType().cast<IntegerType>().getWidth();
      }
      continue;
    }
    if (auto memOp = dyn_cast<AllocMemoryOp>(op)) {
      auto memType = memOp.getType();
      auto intType = memType.getWordType();
      auto &stateInfo = stateInfos.emplace_back();
      stateInfo.type = StateInfo::Memory;
      stateInfo.name = op->getAttrOfType<StringAttr>("name");
      stateInfo.offset =
          op->getAttrOfType<IntegerAttr>("offset").getValue().getZExtValue() +
          offset;
      stateInfo.numBits = intType.getWidth();
      stateInfo.memoryStride =
          op->getAttrOfType<IntegerAttr>("stride").getValue().getZExtValue();
      stateInfo.memoryDepth = memType.getNumWords();
      continue;
    }
  }
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createPrintStateInfoPass() {
  return std::make_unique<PrintStateInfoPass>();
}
} // namespace arc
} // namespace circt
