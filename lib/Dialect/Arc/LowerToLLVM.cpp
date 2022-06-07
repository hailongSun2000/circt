//===- LowerToLLVM.cpp -------------------------------------------------===//
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
// Lowering Patterns
//===----------------------------------------------------------------------===//

namespace {

struct DefineOpLowering : public OpConversionPattern<arc::DefineOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::DefineOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct OutputOpLowering : public OpConversionPattern<arc::OutputOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.operands());
    return success();
  }
};

struct StateOpLowering : public OpConversionPattern<arc::StateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::StateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type> newResultTypes;
    if (failed(
            typeConverter->convertTypes(op.getResultTypes(), newResultTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<func::CallOp>(op, newResultTypes, op.arcAttr(),
                                              adaptor.operands());
    return success();
  }
};

struct ClockTreeOpLowering : public OpConversionPattern<arc::ClockTreeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::ClockTreeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (auto enable = adaptor.enable()) {
      auto ifOp = rewriter.create<scf::IfOp>(op.getLoc(), enable, false);
      auto yield = ifOp.thenYield();
      yield->remove();
      rewriter.mergeBlocks(&op.bodyBlock(), ifOp.thenBlock(), {});
      ifOp.thenBlock()->push_back(yield);
      rewriter.eraseOp(op);
    } else {
      Block *parentBlock = op->getBlock();
      Block *after = rewriter.splitBlock(parentBlock, Block::iterator(op));
      rewriter.mergeBlocks(&op.bodyBlock(), parentBlock, {});
      rewriter.mergeBlocks(after, parentBlock, {});
      rewriter.eraseOp(op);
    }
    return success();
  }
};

struct PassThroughOpLowering : public OpConversionPattern<arc::PassThroughOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::PassThroughOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Block *parentBlock = op->getBlock();
    Block *after = rewriter.splitBlock(parentBlock, Block::iterator(op));
    rewriter.mergeBlocks(&op.bodyBlock(), parentBlock, {});
    rewriter.mergeBlocks(after, parentBlock, {});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

static void populateLegality(ConversionTarget &target) {
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<scf::SCFDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  target.addIllegalOp<arc::DefineOp>();
  target.addIllegalOp<arc::OutputOp>();
  target.addIllegalOp<arc::StateOp>();
  target.addIllegalOp<arc::ClockTreeOp>();
  target.addIllegalOp<arc::PassThroughOp>();
}

static void populateTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion([](Type type) { return type; });
  // typeConverter.addConversion([](mlir::IntegerType type) { return type; });
}

static void populateOpConversion(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  auto *context = patterns.getContext();
  // clang-format off
  patterns.add<
    DefineOpLowering,
    OutputOpLowering,
    StateOpLowering,
    ClockTreeOpLowering,
    PassThroughOpLowering
  >(typeConverter, context);
  // clang-format on

  mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      patterns, typeConverter);
}

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

struct LowerToLLVMPass : public LowerToLLVMBase<LowerToLLVMPass> {
  void runOnOperation() override;
  LogicalResult runOnModel();
  LogicalResult allocateState();
  LogicalResult isolateClockTrees();
  LogicalResult isolateClockTree(Operation *clockTreeOp);

  unsigned prune();

  ModelOp modelOp;
  Namespace globalNamespace;
  DenseSet<Operation *> pruningWorklist;
  DenseMap<ModelOp, ModelInfo> modelInfos;
};
} // namespace

void LowerToLLVMPass::runOnOperation() {
  globalNamespace.clear();
  for (auto &op : *getOperation().getBody())
    if (auto sym = op.getAttrOfType<StringAttr>("sym_name"))
      globalNamespace.newName(sym.getValue());

  // Perform the per-model lowerings.
  SmallVector<ModelOp> modelOps;
  modelInfos.clear();
  for (auto op : getOperation().getOps<ModelOp>()) {
    modelOp = op;
    modelOps.push_back(op);
    if (failed(runOnModel()))
      return signalPassFailure();
  }

  // Emit the model state information if requested.
  if (!stateFile.empty()) {
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
      for (auto modelOp : modelOps) {
        const auto &modelInfo = modelInfos[modelOp];
        json.object([&] {
          json.attribute("name", modelOp.name());
          json.attribute("numStateBytes", modelInfo.numStateBytes);
          json.attributeArray("states", [&] {
            for (const auto &state : modelInfo.states) {
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

  // Erase the model ops.
  for (auto modelOp : modelOps)
    modelOp.erase();

  // Eliminate dead code.
  {
    LLVM_DEBUG(llvm::dbgs() << "Eliminating dead code\n");
    IRRewriter builder(&getContext());
    (void)runRegionDCE(builder, getOperation().getBodyRegion());
  }

  // Perform the lowering to Func and SCF.
  {
    LLVM_DEBUG(llvm::dbgs() << "Lowering arcs to Func/SCF dialects\n");
    ConversionTarget target(getContext());
    TypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    populateLegality(target);
    populateTypeConversion(typeConverter);
    populateOpConversion(patterns, typeConverter);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }

  // Perform lowering to LLVM.
  {
    LLVM_DEBUG(llvm::dbgs() << "Lowering to LLVM dialect\n");
    LLVMConversionTarget target(getContext());
    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<arc::ModelOp>();
    populateSCFToControlFlowConversionPatterns(patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    size_t sigCounter;
    size_t regCounter;
    populateLLHDToLLVMConversionPatterns(typeConverter, patterns, sigCounter,
                                         regCounter);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Fully lowered\n");
}

LogicalResult LowerToLLVMPass::runOnModel() {
  pruningWorklist.clear();
  if (failed(allocateState()))
    return failure();
  if (failed(isolateClockTrees()))
    return failure();
  return success();
}

LogicalResult LowerToLLVMPass::allocateState() {
  LLVM_DEBUG(llvm::dbgs() << "Allocating states in model `" << modelOp.name()
                          << "`\n");

  SmallVector<Operation *> allocsToErase;
  SmallVector<std::pair<Value, unsigned>> allocOffsets;
  SmallVector<std::tuple<MemoryOp, unsigned, unsigned>> memoryOffsets;
  SmallDenseMap<MemoryReadOp, Value> memoryReadBuffers;
  SmallDenseMap<Value, IntegerAttr> memoryAddressLimits;

  // A list of child state allocations to include in a parent block. Given as a
  // list of placeholders for the child state pointer and the expected
  // allocation size.
  SmallDenseMap<
      Block *,
      SmallVector<std::tuple<Value, unsigned, SmallVector<StateInfo>>, 1>>
      childAllocations;

  modelOp.walk([&](Block *block) {
    unsigned numAllocs = 0;
    unsigned numMemories = 0;
    unsigned currentByte = 0; // allocated size in bytes
    auto allocBytes = [&](unsigned numBytes) {
      if (numBytes >= 2)
        currentByte = (currentByte + 1) / 2 * 2;
      if (numBytes >= 4)
        currentByte = (currentByte + 3) / 4 * 4;
      if (numBytes >= 8)
        currentByte = (currentByte + 7) / 8 * 8;
      unsigned offset = currentByte;
      currentByte += numBytes;
      return offset;
    };

    // Allocate space for state and memories in this block.
    allocOffsets.clear();
    allocsToErase.clear();
    memoryOffsets.clear();
    SmallDenseMap<MemoryReadOp, Value> localMemoryReadBuffers;
    SmallVector<StateInfo> stateInfos;
    for (auto &op : *block) {
      if (isa<AllocStateOp, RootInputOp, RootOutputOp>(&op)) {
        allocsToErase.push_back(&op);
        ArrayRef<Attribute> names;
        if (auto namesAttr = op.getAttrOfType<ArrayAttr>("names"))
          names = namesAttr.getValue();
        for (auto result : op.getResults()) {
          ++numAllocs;
          auto intType = result.getType().cast<IntegerType>();
          unsigned numBytes = (intType.getWidth() + 7) / 8;
          unsigned offset = allocBytes(numBytes);
          allocOffsets.push_back({result, offset});

          auto &stateInfo = stateInfos.emplace_back();
          stateInfo.type = StateInfo::Register;
          if (isa<RootInputOp>(&op))
            stateInfo.type = StateInfo::Input;
          if (isa<RootOutputOp>(&op))
            stateInfo.type = StateInfo::Output;
          if (!names.empty())
            stateInfo.name =
                names[result.getResultNumber()].dyn_cast<StringAttr>();
          else
            stateInfo.name = op.getAttrOfType<StringAttr>("name");
          stateInfo.offset = offset;
          stateInfo.numBits = intType.getWidth();
        }
        continue;
      }
      if (auto memOp = dyn_cast<MemoryOp>(&op)) {
        ++numMemories;
        auto intType = memOp.getType();
        unsigned numBytesPerWord = (intType.getWidth() + 7) / 8;
        if (numBytesPerWord > 1)
          numBytesPerWord = (numBytesPerWord + 1) / 2 * 2;
        if (numBytesPerWord > 2)
          numBytesPerWord = (numBytesPerWord + 3) / 4 * 4;
        if (numBytesPerWord > 4)
          numBytesPerWord = (numBytesPerWord + 7) / 8 * 8;
        unsigned numBytes = memOp.numWords() * numBytesPerWord;
        unsigned offset = allocBytes(numBytes);
        memoryOffsets.push_back({memOp, offset, numBytesPerWord});

        auto &stateInfo = stateInfos.emplace_back();
        stateInfo.type = StateInfo::Memory;
        stateInfo.name = op.getAttrOfType<StringAttr>("name");
        stateInfo.offset = offset;
        stateInfo.numBits = intType.getWidth();
        stateInfo.memoryStride = numBytesPerWord;
        stateInfo.memoryDepth = memOp.numWords();
        continue;
      }
      if (auto memReadOp = dyn_cast<MemoryReadOp>(&op)) {
        auto intType = memReadOp.getType();
        unsigned numBytes = (intType.getWidth() + 7) / 8;
        unsigned offset = allocBytes(numBytes);
        OpBuilder builder(memReadOp);
        localMemoryReadBuffers[memReadOp] = builder.create<LLVM::ConstantOp>(
            memReadOp.getLoc(), builder.getI32Type(),
            builder.getI32IntegerAttr(offset));
      }
    }

    // Include child allocations. This rewrites the `unsigned` in the child
    // allocations to mean "offset" instead of "number of bytes".
    unsigned numChildAllocs = 0;
    unsigned totalChildAllocBytes = 0;
    auto &blockChildAllocs = childAllocations[block];
    for (auto &[value, numBytes, childStateInfos] : blockChildAllocs) {
      ++numChildAllocs;
      totalChildAllocBytes += numBytes;
      numBytes = allocBytes(numBytes);
    }

    if (currentByte == 0) {
      childAllocations.erase(block);
      return;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "- Allocated " << currentByte << " bytes for " << numAllocs
               << " states, " << numMemories << " memories, including "
               << totalChildAllocBytes << " bytes for " << numChildAllocs
               << " nested states\n");

    // Create a placeholder for the pointer to the allocated state. This will
    // be provided by the parent later on, but we already have to be able to
    // access various subelements here.
    OpBuilder builder(block, block->begin());
    auto bytePtrType = LLVM::LLVMPointerType::get(builder.getI8Type());
    Value statePtr =
        builder
            .create<UnrealizedConversionCastOp>(block->getParentOp()->getLoc(),
                                                bytePtrType, ValueRange{})
            .getResult(0);

    // For each memory read operation, assemble a pointer to its local scratch
    // memory where it can keep the previously-read data.
    for (auto [memReadOp, offset] : localMemoryReadBuffers) {
      ImplicitLocOpBuilder builder(memReadOp.getLoc(), memReadOp);
      Value ptr = builder.create<LLVM::GEPOp>(bytePtrType, statePtr,
                                              ValueRange{offset});
      auto ptrType = LLVM::LLVMPointerType::get(memReadOp.getType());
      if (ptrType != ptr.getType())
        ptr = builder.create<LLVM::BitcastOp>(ptrType, ptr);
      memoryReadBuffers.insert({memReadOp, ptr});
    }

    // Create a pointer to a byte offset in the state, with a given type.
    auto createOffsetPtr = [&](Location loc, Type type,
                               unsigned offset) -> Value {
      auto offsetOp = builder.create<LLVM::ConstantOp>(
          loc, builder.getI32Type(), builder.getI32IntegerAttr(offset));
      auto gepOp = builder.create<LLVM::GEPOp>(loc, bytePtrType, statePtr,
                                               ValueRange{offsetOp});
      auto ptrType = LLVM::LLVMPointerType::get(type);
      if (ptrType == gepOp.getType())
        return gepOp;
      auto castOp = builder.create<LLVM::BitcastOp>(loc, ptrType, gepOp);
      return castOp;
    };

    // Replace the placeholders for child allocations with a concrete pointer
    // into the current state.
    for (auto &[value, offset, childStateInfos] : blockChildAllocs) {
      builder.setInsertionPoint(value.getDefiningOp());
      auto childStatePtr =
          createOffsetPtr(value.getLoc(), builder.getI8Type(), offset);
      value.replaceAllUsesWith(childStatePtr);
      value.getDefiningOp()->erase();
      for (auto &info : childStateInfos)
        info.offset += offset;
      stateInfos.append(std::move(childStateInfos));
    }
    childAllocations.erase(block);

    // Ensure our parent block will allocate space for our state.
    childAllocations[block->getParentOp()->getBlock()].push_back(
        {statePtr, currentByte, stateInfos});

    // Replace the alloc ops with the corresponding LLVM GEP and load
    // operation.
    for (auto [value, offset] : allocOffsets) {
      builder.setInsertionPoint(value.getDefiningOp());
      auto castOp = createOffsetPtr(value.getLoc(), value.getType(), offset);
      auto loadOp = builder.create<LLVM::LoadOp>(value.getLoc(), castOp);
      Value castForUpdateOps =
          builder
              .create<UnrealizedConversionCastOp>(
                  value.getLoc(), value.getType(), ValueRange{castOp})
              .getResult(0);
      for (auto &use : llvm::make_early_inc_range(value.getUses())) {
        if (auto updateOp = dyn_cast<UpdateStateOp>(use.getOwner())) {
          if (llvm::is_contained(updateOp.states(), use.get())) {
            use.set(castForUpdateOps);
            continue;
          }
        }
        use.set(loadOp);
      }
    }
    for (auto *op : allocsToErase)
      op->erase();

    // Replace the memory ops with the corresponding GEP and cast operations.
    for (auto [memOp, offset, bytesPerWord] : memoryOffsets) {
      builder.setInsertionPoint(memOp);
      auto wordType = builder.getIntegerType(bytesPerWord * 8);
      auto offsetPtr = createOffsetPtr(memOp.getLoc(), wordType, offset);
      Value cast =
          builder
              .create<UnrealizedConversionCastOp>(
                  memOp.getLoc(), memOp.getType(), ValueRange{offsetPtr})
              .getResult(0);
      memoryAddressLimits[cast] = memOp.numWordsAttr();
      memOp.replaceAllUsesWith(cast);
      memOp.erase();
    }
  });

  // Rewrite the memory access and state update ops.
  SmallSetVector<Operation *, 8> castsToDrop;
  auto pruneCasts = [&]() {
    while (!castsToDrop.empty()) {
      auto *op = castsToDrop.pop_back_val();
      if (!op->use_empty())
        continue;
      for (auto operand : op->getOperands())
        if (auto *def = operand.getDefiningOp())
          castsToDrop.insert(def);
      op->erase();
    }
  };

  auto unwrapCastPointer = [&](Value state) {
    // At this point all states must be realized as LLVM pointers cast
    // back to the original type.
    auto castOp = state.getDefiningOp<UnrealizedConversionCastOp>();
    assert(castOp && castOp->getNumOperands() == 1 &&
           "states must be unary casts");
    castsToDrop.insert(castOp);
    auto ptrValue = castOp->getOperand(0);
    assert(ptrValue.getType().isa<LLVM::LLVMPointerType>() &&
           "states must be LLVM pointers");
    return ptrValue;
  };

  auto prepareMemoryAddress = [&](ImplicitLocOpBuilder &builder, Value memory,
                                  Value address) {
    auto zextAddrType = builder.getIntegerType(
        address.getType().cast<IntegerType>().getWidth() + 1);
    auto addr = builder.create<LLVM::ZExtOp>(zextAddrType, address);
    auto addrLimit = builder.create<LLVM::ConstantOp>(
        zextAddrType, memoryAddressLimits[memory]);
    auto withinBounds =
        builder.create<LLVM::ICmpOp>(LLVM::ICmpPredicate::ult, addr, addrLimit);
    return std::make_pair(addr, withinBounds);
  };

  auto getAddressedMemoryPointer = [&](ImplicitLocOpBuilder &builder,
                                       Value memory, Value address,
                                       Type wordType) -> Value {
    auto ptrValue = unwrapCastPointer(memory);
    auto gepOp = builder.create<LLVM::GEPOp>(ptrValue.getType(), ptrValue,
                                             ValueRange{address});
    Type desiredPtrType = LLVM::LLVMPointerType::get(wordType);
    if (gepOp.getType() == desiredPtrType)
      return gepOp;
    return builder.create<LLVM::BitcastOp>(desiredPtrType, gepOp);
  };

  modelOp.walk([&](Operation *op) {
    if (auto updateOp = dyn_cast<UpdateStateOp>(op)) {
      ImplicitLocOpBuilder builder(updateOp.getLoc(), updateOp);
      for (auto [state, value] :
           llvm::zip(updateOp.states(), updateOp.values())) {
        auto ptrValue = unwrapCastPointer(state);
        assert(
            ptrValue.getType().cast<LLVM::LLVMPointerType>().getElementType() ==
            value.getType());
        builder.create<LLVM::StoreOp>(value, ptrValue);
      }
      updateOp.erase();
      pruneCasts();
      return;
    }
    if (auto readOp = dyn_cast<MemoryReadOp>(op)) {
      ImplicitLocOpBuilder builder(readOp.getLoc(), readOp);
      auto [addr, withinBounds] =
          prepareMemoryAddress(builder, readOp.memory(), readOp.address());
      auto memPtrValue = getAddressedMemoryPointer(builder, readOp.memory(),
                                                   addr, readOp.getType());
      auto lastPtrValue = memoryReadBuffers[readOp];
      auto lastValue = builder.create<LLVM::LoadOp>(lastPtrValue);
      auto zeroValue = builder.create<LLVM::ConstantOp>(
          readOp.getType(), builder.getI64IntegerAttr(0));

      // Only attempt to read the memory location if the address is in bounds.
      // Otherwise produce a zero value (Verilator's interpretation of X).
      auto ifOp = builder.create<scf::IfOp>(
          readOp.getType(), withinBounds,
          [&](OpBuilder &builder, Location loc) {
            auto loadOp = builder.create<LLVM::LoadOp>(loc, memPtrValue);
            builder.create<scf::YieldOp>(loc, ValueRange{loadOp});
          },
          [&](OpBuilder &builder, Location loc) {
            builder.create<scf::YieldOp>(loc, ValueRange{zeroValue});
          });

      auto valueIfDisabled = zeroValue; // <-- MFC does this
      // auto valueIfDisabled = lastValue; // <-- also an option
      Value value = builder.create<LLVM::SelectOp>(
          readOp.enable(), ifOp.getResult(0), valueIfDisabled);
      builder.create<LLVM::StoreOp>(value, lastPtrValue);
      readOp.replaceAllUsesWith(value);
      readOp.erase();
      pruneCasts();
      return;
    }
    if (auto writeOp = dyn_cast<MemoryWriteOp>(op)) {
      ImplicitLocOpBuilder builder(writeOp.getLoc(), writeOp);
      auto [addr, withinBounds] =
          prepareMemoryAddress(builder, writeOp.memory(), writeOp.address());
      auto enable = builder.create<LLVM::AndOp>(writeOp.enable(), withinBounds);
      auto ifOp = builder.create<scf::IfOp>(enable, false);
      builder.setInsertionPointToStart(ifOp.thenBlock());
      auto ptrValue = getAddressedMemoryPointer(builder, writeOp.memory(), addr,
                                                writeOp.data().getType());
      builder.create<LLVM::StoreOp>(writeOp.data(), ptrValue);
      writeOp.erase();
      pruneCasts();
      return;
    }
  });

  // At this point we should only have a single child allocation left -- the
  // one for the entire model body. Expose this as a block argument on the
  // model, to be converted to dedicated update functions.
  assert(childAllocations.size() == 1 &&
         "only model-wide allocations should be left");
  assert(childAllocations.begin()->second.size() == 1 &&
         "model should produce only one state placeholder");
  auto [statePlaceholder, numBytes, stateInfos] =
      childAllocations.begin()->second.back();
  auto stateArg = modelOp.bodyBlock().addArgument(statePlaceholder.getType(),
                                                  statePlaceholder.getLoc());
  statePlaceholder.replaceAllUsesWith(stateArg);
  statePlaceholder.getDefiningOp()->erase();

  auto &modelInfo = modelInfos[modelOp];
  modelInfo.numStateBytes = numBytes;
  modelInfo.states.assign(stateInfos.begin(), stateInfos.end());

  return success();
}

LogicalResult LowerToLLVMPass::isolateClockTrees() {
  LLVM_DEBUG(llvm::dbgs() << "Isolating clock trees in model `"
                          << modelOp.name() << "`\n");
  SmallVector<Operation *> treeOps;
  for (auto &op : modelOp.bodyBlock())
    if (isa<ClockTreeOp, PassThroughOp>(&op))
      treeOps.push_back(&op);
  for (auto *op : treeOps)
    if (failed(isolateClockTree(op)))
      return failure();
  return success();
}

LogicalResult LowerToLLVMPass::isolateClockTree(Operation *clockTreeOp) {
  LLVM_DEBUG(llvm::dbgs() << "- Isolating clock tree\n");

  // Mark operations outside the clock tree that are used inside of it to be
  // extracted as well.
  unsigned numOpsInClockTree = 0;
  SmallVector<OpOperand *, 0> externalOperands;
  DenseSet<Operation *> additionalOps;
  SetVector<Value> additionalValues;
  SmallVector<Operation *, 0> worklist;

  clockTreeOp->walk([&](Operation *op) {
    ++numOpsInClockTree;
    for (auto &operand : op->getOpOperands()) {
      auto *def = operand.get().getDefiningOp();
      if (def && def->getParentOp() != modelOp)
        continue;
      externalOperands.push_back(&operand);
      if (!def) {
        additionalValues.insert(operand.get());
      } else {
        if (additionalOps.insert(def).second)
          worklist.push_back(def);
      }
    }
  });
  while (!worklist.empty()) {
    auto *op = worklist.pop_back_val();
    for (auto operand : op->getOperands())
      if (auto *def = operand.getDefiningOp())
        if (additionalOps.insert(def).second)
          worklist.push_back(def);
  }
  LLVM_DEBUG(llvm::dbgs() << "  - Using " << additionalOps.size()
                          << " external ops\n");

  // Migrate the clock tree op and all external ops into a new block.
  auto block = std::make_unique<Block>();
  OpBuilder builder(clockTreeOp);
  builder.setInsertionPointToStart(block.get());

  SmallVector<Value> additionalInputs;
  SmallVector<Type> additionalInputTypes;
  DenseMap<Value, Value> valueMapping;

  for (auto value : additionalValues) {
    auto arg = block->addArgument(value.getType(), value.getLoc());
    LLVM_DEBUG(llvm::dbgs() << "  - Adding " << arg.getType() << " input\n");
    valueMapping.insert({value, arg});
    additionalInputs.push_back(value);
    additionalInputTypes.push_back(arg.getType());
  }

  for (auto &op : llvm::make_early_inc_range(modelOp.bodyBlock())) {
    // Just move over the clock tree op itself.
    if (&op == clockTreeOp) {
      clockTreeOp->remove();
      builder.insert(clockTreeOp);
      continue;
    }

    // Clone the ops we've marked earlier.
    if (!additionalOps.contains(&op))
      continue;
    pruningWorklist.insert(&op);
    auto *clonedOp = op.cloneWithoutRegions();
    builder.insert(clonedOp);
    for (auto &operand : clonedOp->getOpOperands()) {
      if (auto mapping = valueMapping.lookup(operand.get())) {
        operand.set(mapping);
        continue;
      }
      auto arg =
          block->addArgument(operand.get().getType(), operand.get().getLoc());
      LLVM_DEBUG(llvm::dbgs() << "  - Adding " << arg.getType() << " input\n");
      valueMapping.insert({operand.get(), arg});
      additionalInputs.push_back(operand.get());
      additionalInputTypes.push_back(arg.getType());
      operand.set(arg);
    }
    for (auto [oldResult, newResult] :
         llvm::zip(op.getResults(), clonedOp->getResults()))
      valueMapping.insert({oldResult, newResult});
  }

  // Update the ops inside the clock tree to use the cloned ops.
  for (auto *operand : externalOperands) {
    auto mapping = valueMapping.lookup(operand->get());
    assert(mapping && "external operand should have been cloned");
    operand->set(mapping);
  }

  // Create the function for the extracted clock tree.
  builder.create<func::ReturnOp>(clockTreeOp->getLoc());
  StringRef suffix =
      isa<PassThroughOp>(clockTreeOp) ? "_passthrough" : "_clock";
  StringRef funcName = globalNamespace.newName(modelOp.name() + suffix);
  builder.setInsertionPoint(modelOp);
  auto funcOp = builder.create<func::FuncOp>(
      clockTreeOp->getLoc(), funcName,
      builder.getFunctionType(additionalInputTypes, {}));
  funcOp.getBody().push_back(block.release());
  LLVM_DEBUG(llvm::dbgs() << "  - Created function `" << funcName << "`\n");

  // Create a call to the function.
  builder.setInsertionPointToEnd(&modelOp.bodyBlock());
  builder.create<func::CallOp>(clockTreeOp->getLoc(), funcOp, additionalInputs);

  // Prune any ops we've touched.
  unsigned numPruned = prune();
  if (numPruned > 0)
    LLVM_DEBUG(llvm::dbgs() << "  - Pruned " << numPruned
                            << " unused ops outside clock tree\n");

  return success();
}

unsigned LowerToLLVMPass::prune() {
  unsigned numPruned = 0;
  while (!pruningWorklist.empty()) {
    auto *op = *pruningWorklist.begin();
    pruningWorklist.erase(op);
    if (!op->use_empty())
      continue;
    for (auto operand : op->getOperands())
      if (auto *def = operand.getDefiningOp())
        pruningWorklist.insert(def);
    op->erase();
    ++numPruned;
  }
  return numPruned;
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createLowerToLLVMPass() {
  return std::make_unique<LowerToLLVMPass>();
}
} // namespace arc
} // namespace circt
