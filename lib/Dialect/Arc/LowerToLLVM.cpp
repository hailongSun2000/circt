//===- LowerToLLVM.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Conversion/LLHDToLLVM.h"
#include "circt/Dialect/Comb/CombOps.h"
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

struct AllocStorageOpLowering
    : public OpConversionPattern<arc::AllocStorageOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::AllocStorageOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto type = typeConverter->convertType(op.getType());
    Value offset = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), op.offsetAttr());
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, type, adaptor.input(), offset);
    return success();
  }
};

struct AllocStateOpLowering : public ConversionPattern {
  AllocStateOpLowering(TypeConverter &typeConverter, MLIRContext *context,
                       PatternBenefit benefit = 1)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), benefit,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isa<AllocStateOp, RootInputOp, RootOutputOp>(op))
      return failure();

    ArrayRef<Attribute> offsets =
        op->getAttrOfType<ArrayAttr>("offsets").getValue();
    assert(offsets.size() == op->getNumResults());

    SmallVector<Value> values;
    for (auto [result, offsetAttr] : llvm::zip(op->getResults(), offsets)) {
      // Get a pointer to the correct offset in the storage.
      Value offset = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), rewriter.getI32Type(), offsetAttr);
      Value ptr = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), operands[0].getType(), operands[0], offset);

      // Cast the raw storage pointer to a pointer of the state's actual type.
      auto ptrType = LLVM::LLVMPointerType::get(result.getType());
      if (ptrType != ptr.getType())
        ptr = rewriter.create<LLVM::BitcastOp>(op->getLoc(), ptrType, ptr);

      // Load the value.
      Value value = rewriter.create<LLVM::LoadOp>(result.getLoc(), ptr);
      values.push_back(value);
    }

    rewriter.replaceOp(op, values);
    return success();
  }
};

struct UpdateStateOpLowering : public OpConversionPattern<arc::UpdateStateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::UpdateStateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value statsPtr;
    for (auto alloca : op->getParentOfType<func::FuncOp>()
                           .getBody()
                           .front()
                           .getOps<LLVM::AllocaOp>()) {
      if (alloca->hasAttr("stats")) {
        statsPtr = alloca;
        break;
      }
    }
    assert(statsPtr);
    for (auto [state, value] : llvm::zip(adaptor.states(), adaptor.values())) {
      auto ptr = state.getDefiningOp<LLVM::LoadOp>().getOperand();
      auto oldValue = rewriter.create<LLVM::LoadOp>(op.getLoc(), ptr);
      auto valuesMatch = rewriter.create<LLVM::ICmpOp>(
          op.getLoc(), LLVM::ICmpPredicate::eq, value, oldValue);
      auto valuesMatchExt = rewriter.create<LLVM::ZExtOp>(
          op.getLoc(), rewriter.getI32Type(), valuesMatch);
      Value stats = rewriter.create<LLVM::LoadOp>(op.getLoc(), statsPtr);
      stats = rewriter.create<LLVM::AddOp>(op.getLoc(), stats, valuesMatchExt);
      rewriter.create<LLVM::StoreOp>(op.getLoc(), stats, statsPtr);
      rewriter.create<LLVM::StoreOp>(op.getLoc(), value, ptr);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct AllocMemoryOpLowering : public OpConversionPattern<arc::AllocMemoryOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::AllocMemoryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value offset = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), op->getAttr("offset"));
    Value ptr = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), adaptor.storage().getType(), adaptor.storage(), offset);

    auto type = typeConverter->convertType(op.getType());
    if (type != ptr.getType())
      ptr = rewriter.create<LLVM::BitcastOp>(op.getLoc(), type, ptr);

    rewriter.replaceOp(op, ptr);
    return success();
  }
};

struct MemoryAccess {
  Value ptr;
  Value withinBounds;
};

static MemoryAccess prepareMemoryAccess(Location loc, Value memory,
                                        Value address, MemoryType type,
                                        ConversionPatternRewriter &rewriter) {
  auto zextAddrType = rewriter.getIntegerType(
      address.getType().cast<IntegerType>().getWidth() + 1);
  Value addr = rewriter.create<LLVM::ZExtOp>(loc, zextAddrType, address);
  Value addrLimit = rewriter.create<LLVM::ConstantOp>(
      loc, zextAddrType, rewriter.getI32IntegerAttr(type.getNumWords()));
  Value withinBounds = rewriter.create<LLVM::ICmpOp>(
      loc, LLVM::ICmpPredicate::ult, addr, addrLimit);
  auto ptrType = LLVM::LLVMPointerType::get(type.getWordType());
  Value ptr =
      rewriter.create<LLVM::GEPOp>(loc, ptrType, memory, ValueRange{addr});
  return {ptr, withinBounds};
};

struct MemoryReadOpLowering : public OpConversionPattern<arc::MemoryReadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::MemoryReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto type = typeConverter->convertType(op.getType());
    auto access =
        prepareMemoryAccess(op.getLoc(), adaptor.memory(), adaptor.address(),
                            op.memory().getType().cast<MemoryType>(), rewriter);
    auto enable = rewriter.create<LLVM::AndOp>(op.getLoc(), adaptor.enable(),
                                               access.withinBounds);

    // Only attempt to read the memory if the address is within bounds,
    // otherwise produce a zero value.
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, type, enable,
        [&](auto &builder, auto loc) {
          Value loadOp = builder.template create<LLVM::LoadOp>(loc, access.ptr);
          builder.template create<scf::YieldOp>(loc, loadOp);
        },
        [&](auto &builder, auto loc) {
          Value zeroValue = builder.template create<LLVM::ConstantOp>(
              loc, type, builder.getI64IntegerAttr(0));
          builder.template create<scf::YieldOp>(loc, zeroValue);
        });
    return success();
  }
};

struct MemoryWriteOpLowering : public OpConversionPattern<arc::MemoryWriteOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::MemoryWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto access =
        prepareMemoryAccess(op.getLoc(), adaptor.memory(), adaptor.address(),
                            op.memory().getType().cast<MemoryType>(), rewriter);
    auto enable = rewriter.create<LLVM::AndOp>(op.getLoc(), adaptor.enable(),
                                               access.withinBounds);

    // Only attempt to write the memory if the address is within bounds.
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, enable, [&](auto &builder, auto loc) {
          rewriter.create<LLVM::StoreOp>(loc, adaptor.data(), access.ptr);
          rewriter.create<scf::YieldOp>(loc);
        });
    return success();
  }
};

/// A dummy lowering for clock gates to an AND gate.
struct ClockGateOpLowering : public OpConversionPattern<arc::ClockGateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::ClockGateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<comb::AndOp>(op, adaptor.input(),
                                             adaptor.enable());
    return success();
  }
};

} // namespace

static bool isArcType(Type type) {
  return type.isa<StorageType>() || type.isa<MemoryType>();
}

static bool hasArcType(TypeRange types) {
  return llvm::any_of(types, isArcType);
}

static bool hasArcType(ValueRange values) {
  return hasArcType(values.getTypes());
}

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

  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
    auto argsConverted = llvm::none_of(op.getBlocks(), [](auto &block) {
      return hasArcType(block.getArguments());
    });
    auto resultsConverted = !hasArcType(op.getResultTypes());
    return argsConverted && resultsConverted;
  });
}

static void populateTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion([&](StorageType type) {
    return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));
  });
  typeConverter.addConversion([&](MemoryType type) {
    return LLVM::LLVMPointerType::get(
        IntegerType::get(type.getContext(), type.getStride() * 8));
  });
  typeConverter.addConversion([](hw::ArrayType type) { return type; });
  typeConverter.addConversion([](mlir::IntegerType type) { return type; });
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
    PassThroughOpLowering,
    AllocStorageOpLowering,
    AllocStateOpLowering,
    UpdateStateOpLowering,
    AllocMemoryOpLowering,
    MemoryReadOpLowering,
    MemoryWriteOpLowering,
    ClockGateOpLowering
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
  if (failed(isolateClockTrees()))
    return failure();
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

  auto statsType = builder.getI32Type();
  auto statsPtr = builder.create<LLVM::AllocaOp>(
      clockTreeOp->getLoc(), LLVM::LLVMPointerType::get(statsType),
      builder.create<LLVM::ConstantOp>(clockTreeOp->getLoc(),
                                       builder.getI32Type(),
                                       builder.getI32IntegerAttr(1)),
      0);
  builder.create<LLVM::StoreOp>(clockTreeOp->getLoc(),
                                builder.create<LLVM::ConstantOp>(
                                    clockTreeOp->getLoc(), builder.getI32Type(),
                                    builder.getI32IntegerAttr(1)),
                                statsPtr);
  statsPtr->setAttr("stats", builder.getUnitAttr());

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
  Value statsValue =
      builder.create<LLVM::LoadOp>(clockTreeOp->getLoc(), statsPtr);
  builder.create<func::ReturnOp>(clockTreeOp->getLoc(), statsValue);
  StringRef suffix =
      isa<PassThroughOp>(clockTreeOp) ? "_passthrough" : "_clock";
  StringRef funcName = globalNamespace.newName(modelOp.name() + suffix);
  builder.setInsertionPoint(modelOp);
  auto funcOp = builder.create<func::FuncOp>(
      clockTreeOp->getLoc(), funcName,
      builder.getFunctionType(additionalInputTypes, {statsType}));
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
