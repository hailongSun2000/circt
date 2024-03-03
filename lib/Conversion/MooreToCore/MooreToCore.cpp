//===- MooreToCore.cpp - Moore To Core Conversion Pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Moore to Core Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/MooreToCore.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace moore;

Declaration moore::decl;

/// Stores port interface into data structure to help convert moore module
/// structure to hw module structure.
MoorePortInfo::MoorePortInfo(moore::SVModuleOp moduleOp) {
  SmallVector<hw::PortInfo, 4> inputs, outputs;

  for (auto netOp : moduleOp.getBody()->getOps<NetOp>())
    nameMapping[netOp.getNameAttr()] = std::make_pair(netOp, netOp.getType());

  for (auto varOp : moduleOp.getBody()->getOps<VariableOp>())
    nameMapping[varOp.getNameAttr()] = std::make_pair(varOp, varOp.getType());

  // Gather all input or output ports.
  for (auto portOp : moduleOp.getBody()->getOps<PortOp>()) {
    mlir::Type portTy;
    auto argNum = inputs.size();
    auto portLoc = portOp.getLoc();
    auto portName = portOp.getNameAttr();

    if (nameMapping.contains(portName)) {
      portTy = nameMapping[portName].second;
      portBinding[portOp] = nameMapping[portName].first;
    }

    switch (portOp.getDirection()) {
    case Direction::In:
      inputs.push_back(
          hw::PortInfo{{portName, portTy, hw::ModulePort::Direction::Input},
                       argNum,
                       {},
                       portLoc});

      break;
    case Direction::InOut:
      inputs.push_back(hw::PortInfo{{portName, hw::InOutType::get(portTy),
                                     hw::ModulePort::Direction::InOut},
                                    argNum,
                                    {},
                                    portLoc});
      break;
    case Direction::Out:
      if (!portBinding[portOp]->getUsers().empty())
        outputs.push_back(
            hw::PortInfo{{portName, portTy, hw::ModulePort::Direction::Output},
                         argNum,
                         {},
                         portOp.getLoc()});
      break;
    case Direction::Ref:
      // TODO: Support parsing Direction::Ref port into portInfo structure.
      break;
    }
  }
  hwPorts = std::make_unique<hw::ModulePortInfo>(inputs, outputs);
}

namespace {

/// Returns the passed value if the integer width is already correct.
/// Zero-extends if it is too narrow.
/// Truncates if the integer is too wide and the truncated part is zero, if it
/// is not zero it returns the max value integer of target-width.
static Value adjustIntegerWidth(OpBuilder &builder, Value value,
                                uint32_t targetWidth, Location loc) {
  uint32_t intWidth = value.getType().getIntOrFloatBitWidth();
  if (intWidth == targetWidth)
    return value;

  if (intWidth < targetWidth) {
    Value zeroExt = builder.create<hw::ConstantOp>(
        loc, builder.getIntegerType(targetWidth - intWidth), 0);
    return builder.create<comb::ConcatOp>(loc, ValueRange{zeroExt, value});
  }

  Value hi = builder.create<comb::ExtractOp>(loc, value, targetWidth,
                                             intWidth - targetWidth);
  Value zero = builder.create<hw::ConstantOp>(
      loc, builder.getIntegerType(intWidth - targetWidth), 0);
  Value isZero = builder.create<comb::ICmpOp>(loc, comb::ICmpPredicate::eq, hi,
                                              zero, false);
  Value lo = builder.create<comb::ExtractOp>(loc, value, 0, targetWidth);
  Value max = builder.create<hw::ConstantOp>(
      loc, builder.getIntegerType(targetWidth), -1);
  return builder.create<comb::MuxOp>(loc, isZero, lo, max, false);
}

/// Due to the result type of the `lt`, or `le`, or `gt`, or `ge` ops are
/// always unsigned, estimating their operands type.
static bool isSignedType(Operation *op) {
  return TypeSwitch<Operation *, bool>(op)
      .template Case<LtOp, LeOp, GtOp, GeOp>([&](auto op) -> bool {
        return cast<UnpackedType>(op->getOperand(0).getType())
                   .castToSimpleBitVector()
                   .isSigned() &&
               cast<UnpackedType>(op->getOperand(1).getType())
                   .castToSimpleBitVector()
                   .isSigned();
      })
      .Default([&](auto op) -> bool {
        return cast<UnpackedType>(op->getResult(0).getType())
            .castToSimpleBitVector()
            .isSigned();
      });
}

/// Not define the predicate for `relation` and `equality` operations in the
/// MooreDialect, but comb needs it. Return a correct `comb::ICmpPredicate`
/// corresponding to different moore `relation` and `equality` operations.
static comb::ICmpPredicate getCombPredicate(Operation *op) {
  using comb::ICmpPredicate;
  return TypeSwitch<Operation *, ICmpPredicate>(op)
      .Case<LtOp>([&](auto op) {
        return isSignedType(op) ? ICmpPredicate::slt : ICmpPredicate::ult;
      })
      .Case<LeOp>([&](auto op) {
        return isSignedType(op) ? ICmpPredicate::sle : ICmpPredicate::ule;
      })
      .Case<GtOp>([&](auto op) {
        return isSignedType(op) ? ICmpPredicate::sgt : ICmpPredicate::ugt;
      })
      .Case<GeOp>([&](auto op) {
        return isSignedType(op) ? ICmpPredicate::sge : ICmpPredicate::uge;
      })
      .Case<EqOp>([&](auto op) { return ICmpPredicate::eq; })
      .Case<NeOp>([&](auto op) { return ICmpPredicate::ne; })
      .Case<CaseEqOp>([&](auto op) { return ICmpPredicate::ceq; })
      .Case<CaseNeOp>([&](auto op) { return ICmpPredicate::cne; })
      .Case<WildcardEqOp>([&](auto op) { return ICmpPredicate::weq; })
      .Case<WildcardNeOp>([&](auto op) { return ICmpPredicate::wne; });
}

//===----------------------------------------------------------------------===//
// Structure Conversion
//===----------------------------------------------------------------------===//

struct SVModuleOpConv : public OpConversionPattern<SVModuleOp> {
  SVModuleOpConv(TypeConverter &typeConverter, MLIRContext *ctx,
                 MoorePortInfoMap &portMap)
      : OpConversionPattern<SVModuleOp>(typeConverter, ctx), portMap(portMap) {}
  LogicalResult
  matchAndRewrite(SVModuleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    const circt::MoorePortInfo &mp = portMap.at(op.getSymNameAttr());

    // Create the hw.module to replace svmoduleOp
    auto hwModuleOp = rewriter.create<hw::HWModuleOp>(
        op.getLoc(), op.getSymNameAttr(), *mp.hwPorts);
    rewriter.eraseBlock(hwModuleOp.getBodyBlock());
    rewriter.inlineRegionBefore(op.getBodyRegion(), hwModuleOp.getBodyRegion(),
                                hwModuleOp.getBodyRegion().end());
    auto *hwBody = hwModuleOp.getBodyBlock();

    // Replace all relating logic of input port definitions for the input
    // block arguments. And update relating uses chain.
    for (auto [index, input] : llvm::enumerate(mp.hwPorts->getInputs())) {
      BlockArgument newArg;
      auto *portDefOp = mp.nameMapping.at(input.name).first;
      auto inputTy = mp.nameMapping.at(input.name).second;
      rewriter.modifyOpInPlace(hwModuleOp, [&]() {
        newArg = hwBody->addArgument(inputTy, portDefOp->getLoc());
      });
      rewriter.replaceAllUsesWith(portDefOp->getResult(0), newArg);
    }

    // Adjust all relating logic of output port definitions for rewriting
    // hw.output op.
    SmallVector<Value> outputValues;
    for (auto [index, output] : llvm::enumerate(mp.hwPorts->getOutputs())) {
      auto *portDefOp = mp.nameMapping.at(output.name).first;
      outputValues.push_back(portDefOp->getResult(0));
    }

    // Rewrite the hw.output op
    rewriter.setInsertionPointToEnd(hwBody);
    rewriter.create<hw::OutputOp>(op.getLoc(), outputValues);

    // Erase the original op
    rewriter.eraseOp(op);
    return success();
  }
  MoorePortInfoMap &portMap;
};

struct ProcedureOpConv : public OpConversionPattern<ProcedureOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ProcedureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    switch (adaptor.getKind()) {
    case ProcedureKind::AlwaysComb:
      rewriter.setInsertionPointAfter(op->getPrevNode());
      rewriter.inlineBlockBefore(op.getBody(), op);
      rewriter.eraseOp(op);
      return success();
    case ProcedureKind::Always:
    case ProcedureKind::AlwaysFF:
    case ProcedureKind::AlwaysLatch:
    case ProcedureKind::Initial:
    case ProcedureKind::Final:
      return emitError(op->getLoc(), "Unsupported procedure operation");
    };
    return success();
  }
};

struct InstanceOpConv : public OpConversionPattern<InstanceOp> {
  InstanceOpConv(TypeConverter &typeConverter, MLIRContext *ctx,
                 MoorePortInfoMap &portMap)
      : OpConversionPattern<InstanceOp>(typeConverter, ctx), portMap(portMap) {}

  LogicalResult
  matchAndRewrite(InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto instName = op.getInstanceNameAttr();
    auto moduleName = op.getModuleNameAttr();
    auto moduleNameAttr = moduleName.getAttr();

    SmallVector<Type> resultTypes;
    SmallVector<Attribute> argNames, resNames;
    SmallVector<Value> operands, convertedOperands;
    for (auto operand : op.getInputs())
      operands.push_back(operand);

    for (auto arg : portMap.at(moduleNameAttr).hwPorts->getInputs())
      argNames.push_back(arg.name);

    for (auto res : portMap.at(moduleNameAttr).hwPorts->getOutputs()) {
      resultTypes.push_back(res.type);
      resNames.push_back(res.name);
    }

    rewriter.setInsertionPoint(op);
    auto instOp = rewriter.create<hw::InstanceOp>(
        op.getLoc(), resultTypes, instName, moduleName, operands,
        rewriter.getArrayAttr(argNames), rewriter.getArrayAttr(resNames),
        rewriter.getArrayAttr({}), nullptr);

    // Update uses chain of output ports within instanceOp from original
    // moore instance op.
    for (auto [index, res] : llvm::enumerate(op.getOutputs())) {
      auto *defineOp = res.getDefiningOp();
      defineOp->getResult(0).replaceAllUsesWith(instOp->getResult(index));
    }
    rewriter.eraseOp(op);
    return success();
  }

  MoorePortInfoMap &portMap;
};

struct PortOpConv : public OpConversionPattern<PortOp> {
  PortOpConv(TypeConverter &typeConverter, MLIRContext *ctx,
             MoorePortInfoMap &portMap)
      : OpConversionPattern<PortOp>(typeConverter, ctx), portMap(portMap) {}

  LogicalResult
  matchAndRewrite(PortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto hwmod = op->getParentOfType<hw::HWModuleOp>();
    auto binding = portMap.at(hwmod.getModuleNameAttr()).portBinding;

    switch (op.getDirection()) {
    // The users of definition operation (The In / InOut direction port's
    // definition op) are replaced in the HWModuleOp's generation. When it has
    // no users left, erase itself.
    case Direction::In:
    case Direction::InOut:
      if (binding[op]->getUsers().empty())
        rewriter.eraseOp(binding[op]);
      rewriter.eraseOp(op);
      break;

      // Handle the users of definition operation (The Out direction port's
      // definition op), if there is no user left, erase itself.
    case Direction::Out:
      if (!binding[op]->getUsers().empty()) {
        binding[op]->getResult(0).replaceAllUsesWith(
            decl.getOutputValue(op)->getOperand(1));
        rewriter.eraseOp(binding[op]);
      }
      rewriter.eraseOp(op);
      break;

      // TODO: Not support handling port operation of Ref direction.
    case Direction::Ref:
      return op.emitOpError(
          "Not support conversion of port (Ref direction) operation.");
    }
    return success();
  }

  MoorePortInfoMap &portMap;
};

//===----------------------------------------------------------------------===//
// Declaration Conversion
//===----------------------------------------------------------------------===//

template <typename OpTy>
struct DeclOpConv : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value value = decl.getValue(op);
    if (!value) {
      rewriter.eraseOp(op);
      return success();
    }

    value.setType(ConversionPattern::typeConverter->convertType(
        op.getResult().getType()));
    rewriter.replaceOpWithNewOp<hw::WireOp>(op, value, op.getNameAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Expression Conversion
//===----------------------------------------------------------------------===//

struct ConstantOpConv : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, op.getValueAttr());
    return success();
  }
};

struct ConcatOpConv : public OpConversionPattern<ConcatOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, adaptor.getValues());
    return success();
  }
};

template <typename OpTy>
struct UnaryOpConv : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comb::ParityOp>(op, adaptor.getInput());
    return success();
  }
};

struct NotOpConv : public OpConversionPattern<NotOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());
    Value max = rewriter.create<hw::ConstantOp>(op.getLoc(), resultType, -1);
    rewriter.replaceOpWithNewOp<comb::XorOp>(op, adaptor.getInput(), max);
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct BinaryOpConv : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<DivOp>(op) && isSignedType(op)) {
      rewriter.replaceOpWithNewOp<comb::DivSOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs(), false);
      return success();
    }
    if (isa<ModOp>(op) && isSignedType(op)) {
      rewriter.replaceOpWithNewOp<comb::ModSOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs(), false);
      return success();
    }

    rewriter.replaceOpWithNewOp<TargetOp>(op, adaptor.getLhs(),
                                          adaptor.getRhs(), false);
    return success();
  }
};

template <typename SourceOp>
struct ICmpOpConv : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());
    comb::ICmpPredicate pred = getCombPredicate(op);

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(
        op, resultType, pred, adapter.getLhs(), adapter.getRhs());
    return success();
  }
};

struct ExtractOpConv : public OpConversionPattern<ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value value = adaptor.getInput();
    uint32_t lowBit = adaptor.getLowBit();
    uint32_t width = cast<UnpackedType>(op.getResult().getType())
                         .castToSimpleBitVectorOrNull()
                         .size;

    rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, value, lowBit, width);
    return success();
  }
};

struct ConversionOpConv : public OpConversionPattern<ConversionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConversionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getInput(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<hw::BitcastOp>(op, resultType, amount);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Statement Conversion
//===----------------------------------------------------------------------===//

/// TODO: The `PAssign`, `PCAssign` ops need to convert.
template <typename SourceOp>
struct AssignOpConv : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.eraseOp(op);
    return success();
  }
};

struct HWOutputOpConv : public OpConversionPattern<hw::OutputOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hw::OutputOp>(op, adaptor.getOperands());
    return success();
  }
};

struct HWInstanceOpConv : public OpConversionPattern<hw::InstanceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(hw::InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = adaptor.getInputs();
    auto *refeModule = op->getParentOfType<ModuleOp>().lookupSymbol(
        op.getReferencedModuleNameAttr());

    SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();

    rewriter.replaceOpWithNewOp<hw::InstanceOp>(
        op, refeModule, adaptor.getInstanceNameAttr(), inputs,
        rewriter.getArrayAttr({}), nullptr);

    return success();
  }
};

struct CondBranchOpConv : public OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), adaptor.getTrueDestOperands(),
        adaptor.getFalseDestOperands(), op.getTrueDest(), op.getFalseDest());
    return success();
  }
};

struct BranchOpConv : public OpConversionPattern<cf::BranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(),
                                              adaptor.getDestOperands());
    return success();
  }
};

struct UnrealizedConversionCastConv
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();

    // Drop the cast if the operand and result types agree after type
    // conversion.
    if (convResTypes == adaptor.getOperands().getTypes()) {
      rewriter.replaceOp(op, adaptor.getOperands());
      return success();
    }

    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, convResTypes, adaptor.getOperands());
    return success();
  }
};

struct ShlOpConv : public OpConversionPattern<ShlOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Comb shift operations require the same bit-width for value and amount
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getAmount(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<comb::ShlOp>(op, resultType, adaptor.getValue(),
                                             amount, false);
    return success();
  }
};

struct ShrOpConv : public OpConversionPattern<ShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Comb shift operations require the same bit-width for value and amount
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getAmount(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<comb::ShrUOp>(
        op, resultType, adaptor.getValue(), amount, false);
    return success();
  }
};

struct AShrOpConv : public OpConversionPattern<AShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Comb shift operations require the same bit-width for value and amount
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getAmount(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<comb::ShrSOp>(
        op, resultType, adaptor.getValue(), amount, false);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Conversion Infrastructure
//===----------------------------------------------------------------------===//

static bool isMooreType(Type type) { return type.isa<UnpackedType>(); }

static bool hasMooreType(TypeRange types) {
  return llvm::any_of(types, isMooreType);
}

static bool hasMooreType(ValueRange values) {
  return hasMooreType(values.getTypes());
}

template <typename Op>
static void addGenericLegality(ConversionTarget &target) {
  target.addDynamicallyLegalOp<Op>([](Op op) {
    return !hasMooreType(op->getOperands()) && !hasMooreType(op->getResults());
  });
}

static void populateLegality(ConversionTarget &target) {
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addLegalDialect<scf::SCFDialect>();
  target.addIllegalDialect<MooreDialect>();

  addGenericLegality<cf::CondBranchOp>(target);
  addGenericLegality<cf::BranchOp>(target);
  addGenericLegality<UnrealizedConversionCastOp>(target);

  target.addDynamicallyLegalOp<hw::HWModuleOp>([](hw::HWModuleOp op) {
    return !hasMooreType(op.getInputTypes()) &&
           !hasMooreType(op.getOutputTypes()) &&
           !hasMooreType(op.getBody().getArgumentTypes());
  });
  target.addDynamicallyLegalOp<hw::OutputOp>(
      [](hw::OutputOp op) { return !hasMooreType(op.getOutputs()); });
  target.addDynamicallyLegalOp<hw::InstanceOp>([](hw::InstanceOp op) {
    return !hasMooreType(op.getInputs()) && !hasMooreType(op->getResults());
  });
}

static void populateTypeConversion(TypeConverter &typeConverter) {
  // Directly map simple bit vector types to a compact integer type. This needs
  // to be added after all of the other conversions above, such that SBVs
  // conversion gets tried first before any of the others.
  typeConverter.addConversion([&](UnpackedType type) -> std::optional<Type> {
    if (auto sbv = type.getSimpleBitVectorOrNull())
      return mlir::IntegerType::get(type.getContext(), sbv.size);
    return std::nullopt;
  });

  typeConverter.addConversion([](mlir::IntegerType type) { return type; });
  typeConverter.addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return inputs[0];
      });
  typeConverter.addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return inputs[0];
      });
}

void circt::populateMooreStructureConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    circt::MoorePortInfoMap &portInfoMap) {
  auto *context = patterns.getContext();

  patterns.add<SVModuleOpConv, InstanceOpConv, PortOpConv>(
      typeConverter, context, portInfoMap);
}

void circt::populateMooreToCoreConversionPatterns(TypeConverter &typeConverter,
                                                  RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  // clang-format off
  patterns.add<
    // Patterns of assignment operations.
    AssignOpConv<ContinuousAssignOp>, AssignOpConv<BlockingAssignOp>,

    // Patterns of branch operations.
    BranchOpConv, CondBranchOpConv,

    // Patterns of declaration operations.
    DeclOpConv<NetOp>, DeclOpConv<VariableOp>,

    // Patterns of miscellaneous operations.
    ConstantOpConv, ConcatOpConv, ExtractOpConv, ConversionOpConv, 
    ProcedureOpConv,

    // Patterns of shifting operations.
    ShlOpConv, ShrOpConv, AShrOpConv,

    // Patterns of binary operations.
    BinaryOpConv<AddOp, comb::AddOp>, BinaryOpConv<SubOp, comb::SubOp>,
    BinaryOpConv<MulOp, comb::MulOp>, BinaryOpConv<DivOp, comb::DivUOp>,
    BinaryOpConv<ModOp, comb::ModUOp>, BinaryOpConv<AndOp, comb::AndOp>,
    BinaryOpConv<OrOp, comb::OrOp>, BinaryOpConv<XorOp, comb::XorOp>,
    
    // Patterns of relational operations.
    ICmpOpConv<LtOp>, ICmpOpConv<LeOp>, ICmpOpConv<GtOp>, ICmpOpConv<GeOp>,
    ICmpOpConv<EqOp>, ICmpOpConv<NeOp>, ICmpOpConv<CaseEqOp>,
    ICmpOpConv<CaseNeOp>, ICmpOpConv<WildcardEqOp>, ICmpOpConv<WildcardNeOp>,

    // Patterns of unary operations.
    UnaryOpConv<BoolCastOp>, UnaryOpConv<ReduceAndOp>, UnaryOpConv<ReduceOrOp>,
    UnaryOpConv<ReduceXorOp>, NotOpConv,
    
    // Patterns of other operations outside Moore dialect.
    HWInstanceOpConv, HWOutputOpConv, UnrealizedConversionCastConv>(
          typeConverter, context);

  hw::populateHWModuleLikeTypeConversionPattern(
      hw::HWModuleOp::getOperationName(), patterns, typeConverter);
}

//===----------------------------------------------------------------------===//
// Moore to Core Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct MooreToCorePass : public ConvertMooreToCoreBase<MooreToCorePass> {
  void runOnOperation() override;
};
} // namespace

/// Create a Moore to core dialects conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertMooreToCorePass() {
  return std::make_unique<MooreToCorePass>();
}

/// This is the main entrypoint for the Moore to Core conversion pass.
void MooreToCorePass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  // This is used to collect the net/variable/port declarations with their
  // assigned value and identify the assignment statements that had been used.
  auto pm = PassManager::on<ModuleOp>(&context);
  pm.addPass(moore::createMooreDeclarationsPass());
  if (failed(pm.run(module)))
    return signalPassFailure();

  ConversionTarget target(context);
  TypeConverter typeConverter;
  RewritePatternSet patterns(&context);

  // Generate moore module signatures.
  MoorePortInfoMap portInfoMap;
  for (auto svModuleOp : module.getOps<SVModuleOp>())
    portInfoMap.try_emplace(svModuleOp.getSymNameAttr(),
                            MoorePortInfo(svModuleOp));

  populateTypeConversion(typeConverter);
  populateLegality(target);

  // Convert moore operations to core ir.
  populateMooreStructureConversionPatterns(typeConverter, patterns,
                                           portInfoMap);
  populateMooreToCoreConversionPatterns(typeConverter, patterns);

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
