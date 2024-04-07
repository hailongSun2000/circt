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
                         outputs.size(),
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

/// Lower `scf.if` into `comb.mux`.
/// Example : if (cond1) a = 1; else if (cond2) a = 2; else a=3;
/// post-conv: %0 = comb.mux %cond1, a=1, %1; %1 = comb.mux %cond2, a=2, a=3;
static LogicalResult lowerSCFIfOp(scf::IfOp ifOp,
                                  ConversionPatternRewriter &rewriter,
                                  DenseMap<Operation *, Value> &muxMap,
                                  SmallVector<Value> &allConds) {

  Value cond, trueValue, falseValue;

  // Traverse the 'else' region.
  for (auto &elseOp : ifOp.getElseRegion().getOps()) {
    // First to find the innermost 'else if' statement.
    if (isa<scf::IfOp>(elseOp))
      if (failed(lowerSCFIfOp(dyn_cast<scf::IfOp>(elseOp), rewriter, muxMap,
                              allConds)))
        return failure();
    if (isa<BlockingAssignOp, NonBlockingAssignOp>(elseOp)) {
      auto *lhs = elseOp.getOperand(0).getDefiningOp();
      auto elseValue = elseOp.getOperand(1);
      muxMap[lhs] = elseValue;
    }

    // Handle '? :'.
    if (isa<scf::YieldOp>(elseOp) && elseOp.getNumOperands()) {
      falseValue = elseOp.getOperand(0);
    }
  }

  // Traverse the 'then' region.
  for (auto &thenOp : ifOp.getThenRegion().getOps()) {

    // First to find the innermost 'if' statement.
    if (isa<scf::IfOp>(thenOp)) {
      if (allConds.empty())
        allConds.push_back(ifOp.getCondition());
      allConds.push_back(thenOp.getOperand(0));
      if (failed(lowerSCFIfOp(dyn_cast<scf::IfOp>(thenOp), rewriter, muxMap,
                              allConds)))
        return failure();
    }

    cond = ifOp.getCondition();
    if (isa<BlockingAssignOp, NonBlockingAssignOp>(thenOp)) {
      auto *lhs = thenOp.getOperand(0).getDefiningOp();
      trueValue = thenOp.getOperand(1);

      // Maybe just the 'then' region exists. Like if(); no 'else'.
      if (muxMap.lookup(lhs))
        falseValue = muxMap.lookup(lhs);
      else
        falseValue = decl.getValue(lhs);

      if (!falseValue)
        falseValue = lhs->getResult(0);

      auto b = ifOp.getThenBodyBuilder();
      trueValue = rewriter.getRemappedValue(trueValue);
      falseValue = rewriter.getRemappedValue(falseValue);

      if (allConds.size() > 1) {
        cond = b.create<comb::AndOp>(ifOp.getLoc(), allConds, false);
        allConds.pop_back();
      }
      auto muxOp =
          b.create<comb::MuxOp>(ifOp.getLoc(), cond, trueValue, falseValue);
      muxMap[lhs] = muxOp;
    }

    // Handle '? :'.
    if (isa<scf::YieldOp>(thenOp) && thenOp.getNumOperands()) {
      trueValue = thenOp.getOperand(0);
      rewriter.inlineBlockBefore(&ifOp.getThenRegion().front(), ifOp);
      rewriter.inlineBlockBefore(&ifOp.getElseRegion().front(), ifOp);

      trueValue = rewriter.getRemappedValue(trueValue);
      falseValue = rewriter.getRemappedValue(falseValue);

      if (allConds.size() > 1) {
        cond = rewriter.create<comb::AndOp>(ifOp.getLoc(), allConds, false);
        allConds.pop_back();
      }
      rewriter.replaceOpWithNewOp<comb::MuxOp>(ifOp, cond, trueValue,
                                               falseValue);
      return success();
    }
  }
  rewriter.inlineBlockBefore(&ifOp.getThenRegion().front(), ifOp);
  if (!ifOp.getElseRegion().empty()) {
    rewriter.inlineBlockBefore(&ifOp.getElseRegion().front(), ifOp);
  }
  rewriter.eraseOp(ifOp);
  return success();
}

/// Handle "always_comb".
static LogicalResult lowerAlwaysComb(ProcedureOp procedureOp,
                                     ConversionPatternRewriter &rewriter) {
  DenseMap<Operation *, Value> muxMap;
  for (auto &nestOp : procedureOp.getBodyRegion().getOps()) {
    SmallVector<Value> allConds;
    if (isa<scf::IfOp>(nestOp))
      if (failed(lowerSCFIfOp(dyn_cast<scf::IfOp>(nestOp), rewriter, muxMap,
                              allConds)))
        return failure();
  }

  // Update the values of the variables/nets.
  for (auto muxOp : muxMap) {
    auto from = rewriter.getRemappedValue(muxOp.first->getResult(0));
    if (!from)
      return failure();
    auto name = dyn_cast<VariableOp>(muxOp.first).getNameAttr();
    rewriter.setInsertionPoint(muxOp.first);
    auto wireOp =
        rewriter.create<hw::WireOp>(muxOp.first->getLoc(), muxOp.second, name);
    rewriter.replaceOp(from.getDefiningOp(), wireOp);
    rewriter.clearInsertionPoint();
  }

  rewriter.inlineBlockBefore(procedureOp.getBody(), procedureOp);
  return success();
}

/// Lower `always` with an explicit sensitivity list and clock edges to `seq`,
/// otherwise to `comb`.
static LogicalResult lowerAlways(ProcedureOp procedureOp,
                                 ConversionPatternRewriter &rewriter) {
  // The default is a synchronization.
  bool isAsync = false;

  // The default is the combinational logic unit.
  bool isCombination = true;

  // Assume the reset value is at the then region.
  bool atThenRegion = true;

  // Assume we can correctly lower `scf.if`.
  bool falseLowerIf = false;

  Value clk, rst, input, rstValue;

  // Collect all signals.
  DenseMap<Value, moore::Edge> senLists;

  // Collect `comb.mux`.
  DenseMap<Operation *, Value> muxMap;

  for (auto &nestOp : procedureOp.getBodyRegion().getOps()) {
    TypeSwitch<Operation *, void>(&nestOp)
        .Case<EventControlOp>([&](auto op) {
          if (op.getEdge() != moore::Edge::None)
            isCombination = false;
          senLists.insert({op.getInput(), op.getEdge()});
        })
        .Case<NotOp, ConversionOp>([&](auto op) {
          auto operand = op.getOperand();
          if (senLists.contains(operand))
            isAsync = true;

          // Represent reset value doesn't at `then` region.
          if ((senLists.lookup(operand) == Edge::PosEdge && isa<NotOp>(op)) ||
              (senLists.lookup(operand) == Edge::NegEdge &&
               isa<ConversionOp>(op)) ||
              (!isAsync && isa<NotOp>(op))) {
            atThenRegion = false;
            rst = rewriter.getRemappedValue(operand);
          }

          // Erase the reset signal.
          senLists.erase(operand);
        })
        .Case<NonBlockingAssignOp>([&](auto op) {
          input = rewriter.getRemappedValue(op.getSrc());

          // Get the clock signal.
          if (!clk) {
            clk = senLists.begin()->first;
            clk = rewriter.getRemappedValue(clk);
            clk = rewriter.create<seq::ToClockOp>(procedureOp->getLoc(), clk);
          }

          auto name =
              dyn_cast<VariableOp>(op.getDst().getDefiningOp()).getNameAttr();
          auto regOp = rewriter.create<seq::FirRegOp>(procedureOp->getLoc(),
                                                      input, clk, name);

          // Update the variables.
          auto from = rewriter.getRemappedValue(op.getDst());
          rewriter.replaceOp(from.getDefiningOp(), regOp);
        })
        .Case<scf::IfOp>([&](auto op) {
          SmallVector<Value> allConds;
          if (failed(lowerSCFIfOp(op, rewriter, muxMap, allConds))) {
            falseLowerIf = true;
            return;
          }

          // Get the clock signal.
          clk = senLists.begin()->first;
          clk = rewriter.getRemappedValue(clk);
          clk = rewriter.create<seq::ToClockOp>(procedureOp->getLoc(), clk);
          if (!rst)
            rst = op.getCondition();

          for (auto muxOp : muxMap) {
            auto name = dyn_cast<VariableOp>(muxOp.first).getNameAttr();
            auto *defOp = muxOp.second.getDefiningOp();

            input = atThenRegion ? defOp->getOperand(2) : defOp->getOperand(1);
            rstValue =
                atThenRegion ? defOp->getOperand(1) : defOp->getOperand(2);

            auto regOp = rst ? rewriter.create<seq::FirRegOp>(
                                   procedureOp->getLoc(), input, clk, name, rst,
                                   rstValue, hw::InnerSymAttr{}, isAsync)
                             : rewriter.create<seq::FirRegOp>(
                                   procedureOp->getLoc(), input, clk, name);

            // Update the variables.
            auto from = rewriter.getRemappedValue(muxOp.first->getResult(0));
            rewriter.replaceOp(from.getDefiningOp(), regOp);
          }
        });

    if (falseLowerIf)
      return failure();

    // Always represents a combinational logic unit.
    // Like `always @(a, b, ...)` and `always @(*)`.
    if (isCombination) {
      if (failed(lowerAlwaysComb(procedureOp, rewriter)))
        return failure();
      return success();
    }
  }

  rewriter.inlineBlockBefore(procedureOp.getBody(), procedureOp);
  return success();
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
      if (failed(lowerAlwaysComb(op, rewriter)))
        return failure();
      rewriter.eraseOp(op);
      return success();
    case ProcedureKind::Always:
      if (failed(lowerAlways(op, rewriter)))
        return failure();
      rewriter.eraseOp(op);
      return success();
    case ProcedureKind::AlwaysFF:
    case ProcedureKind::AlwaysLatch:
    case ProcedureKind::Initial:
    case ProcedureKind::Final:
      return emitError(op->getLoc(), "Unsupported procedure operation");
    };
    return success();
  }
};

struct SCFIfOpConv : public OpConversionPattern<scf::IfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    DenseMap<Operation *, Value> muxMap;
    SmallVector<Value> allConds;
    if (failed(lowerSCFIfOp(op, rewriter, muxMap, allConds)))
      return failure();

    return success();
  }
};

struct SCFYieldOpConv : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.eraseOp(op);
    return success();
  }
};

struct EventControlOpConv : public OpConversionPattern<EventControlOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(EventControlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.eraseOp(op);
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
      resultTypes.push_back(typeConverter->convertType(res.type));
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

    rewriter.eraseOp(op);
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

    value = rewriter.getRemappedValue(value);
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

struct ReplicateOpConv : public OpConversionPattern<ReplicateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReplicateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<comb::ReplicateOp>(op, resultType,
                                                   adaptor.getValue());
    return success();
  }
};

struct ExtractOpConv : public OpConversionPattern<ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    auto width = typeConverter->convertType(op.getInput().getType())
                     .getIntOrFloatBitWidth();
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getLowBit(), width, op->getLoc());
    Value value =
        rewriter.create<comb::ShrUOp>(op->getLoc(), adaptor.getInput(), amount);

    rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, resultType, value, 0);
    return success();
  }
};

struct ReduceAndOpConv : public OpConversionPattern<ReduceAndOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceAndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getInput().getType());
    Value max = rewriter.create<hw::ConstantOp>(op->getLoc(), resultType, -1);

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::eq,
                                              adaptor.getInput(), max);
    return success();
  }
};

struct ReduceOrOpConv : public OpConversionPattern<ReduceOrOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getInput().getType());
    Value zero = rewriter.create<hw::ConstantOp>(op->getLoc(), resultType, 0);

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::ne,
                                              adaptor.getInput(), zero);
    return success();
  }
};

struct ReduceXorOpConv : public OpConversionPattern<ReduceXorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceXorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<comb::ParityOp>(op, adaptor.getInput());
    return success();
  }
};

struct BoolCastOpConv : public OpConversionPattern<BoolCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(BoolCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (cast<UnpackedType>(op.getInput().getType())
            .castToSimpleBitVectorOrNull()) {
      Type resultType = typeConverter->convertType(op.getInput().getType());
      Value zero = rewriter.create<hw::ConstantOp>(op->getLoc(), resultType, 0);

      rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::ne,
                                                adaptor.getInput(), zero);
      return success();
    }

    return failure();
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

template <typename SourceOp, typename UnsignedOp,
          typename SignedOp = UnsignedOp>
struct BinaryOpConv : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    isSignedType(op)
        ? rewriter.replaceOpWithNewOp<SignedOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs(), false)
        : rewriter.replaceOpWithNewOp<UnsignedOp>(op, adaptor.getLhs(),
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
  target.addLegalDialect<seq::SeqDialect>();
  target.addLegalDialect<mlir::BuiltinDialect>();
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
  typeConverter.addConversion([&](IntType type) {
    return mlir::IntegerType::get(type.getContext(), type.getBitSize());
  });
  // Directly map simple bit vector types to a compact integer type. This needs
  // to be added after all of the other conversions above, such that SBVs
  // conversion gets tried first before any of the others.
  typeConverter.addConversion([&](UnpackedType type) -> std::optional<Type> {
    if (auto sbv = type.getSimpleBitVectorOrNull())
      return mlir::IntegerType::get(type.getContext(), sbv.size);
    if (isa<UnpackedRangeDim, PackedRangeDim>(type)) {
      return mlir::IntegerType::get(type.getContext(),
                                    type.getBitSize().value());
    }
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

void circt::populateMooreToCoreConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    circt::MoorePortInfoMap &portInfoMap) {
  auto *context = patterns.getContext();
  // clang-format off
  patterns.add<
    // Patterns of assignment operations.
    AssignOpConv<ContinuousAssignOp>, AssignOpConv<BlockingAssignOp>,
    AssignOpConv<NonBlockingAssignOp>,

    // Patterns of branch operations.
    BranchOpConv, CondBranchOpConv, SCFIfOpConv, SCFYieldOpConv,

    // Patterns of declaration operations.
    DeclOpConv<NetOp>, DeclOpConv<VariableOp>,

    // Patterns of miscellaneous operations.
    ConstantOpConv, ConcatOpConv, ReplicateOpConv, ExtractOpConv, 
    ConversionOpConv, ProcedureOpConv, EventControlOpConv,

    // Patterns of shifting operations.
    ShlOpConv, ShrOpConv, AShrOpConv,

    // Patterns of binary operations.
    BinaryOpConv<AddOp, comb::AddOp>, BinaryOpConv<SubOp, comb::SubOp>,
    BinaryOpConv<MulOp, comb::MulOp>, BinaryOpConv<AndOp, comb::AndOp>,
    BinaryOpConv<OrOp, comb::OrOp>, BinaryOpConv<XorOp, comb::XorOp>,
    BinaryOpConv<DivOp, comb::DivUOp, comb::DivSOp>,
    BinaryOpConv<ModOp, comb::ModUOp, comb::ModSOp>,
    
    // Patterns of relational operations.
    ICmpOpConv<LtOp>, ICmpOpConv<LeOp>, ICmpOpConv<GtOp>, ICmpOpConv<GeOp>,
    ICmpOpConv<EqOp>, ICmpOpConv<NeOp>, ICmpOpConv<CaseEqOp>,
    ICmpOpConv<CaseNeOp>, ICmpOpConv<WildcardEqOp>, ICmpOpConv<WildcardNeOp>,

    // Patterns of unary operations.
    ReduceAndOpConv, ReduceOrOpConv, ReduceXorOpConv, 
    BoolCastOpConv, NotOpConv,
    
    // Patterns of other operations outside Moore dialect.
    HWInstanceOpConv, HWOutputOpConv, UnrealizedConversionCastConv>(
          typeConverter, context);

  patterns.add<SVModuleOpConv, InstanceOpConv, PortOpConv>(
      typeConverter, context, portInfoMap);

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
  pm.addPass(moore::createInfoCollectionPass());
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

  populateMooreToCoreConversionPatterns(typeConverter, patterns,portInfoMap);

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
