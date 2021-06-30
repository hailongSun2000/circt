//===- circt-verify.cpp - The CIRCT verification driver -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the `circt-verify` tool, which is the main driver for
// verification of CIRCT designs.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Verif/Verif.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

#define DEBUG_TYPE "circt-verify"

using namespace llvm;
using namespace mlir;
using namespace circt;
using circt::comb::ICmpPredicate;

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<bool>
    splitInputFile("split-input-file",
                   cl::desc("Split the input file into pieces and process each "
                            "chunk independently"),
                   cl::init(false));

static cl::opt<bool>
    verifyDiagnostics("verify-diagnostics",
                      cl::desc("Check that emitted diagnostics match "
                               "expected-* lines on the corresponding line"),
                      cl::init(false));

//===----------------------------------------------------------------------===//
// Verification Interpreter
//===----------------------------------------------------------------------===//

// The following set of functions is a temporary proof-of-concept to show how a
// handful of verification constructs can be implemented and used in designs. It
// is expected for this interpreting behaviour to be replaced with a proper
// lowering to LLVM for fast execution of multi-cycle tests, and a potential
// lowering/mapping to a SMT solver like Z3 for formal proofs.

struct State {
  FailureOr<APInt> evaluate(Value value);
  LogicalResult evaluate(Operation *op, ArrayRef<APInt> operands);

  APInt &operator[](Value value) { return evals.find(value)->second; }

private:
  /// Mapping of IR values to the actual evaluated result.
  llvm::DenseMap<Value, APInt> evals;
  void store(Value value, APInt eval) { evals.insert({value, eval}); }
};

FailureOr<APInt> State::evaluate(Value value) {
  // If we already have a result, return that immediately.
  auto it = evals.find(value);
  if (it != evals.end())
    return it->second;

  // Otherwise push the value onto the worklist, and start chugging away.
  SmallVector<Value> worklist;
  SmallVector<APInt> operands;
  worklist.push_back(value);
  while (!worklist.empty()) {
    // Fetch the next element off the worklist. If we already know it's value,
    // no need to continue.
    auto value = worklist.back();
    if (evals.count(value)) {
      worklist.pop_back();
      continue;
    }

    // Otherwise we grab the instruction that defines this value. We need this
    // value to come from an op; all other values (like block args) should be
    // pre-populated in the state (e.g. to evaluate an instance).
    auto op = value.getDefiningOp();
    assert(op && "non-op values must be prepopulated in the state");
    LLVM_DEBUG(llvm::dbgs() << "- Evaluating " << value << "\n");

    // Gather the operands. Any values that are missing go onto the worklist. If
    // any were missing, we jump back up to the beginning of the loop body,
    // which will cause the inserted operand values to be evaluated first.
    // Otherwise we pop the operation off the worklist.
    bool anyMissing = false;
    operands.clear();
    for (auto operand : op->getOperands()) {
      auto it = evals.find(operand);
      if (it == evals.end()) {
        worklist.push_back(operand);
        anyMissing = true;
      } else {
        operands.push_back(it->second);
      }
    }
    if (anyMissing)
      continue;
    else
      worklist.pop_back();

    // Handle the actual operation.
    if (failed(evaluate(op, operands)))
      return failure();
  }

  return evals.find(value)->second;
}

LogicalResult State::evaluate(Operation *op, ArrayRef<APInt> operands) {
  bool didFail = false;
  llvm::TypeSwitch<Operation *>(op)
      .Case<hw::ConstantOp>([&](auto op) { store(op.getResult(), op.value()); })
      .Case<hw::InstanceOp>([&](auto op) {
        // Look up what we're actually instantiating.
        auto target =
            dyn_cast_or_null<hw::HWModuleOp>(op.getReferencedModule());
        if (!target) {
          op->emitOpError("instantiates unsupported module");
          didFail = true;
          return;
        }
        auto targetOutput =
            cast<hw::OutputOp>(target.getBodyBlock()->getTerminator());

        // Create a new interpreter state and populate it with the values passed
        // into the instance as arguments.
        State state;
        for (auto it : llvm::zip(target.getArguments(), operands))
          state.store(std::get<0>(it), std::get<1>(it));

        // Query the results of the instantiated module and copy them over to
        // this interpreter state.
        for (auto it : llvm::zip(targetOutput.getOperands(), op.getResults())) {
          auto result = state.evaluate(std::get<0>(it));
          if (failed(result)) {
            didFail = true;
            return;
          }
          store(std::get<1>(it), *result);
        }
      })
      .Case<comb::AddOp>([&](auto op) {
        auto result = operands[0];
        for (auto operand : operands.drop_front())
          result += operand;
        store(op.getResult(), result);
      })
      .Case<comb::ICmpOp>([&](auto op) {
        auto result =
            comb::applyCmpPredicate(op.predicate(), operands[0], operands[1]);
        store(op.getResult(), APInt(1, result));
      })
      .Default([&](auto op) {
        op->emitOpError("unsupported for verification");
        didFail = true;
      });
  return failure(didFail);
}

//===----------------------------------------------------------------------===//
// Command Line Implementation
//===----------------------------------------------------------------------===//

static const char *symbolizeCmpPredicate(ICmpPredicate predicate) {
  switch (predicate) {
  case ICmpPredicate::eq:
    return "==";
  case ICmpPredicate::ne:
    return "!=";
  case ICmpPredicate::slt:
    return "<";
  case ICmpPredicate::sle:
    return "<=";
  case ICmpPredicate::sgt:
    return ">";
  case ICmpPredicate::sge:
    return ">=";
  case ICmpPredicate::ult:
    return "<";
  case ICmpPredicate::ule:
    return "<=";
  case ICmpPredicate::ugt:
    return ">";
  case ICmpPredicate::uge:
    return ">=";
  }
}

/// Try to look up the name of an output port on an instance.
static StringAttr getResultName(hw::InstanceOp instOp, unsigned resultIdx) {
  auto targetOp = instOp.getReferencedModule();
  if (!targetOp)
    return {};
  auto names = targetOp->getAttrOfType<ArrayAttr>("resultNames");
  if (!names || resultIdx >= names.size())
    return {};
  return names[resultIdx].dyn_cast<StringAttr>();
}

/// Executes a single `CheckOp` in the IR.
static LogicalResult execute(verif::CheckOp checkOp, State &state) {
  LLVM_DEBUG(llvm::dbgs() << "Running " << checkOp << "\n");

  // Evaluate the condition of the check.
  auto cond = state.evaluate(checkOp.cond());
  if (failed(cond))
    return failure();

  // If the condition is true, we can simply return a success here.
  if ((*cond).isOneValue())
    return success();

  // Otherwise report an error to the user.
  auto diag = mlir::emitError(checkOp.getLoc(), "check failed");
  if (auto note = checkOp->getAttrOfType<StringAttr>("note"))
    diag << ": " << note;

  // Try to look through one level of comparison so we can report better errors
  // to the user.
  if (auto cmp =
          dyn_cast_or_null<comb::ICmpOp>(checkOp.cond().getDefiningOp())) {
    // Attach the comparison as a note, making instance ports symbolic.
    SmallVector<std::pair<SmallString<32>, Value>, 2> symbolicValues;
    auto &note = diag.attachNote(cmp.getLoc());
    auto print = [&](Value value) {
      auto *op = value.getDefiningOp();
      if (auto instOp = dyn_cast_or_null<hw::InstanceOp>(op)) {
        auto result = value.cast<OpResult>();
        if (auto name = getResultName(instOp, result.getResultNumber())) {
          auto instName = instOp.getName();
          SmallString<32> str;
          if (!instName.empty()) {
            str.append(instName);
            str.push_back('.');
          }
          str.append(name.getValue());
          note << str;
          symbolicValues.push_back({std::move(str), value});
          return;
        }
      }
      SmallString<32> str;
      state[value].toStringUnsigned(str);
      note << str;
    };
    note << "comparison ";
    print(cmp.lhs());
    note << " " << symbolizeCmpPredicate(cmp.predicate()) << " ";
    print(cmp.rhs());
    note << " failed";

    // Print the values for the symbolic values.
    for (auto pair : symbolicValues) {
      auto &symbolNote = diag.attachNote(cmp.getLoc());
      SmallString<32> str;
      state[pair.second].toStringUnsigned(str);
      symbolNote << "where " << pair.first << " = " << str;
    }
  }

  return failure();
}

/// Executes the verification operations in an MLIR module.
static LogicalResult execute(ModuleOp module) {
  bool anyFailed = false;
  module.walk([&](verif::TestOp testOp) {
    LLVM_DEBUG(llvm::dbgs() << "Running test\n");
    State state;
    testOp.walk([&](verif::CheckOp checkOp) {
      if (failed(execute(checkOp, state)))
        anyFailed = true;
    });
    return WalkResult::skip();
  });
  return failure(anyFailed);
}

/// Implement the actual processing.
static LogicalResult execute(MLIRContext &context, llvm::SourceMgr &sourceMgr,
                             TimingScope &ts) {
  // Parse the input.
  auto parserTimer = ts.nest("Parser");
  OwningModuleRef module = parseSourceFile(sourceMgr, &context);
  parserTimer.stop();
  if (!module)
    return failure();

  // Setup a pass manager and apply command line options.
  PassManager pm(&context);
  pm.enableTiming(ts);
  applyPassManagerCLOptions(pm);

  pm.addPass(createCSEPass());
  pm.addPass(createSimpleCanonicalizerPass());

  // Run the pass manager.
  if (failed(pm.run(module.get())))
    return failure();

  // Run the verification stuff.
  LLVM_DEBUG(llvm::dbgs() << *module.get() << "\n");
  return execute(module.get());
}

static LogicalResult execute(MLIRContext &context,
                             std::unique_ptr<MemoryBuffer> input,
                             TimingScope &ts) {
  // Setup the source manager and perform actions.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  if (verifyDiagnostics) {
    SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
    context.printOpOnDiagnostic(false);
    (void)execute(context, sourceMgr, ts);
    return sourceMgrHandler.verify();
  } else {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return execute(context, sourceMgr, ts);
  }
}

/// This implements the top-level logic for the `circt-verify` command,
/// invoked once command line options are parsed and LLVM/MLIR are all set up
/// and ready to go.
static LogicalResult execute(MLIRContext &context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Register our dialects.
  context.loadDialect<hw::HWDialect, comb::CombDialect, verif::VerifDialect>();

  // The split-input-file mode is a very specific mode that slices the file
  // up into small pieces and checks each independently.
  if (splitInputFile)
    return splitAndProcessBuffer(
        std::move(input),
        [&](std::unique_ptr<MemoryBuffer> buffer, raw_ostream &os) {
          return execute(context, std::move(buffer), ts);
        },
        llvm::outs());
  else
    return execute(context, std::move(input), ts);
}

/// Main driver for the circt-verify command. This sets up LLVM and MLIR,
/// parses command line options, and then passes off to `execute'.
int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "CIRCT verification utility\n");

  // Do the actual processing.
  MLIRContext context;
  auto result = execute(context);

  // Use "exit" instead of return'ing to signal completion. This avoids
  // invoking the `MLIRContext` destructor, which spends a bunch of time
  // deallocating memory, which process exit will do for us.
  exit(failed(result));
}
