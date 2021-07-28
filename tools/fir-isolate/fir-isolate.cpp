//===- fir-isolate.cpp - The fir-isolate utility --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements 'fir-isolate', which reduces a FIR input to a selected
// set of nodes, facilitating debugging and test case isolation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/FieldRef.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Translation/ExportVerilog.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

namespace cl = llvm::cl;
using namespace mlir;
using namespace circt;
using namespace firrtl;

/// Allow the user to specify the input/output file format. This allows the user
/// to specify ambiguous cases like stdin/stdout.
enum class FormatKind { Unspecified, FIR, MLIR };

static cl::opt<FormatKind> inputFormat(
    "input-format", cl::desc("Specify input file format:"),
    cl::values(clEnumValN(FormatKind::Unspecified, "autodetect",
                          "Autodetect input format"),
               clEnumValN(FormatKind::FIR, "fir", "Parse as FIR file"),
               clEnumValN(FormatKind::MLIR, "mlir", "Parse as MLIR file")),
    cl::init(FormatKind::Unspecified));

static cl::opt<FormatKind> outputFormat(
    "output-format", cl::desc("Specify output file format:"),
    cl::values(clEnumValN(FormatKind::Unspecified, "autodetect",
                          "Autodetect output format"),
               clEnumValN(FormatKind::FIR, "fir", "Emit FIR file"),
               clEnumValN(FormatKind::MLIR, "mlir", "Emit MLIR file")),
    cl::init(FormatKind::Unspecified));

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output file name"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool>
    ignoreFIRLocations("ignore-fir-locators",
                       cl::desc("ignore the @info locations in the .fir file"),
                       cl::init(false));

static cl::list<std::string> isolateObjects("i", cl::desc("Isolate objects"));

namespace {
struct Marker {
  Marker(ModuleOp rootOp) : rootOp(rootOp) {}
  LogicalResult findAndMarkObject(StringRef path);
  void expandFanIn();
  void expandConnects(Value subaccess, FieldRef within, FieldRef matches);

  void preserve(Operation *op) {
    if (preservedOps.insert(op).second)
      worklistOps.push_back(op);
  }
  void preserve(FieldRef field) {
    if (preservedFields.insert(field).second)
      worklistFields.push_back(field);
  }

  ModuleOp rootOp;
  DenseSet<Operation *> preservedOps;
  DenseSet<FieldRef> preservedFields;
  SmallVector<Operation *> worklistOps;
  SmallVector<FieldRef> worklistFields;
};
} // namespace

LogicalResult Marker::findAndMarkObject(StringRef path) {
  Operation *currentOp = rootOp;
  StringRef currentPath = path;

  // Resolve path segments.
  size_t nextSep;
  do {
    // Chop off the next name in the path.
    nextSep = currentPath.find_first_of("/.[");
    auto name = currentPath.slice(0, nextSep);
    currentPath = currentPath.slice(nextSep, currentPath.npos);

    // Look for an operation with that name.
    Operation *found = nullptr;
    currentOp->walk([&](Operation *op) {
      TypeSwitch<Operation *>(op)
          .Case<FModuleOp>([&](auto op) {
            if (op.getName() == name)
              found = op;
          })
          .Case<InstanceOp, RegOp, RegResetOp, WireOp, NodeOp>([&](auto op) {
            if (op.name() == name)
              found = op;
          });
      if (found)
        return WalkResult::interrupt();
      if (isa<FModuleOp>(op))
        return WalkResult::skip();
      return WalkResult::advance();
    });

    // Complain if we can't find the object.
    if (!found) {
      mlir::emitError(currentOp->getLoc(), "unknown object `") << name << "`";
      return failure();
    }
    currentOp = found;
    preserve(currentOp);
    if (auto inst = dyn_cast<InstanceOp>(currentOp)) {
      currentOp = inst.getReferencedModule();
      preserve(currentOp);
    }
  } while (currentPath.consume_front("/"));

  mlir::emitRemark(currentOp->getLoc(), "resolved `")
      << path << "` to this operation";

  // If the remaining path is empty, there are now field accesses to resolve.
  // Return immediately in this case.
  if (currentPath.empty()) {
    for (auto result : currentOp->getResults())
      preserve(FieldRef(result, 0));
    return success();
  }

  // TODO: Resolve into fields.

  mlir::emitError(currentOp->getLoc(), "subaccess `")
      << currentPath << "` not yet supported";
  return failure();
}

void Marker::expandFanIn() {
  while (!worklistFields.empty() || !worklistOps.empty()) {
    while (!worklistFields.empty()) {
      FieldRef field = worklistFields.pop_back_val();
      // llvm::errs() << "Processing " << field.getValue() << "\n";
      auto value = field.getValue();
      auto *op = value.getDefiningOp();

      // If this value has wire semantics, consider connections to it.
      if (isa_and_nonnull<WireOp, RegOp, RegResetOp>(op) ||
          value.isa<BlockArgument>())
        expandConnects(value, FieldRef(value, 0), field);

      if (!op)
        continue;

      // Ensure we preserve the operation that defines the value.
      preserve(op);

      // Instance results marked as to be preserved in turn mark the
      // corresponding module port to be preserved.
      if (auto instOp = dyn_cast<InstanceOp>(op)) {
        auto target = dyn_cast<FModuleOp>(instOp.getReferencedModule());
        auto resultNumber = value.cast<OpResult>().getResultNumber();
        preserve(
            FieldRef(target.getArgument(resultNumber), field.getFieldID()));
      }
    }
    while (!worklistOps.empty()) {
      Operation *op = worklistOps.pop_back_val();

      // Preserve all the inputs that contribute to the operation.
      for (const auto &operand : op->getOperands())
        preserve(FieldRef(operand, 0));

      // Preserve the parent operation.
      if (auto parentOp = op->getParentOp())
        preserve(parentOp);

      // Instances preserve the instantiated module.
      if (auto instOp = dyn_cast<InstanceOp>(op))
        preserve(instOp.getReferencedModule());
    }
  }
}

/// Considers all drives to subaccess, looking through further subaccesses, and
/// preserves drives that touch the given `matches` field in any way.
void Marker::expandConnects(Value subaccess, FieldRef within,
                            FieldRef matches) {
  for (const auto &use : subaccess.getUses()) {
    Operation *op = use.getOwner();
    // llvm::errs() << "- Use " << *op << "\n";

    TypeSwitch<Operation *>(op)
        .Case<SubindexOp, SubfieldOp, SubaccessOp>([&](auto op) {
          // TODO: Update the `within` field to point at the accessed field.
          expandConnects(op, within, matches);
        })
        .Case<ConnectOp, PartialConnectOp>([&](auto op) {
          // TODO: Go through the pairs of connected fields and only consider
          // the ones that actually match.
          preserve(FieldRef(op.src(), 0));
        });
  }
}

static LogicalResult executeTool(ModuleOp module) {
  // Find the objects in the IR that we are supposed to isolate, and seed the
  // set of reachable nodes with them.
  Marker marker{module};
  for (const auto &path : isolateObjects) {
    auto result = marker.findAndMarkObject(path);
    if (failed(result))
      return failure();
  }

  // Mark the fan-in cones to the marked objects as to be preserved.
  marker.expandFanIn();

  llvm::errs() << "Preserving " << marker.preservedOps.size() << " ops and "
               << marker.preservedFields.size() << " fields\n";

  // Remove all operations in the module not marked as to be preserved.
  unsigned numberRemoved = 0;
  DenseMap<Identifier, unsigned> stats;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!marker.preservedOps.contains(op)) {
      ++stats[op->getName().getIdentifier()];
      ++numberRemoved;
      op->dropAllDefinedValueUses();
      op->erase();
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  SmallVector<std::pair<Identifier, unsigned>> statsSorted(stats.begin(),
                                                           stats.end());
  llvm::sort(statsSorted,
             [](const auto &a, const auto &b) { return a.second > b.second; });
  llvm::errs() << "Removed " << numberRemoved << " operations:\n";
  for (auto pair : statsSorted) {
    llvm::errs() << "  " << pair.first << ": " << pair.second << "\n";
  }

  return success();
}

static LogicalResult executeTool(MLIRContext &context) {
  // Figure out the input format if unspecified.
  if (inputFormat == FormatKind::Unspecified) {
    if (StringRef(inputFilename).endswith(".fir"))
      inputFormat = FormatKind::FIR;
    else if (StringRef(inputFilename).endswith(".mlir"))
      inputFormat = FormatKind::MLIR;
    else {
      llvm::errs()
          << "unknown input format: specify with --input-format={fir,mlir}\n";
      return failure();
    }
  }

  // Figure out the output format if unspecified.
  if (outputFormat == FormatKind::Unspecified) {
    if (StringRef(outputFilename).endswith(".fir"))
      outputFormat = FormatKind::FIR;
    else if (StringRef(outputFilename).endswith(".mlir"))
      outputFormat = FormatKind::MLIR;
    else {
      outputFormat = inputFormat.getValue();
    }
  }

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Set up the output file.
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Register our dialects.
  context.loadDialect<firrtl::FIRRTLDialect, hw::HWDialect, comb::CombDialect,
                      sv::SVDialect>();

  // Move the input into the source manager.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  // Parse the input.
  OwningModuleRef module;
  if (inputFormat == FormatKind::FIR) {
    firrtl::FIRParserOptions options;
    options.ignoreInfoLocators = ignoreFIRLocations;
    module = importFIRRTL(sourceMgr, &context, options);
  } else {
    assert(inputFormat == FormatKind::MLIR);
    module = parseSourceFile(sourceMgr, &context);
  }
  if (!module)
    return failure();

  // Do some magic with the module.
  if (failed(executeTool(module.get())))
    return failure();

  // Emit the output.
  switch (outputFormat) {
  case FormatKind::MLIR:
    module->print(output->os());
    break;
  case FormatKind::FIR:
    llvm_unreachable("FIR output not implemented yet");
    break;
  case FormatKind::Unspecified:
    llvm_unreachable("handled above");
    return failure();
  }

  // If the result succeeded and we're emitting a file, close it.
  output->keep();
  return success();
}

/// Main driver for firtool command.  This sets up LLVM and MLIR, and parses
/// command line options before passing off to 'executeFirtool'.  This is set up
/// so we can `exit(0)` at the end of the program to avoid teardown of the
/// MLIRContext and modules inside of it (reducing compile time).
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  registerLoweringCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "FIR fan-in/out isolator\n");

  // Execute and then exit immediately to don't run the slow MLIRContext
  // destructor.
  MLIRContext context;
  exit(failed(executeTool(context)));
}
