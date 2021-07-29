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
#include "circt/Dialect/FIRRTL/FIREmitter.h"
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
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

namespace cl = llvm::cl;
using llvm::SmallSet;
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
  void expandConnects(Value subaccess, FieldRef within, FieldRef target);

  void preserve(Operation *op) {
    if (preservedOps.insert(op).second)
      worklistOps.push_back(op);
  }
  void preserve(FieldRef field) {
    if (preservedFields.insert(field).second) {
      preservedFieldsByValue[field.getValue()].insert(field.getFieldID());
      worklistFields.push_back(field);
    }
  }

  ModuleOp rootOp;
  DenseSet<Operation *> preservedOps;
  DenseSet<FieldRef> preservedFields;
  SmallVector<Operation *> worklistOps;
  SmallVector<FieldRef> worklistFields;

  /// The set of preserved fields for each value.
  DenseMap<Value, SmallSet<unsigned, 1>> preservedFieldsByValue;
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

  mlir::emitRemark(currentOp->getLoc(), "resolved `") << path << "` here";

  // If the remaining path is empty, there are now field accesses to resolve.
  // Return immediately in this case.
  if (currentPath.empty()) {
    for (auto result : currentOp->getResults())
      preserve(FieldRef(result, 0));
    return success();
  }

  // Process fields and indices.
  FieldRef currentField(currentOp->getResult(0), 0);
  FIRRTLType currentType = currentField.getValue().getType().cast<FIRRTLType>();
  while (!currentPath.empty()) {
    // Handle field accesses.
    if (currentPath.consume_front(".")) {
      // Extract the field name.
      nextSep = currentPath.find_first_of(".[");
      auto name = currentPath.slice(0, nextSep);
      currentPath = currentPath.slice(nextSep, currentPath.npos);

      // Ensure we have a bundle we can access into.
      auto bundleType = currentType.dyn_cast<BundleType>();
      if (!bundleType) {
        mlir::emitError(currentOp->getLoc(), "cannot access `")
            << name << "`: `" << getFieldName(currentField)
            << "` is not a bundle";
        return failure();
      }

      // Resolve the accessed field.
      auto index = bundleType.getElementIndex(name);
      if (!index.hasValue()) {
        mlir::emitError(currentOp->getLoc(), "no field `")
            << name << "` in bundle `" << getFieldName(currentField) << "`";
        return failure();
      }
      currentField =
          FieldRef(currentField.getValue(),
                   currentField.getFieldID() + bundleType.getFieldID(*index));
      currentType = bundleType.getElement(*index).type;
      continue;
    }

    // Handle index accesses.
    if (currentPath.consume_front("[")) {
      // Extract the index.
      unsigned index;
      if (currentPath.consumeInteger(0, index)) {
        mlir::emitError(currentOp->getLoc(),
                        "invalid element index in object path `")
            << currentPath << "`";
        return failure();
      }
      if (!currentPath.consume_front("]")) {
        mlir::emitError(currentOp->getLoc(), "expected `]` in object path `")
            << currentPath << "`";
        return failure();
      }

      // Ensure we have a vector we can access into.
      auto vectorType = currentType.dyn_cast<FVectorType>();
      if (!vectorType) {
        mlir::emitError(currentOp->getLoc(), "cannot access index `")
            << index << "`: `" << getFieldName(currentField)
            << "` is not a vector";
        return failure();
      }

      // Resolve the accessed index.
      if (index >= vectorType.getNumElements()) {
        mlir::emitError(currentOp->getLoc(), "no index `")
            << index << "` in vector `" << getFieldName(currentField) << "`";
        return failure();
      }
      currentField =
          FieldRef(currentField.getValue(),
                   currentField.getFieldID() + vectorType.getFieldID(index));
      currentType = vectorType.getElementType();
      continue;
    }

    mlir::emitError(currentOp->getLoc(), "expected `.` or `[` in object path `")
        << currentPath << "`";
    return failure();
  }

  // Mark this field as to be preserved.
  preserve(currentField);
  return success();
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

      // Preserve all the inputs that contribute to the operation. Subindex and
      // subfield are special in that they explicitly only preserve the accessed
      // field.
      TypeSwitch<Operation *>(op)
          .Case<SubfieldOp, SubindexOp>(
              [&](auto op) { preserve(op.getAccessedField()); })
          .Default([&](auto op) {
            for (const auto &operand : op->getOperands())
              preserve(FieldRef(operand, 0));
          });

      // Preserve the parent operation.
      if (auto parentOp = op->getParentOp())
        preserve(parentOp);

      // Instances preserve the instantiated module.
      if (auto instOp = dyn_cast<InstanceOp>(op))
        preserve(instOp.getReferencedModule());
    }
  }
}

/// Get the type of a nested field within a type.
static FIRRTLType getFieldType(FIRRTLType type, unsigned fieldID) {
  while (fieldID) {
    if (auto bundleType = type.dyn_cast<BundleType>()) {
      auto index = bundleType.getIndexForFieldID(fieldID);
      type = bundleType.getElement(index).type;
      fieldID = fieldID - bundleType.getFieldID(index);
    } else if (auto vectorType = type.dyn_cast<FVectorType>()) {
      auto index = vectorType.getIndexForFieldID(fieldID);
      type = vectorType.getElementType();
      fieldID = fieldID - vectorType.getFieldID(index);
    } else {
      // If we reach here, the field ref is pointing inside some aggregate type
      // that isn't a bundle or a vector. If the type is a ground type, then the
      // fieldID should be 0 at this point, and we should have broken from the
      // loop.
      llvm::errs() << "Accessing " << fieldID << " inside " << type << "\n";
      llvm_unreachable("unsupported type");
    }
  }
  return type;
}

static void forEachConnectedField(
    FIRRTLType lhsType, FieldRef lhs, FIRRTLType rhsType, FieldRef rhs,
    bool flipped, llvm::function_ref<void(FieldRef, FieldRef, bool)> callback) {
  if (auto lhsBundle = lhsType.dyn_cast<BundleType>()) {
    auto rhsBundle = rhsType.cast<BundleType>();
    for (unsigned lhsIndex = 0, e = lhsBundle.getNumElements(); lhsIndex < e;
         ++lhsIndex) {
      auto lhsField = lhsBundle.getElements()[lhsIndex].name.getValue();
      auto rhsIndex = rhsBundle.getElementIndex(lhsField);
      if (!rhsIndex)
        continue;
      auto &lhsElt = lhsBundle.getElements()[lhsIndex];
      auto &rhsElt = rhsBundle.getElements()[*rhsIndex];
      forEachConnectedField(
          lhsElt.type, lhs.getSubField(lhsBundle.getFieldID(lhsIndex)),
          rhsElt.type, rhs.getSubField(rhsBundle.getFieldID(*rhsIndex)),
          flipped ^ lhsElt.isFlip, callback);
    }
  } else if (auto lhsVecType = lhsType.dyn_cast<FVectorType>()) {
    auto rhsVecType = rhsType.cast<FVectorType>();
    auto numElements =
        std::min(lhsVecType.getNumElements(), rhsVecType.getNumElements());
    for (unsigned i = 0; i < numElements; ++i)
      forEachConnectedField(lhsVecType.getElementType(),
                            lhs.getSubField(lhsVecType.getFieldID(i)),
                            rhsVecType.getElementType(),
                            rhs.getSubField(rhsVecType.getFieldID(i)), flipped,
                            callback);
  } else if (lhsType.isGround()) {
    // Leaf element, look up their expressions, and create the constraint.
    callback(lhs, rhs, flipped);
  } else {
    llvm_unreachable("unknown type inside a bundle!");
  }
}

/// Invoke a callback for each leaf field that is connected.
static void forEachConnectedField(
    FieldRef lhs, FieldRef rhs,
    llvm::function_ref<void(FieldRef, FieldRef, bool)> callback) {
  auto lhsType = getFieldType(lhs.getValue().getType().cast<FIRRTLType>(),
                              lhs.getFieldID());
  auto rhsType = getFieldType(rhs.getValue().getType().cast<FIRRTLType>(),
                              rhs.getFieldID());
  forEachConnectedField(lhsType, lhs, rhsType, rhs, false, callback);
}

/// Considers all drives to subaccess, looking through further subaccesses, and
/// preserves drives that touch the given target field in any way.
void Marker::expandConnects(Value subaccess, FieldRef within, FieldRef target) {
  for (const auto &use : subaccess.getUses()) {
    TypeSwitch<Operation *>(use.getOwner())
        .Case<SubfieldOp, SubindexOp>([&](auto op) {
          // Check if this subfield contains the target field, in which case we
          // are interested in preserving it. Otherwise skip.
          FieldRef newField =
              within.getSubField(op.getAccessedField().getFieldID());
          auto fieldType = op.getType();
          if (target.getFieldID() < newField.getFieldID() ||
              target.getFieldID() >
                  newField.getFieldID() + fieldType.getMaxFieldID())
            return;

          // This subfield is interesting. Preserve it.
          expandConnects(op, newField, target);
        })
        .Case<SubaccessOp>([&](auto op) {
          // Subaccesses are weird because they can target any element of a
          // vector. Thus we just assume that this also touches the target
          // field.
          expandConnects(op, FieldRef(op, 0), target);
        })
        .Case<ConnectOp, PartialConnectOp>([&](auto op) {
          if (op.dest() != subaccess)
            return;
          bool anyPreserved = false;
          forEachConnectedField(within, FieldRef(op.src(), 0),
                                [&](FieldRef lhs, FieldRef rhs, bool flipped) {
                                  preserve(flipped ? lhs : rhs);
                                  anyPreserved = true;
                                });
          if (anyPreserved)
            preserve(op);
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
  DenseSet<Value> possiblyUninitializedValues;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!marker.preservedOps.contains(op)) {
      ++stats[op->getName().getIdentifier()];
      ++numberRemoved;
      op->dropAllDefinedValueUses();
      op->erase();
      return WalkResult::skip();
    }
    if (isa<WireOp, RegOp, RegResetOp>(op))
      for (const auto &result : op->getResults())
        possiblyUninitializedValues.insert(result);
    else if (auto moduleOp = dyn_cast<FModuleOp>(op))
      for (unsigned i = 0, e = moduleOp.getNumArguments(); i < e; ++i)
        if (getModulePortDirection(moduleOp, i) == Direction::Output)
          possiblyUninitializedValues.insert(moduleOp.getArgument(i));
    return WalkResult::advance();
  });

  // For partially preserved bundles and vectors, stub out all the non-preserved
  // fields.
  for (auto value : possiblyUninitializedValues) {
    const auto &preservedFields = marker.preservedFieldsByValue[value];

    // Setup a builder that inserts after the operation that defines the current
    // value, or at the end of the module if the value is a port.
    ImplicitLocOpBuilder builder(value.getLoc(), module.getContext());
    if (auto blockArg = value.dyn_cast<BlockArgument>())
      builder.setInsertionPointToEnd(blockArg.getOwner());
    else
      builder.setInsertionPointAfter(value.getDefiningOp());

    // Initialize each field as necessary.
    SmallDenseMap<Type, Value> invalidValues;
    llvm::function_ref<void(Value, unsigned, bool)> init =
        [&](Value value, unsigned fieldID, bool flipped) {
          auto type = value.getType().cast<FIRRTLType>();
          if (auto bundle = type.dyn_cast<BundleType>()) {
            for (unsigned i = 0, e = bundle.getNumElements(); i < e; ++i) {
              auto subOp = builder.create<SubfieldOp>(value, i);
              init(subOp, fieldID + bundle.getFieldID(i),
                   flipped ^ bundle.getElement(i).isFlip);
              if (subOp->use_empty())
                subOp->erase();
            }
          } else if (auto vectorType = type.dyn_cast<FVectorType>()) {
            for (unsigned i = 0, e = vectorType.getNumElements(); i < e; ++i) {
              auto subOp = builder.create<SubindexOp>(value, i);
              init(subOp, fieldID + vectorType.getFieldID(i), flipped);
              if (subOp->use_empty())
                subOp->erase();
            }
          } else if (type.isGround()) {
            if (!flipped && !preservedFields.contains(fieldID)) {
              auto &stub = invalidValues[value.getType()];
              if (!stub) {
                auto ip = builder.saveInsertionPoint();
                builder.setInsertionPointToStart(builder.getInsertionBlock());
                stub = builder.create<InvalidValueOp>(value.getType());
                builder.restoreInsertionPoint(ip);
              }
              builder.create<ConnectOp>(value, stub);
            }
          } else {
            llvm_unreachable("unknown type inside a bundle!");
          }
        };
    init(value, 0, false);
  }

  // Print some statistics on the ops that were removed.
  SmallVector<std::pair<Identifier, unsigned>> statsSorted(stats.begin(),
                                                           stats.end());
  llvm::sort(statsSorted,
             [](const auto &a, const auto &b) { return a.second > b.second; });
  llvm::errs() << "Removed " << numberRemoved << " operations:\n";
  for (auto pair : statsSorted) {
    llvm::errs() << "  " << pair.first << ": " << pair.second << "\n";
  }

  // Verify that we didn't break the module.
  return module.verify();
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
  context.loadDialect<firrtl::FIRRTLDialect>();

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
    if (failed(exportFIRRTL(module.get(), output->os())))
      return failure();
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
  cl::ParseCommandLineOptions(argc, argv, "FIR fan-in/out isolator\n");

  // Execute and then exit immediately to don't run the slow MLIRContext
  // destructor.
  MLIRContext context;
  exit(failed(executeTool(context)));
}
