//===- circt-vcd-testbench.cpp - The trace-based testbench utility --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'circt-vcd-testbench' tool, which reads VCD files
// and creates a testbench that can be used to verify if a circuit would
// generate the exact same output waveforms given the input waveforms in the VCD
// file.
//
//===----------------------------------------------------------------------===//

#include "VCDParser.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "circt-vcd-testbench"

using namespace llvm;
using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//

static cl::opt<std::string>
    inputCircuitFile(cl::Positional, cl::desc("<circuit>"), cl::Required);
static cl::opt<std::string> inputTraceFile(cl::Positional, cl::desc("<trace>"),
                                           cl::Required);

static cl::opt<std::string> modulePath(cl::Positional,
                                       cl::desc("<module-path>"), cl::Required);

static cl::opt<std::string> moduleName(cl::Positional,
                                       cl::desc("<module-name>"), cl::Required);

//===----------------------------------------------------------------------===//
// Tool Implementation
//===----------------------------------------------------------------------===//

namespace {
struct RootPort {
  StringRef name;
  VCDVariable *var;
  Type type;
  bool isInput;
};
} // namespace

/// Execute the main chunk of work of the tool. This function reads the input
/// and generates a testbench.
static LogicalResult execute(MLIRContext *context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Open the circuit file.
  std::string errorMessage;
  auto inputCircuit = openInputFile(inputCircuitFile, &errorMessage);
  if (!inputCircuit) {
    mlir::emitError(UnknownLoc::get(context), "unable to open circuit: ")
        << errorMessage << "\n";
    return failure();
  }

  // Open the trace file.
  auto inputTrace = openInputFile(inputTraceFile, &errorMessage);
  if (!inputTrace) {
    mlir::emitError(UnknownLoc::get(context), "unable to open trace: ")
        << errorMessage << "\n";
    return failure();
  }

  // Setup the source manager.
  SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, context);

  // Parse the module.
  OwningModuleRef module;
  {
    auto parserTimer = ts.nest("MLIR Parser");
    sourceMgr.AddNewSourceBuffer(std::move(inputCircuit), llvm::SMLoc());
    module = parseSourceFile(sourceMgr, context);
  }

  // Setup the input parser and process the header.
  auto traceFileID =
      sourceMgr.AddNewSourceBuffer(std::move(inputTrace), llvm::SMLoc());
  VCDParser parser(sourceMgr, traceFileID, context);
  {
    auto parserTimer = ts.nest("VCD Header Parser");
    if (failed(parser.parseHeader()))
      return failure();
  }

  // TODO: Show some information about the VCD file if requested.

  // Resolve the module path provided by the user.
  const VCDScope *currentScope = nullptr;
  StringRef modulePathTail = modulePath;
  while (!modulePathTail.empty()) {
    StringRef moduleName;
    std::tie(moduleName, modulePathTail) = modulePathTail.split('.');
    const VCDScope *found = currentScope
                                ? currentScope->findScope(moduleName)
                                : parser.getHeader().findScope(moduleName);
    if (!found) {
      auto d = mlir::emitError(UnknownLoc::get(context), "unknown module `")
               << moduleName << "` not found in trace";
      return failure();
    }
    currentScope = found;
  }
  if (!currentScope) {
    mlir::emitError(UnknownLoc::get(context), "no module path");
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Subscopes:\n";
    for (auto *s : currentScope->scopes) {
      llvm::dbgs() << "  - " << s->name << "\n";
    }
  });

  // Look up the module in the circuit.
  hw::HWModuleOp moduleOp;
  module->walk([&](hw::HWModuleOp op) {
    if (op.getName() == moduleName)
      moduleOp = op;
  });
  if (!moduleOp) {
    mlir::emitError(UnknownLoc::get(context), "unknown module `")
        << moduleName << "` not found in circuit";
    return failure();
  }

  // Pair up the module ports with variables in the selected trace scope.
  llvm::DenseMap<StringRef, VCDVariable *> variablesByName;
  llvm::DenseMap<StringRef, SmallVector<VCDVariable *, 1>> variablesByAbbrev;
  for (auto *var : currentScope->variables) {
    variablesByName.insert({var->name, var});
    variablesByAbbrev[var->abbrev].push_back(var);
  }

  LLVM_DEBUG(llvm::dbgs() << "Pairing up module ports\n");
  auto portInfos = hw::getAllModulePortInfos(moduleOp);
  bool anyMissing = false;
  SmallVector<RootPort> rootPorts;
  rootPorts.reserve(portInfos.size());
  for (auto portInfo : portInfos) {
    auto *var = variablesByName.lookup(portInfo.name.getValue());
    if (!var) {
      if (portInfo.direction == hw::INPUT) {
        mlir::emitError(UnknownLoc::get(context), "unmatched input `")
            << portInfo.name.getValue() << "`: port missing in trace file\n";
        anyMissing = true;
        continue;
      } else {
        mlir::emitWarning(UnknownLoc::get(context), "unmatched output `")
            << portInfo.name.getValue()
            << "`: port missing in trace file; will remain unverified\n";
      }
    }
    rootPorts.push_back({portInfo.name.getValue(), var, portInfo.type,
                         portInfo.direction == hw::INPUT});
  }
  llvm::DenseMap<VCDVariable *, RootPort *> portByVariable;
  for (auto &port : rootPorts)
    if (port.var)
      portByVariable.insert({port.var, &port});
  // TODO: The following is an error. There should be a flag to deal with this.
  // if (anyMissing)
  //   return failure();

  // LLVM_DEBUG({
  //   llvm::dbgs() << "Variables:\n";
  //   for (auto *var : currentScope->variables) {
  //     llvm::dbgs() << "  - " << var->name << "\n";
  //   }
  // });

  auto &os = llvm::outs();
  os << "module circt_vcd_testbench;\n";
  os.indent(2) << "timeunit " << parser.getHeader().timescale
               << parser.getHeader().timescaleUnit << " / "
               << parser.getHeader().timescale
               << parser.getHeader().timescaleUnit << ";\n";

  // Emit the local wires to hook up to the ports of the module under test.
  for (auto &port : rootPorts) {
    auto width = port.type.cast<IntegerType>().getWidth();
    os.indent(2) << "wire ";
    if (width > 1)
      os << "[" << (width - 1) << ":0] ";
    os << "dut_" << port.name << ";\n";
  }
  os << "\n";

  // Instantiate the module under test.
  os.indent(2) << moduleName << " dut(";
  bool isFirst = true;
  for (auto &port : rootPorts) {
    if (!isFirst)
      os << ",";
    isFirst = false;
    os << "\n";
    os.indent(4) << "." << port.name << " (dut_" << port.name << ")";
  }
  os << "\n";
  os.indent(2) << ");\n\n";

  // Create a stimulus application and checking process.
  os.indent(2) << "initial begin\n";
  uint64_t currentTime = 0;
  SmallVector<std::pair<RootPort *, StringRef>> updates;

  auto parseChanges = [&](VCDVariable *var, StringRef value) {
    auto it = portByVariable.find(var);
    if (it == portByVariable.end())
      return;
    updates.push_back({it->second, value});
  };

  for (;;) {
    updates.clear();
    if (failed(parser.parseSignalChanges(parseChanges)))
      return failure();
    for (auto &pair : updates) {
      if (!pair.first->isInput)
        continue;
      os.indent(4) << "dut_" << pair.first->name << " = 'b" << pair.second
                   << ";\n";
    }
    for (auto &pair : updates) {
      if (pair.first->isInput)
        continue;
      os.indent(4) << "assert(dut_" << pair.first->name << " ==? 'b"
                   << pair.second << ");\n";
    }

    Optional<uint64_t> nextTime;
    if (failed(parser.parseNextTimestamp(nextTime)))
      return failure();
    if (!nextTime)
      break;
    if (updates.empty())
      continue;
    uint64_t delta = *nextTime - currentTime;
    os << "\n";
    os.indent(4) << "#" << delta << ";\n";
  }

  os.indent(2) << "end\n";
  os << "endmodule\n";

  return success();
}

/// The entry point for the `circt-vcd-testbench` tool. Configures and parses
/// the command line options, registers all dialects with a context, and calls
/// the `execute` function to do the actual work.
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Parse the command line options provided by the user.
  registerMLIRContextCLOptions();
  registerDefaultTimingManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "CIRCT trace-based testbench tool\n");

  // Create a context to work wtih.
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  mlir::MLIRContext context(registry);

  // Do the actual processing and use `exit` to avoid the slow teardown of the
  // context.
  exit(failed(execute(&context)));
}
