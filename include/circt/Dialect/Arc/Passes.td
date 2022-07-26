//===- Passes.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_PASSES_TD
#define CIRCT_DIALECT_ARC_PASSES_TD

include "mlir/Pass/PassBase.td"

def StripSV : Pass<"arc-strip-sv", "mlir::ModuleOp"> {
  let summary = "Remove SV wire, reg, and assigns";
  let constructor = "createStripSVPass()";
  let dependentDialects = ["seq::SeqDialect", "arc::ArcDialect"];
}

def PushOutputLogic : Pass<"arc-push-output-logic", "mlir::ModuleOp"> {
  let summary = "Push logic through module outputs";
  let description = [{
    Makes arc breaking ops line up inside modules at the output op.
  }];
  let constructor = "createPushOutputLogicPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def PullInputLogic : Pass<"arc-pull-input-logic", "mlir::ModuleOp"> {
  let summary = "Pull logic through module inputs";
  let description = [{
    Makes arc breaking ops line up outside modules at their instance operands.
  }];
  let constructor = "createPullInputLogicPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def RegToArc : Pass<"reg-to-arc", "mlir::ModuleOp"> {
  let summary = "Lower registers to state transfer arcs";
  let constructor = "createRegToArcPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def InferMemories : Pass<"arc-infer-mems", "mlir::ModuleOp"> {
  let summary = "Convert `FIRRTL_Memory` instances to dedicated memory ops";
  let constructor = "createInferMemoriesPass()";
  let dependentDialects = ["arc::ArcDialect", "seq::SeqDialect"];
}

def ConvertToArcs : Pass<"convert-to-arcs", "mlir::ModuleOp"> {
  let summary = "Lower logic to state transfer arcs";
  let constructor = "createConvertToArcsPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def InlineModules : Pass<"arc-inline-modules", "mlir::ModuleOp"> {
  let summary = "Inline modules that are instantiated once";
  let constructor = "createInlineModulesPass()";
}

def InlineArcInputs : Pass<"arc-inline-inputs", "mlir::ModuleOp"> {
  let summary = "Inline constant inputs into arcs";
  let constructor = "createInlineArcInputsPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def Dedup : Pass<"arc-dedup", "mlir::ModuleOp"> {
  let summary = "Deduplicate identical arc definitions";
  let constructor = "createDedupPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def DumpArcGraph : Pass<"dump-arc-graph", "mlir::ModuleOp"> {
  let summary = "Dump a transfer arc DOT graph";
  let constructor = "createDumpArcGraphPass()";
  let options = [
    Option<"plotArcUses", "arc-uses", "bool", "false", "">,
    Option<"plotArcDefs", "arc-defs", "bool", "false", "">,
    Option<"plotStepDeps", "step-deps", "bool", "false", "">
  ];
  let dependentDialects = ["arc::ArcDialect"];
}

def PrintArcInfo : Pass<"print-arc-info", "mlir::ModuleOp"> {
  let summary = "Print information about arc definitions and uses";
  let constructor = "createPrintArcInfoPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def MakeLookupTables : Pass<"arc-lookup-tables", "mlir::ModuleOp"> {
  let summary = "Transform appropriate arc logic into lookup tables";
  let constructor = "createMakeLookupTablesPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def LowerState : Pass<"arc-lower-state", "mlir::ModuleOp"> {
  let summary = "Split state into read and write ops grouped by clock tree";
  let constructor = "createLowerStatePass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def ScheduleModels : Pass<"arc-schedule", "mlir::ModuleOp"> {
  let summary = "Bring ops in models into a schedulable order";
  let constructor = "createScheduleModelsPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def SplitLoops : Pass<"split-arc-loops", "mlir::ModuleOp"> {
  let summary = "Split arcs to break zero latency loops";
  let constructor = "createSplitLoopsPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def AllocateState : Pass<"arc-alloc-state", "arc::ModelOp"> {
  let summary = "Allocate and layout the global simulation state";
  let constructor = "createAllocateStatePass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def PrintStateInfo : Pass<"print-arc-state-info", "mlir::ModuleOp"> {
  let summary = "Print the state storage layout in JSON format";
  let constructor = "createPrintStateInfoPass()";
  let options = [
    Option<"stateFile", "state-file", "std::string", "",
      "Emit file with state description">
  ];
}

def LowerToLLVM : Pass<"lower-arc-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower state transfer arc representation to LLVM";
  let constructor = "createLowerToLLVMPass()";
  let dependentDialects = [
    "arc::ArcDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect",
    "mlir::func::FuncDialect"
  ];
  let options = [
    Option<"stateFile", "state-file", "std::string", "",
      "Emit file with state description">
  ];
}

#endif // CIRCT_DIALECT_ARC_PASSES_TD
