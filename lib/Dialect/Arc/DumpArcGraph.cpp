//===- DumpArcGraph.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#define DEBUG_TYPE "dump-arc-graph"

using namespace circt;
using namespace arc;
using namespace hw;
using llvm::SetVector;
using llvm::SmallDenseSet;
using llvm::SmallMapVector;
using llvm::SmallSetVector;

namespace {
struct DumpArcGraphPass : public DumpArcGraphBase<DumpArcGraphPass> {
  void runOnOperation() override;
  void plotDefinitionGraph();
  void plotCallGraph();
  void plotStepDependencies();
};
} // namespace

static void traceToPrimaryInputs(Value from, SmallVectorImpl<Value> &into) {
  SmallDenseSet<Value> visited;
  SmallVector<Value> worklist;
  visited.insert(from);
  worklist.push_back(from);
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    auto *op = value.getDefiningOp();
    if (!op || isArcBreakingOp(op)) {
      into.push_back(value);
      continue;
    }
    for (auto operand : op->getOperands())
      if (!visited.insert(operand).second)
        worklist.push_back(operand);
  }
}

void DumpArcGraphPass::runOnOperation() {
  if (plotArcDefs)
    plotDefinitionGraph();
  if (plotArcUses)
    plotCallGraph();
  if (plotStepDeps)
    plotStepDependencies();
}

void DumpArcGraphPass::plotDefinitionGraph() {
  SmallVector<Value> inputs;
  SetVector<DefineOp> deps;
  llvm::outs() << "digraph {\n";
  llvm::outs() << "layout=sfdp\n";
  llvm::outs() << "graph [ranksep=3, overlap=prism]\n";
  llvm::outs() << "edge [color=\"#00000040\"]\n";
  llvm::outs() << "node [shape=circle, style=filled, "
                  "colorscheme=gnbu9, color=\"#FF0000\"]\n";

  SmallVector<DefineOp> defs;
  DenseMap<StringAttr, DefineOp> defByName;
  DenseMap<StringAttr, DenseSet<StateOp>> defUses;
  unsigned maxNumUses = 0;
  getOperation().walk([&](Operation *op) {
    if (auto defOp = dyn_cast<DefineOp>(op)) {
      defs.push_back(defOp);
      defByName[defOp.sym_nameAttr()] = defOp;
      return;
    }
    if (auto stateOp = dyn_cast<StateOp>(op)) {
      auto &uses = defUses[stateOp.arcAttr().getAttr()];
      uses.insert(stateOp);
      if (uses.size() > maxNumUses)
        maxNumUses = uses.size();
      return;
    }
  });

  for (auto def : defs) {
    deps.clear();
    bool selfReferential = false;
    auto &uses = defUses[def.sym_nameAttr()];
    for (auto stateOp : uses) {
      for (auto operand : stateOp.operands()) {
        inputs.clear();
        traceToPrimaryInputs(operand, inputs);
        for (auto input : inputs) {
          if (auto depStateOp = input.getDefiningOp<StateOp>()) {
            auto depDef = defByName[depStateOp.arcAttr().getAttr()];
            if (depDef == def)
              selfReferential = true;
            else
              deps.insert(depDef);
          }
        }
      }
    }

    float colorScale = log2(uses.size()) / log2(maxNumUses) * 6 + 3;
    float sizeScale = log(def.bodyBlock().getOperations().size() + 1);

    llvm::outs() << "N" << (void *)def.getOperation() << " [";
    llvm::outs() << "fillcolor=" << (unsigned)colorScale << ", ";
    llvm::outs() << "label=" << uses.size() << ", ";
    llvm::outs() << "penwidth=" << (selfReferential ? "0" : "3") << ", ";
    llvm::outs() << "width=" << llvm::format("%0.4f", sizeScale * 0.2);
    llvm::outs() << "]\n";
    for (auto depDef : deps) {
      llvm::outs() << "N" << (void *)depDef.getOperation() << " -> N"
                   << (void *)def.getOperation() << "\n";
    }
  }

  llvm::outs() << "}\n";
}

void DumpArcGraphPass::plotCallGraph() {
  SmallVector<Value> inputs;
  SetVector<Operation *> opSet;
  llvm::outs() << "digraph {\n";
  llvm::outs() << "layout=sfdp\n";
  llvm::outs() << "graph [ranksep=3, overlap=prism]\n";
  llvm::outs() << "edge [color=\"#00000040\"]\n";
  llvm::outs() << "node [shape=circle, style=filled, width=1.0, "
                  "colorscheme=gnbu9, color=\"#FF0000\"]\n";
  getOperation().walk([&](Operation *op) {
    if (auto stateOp = dyn_cast<StateOp>(op)) {
      bool selfReferential = false;
      opSet.clear();
      for (auto operand : stateOp.operands()) {
        inputs.clear();
        traceToPrimaryInputs(operand, inputs);
        for (auto input : inputs) {
          if (auto depStateOp = input.getDefiningOp<StateOp>()) {
            if (depStateOp == stateOp)
              selfReferential = true;
            else
              opSet.insert(depStateOp);
          }
        }
      }
      auto uses = stateOp->getUsers();
      unsigned numUsers = std::distance(uses.begin(), uses.end());
      unsigned color = std::min<unsigned>(numUsers / 2, 6) + 3;
      llvm::outs() << "N" << (void *)op;
      llvm::outs() << " [";
      llvm::outs() << "fontcolor=\"" << (color < 6 ? "#000000" : "#FFFFFF")
                   << "\", ";
      llvm::outs() << "fillcolor=" << color << ", ";
      llvm::outs() << "penwidth=" << (selfReferential ? "0" : "3") << ", ";
      llvm::outs() << "label=\"" << numUsers << "\"";
      llvm::outs() << "]\n";
      for (auto *depArc : opSet) {
        llvm::outs() << "N" << (void *)depArc << " -> N" << (void *)op << "\n";
      }
    }
  });
  llvm::outs() << "}\n";
}

void DumpArcGraphPass::plotStepDependencies() {
  llvm::outs() << "digraph {\n";
  llvm::outs() << "graph [ranksep=3, overlap=prism]\n";
  llvm::outs() << "node [shape=circle, style=filled]\n";

  unsigned stateIndex = 0;
  unsigned arcIndex = 0;

  getOperation().walk([&](Operation *op) {
    if (auto stateOp = dyn_cast<StateOp>(op)) {
      StringRef nodePrefix = "N";
      if (stateOp.latency() > 0) {
        unsigned index = stateIndex++;
        llvm::outs() << "N" << (void *)op << " [label=\"S" << index
                     << "\", fillcolor=\"#FFCCCC\"]\n";
        llvm::outs() << "NQ" << (void *)op << " [label=\"S" << index
                     << "\", fillcolor=\"#FF4444\"]\n";
        nodePrefix = "NQ";
      } else {
        unsigned index = arcIndex++;
        llvm::outs() << "N" << (void *)op << " [label=\"" << index
                     << "\", fillcolor=\"#CCDDFF\"]\n";
      }
      for (auto operand : stateOp.getOperands()) {
        if (operand == stateOp.clock())
          continue;
        if (auto blockArg = operand.dyn_cast<BlockArgument>())
          llvm::outs() << "I" << blockArg.getAsOpaquePointer();
        else
          llvm::outs() << "N" << (void *)operand.getDefiningOp();
        llvm::outs() << " -> " << nodePrefix << (void *)op << "\n";
      }
    }
  });

  llvm::outs() << "}\n";
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createDumpArcGraphPass() {
  return std::make_unique<DumpArcGraphPass>();
}
} // namespace arc
} // namespace circt
