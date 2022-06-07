//===- PrintArcInfo.cpp ---------------------------------------------------===//
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
struct PrintArcInfoPass : public PrintArcInfoBase<PrintArcInfoPass> {
  void runOnOperation() override;
};
} // namespace

void PrintArcInfoPass::runOnOperation() {
  SmallVector<DefineOp, 0> arcs;
  DenseMap<StringAttr, DefineOp> arcsByName;
  DenseMap<StringAttr, SmallVector<StateOp>> usesByArc;

  getOperation().walk([&](Operation *op) {
    if (auto defineOp = dyn_cast<DefineOp>(op)) {
      arcs.push_back(defineOp);
      arcsByName.insert({defineOp.sym_nameAttr(), defineOp});
      return WalkResult::skip();
    }
    if (auto arcOp = dyn_cast<StateOp>(op)) {
      usesByArc[arcOp.arcAttr().getAttr()].push_back(arcOp);
    }
    return WalkResult::advance();
  });

  llvm::outs() << "Unique Arcs: " << arcs.size() << "\n";

  llvm::outs() << "Reuse Histogram:\n";
  SmallMapVector<unsigned, unsigned, 8> histogram;
  for (auto [arcName, uses] : usesByArc)
    ++histogram[uses.size()];
  SmallVector<std::pair<unsigned, unsigned>> sortedHistogram;
  sortedHistogram.assign(histogram.begin(), histogram.end());
  llvm::sort(sortedHistogram);
  for (auto [numUses, numArcs] : llvm::reverse(sortedHistogram))
    llvm::outs() << "  " << numUses << " reuses: " << numArcs << " arcs\n";

  llvm::outs() << "Complexity Histogram:\n";
  histogram.clear();
  unsigned numOpsIfFlattened = 0;
  unsigned numOpsIfReused = 0;
  for (auto arc : arcs) {
    auto numOps = arc.bodyBlock().getOperations().size();
    numOpsIfFlattened += numOps;
    numOpsIfReused += numOps / usesByArc[arc.sym_nameAttr()].size();
    ++histogram[numOps];
  }
  sortedHistogram.assign(histogram.begin(), histogram.end());
  llvm::sort(sortedHistogram);
  for (auto [numOps, numArcs] : llvm::reverse(sortedHistogram))
    llvm::outs() << "  " << numOps << " ops in " << numArcs << " arc defs\n";
  llvm::outs() << "Reuse Benefits:\n";
  llvm::outs() << "  Ops without reuse: " << numOpsIfFlattened << "\n";
  llvm::outs() << "  Ops with reuse:    " << numOpsIfReused << "\n";
  llvm::outs() << "  Gain:              "
               << llvm::format("%.4fx",
                               (float(numOpsIfFlattened) / numOpsIfReused))
               << "\n";
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createPrintArcInfoPass() {
  return std::make_unique<PrintArcInfoPass>();
}
} // namespace arc
} // namespace circt
