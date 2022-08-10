//===- MakeLookupTables.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "arc-lookup-tables"

using namespace circt;
using namespace arc;
using namespace hw;

namespace {

static constexpr int tableMinOpCount = 5;
static constexpr int tableMaxBytes = 32;
static constexpr int tableTotalBytes = 1024 * 16;

struct MakeLookupTablesPass : public MakeLookupTablesBase<MakeLookupTablesPass> {
  void runOnOperation() override;
  void runOnDefine();
  DefineOp defineOp;
  size_t totalBytes = 0;
};
} // namespace

static inline uint32_t bitsMask(uint32_t nbits) {
    if (nbits == 32)
        return ~0;
    return (1 << nbits) - 1;
}

static inline uint32_t bitsGet(uint32_t x, uint32_t lb, uint32_t ub) {
    return (x >> lb) & bitsMask(ub - lb + 1);
}

void MakeLookupTablesPass::runOnOperation() {
  auto module = getOperation();
  for (auto op : llvm::make_early_inc_range(module.getOps<DefineOp>())) {
    defineOp = op;
    runOnDefine();
  }
}

void MakeLookupTablesPass::runOnDefine() {
  LLVM_DEBUG(llvm::dbgs() << "Making lookup tables in `" << defineOp.getName()
                          << "`\n");

  unsigned inbits = 0;
  unsigned outbits = 0;
  unsigned opcount = defineOp.bodyBlock().getOperations().size();
  unsigned invars = 0;
  unsigned outvars = 0;

  // inputs
  bool noninteger = false;
  auto types = defineOp.getArgumentTypes();
  for (auto &t : types) {
    invars++;
    if (auto i = t.dyn_cast<IntegerType>()) {
      inbits += i.getWidth();
    } else {
      noninteger = true;
    }
  }
  if (noninteger) {
    // only make lookup tables if all inputs are integers
    return;
  }
  // outputs
  arc::OutputOp outputOp;
  for (auto op : llvm::make_early_inc_range(defineOp.getOps<arc::OutputOp>())) {
    outputOp = op;
    opcount--; // don't count output ops
    auto types = op.getOperandTypes();
    for (auto t : types) {
      outvars++;
      if (auto i = t.dyn_cast<IntegerType>()) {
        outbits += i.getWidth();
      }
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "lookup table analysis: inbits: " << inbits << ", outbits: " << outbits << ", num ops: " << opcount << "\n");

  if (inbits > 26 || outbits > 26) {
    LLVM_DEBUG(llvm::dbgs() << "no table created: table is too large" << "\n");
    return;
  }

  unsigned insize = 1U << inbits;
  const unsigned space = insize * outbits / 8;

  if (opcount < tableMinOpCount) {
    LLVM_DEBUG(llvm::dbgs() << "no table created: too few nodes involved" << "\n");
    return;
  }
  if (space > tableMaxBytes) {
    LLVM_DEBUG(llvm::dbgs() << "no table created: table is too large" << "\n");
    return;
  }
  if (totalBytes > tableTotalBytes) {
    LLVM_DEBUG(llvm::dbgs() << "no table created: out of table memory" << "\n");
    return;
  }
  if (!outbits || !inbits) {
    LLVM_DEBUG(llvm::dbgs() << "no table created: no outputs or no inputs" << "\n");
    return;
  }

  totalBytes += space;

  LLVM_DEBUG(llvm::dbgs() << "creating table of size " << space << ", total bytes: " << totalBytes << "\n");

  SmallVector<Operation*, 64> origBody;
  for (auto &op : defineOp.bodyBlock()) {
    if (isa<arc::OutputOp>(op)) {
      continue;
    }
    origBody.push_back(&op);
  }

  auto b = ImplicitLocOpBuilder::atBlockBegin(defineOp.getLoc(), &defineOp.bodyBlock());

  auto ins = defineOp.getArguments();
  Value inconcat = ins[0];
  for (size_t i = 1; i < ins.size(); i++) {
    inconcat = b.create<comb::ConcatOp>(ins[i], inconcat);
  }

  SmallVector<Value, 64> lookups;
  for (auto op : outputOp.getOperands()) {
    SmallVector<Value, 64> outtbl;
    for (int i = (1U << inbits) - 1; i >= 0; i--) {
      DenseMap<Value, Attribute> vals;
      unsigned bits = 0;
      for (auto arg : ins) {
        auto w = arg.getType().dyn_cast<IntegerType>().getWidth();
        vals[arg] = b.getIntegerAttr(arg.getType(), bitsGet(i, bits, bits+w-1));
        bits += w;
      }
      for (auto *operation : origBody) {
        std::vector<Attribute> constants;
        for (auto operand : operation->getOperands()) {
          constants.push_back(vals[operand]);
        }
        ArrayRef<Attribute> attrs = constants;
        SmallVector<OpFoldResult, 8> results;
        if (failed(operation->fold(attrs, results))) {
          LLVM_DEBUG(llvm::dbgs() << "no table created: computation failed" << "\n");
          return;
        }
        unsigned j = 0;
        for (auto result : operation->getResults()) {
          if (Attribute foldAttr = results[j].dyn_cast<Attribute>()) {
            vals[result] = foldAttr;
          } else if (Value foldVal = results[j].dyn_cast<Value>()) {
            vals[result] = vals[foldVal];
          }
          j++;
        }
      }
      outtbl.push_back(b.create<ConstantOp>(op.getType(), vals[op].dyn_cast<Attribute>().dyn_cast<IntegerAttr>().getInt()));
    }
    auto arr = b.create<hw::ArrayCreateOp>(ArrayType::get(op.getType(), insize), outtbl);
    lookups.push_back(b.create<hw::ArrayGetOp>(arr, inconcat));
  }

  unsigned i = 0;
  for (auto op : outputOp.getOperands()) {
    op.replaceUsesWithIf(lookups[i], [&](OpOperand &use) {
        return use.getOwner() == outputOp;
    });
    i++;
  }
  for (auto *op : origBody) {
    op->dropAllUses();
    op->erase();
  }
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createMakeLookupTablesPass() {
  return std::make_unique<MakeLookupTablesPass>();
}
} // namespace arc
} // namespace circt
