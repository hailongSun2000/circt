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

// from https://github.com/verilator/verilator/blob/master/src/V3Table.cpp
// 1MB is max table size (better be lots of instructs to be worth it!)
static constexpr int TABLE_MAX_BYTES = 1 * 1024 * 1024;
// 64MB is close to max memory of some systems (256MB or so), so don't get out of control
static constexpr int TABLE_TOTAL_BYTES = 64 * 1024 * 1024;
// Worth no more than 8 bytes of data to replace an instruction
static constexpr int TABLE_SPACE_TIME_MULT = 8;
// If < 32 instructions, not worth the effort
// static constexpr int TABLE_MIN_NODE_COUNT = 32;
static constexpr int TABLE_MIN_NODE_COUNT = 5;
// Assume an instruction is 4 bytes
static constexpr int TABLE_BYTES_PER_INST = 4;

namespace {
struct MakeLookupTablesPass : public MakeLookupTablesBase<MakeLookupTablesPass> {
  void runOnOperation() override;
  void runOnDefine();
  DefineOp defineOp;
  size_t totalBytes;
};
} // namespace

static inline uint32_t bits_mask(uint32_t nbits) {
    if (nbits == 32)
        return ~0;
    return (1 << nbits) - 1;
}

static inline uint32_t bits_get(uint32_t x, uint32_t lb, uint32_t ub) {
    return (x >> lb) & bits_mask(ub - lb + 1);
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
  LLVM_DEBUG(llvm::dbgs() << "inbits: " << inbits << ", outbits: " << outbits << ", num ops: " << opcount << "\n");

  if (inbits > 12 || outbits > 12) {
    LLVM_DEBUG(llvm::dbgs() << "no table created: table is too large" << "\n");
    return;
  }

  unsigned insize = 1 << inbits;
  const unsigned space = insize * outbits;
  const unsigned time = std::max<unsigned>(opcount * TABLE_BYTES_PER_INST, 1);

  // if (opcount < TABLE_MIN_NODE_COUNT) {
  //   LLVM_DEBUG(llvm::dbgs() << "no table created: too few nodes involved" << "\n");
  //   return;
  // }
  if (space > TABLE_MAX_BYTES) {
    LLVM_DEBUG(llvm::dbgs() << "no table created: table is too large" << "\n");
    return;
  }
  // if (space > time * TABLE_SPACE_TIME_MULT) {
  //   LLVM_DEBUG(llvm::dbgs() << "no table created: table has bad tradeoff" << "\n");
  //   return;
  // }
  if (totalBytes > TABLE_TOTAL_BYTES) {
    LLVM_DEBUG(llvm::dbgs() << "no table created: out of table memory" << "\n");
    return;
  }
  if (!outbits || !inbits) {
    LLVM_DEBUG(llvm::dbgs() << "no table created: no outputs or no inputs" << "\n");
    return;
  }

  totalBytes += space;

  LLVM_DEBUG(llvm::dbgs() << "creating table of size " << space << "\n");

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
    for (uint32_t i = 0; i < (1U << inbits); i++) {
      DenseMap<Value, Attribute> vals;
      unsigned bits = 0;
      for (auto arg : ins) {
        auto w = arg.getType().dyn_cast<IntegerType>().getWidth();
        vals[arg] = b.getIntegerAttr(arg.getType(), bits_get(i, bits, bits+w-1));
        bits += w;
      }
      for (auto *operation : origBody) {
        std::vector<Attribute> constants;
        for (auto operand : operation->getOperands()) {
          constants.push_back(vals[operand]);
        }
        ArrayRef<Attribute> attrs = constants;
        SmallVector<OpFoldResult, 8> resultAttrs;
        if (failed(operation->fold(attrs, resultAttrs)))
          signalPassFailure();
        unsigned j = 0;
        for (auto result : operation->getResults()) {
          if (Attribute foldAttr = resultAttrs[j].dyn_cast<Attribute>()) {
            vals[result] = foldAttr;
          } else if (Value foldVal = resultAttrs[j].dyn_cast<Value>()) {
            vals[result] = vals[foldVal];
          }
          j++;
        }
      }
      // evaluate op when the inputs are 'i'
      // put that value into outtbl
      // TODO
      outtbl.push_back(b.create<ConstantOp>(op.getType(), vals[op].dyn_cast<Attribute>().dyn_cast<IntegerAttr>().getInt()));
      // outtbl.push_back(b.create<ConstantOp>(op.getType(), 0));
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
