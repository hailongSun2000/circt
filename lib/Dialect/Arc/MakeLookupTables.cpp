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
  auto types = defineOp.getArgumentTypes();
  for (auto &t : types) {
    invars++;
    if (auto i = t.dyn_cast<IntegerType>()) {
      inbits += i.getWidth();
    }
  }
  // outputs
  auto outs = defineOp.getOps<arc::OutputOp>();
  for (auto op : llvm::make_early_inc_range(outs)) {
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

  if (inbits > 28 || outbits > 28) {
    LLVM_DEBUG(llvm::dbgs() << "no table created: table is too large" << "\n");
    return;
  }

  unsigned insize = 1 << inbits;
  const unsigned space = insize * outbits;
  const unsigned time = std::max<unsigned>(opcount * TABLE_BYTES_PER_INST, 1);

  if (opcount < TABLE_MIN_NODE_COUNT) {
    LLVM_DEBUG(llvm::dbgs() << "no table created: too few nodes involved" << "\n");
    return;
  }
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

  auto b = ImplicitLocOpBuilder::atBlockBegin(defineOp.getLoc(), &defineOp.bodyBlock());

  auto ins = defineOp.getArguments();
  Value inconcat = ins[0];
  for (size_t i = 1; i < ins.size(); i++) {
    inconcat = b.create<comb::ConcatOp>(inconcat, ins[i]);
  }

  for (auto out : llvm::make_early_inc_range(outs)) {
    for (auto op : out.getOperands()) {
      SmallVector<Value, 64> vals;
      for (uint32_t i = 0; i < (1U << inbits); i++) {
        // evaluate op when the inputs are 'i'
        // put that value into vals
        // TODO
        vals.push_back(b.create<ConstantOp>(op.getType(), 0));
      }
      auto arr = b.create<hw::ArrayCreateOp>(ArrayType::get(op.getType(), insize), vals);
      Value lookup = b.create<hw::ArrayGetOp>(arr, inconcat);
      op.replaceAllUsesWith(lookup);
    }
  }
}

namespace circt {
namespace arc {
std::unique_ptr<Pass> createMakeLookupTablesPass() {
  return std::make_unique<MakeLookupTablesPass>();
}
} // namespace arc
} // namespace circt
