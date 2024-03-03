//===- Passes.h - Moore pass entry points -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREPASSES_H
#define CIRCT_DIALECT_MOORE_MOOREPASSES_H

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include <memory>

namespace circt {
namespace moore {

class Declaration {
  // A map used to collect output ports, nets or variables and their values.
  // Only record the value produced by the last assignment op,
  // including the declaration assignment.
  DenseMap<Operation *, Value> assignmentChains;

  // Record chains of output port op and definition op (netOp or variable op)
  // with the last user of definition op (It faces the situation when has
  // multiple assignment to a same output port). It only records port which is
  // output direction.
  DenseMap<Operation *, std::pair<Operation *, Operation *>> outPortChains;

  // Provide a container which is used on mapping the defined name of net or
  // variable operation with the operation itself. It is useful when need to
  // find related definition of port operation via the same name.
  DenseMap<StringRef, Operation *> nameBindings;

public:
  void addValue(Operation *op);

  auto getValue(Operation *op) { return assignmentChains.lookup(op); }

  Operation *getOutputValue(Operation *op) {
    return outPortChains.lookup(op).second;
  }

  void buildPortBending(PortOp op);
};

extern Declaration decl;
std::unique_ptr<mlir::Pass> createMooreDeclarationsPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Moore/MoorePasses.h.inc"

} // namespace moore
} // namespace circt

#endif // CIRCT_DIALECT_MOORE_MOOREPASSES_H
