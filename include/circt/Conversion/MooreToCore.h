//===- MooreToCore.h - Moore to Core pass entry point -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the MooreToCore pass
// constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_MOORETOCORE_H
#define CIRCT_CONVERSION_MOORETOCORE_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {
/// Stores port interface into data structure to help convert moore module
/// structure to hw module structure.
struct MoorePortInfo {
  std::unique_ptr<hw::ModulePortInfo> hwPorts;

  // A mapping between the port name, port op and port type in moore module.
  DenseMap<StringAttr, std::pair<Operation *, Type>> nameMapping;

  DenseMap<moore::PortOp, Operation *> portBinding;

  // Constructor
  MoorePortInfo(moore::SVModuleOp moduleOp);
};

using MoorePortInfoMap = DenseMap<StringAttr, MoorePortInfo>;

/// Get the Moore structure operations to HW conversion patterns.
void populateMooreStructureConversionPatterns(TypeConverter &typeConverter,
                                              RewritePatternSet &patterns,
                                              MoorePortInfoMap &portInfoMap);

/// Get the Moore to HW/Comb/Seq conversion patterns.
void populateMooreToCoreConversionPatterns(TypeConverter &typeConverter,
                                           RewritePatternSet &patterns);

/// Create an Moore to HW/Comb/Seq conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> createConvertMooreToCorePass();

} // namespace circt

#endif // CIRCT_CONVERSION_MOORETOCORE_H
