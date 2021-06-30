//===- VerifOps.cpp - Implement the Verif operations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Verif ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Verif/Verif.h"

using namespace circt;
using namespace verif;

#define GET_OP_CLASSES
#include "circt/Dialect/Verif/Verif.cpp.inc"
