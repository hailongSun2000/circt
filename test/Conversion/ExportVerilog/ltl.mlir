// RUN: circt-opt %s --export-verilog --verify-diagnostics | FileCheck %s

// CHECK-LABEL: module Verification(
hw.module @Verification(%ready: i1, %start: i1, %go: i1, %done: i1) {
  // %startDelay = ltl.delay %start, 1 : i1
  // %doneDelay = ltl.delay %done, 1 : i1
  // %property = ltl.implication %startDelay, %doneDelay : !ltl.sequence, !ltl.sequence
  hw.output
}
