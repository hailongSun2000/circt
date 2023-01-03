// RUN: circt-opt %s --verify-diagnostics | circt-opt --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @BasicOps
func.func @BasicOps(%arg0: i1) {
  // CHECK: ltl.delay %arg0, 3 : i1
  %0 = ltl.delay %arg0, 3 : i1

  // CHECK: ltl.implication %arg0, %0 : i1, !ltl.sequence
  %1 = ltl.implication %arg0, %0 : i1, !ltl.sequence

  return
}
