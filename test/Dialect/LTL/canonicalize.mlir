// RUN: circt-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: @NoDelay
func.func @NoDelay(%arg0: !ltl.sequence) -> !ltl.sequence {
  %0 = ltl.delay %arg0, 0 : !ltl.sequence
  return %0 : !ltl.sequence
  // CHECK-NEXT: return %arg0 :
}

// CHECK-LABEL: @CompoundDelays
func.func @CompoundDelays(%arg0: i1) -> !ltl.sequence {
  %0 = ltl.delay %arg0, 1 : i1
  %1 = ltl.delay %0, 2 : !ltl.sequence
  %2 = ltl.delay %1, 3 : !ltl.sequence
  return %2 : !ltl.sequence
  // CHECK-NEXT: %0 = ltl.delay %arg0, 6 :
  // CHECK-NEXT: return %0 :
}
