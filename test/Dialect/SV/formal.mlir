// RUN: circt-opt %s | FileCheck %s
// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: hw.module @Properties(%arg0: i1, %arg1: i1) {
hw.module @Properties(%arg0: i1, %arg1: i1) {
  // CHECK-NEXT: [[pnot:%.+]] = sv.property.not %arg0 : i1
  %pnot = sv.property.not %arg0 : i1

  // CHECK-NEXT: [[pand:%.+]] = sv.property.and [[pnot]], %arg1
  // CHECK-NEXT: [[por:%.+]] = sv.property.or [[pnot]], [[pand]]
  %pand = sv.property.and %pnot, %arg1 : !sv.property, i1
  %por = sv.property.or %pnot, %pand : !sv.property, !sv.property

  // CHECK-NEXT: [[pimpl0:%.+]] = sv.property.implication %arg1, [[por]]
  // CHECK-NEXT: [[pimpl1:%.+]] = sv.property.implication nonoverlap %arg1, [[por]]
  %pimpl0 = sv.property.implication %arg1, %por : i1, !sv.property
  %pimpl1 = sv.property.implication nonoverlap %arg1, %por : i1, !sv.property
}
