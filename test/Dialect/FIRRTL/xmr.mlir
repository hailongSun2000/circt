// RUN: circt-opt --lower-firrtl-to-hw %s | FileCheck %s

firrtl.circuit "A" {
  // Use "a.b.x" as rvalue.
  // Use "a.b.y" as lvalue.

  // CHECK: hw.module @B()
  firrtl.module @B(out %xmr_x: !firrtl.xmr<uint<42>>, out %xmr_y: !firrtl.xmr<uint<17>>) {
    // CHECK: %x = sv.wire
    // CHECK: %y = sv.wire
    %x = firrtl.wire : !firrtl.uint<42>
    %y = firrtl.wire : !firrtl.uint<17>
    firrtl.xmr.source %xmr_x, %x : !firrtl.uint<42>
    firrtl.xmr.source %xmr_y, %y : !firrtl.uint<17>
  }
  // CHECK: hw.module @A()
  firrtl.module @A(out %xmr_x: !firrtl.xmr<uint<42>>, out %xmr_y: !firrtl.xmr<uint<17>>) {
    // CHECK: hw.instance "b" @B()
    %0, %1 = firrtl.instance @B {name = "b"} : !firrtl.xmr<uint<42>>, !firrtl.xmr<uint<17>>
    firrtl.xmr.attach %xmr_x, %0 : !firrtl.xmr<uint<42>>
    firrtl.xmr.attach %xmr_y, %1 : !firrtl.xmr<uint<17>>
  }
  // CHECK: hw.module @C() -> (%out: i42)
  firrtl.module @C(in %in: !firrtl.uint<17>, out %out: !firrtl.uint<42>) {
    // CHECK: hw.instance "a" @A()
    // CHECK: [[T1:%.+]] = sv.verbatim.expr "a.b.x"
    // CHECK: hw.output [[T1]]
    %0, %1 = firrtl.instance @A {name = "a"} : !firrtl.xmr<uint<42>>, !firrtl.xmr<uint<17>>
    %2 = firrtl.xmr.deref %0 : !firrtl.uint<42>
    %3 = firrtl.xmr.deref %1 : !firrtl.uint<17>
    firrtl.connect %out, %2 : !firrtl.uint<42>, !firrtl.uint<42>
    firrtl.connect %3, %in : !firrtl.uint<17>, !firrtl.uint<17>
  }
}
