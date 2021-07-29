// RUN: circt-translate --export-firrtl --verify-diagnostics %s | FileCheck %s --strict-whitespace

// CHECK-LABEL: circuit Foo :
firrtl.circuit "Foo" {
  // CHECK-LABEL: module Foo :
  firrtl.module @Foo() {}

  // CHECK-LABEL: module PortsAndTypes :
  firrtl.module @PortsAndTypes(
    // CHECK-NEXT: input a00 : Clock
    // CHECK-NEXT: input a01 : Reset
    // CHECK-NEXT: input a02 : AsyncReset
    // CHECK-NEXT: input a03 : UInt
    // CHECK-NEXT: input a04 : SInt
    // CHECK-NEXT: input a05 : Analog
    // CHECK-NEXT: input a06 : UInt<42>
    // CHECK-NEXT: input a07 : SInt<42>
    // CHECK-NEXT: input a08 : Analog<42>
    // CHECK-NEXT: input a09 : { a : UInt, flip b : UInt }
    // CHECK-NEXT: input a10 : UInt[42]
    // CHECK-NEXT: output b0 : UInt
    in %a00: !firrtl.clock,
    in %a01: !firrtl.reset,
    in %a02: !firrtl.asyncreset,
    in %a03: !firrtl.uint,
    in %a04: !firrtl.sint,
    in %a05: !firrtl.analog,
    in %a06: !firrtl.uint<42>,
    in %a07: !firrtl.sint<42>,
    in %a08: !firrtl.analog<42>,
    in %a09: !firrtl.bundle<a: uint, b flip: uint>,
    in %a10: !firrtl.vector<uint, 42>,
    out %b0: !firrtl.uint
  ) {}

  // CHECK-LABEL: module Simple :
  // CHECK:         input someIn : UInt<1>
  // CHECK:         output someOut : UInt<1>
  firrtl.module @Simple(in %someIn: !firrtl.uint<1>, out %someOut: !firrtl.uint<1>) {
    firrtl.skip
  }

  // CHECK-LABEL: module Statements :
  firrtl.module @Statements(in %ui1: !firrtl.uint<1>, in %someClock: !firrtl.clock, in %someReset: !firrtl.reset, out %someOut: !firrtl.uint<1>) {
    // CHECK: when ui1 :
    // CHECK:   skip
    firrtl.when %ui1 {
      firrtl.skip
    }
    // CHECK: when ui1 :
    // CHECK:   skip
    // CHECK: else :
    // CHECK:   skip
    firrtl.when %ui1 {
      firrtl.skip
    } else {
      firrtl.skip
    }
    // CHECK: when ui1 :
    // CHECK:   skip
    // CHECK: else when ui1 :
    // CHECK:   skip
    firrtl.when %ui1 {
      firrtl.skip
    } else {
      firrtl.when %ui1 {
        firrtl.skip
      }
    }
    // CHECK: wire someWire : UInt<1>
    %someWire = firrtl.wire : !firrtl.uint<1>
    // CHECK: reg someReg : UInt<1>, someClock
    %someReg = firrtl.reg %someClock : (!firrtl.clock) -> !firrtl.uint<1>
    // CHECK: reg someReg2 : UInt<1>, someClock with :
    // CHECK:   reset => (someReset, ui1)
    %someReg2 = firrtl.regreset %someClock, %someReset, %ui1 : (!firrtl.clock, !firrtl.reset, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK: node someNode = ui1
    %someNode = firrtl.node %ui1 : !firrtl.uint<1>
    // CHECK: stop(someClock, ui1, 42) : foo
    firrtl.stop %someClock, %ui1, 42 {name = "foo"}
    // CHECK: skip
    firrtl.skip
    // CHECK: printf(someClock, ui1, "some\n magic\"stuff\"", ui1, someReset) : foo
    firrtl.printf %someClock, %ui1, "some\n magic\"stuff\"" {name = "foo"} (%ui1, %someReset) : !firrtl.uint<1>, !firrtl.reset
    // CHECK: assert(someClock, ui1, ui1, "msg") : foo
    // CHECK: assume(someClock, ui1, ui1, "msg") : foo
    // CHECK: cover(someClock, ui1, ui1, "msg") : foo
    firrtl.assert %someClock, %ui1, %ui1, "msg" {name = "foo"}
    firrtl.assume %someClock, %ui1, %ui1, "msg" {name = "foo"}
    firrtl.cover %someClock, %ui1, %ui1, "msg" {name = "foo"}
    // CHECK: someOut <= ui1
    // CHECK: someOut <- ui1
    firrtl.connect %someOut, %ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.partialconnect %someOut, %ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: inst someInst of Simple
    // CHECK: someInst.someIn <= ui1
    // CHECK: someOut <= someInst.someOut
    %someInst_someIn, %someInst_someOut = firrtl.instance @Simple {name = "someInst"} : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %someInst_someIn, %ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %someOut, %someInst_someOut : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: someOut is invalid
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %someOut, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: attach(an0, an1)
    %an0 = firrtl.wire : !firrtl.analog<1>
    %an1 = firrtl.wire : !firrtl.analog<1>
    firrtl.attach %an0, %an1 : !firrtl.analog<1>, !firrtl.analog<1>

    // TODO: Add tests for the following:
    //
    // - ConstantOp
    // - SpecialConstantOp
    // - SubfieldOp
    // - SubindexOp
    // - SubaccessOp
    //
    // - AddPrimOp
    // - SubPrimOp
    // - MulPrimOp
    // - DivPrimOp
    // - RemPrimOp
    // - AndPrimOp
    // - OrPrimOp
    // - XorPrimOp
    // - LEQPrimOp
    // - LTPrimOp
    // - GEQPrimOp
    // - GTPrimOp
    // - EQPrimOp
    // - NEQPrimOp
    // - CatPrimOp
    // - DShlPrimOp
    // - DShlwPrimOp
    // - DShrPrimOp
    //
    // - AsSIntPrimOp
    // - AsUIntPrimOp
    // - AsAsyncResetPrimOp
    // - AsClockPrimOp
    // - CvtPrimOp
    // - NegPrimOp
    // - NotPrimOp
    // - AndRPrimOp
    // - OrRPrimOp
    // - XorRPrimOp
    //
    // - BitsPrimOp
    // - HeadPrimOp
    // - TailPrimOp
    // - PadPrimOp
    // - MuxPrimOp
    // - ShlPrimOp
    // - ShrPrimOp
  }

  firrtl.extmodule @MyParameterizedExtModule(in %in: !firrtl.uint, out %out: !firrtl.uint<8>) attributes {defname = "name_thing", parameters = {DEFAULT = 0 : i64, DEPTH = 3.242000e+01 : f64, FORMAT = "xyz_timeout=%d\0A", WIDTH = 32 : i8}}
  // CHECK-LABEL: extmodule MyParameterizedExtModule :
  // CHECK-NEXT:    input in : UInt
  // CHECK-NEXT:    output out : UInt<8>
  // CHECK-NEXT:    defname = name_thing
  // CHECK-NEXT:    parameter DEFAULT = 0
  // CHECK-NEXT:    parameter DEPTH = 32.42
  // CHECK-NEXT:    parameter FORMAT = "xyz_timeout=%d\n"
  // CHECK-NEXT:    parameter WIDTH = 32
}
