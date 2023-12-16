// RUN: circt-translate --import-verilog %s | FileCheck %s

// CHECK-LABEL: moore.module @arith {
module arith();
  // CHECK:  %a = moore.variable : !moore.int
  // CHECK:  %0 = moore.mir.constant 1 : !moore.int
  // CHECK:  moore.mir.bpassign %a, %0 : !moore.int
  // CHECK:  %b = moore.variable : !moore.int
  // CHECK:  %1 = moore.mir.constant 2 : !moore.int
  // CHECK:  moore.mir.bpassign %b, %1 : !moore.int
  int a = 1;
  int b = 2;

  initial begin
    // CHECK:      %2 = moore.mir.constant 1 : !moore.int
    // CHECK:      %3 = moore.mir.constant 2 : !moore.int
    // CHECK:      %4 = moore.mir.add %2, %3 : !moore.int
    // CHECK:      moore.mir.bpassign %a, %4 : !moore.int
    // CHECK:      %5 = moore.mir.add %a, %b : !moore.int
    // CHECK:      moore.mir.bpassign %a, %5 : !moore.int
    a = 1 + 2;
    a = a + b;
    // CHECK:      %6 = moore.mir.constant 1 : !moore.int
    // CHECK:      %7 = moore.mir.constant 2 : !moore.int
    // CHECK:      %8 = moore.mir.mul %6, %7 : !moore.int
    // CHECK:      moore.mir.bpassign %a, %8 : !moore.int
    // CHECK:      %9 = moore.mir.mul %a, %b : !moore.int
    // CHECK:      moore.mir.bpassign %a, %9 : !moore.int
    a=1*2;
    a=a*b;
    // CHECK:      %10 = moore.mir.constant 1 : !moore.int
    // CHECK:      %11 = moore.mir.add %10, %a : !moore.int
    // CHECK:      moore.mir.bpassign %a, %11 : !moore.int
    // CHECK:      %12 = moore.mir.constant 1 : !moore.int
    // CHECK:      %13 = moore.mir.mul %12, %a : !moore.int
    // CHECK:      moore.mir.bpassign %a, %13 : !moore.int
    a=1+a;
    a=1*a;
    // CHECK:      %14 = moore.mir.constant 1 : !moore.int
    // CHECK:      %15 = moore.mir.constant 2 : !moore.int
    // CHECK:      %16 = moore.mir.add %14, %15 : !moore.int
    // CHECK:      %17 = moore.mir.constant 3 : !moore.int
    // CHECK:      %18 = moore.mir.add %16, %17 : !moore.int
    // CHECK:      moore.mir.bpassign %a, %18 : !moore.int
    a=1+2+3;
    // CHECK:      %19 = moore.mir.constant 1 : !moore.int
    // CHECK:      %20 = moore.mir.add %19, %a : !moore.int
    // CHECK:      %21 = moore.mir.constant 1 : !moore.int
    // CHECK:      %22 = moore.mir.add %20, %21 : !moore.int
    // CHECK:      moore.mir.bpassign %a, %22 : !moore.int
    a=1+a+1;
    // CHECK:      %23 = moore.mir.add %a, %a : !moore.int
    // CHECK:      %24 = moore.mir.add %23, %a : !moore.int
    // CHECK:      moore.mir.bpassign %a, %24 : !moore.int
    a=a+a+a;
    // CHECK:      %25 = moore.mir.constant 1 : !moore.int
    // CHECK:      %26 = moore.mir.constant 2 : !moore.int
    // CHECK:      %27 = moore.mir.mul %25, %26 : !moore.int
    // CHECK:      %28 = moore.mir.constant 3 : !moore.int
    // CHECK:      %29 = moore.mir.mul %27, %28 : !moore.int
    // CHECK:      moore.mir.bpassign %a, %29 : !moore.int
    a=1*2*3;
    // CHECK:      %30 = moore.mir.constant 1 : !moore.int
    // CHECK:      %31 = moore.mir.mul %30, %a : !moore.int
    // CHECK:      %32 = moore.mir.constant 1 : !moore.int
    // CHECK:      %33 = moore.mir.mul %31, %32 : !moore.int
    // CHECK:      moore.mir.bpassign %a, %33 : !moore.int
    a=1*a*1;
    // CHECK:      %34 = moore.mir.mul %a, %a : !moore.int
    // CHECK:      %35 = moore.mir.mul %34, %a : !moore.int
    // CHECK:      moore.mir.bpassign %a, %35 : !moore.int
    a=a*a*a;
  end
endmodule
