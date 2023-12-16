// RUN: circt-translate --import-verilog %s | FileCheck %s

// CHECK-LABEL: moore.module @bpassign {
module bpassign();
  // CHECK: %bpassign_a = moore.variable : !moore.logic
  // CHECK: %bpassign_b = moore.variable : !moore.logic
  // CHECK: %0 = moore.mir.constant 2 : !moore.int
  // CHECK: %1 = moore.conversion %0 : (!moore.int) -> !moore.logic
  // CHECK: moore.mir.bpassign %bpassign_b, %1 : !moore.logic
  logic bpassign_a;
  logic bpassign_b = 2;

  initial begin
    // CHECK:    %2 = moore.mir.constant 1 : !moore.int
    // CHECK:    %3 = moore.conversion %2 : (!moore.int) -> !moore.logic
    // CHECK:    moore.mir.bpassign %bpassign_a, %3 : !moore.logic
    // CHECK:    moore.mir.bpassign %bpassign_b, %bpassign_a : !moore.logic
    bpassign_a = 1;
    bpassign_b = bpassign_a;
  end
endmodule

// CHECK-LABEL: moore.module @cassign {
module cassign();
  // CHECK:    %cassign_a = moore.variable : !moore.int
  // CHECK:    %0 = moore.mir.constant 2 : !moore.int
  // CHECK:    moore.mir.cassign %cassign_a, %0 : !moore.int
  // CHECK:  }
  int cassign_a;
  assign cassign_a = 2;
endmodule

// CHECK-LABEL: moore.module @passign {
module passign();
  // CHECK:    %passign_a = moore.variable : !moore.logic
  logic passign_a;
  initial begin
    // CHECK:      %0 = moore.mir.constant 2 : !moore.int
    // CHECK:      %1 = moore.conversion %0 : (!moore.int) -> !moore.logic
    // CHECK:      moore.mir.passign %passign_a, %1 : !moore.logic
    passign_a <= 2;
  end
endmodule


// CHECK-LABEL: moore.module @pcassign {
module pcassign();
  // CHECK:  %pcassign = moore.variable : !moore.int
  int pcassign;
  initial begin
    // CHECK:    %0 = moore.mir.constant 2 : !moore.int
    // CHECK:    moore.mir.pcassign %pcassign, %0 : !moore.int
    assign pcassign=2;
  end
endmodule
