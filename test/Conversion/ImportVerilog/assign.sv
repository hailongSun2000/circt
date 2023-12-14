// RUN: circt-translate --import-verilog %s | FileCheck %s

// CHECK-LABEL: moore.module @bpassign {
module bpassign();
  // CHECK-NEXT: %bpassign_a = moore.variable : !moore.logic
  // CHECK-NEXT: %bpassign_b = moore.variable : !moore.logic
  // CHECK-NEXT: %0 = moore.mir.constant 2 : !moore.int
  // CHECK-NEXT: %1 = moore.conversion %0 : (!moore.int) -> !moore.logic
  // CHECK-NEXT: moore.mir.bpassign %bpassign_b, %1 : !moore.logic
logic bpassign_a;
logic bpassign_b = 2;

// CHECK-LABEL: moore.procedure(initial) {
initial begin
  // CHECK-NEXT:    %2 = moore.mir.constant 1 : !moore.int
  // CHECK-NEXT:    %3 = moore.conversion %2 : (!moore.int) -> !moore.logic
  // CHECK-NEXT:    moore.mir.bpassign %bpassign_a, %3 : !moore.logic
  // CHECK-NEXT:    moore.mir.bpassign %bpassign_b, %bpassign_a : !moore.logic
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
	bpassign_a = 1;
	bpassign_b = bpassign_a;
end
endmodule

// CHECK-LABEL: moore.module @cassign {
module cassign();
  // CHECK-NEXT:    %cassign_a = moore.variable : !moore.int
  // CHECK-NEXT:    %0 = moore.mir.constant 2 : !moore.int
  // CHECK-NEXT:    moore.mir.cassign %cassign_a, %0 : !moore.int
  // CHECK-NEXT:  }
int cassign_a;
assign cassign_a = 2;
endmodule

// CHECK-LABEL: moore.module @passign {
module passign();
  // CHECK-NEXT:    %passign_a = moore.variable : !moore.logic
  // CHECK-LABEL:   moore.procedure(initial) {
  // CHECK-NEXT:      %0 = moore.mir.constant 2 : !moore.int
  // CHECK-NEXT:      %1 = moore.conversion %0 : (!moore.int) -> !moore.logic
  // CHECK-NEXT:      moore.mir.passign %passign_a, %1 : !moore.logic
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  logic passign_a;
initial begin
	passign_a <= 2;
end
endmodule


// CHECK-LABEL: moore.module @pcassign {
module pcassign();
  // CHECK-NEXT:  %pcassign = moore.variable : !moore.int
  // CHECK-LABEL: moore.procedure(initial) {
  // CHECK-NEXT:    %0 = moore.mir.constant 2 : !moore.int
  // CHECK-NEXT:    moore.mir.pcassign %pcassign, %0 : !moore.int
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:}
int pcassign;
initial begin
  assign pcassign=2;
end
endmodule