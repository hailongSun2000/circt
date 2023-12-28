// RUN: circt-translate --import-verilog %s | FileCheck %s


// CHECK-LABEL: moore.module @Variables
module Variables();
  // CHECK: %var1 = moore.variable : !moore.int
  // CHECK: %var2 = moore.variable %var1 : !moore.int
  int var1;
  int var2 = var1;
endmodule

// CHECK-LABEL: moore.module @Procedures
module Procedures();
  // CHECK: moore.procedure initial {
  initial;
  // CHECK: moore.procedure final {
  final begin end;
  // CHECK: moore.procedure always {
  always begin end;
  // CHECK: moore.procedure always_comb {
  always_comb begin end;
  // CHECK: moore.procedure always_latch {
  always_latch begin end;
  // CHECK: moore.procedure always_ff {
  always_ff @* begin end;
endmodule


// CHECK-LABEL: moore.module @Expressions {
module Expressions();
  // CHECK: %a = moore.variable : !moore.int
  // CHECK: %b = moore.variable : !moore.int
  // CHECK: %c = moore.variable : !moore.int
  // CHECK: %ua = moore.net "wire" : !moore.logic
  // CHECK: %ub = moore.net "wire" : !moore.logic
  int a, b, c;
  wire ua,ub;

  initial begin
    // CHECK: moore.mir.unary plus %a : !moore.int
    c = +a;
    // CHECK: moore.mir.unary minus %a : !moore.int
    c = -a;
    // CHECK: moore.mir.unary not %a : !moore.int
    c = !a;

    // CHECK: moore.mir.reduction xnor %a : !moore.int
    c = ~^a;

    // CHECK: moore.mir.add %a, %b : !moore.int
    c = a + b;
    // CHECK: moore.mir.mul %a, %b : !moore.int
    c = a * b;
    // CHECK: moore.mir.sub %a, %b : !moore.int
    c = a - b;
    // CHECK: moore.mir.pow %a, %b : !moore.int
    c = a ** b;
    // CHECK: moore.mir.divs %a, %b : !moore.int
    c = a / b;
    // CHECK: moore.mir.mods %a, %b : !moore.int
    c = a % b;
    // CHECK: moore.mir.divu %ua, %ub : !moore.logic
    c = ua / ub;
    // CHECK: moore.mir.modu %ua, %ub : !moore.logic
    c = ua % ub;

    // CHECK: moore.mir.logic and %a, %b : !moore.int, !moore.int
    c = a && b;
    // CHECK: moore.mir.logic equiv %a, %b : !moore.int, !moore.int
    c = a <-> b;
    // CHECK: moore.mir.logic impl %a, %b : !moore.int, !moore.int
    c = a -> b;
    // CHECK: moore.mir.logic or %a, %b : !moore.int, !moore.int
    c = a || b;

    // CHECK: moore.mir.bin_bitwise and %a, %b : !moore.int, !moore.int
    c = a & b;
    // CHECK: moore.mir.bin_bitwise or %a, %b : !moore.int, !moore.int
    c = a | b;
    // CHECK: moore.mir.bin_bitwise xor %a, %b : !moore.int, !moore.int
    c = a ^ b;
    // CHECK: moore.mir.bin_bitwise xnor %a, %b : !moore.int, !moore.int
    c = a ~^ b;

    // CHECK: moore.mir.eq case %a, %b : !moore.int, !moore.int
    c = a === b;
    // CHECK: moore.mir.ne case %a, %b : !moore.int, !moore.int
    c = a !== b;
    // CHECK: moore.mir.eq %a, %b : !moore.int, !moore.int
    c = a == b;
    // CHECK: moore.mir.ne %a, %b : !moore.int, !moore.int
    c = a != b ;
    // CHECK: moore.mir.icmp gte %a, %b : !moore.int, !moore.int
    c = a >= b;
    // CHECK: moore.mir.icmp gt %a, %b : !moore.int, !moore.int
    c = a > b;
    // CHECK: moore.mir.icmp lte %a, %b : !moore.int, !moore.int
    c = a <= b;
    // CHECK: moore.mir.icmp lt %a, %b : !moore.int, !moore.int
    c = a < b;

    // CHECK: moore.mir.shl %a, %b : !moore.int, !moore.int
    c = a << b;
    // CHECK: moore.mir.shr %a, %b : !moore.int, !moore.int
    c = a >> b;
    // CHECK: moore.mir.shl arithmetic %a, %b : !moore.int, !moore.int
    c = a <<< b;
    // CHECK: moore.mir.shr arithmetic %a, %b : !moore.int, !moore.int
    c = a >>> b;
    // CHECK: moore.mir.add %a, %b : !moore.int
    a += b;
    // CHECK: moore.mir.sub %a, %b : !moore.int
    a -= b;
    // CHECK: moore.mir.mul %a, %b : !moore.int
    a *= b;
    // CHECK: moore.mir.divs %a, %b : !moore.int
    a /= b;
    // CHECK: moore.mir.mods %a, %b : !moore.int
    a %= b;
    // CHECK: moore.mir.binBitwise and %a, %b : !moore.int, !moore.int
    a &= b;
    // CHECK: moore.mir.binBitwise or %a, %b : !moore.int, !moore.int
    a |= b;
    // CHECK: moore.mir.binBitwise xor %a, %b : !moore.int, !moore.int
    a ^= b;
    // CHECK: moore.mir.shl %a, %b : !moore.int, !moore.int
    a <<= b;
    // CHECK: moore.mir.shl arithmetic %a, %b : !moore.int, !moore.int
    a <<<= b;
    // CHECK: moore.mir.shr %a, %b : !moore.int, !moore.int
    a >>= b;
    // CHECK: moore.mir.shr arithmetic %a, %b : !moore.int, !moore.int
    a >>>= b;
    // CHECK: moore.mir.concat %a, %b : (!moore.int, !moore.int) -> !moore.packed<range<bit, 63:0>>
    c = {a,b};
  end
endmodule


// CHECK-LABEL: moore.module @Assignments {
module Assignments();
  // CHECK: %a = moore.variable : !moore.int
  // CHECK: %b = moore.variable : !moore.int
  int a, b;

  initial begin
    // CHECK: moore.mir.bpassign %a, %b : !moore.int
    a = b;
    // CHECK: moore.mir.passign %a, %b : !moore.int
    a <= b;
    // CHECK: moore.mir.pcassign %a, %b : !moore.int
    assign a = b;
  end
endmodule


// CHECK-LABEL: moore.module @Statements {
module Statements();
  // CHECK: %a = moore.variable : !moore.int
  // CHECK: %b = moore.variable : !moore.int
  int a, b;

  initial begin
    // CHECK: [[ZERO:%.+]] = moore.mir.constant 0 : !moore.int
    // CHECK: [[COND:%.+]] = moore.mir.ne %a, [[ZERO]] : !moore.int, !moore.int
    // CHECK: scf.if [[COND]]
    if (a)
      ;
  end
endmodule

// CHECK-LABEL: moore.module @Array {
module Array();
  // CHECK: %a = moore.variable : !moore.packed<range<logic, 3:0>>
  // CHECK: %b = moore.variable : !moore.packed<range<logic, 1:0>>
  // CHECK: %c = moore.variable : !moore.logic
  logic [3:0] a;
  logic [1:0] b;
  logic c;

  initial begin
    // CHECK: moore.mir.replicate %0, %1 : (!moore.int, !moore.packed<range<logic, 1:0>>) -> !moore.packed<range<logic, 3:0>>
    a = {2{b}};
    // CHECK: moore.mir.part_select simple %a, %3, %4 : (!moore.packed<range<logic, 3:0>>, !moore.int, !moore.int) -> !moore.packed<range<logic, 1:0>>
    b = a[1:0];
    // CHECK: moore.mir.part_select index_down %a, %6, %7 : (!moore.packed<range<logic, 3:0>>, !moore.int, !moore.int) -> !moore.packed<range<logic, 3:2>>
    b = a[3-:2];
    // CHECK: moore.mir.bit_select %a, %10 : (!moore.packed<range<logic, 3:0>>, !moore.int) -> !moore.logic
    c = a[1];
  end
endmodule
