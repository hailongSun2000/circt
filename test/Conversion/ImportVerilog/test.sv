// RUN: circt-translate --import-verilog %s | FileCheck %s

// CHECK-LABEL: moore.module @BinaryOp
module BinaryOp();
// CHECK-NEXT: %a = moore.variable : !moore.int
// CHECK-NEXT: %0 = moore.mir.constant 12 : !moore.int
// CHECK-NEXT: moore.mir.bpassign %a, %0 : !moore.int
  int a = 12;
// CHECK-NEXT: %b = moore.variable : !moore.int
// CHECK-NEXT: %1 = moore.mir.constant 5 : !moore.int
// CHECK-NEXT: moore.mir.bpassign %b, %1 : !moore.int
  int b = 5;
// CHECK-NEXT: %log_and = moore.variable : !moore.int
// CHECK-NEXT: %log_equiv = moore.variable : !moore.int
// CHECK-NEXT: %log_impl = moore.variable : !moore.int
// CHECK-NEXT: %log_or = moore.variable : !moore.int
  int log_and, log_equiv, log_impl, log_or;
// CHECK-NEXT: %bin_bit_and = moore.variable : !moore.int
// CHECK-NEXT: %bin_bit_or = moore.variable : !moore.int
// CHECK-NEXT: %bin_bit_xor = moore.variable : !moore.int
// CHECK-NEXT: %bin_bit_xnor = moore.variable : !moore.int
  int bin_bit_and, bin_bit_or, bin_bit_xor, bin_bit_xnor;
// CHECK-NEXT: %case_eq = moore.variable : !moore.int
// CHECK-NEXT: %case_neq = moore.variable : !moore.int
// CHECK-NEXT: %log_eq = moore.variable : !moore.int
// CHECK-NEXT: %log_neq = moore.variable : !moore.int
  int case_eq, case_neq, log_eq, log_neq;
// CHECK-NEXT: %gte = moore.variable : !moore.int
// CHECK-NEXT: %gt = moore.variable : !moore.int
// CHECK-NEXT: %lte = moore.variable : !moore.int
// CHECK-NEXT: %lt = moore.variable : !moore.int
  int gte, gt, lte, lt;
// CHECK-NEXT: %arith_shl = moore.variable : !moore.int
// CHECK-NEXT: %arith_shr = moore.variable : !moore.int
// CHECK-NEXT: %log_shl = moore.variable : !moore.int
// CHECK-NEXT: %log_shr = moore.variable : !moore.int
  int arith_shl, arith_shr, log_shl, log_shr;
// CHECK-NEXT: moore.procedure(initial) 
  initial begin
// CHECK-NEXT:   %2 = moore.mir.logic and %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   moore.mir.bpassign %log_and, %2 : !moore.int
    log_and = a && b;
// CHECK-NEXT:   %3 = moore.mir.logic equiv %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   moore.mir.bpassign %log_equiv, %3 : !moore.int
    log_equiv = a <-> b;
// CHECK-NEXT:   %4 = moore.mir.logic impl %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   moore.mir.bpassign %log_impl, %4 : !moore.int
    log_impl = a ->b;
// CHECK-NEXT:   %5 = moore.mir.logic or %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   moore.mir.bpassign %log_or, %5 : !moore.int
    log_or = a || b;
// CHECK-NEXT:   %6 = moore.mir.binBitwise and %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   moore.mir.bpassign %bin_bit_and, %6 : !moore.int
    bin_bit_and = a & b;
// CHECK-NEXT:   %7 = moore.mir.binBitwise or %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   moore.mir.bpassign %bin_bit_or, %7 : !moore.int
    bin_bit_or = a | b;
// CHECK-NEXT:   %8 = moore.mir.binBitwise xor %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   moore.mir.bpassign %bin_bit_xor, %8 : !moore.int
    bin_bit_xor = a ^ b;
// CHECK-NEXT:   %9 = moore.mir.binBitwise xnor %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   moore.mir.bpassign %bin_bit_xnor, %9 : !moore.int
    bin_bit_xnor = a ~^ b;
// CHECK-NEXT:   %10 = moore.mir.eq case %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   %11 = moore.conversion %10 : (i1) -> !moore.int
// CHECK-NEXT:   moore.mir.bpassign %case_eq, %11 : !moore.int
    case_eq = a === b;
// CHECK-NEXT:   %12 = moore.mir.ne case %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   %13 = moore.conversion %12 : (i1) -> !moore.int
// CHECK-NEXT:   moore.mir.bpassign %case_neq, %13 : !moore.int
    case_neq = a !== b;
// CHECK-NEXT:   %14 = moore.mir.eq %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   %15 = moore.conversion %14 : (i1) -> !moore.int
// CHECK-NEXT:   moore.mir.bpassign %log_eq, %15 : !moore.int
    log_eq = a == b;
// CHECK-NEXT:   %16 = moore.mir.ne %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   %17 = moore.conversion %16 : (i1) -> !moore.int
// CHECK-NEXT:   moore.mir.bpassign %log_neq, %17 : !moore.int
    log_neq = a != b ;
// CHECK-NEXT:   %18 = moore.mir.icmp gte %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   %19 = moore.conversion %18 : (i1) -> !moore.int
// CHECK-NEXT:   moore.mir.bpassign %gte, %19 : !moore.int
    gte = a >= b;
// CHECK-NEXT:   %20 = moore.mir.icmp gt %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   %21 = moore.conversion %20 : (i1) -> !moore.int
// CHECK-NEXT:   moore.mir.bpassign %gt, %21 : !moore.int
    gt = a > b;
// CHECK-NEXT:   %22 = moore.mir.icmp lte %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   %23 = moore.conversion %22 : (i1) -> !moore.int
// CHECK-NEXT:   moore.mir.bpassign %lte, %23 : !moore.int
    lte = a <= b;
// CHECK-NEXT:   %24 = moore.mir.icmp lt %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   %25 = moore.conversion %24 : (i1) -> !moore.int
// CHECK-NEXT:   moore.mir.bpassign %lt, %25 : !moore.int
    lt = a < b;
// CHECK-NEXT:   %26 = moore.mir.shl arithmetic %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   moore.mir.bpassign %arith_shl, %26 : !moore.int
    arith_shl = a <<< b;
// CHECK-NEXT:   %27 = moore.mir.shr arithmetic %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   moore.mir.bpassign %arith_shr, %27 : !moore.int
    arith_shr = a >>> b;
// CHECK-NEXT:   %28 = moore.mir.shl %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   moore.mir.bpassign %log_shl, %28 : !moore.int
    log_shl = a << b;
// CHECK-NEXT:   %29 = moore.mir.shr %a, %b : !moore.int, !moore.int
// CHECK-NEXT:   moore.mir.bpassign %log_shr, %29 : !moore.int
    log_shr = a >> b;
  end
endmodule

// CHECK-LABEL: moore.module @Conditional
module Conditional();
// CHECK-NEXT: %a = moore.net  "wire" : !moore.logic
// CHECK-NEXT: %0 = moore.mir.constant 0 : !moore.int
// CHECK-NEXT: %1 = moore.conversion %0 : (!moore.int) -> !moore.logic
// CHECK-NEXT: moore.mir.cassign %a, %1 : !moore.logic
	wire a = 0;
// CHECK-NEXT: %b = moore.variable : !moore.reg
// CHECK-NEXT: %2 = moore.mir.constant 0 : !moore.int
// CHECK-NEXT: %3 = moore.conversion %2 : (!moore.int) -> !moore.reg
// CHECK-NEXT: moore.mir.bpassign %b, %3 : !moore.reg
	reg b = 0;
// CHECK-NEXT: moore.procedure(always) 
// CHECK-NEXT:   %4 = moore.mir.constant 0 : !moore.logic
// CHECK-NEXT:   %5 = moore.mir.ne %a, %4 : !moore.logic, !moore.logic
// CHECK-NEXT:   scf.if %5 
// CHECK-NEXT:     %6 = moore.mir.constant 1 : !moore.int
// CHECK-NEXT:     %7 = moore.conversion %6 : (!moore.int) -> !moore.reg
// CHECK-NEXT:     moore.mir.bpassign %b, %7 : !moore.reg
	always @* begin
		if(a) b = 1;
	end
endmodule

// CHECK-LABEL: moore.module @Port
// CHECK-NEXT: moore.port In "a"
// CHECK-NEXT: %a = moore.net  "wire" : !moore.packed<range<logic, 3:0>>
// CHECK-NEXT: moore.port Out "b"
// CHECK-NEXT: %b = moore.net  "wire" : !moore.packed<range<logic, 3:0>>
module Port(input [3:0] a, output [3:0] b);
endmodule

// CHECK-LABEL: moore.module @UnaryOp
// CHECK-NEXT: moore.port In "plus_a"
// CHECK-NEXT: %plus_a = moore.net  "wire" : !moore.packed<range<logic, 3:0>>
// CHECK-NEXT: moore.port In "minus_a"
// CHECK-NEXT: %minus_a = moore.net  "wire" : !moore.packed<range<logic, 3:0>>
// CHECK-NEXT: moore.port In "red_not_a"
// CHECK-NEXT: %red_not_a = moore.net  "wire" : !moore.packed<range<logic, 3:0>>
// CHECK-NEXT: moore.port In "red_xnor_a"
// CHECK-NEXT: %red_xnor_a = moore.net  "wire" : !moore.packed<range<logic, 3:0>>
// CHECK-NEXT: moore.port Out "plus_b"
// CHECK-NEXT: %plus_b = moore.net  "wire" : !moore.packed<range<logic, 3:0>>
// CHECK-NEXT: moore.port Out "minus_b"
// CHECK-NEXT: %minus_b = moore.net  "wire" : !moore.packed<range<logic, 3:0>>
// CHECK-NEXT: moore.port Out "red_not_b"
// CHECK-NEXT: %red_not_b = moore.net  "wire" : !moore.packed<range<logic, 3:0>>
// CHECK-NEXT: moore.port Out "red_xnor_b"
// CHECK-NEXT: %red_xnor_b = moore.net  "wire" : !moore.packed<range<logic, 3:0>>
module UnaryOp(input [3:0] plus_a, minus_a, red_not_a, red_xnor_a,
               output [3:0] plus_b, minus_b, red_not_b, red_xnor_b);
// CHECK-NEXT: %0 = moore.mir.unary plus %minus_a : !moore.packed<range<logic, 3:0>>
// CHECK-NEXT: moore.mir.cassign %plus_b, %0 : !moore.packed<range<logic, 3:0>>
  assign plus_b = +minus_a;
// CHECK-NEXT: %1 = moore.mir.unary minus %minus_a : !moore.packed<range<logic, 3:0>>
// CHECK-NEXT: moore.mir.cassign %minus_b, %1 : !moore.packed<range<logic, 3:0>>
  assign minus_b = -minus_a;
// CHECK-NEXT: %2 = moore.mir.unary not %red_not_a : !moore.packed<range<logic, 3:0>>
// CHECK-NEXT: moore.mir.cassign %red_not_b, %2 : !moore.packed<range<logic, 3:0>>
  assign red_not_b = !red_not_a;
// CHECK-NEXT: %3 = moore.mir.reduction xnor %red_xnor_a : !moore.packed<range<logic, 3:0>>
// CHECK-NEXT: moore.mir.cassign %red_xnor_b, %3 : !moore.packed<range<logic, 3:0>>
  assign red_xnor_b = ~^red_xnor_a;
endmodule