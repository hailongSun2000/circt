// RUN: circt-verify %s --verify-diagnostics --split-input-file

hw.module @Adder(%a: i8, %b: i8) -> (%z: i8) {
  %c1_i8 = hw.constant 1 : i8
  %0 = comb.add %a, %b, %c1_i8 : i8
  hw.output %0 : i8
}

verif.test {
  %c1_i8 = hw.constant 1 : i8
  %c3_i8 = hw.constant 2 : i8
  %0 = hw.instance "adder" @Adder (%c1_i8, %c1_i8) : (i8, i8) -> i8
  // expected-note @+2 {{comparison adder.z == 2 failed}}
  // expected-note @+1 {{where adder.z = 3}}
  %1 = comb.icmp eq %0, %c3_i8 : i8
  // expected-error @+1 {{check failed: "1+1 should be 2"}}
  verif.check %1 {note = "1+1 should be 2"}
}
