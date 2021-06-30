// RUN: circt-verify %s --verify-diagnostics

hw.module @Adder(%a: i8, %b: i8) -> (%z: i8) {
  %0 = comb.add %a, %b : i8
  hw.output %0 : i8
}

verif.test {
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %0 = hw.instance "adder0" @Adder (%c0_i8, %c0_i8) : (i8, i8) -> i8
  %1 = hw.instance "adder1" @Adder (%c1_i8, %c0_i8) : (i8, i8) -> i8
  %2 = hw.instance "adder2" @Adder (%c0_i8, %c1_i8) : (i8, i8) -> i8
  %3 = comb.icmp eq %0, %c0_i8 : i8
  %4 = comb.icmp eq %1, %c1_i8 : i8
  %5 = comb.icmp eq %2, %c1_i8 : i8
  verif.check %3 {note = "0+0 = 0"}
  verif.check %4 {note = "1+0 = 1"}
  verif.check %5 {note = "0+1 = 1"}
}

verif.test {
  %c12_i8 = hw.constant 12 : i8
  %c30_i8 = hw.constant 30 : i8
  %c42_i8 = hw.constant 42 : i8
  %0 = hw.instance "adder" @Adder (%c12_i8, %c30_i8) : (i8, i8) -> i8
  %1 = comb.icmp eq %0, %c42_i8 : i8
  verif.check %1 {note = "12+30 = 42"}
}

verif.test {
  %c255_i8 = hw.constant 255 : i8
  %c2_i8 = hw.constant 2 : i8
  %c1_i8 = hw.constant 1 : i8
  %0 = hw.instance "adder" @Adder (%c255_i8, %c2_i8) : (i8, i8) -> i8
  %1 = comb.icmp eq %0, %c1_i8 : i8
  verif.check %1 {note = "255+2 = 1 (overflow)"}
}
