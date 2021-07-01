// RUN: circt-verify %s --verify-diagnostics

hw.module @Adder(%a: i1, %b: i1, %cin: i1) -> (%sum: i1, %cout: i1) {
  %0 = comb.xor %a, %b : i1
  %1 = comb.xor %0, %cin : i1
  %2 = comb.and %a, %b : i1
  %3 = comb.and %cin, %0 : i1
  %4 = comb.or %2, %3 : i1
  hw.output %1, %4 : i1, i1
}

verif.test {
  // Stimuli and expected response
  %a = hw.constant 0 : i1
  %b = hw.constant 0 : i1
  %cin = hw.constant 0 : i1
  %total_exp = hw.constant 0 : i2

  // Instantiate DUT and check result
  %sum, %cout = hw.instance "dut" @Adder (%a, %b, %cin) : (i1, i1, i1) -> (i1, i1)
  %total_act = comb.concat %cout, %sum : (i1, i1) -> i2
  %0 = comb.icmp eq %total_act, %total_exp : i2
  verif.check %0 {note = "addition failed"}
}

verif.test {
  // Stimuli
  %a = hw.constant 1 : i1
  %b = hw.constant 1 : i1
  %cin = hw.constant 1 : i1

  // Compute expected response
  %false = hw.constant 0 : i1
  %a_ext = comb.concat %false, %a : (i1, i1) -> i2
  %b_ext = comb.concat %false, %b : (i1, i1) -> i2
  %cin_ext = comb.concat %false, %cin : (i1, i1) -> i2
  %total_exp = comb.add %a_ext, %b_ext, %cin_ext : i2

  // Instantiate DUT and check result
  %sum, %cout = hw.instance "dut" @Adder (%a, %b, %cin) : (i1, i1, i1) -> (i1, i1)
  %total_act = comb.concat %cout, %sum : (i1, i1) -> i2
  %0 = comb.icmp eq %total_act, %total_exp : i2
  verif.check %0 {note = "addition failed"}
}
