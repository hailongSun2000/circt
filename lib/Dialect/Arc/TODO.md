- [x] Track register resets in CombRegOp
- [x] Map `EICG_wrapper` to a `arc.clock_gate` op
- [x] Add enable to `arc.state`
- [x] Add `arc.memory` op and map "FIRRTL_Memory" generators to it
- [x] Sort ops in `hw.module` to observe use-before-def
- [x] Lower arcs to funcs, modules to func with calls to arcs, state behind ptr
- [x] Generate C header of the func API
- [ ] Merge `arc.clock_gate` condition into `arc.state` enable

Full invocation:
```
firtool ~/sifive/arcs/x280.fir --ir-hw -o ~/sifive/arcs/x280.mlir --verbose-pass-executions --imconstprop=0 --dedup=1
circt-opt ~/sifive/arcs/x280.mlir -o ~/sifive/arcs/x280.arcs.mlir --arc-strip-sv --arc-infer-mems --cse --canonicalize --convert-to-arcs --arc-dedup --arc-inline-modules --cse --canonicalize
circt-opt ~/sifive/arcs/x280.arcs.mlir -o ~/sifive/arcs/x280.split.mlir --split-arc-loops --arc-dedup
circt-opt ~/sifive/arcs/x280.split.mlir -o ~/sifive/arcs/x280.state.mlir --arc-lower-state --cse --canonicalize
circt-opt ~/sifive/arcs/x280.state.mlir -o ~/sifive/arcs/x280.schedule.mlir --arc-schedule --debug-only=arc-schedule
circt-opt ~/sifive/arcs/x280.schedule.mlir -o ~/sifive/arcs/x280.llvm.mlir --lower-arc-to-llvm=state-file=$HOME/sifive/arcs/x280.state.json --cse --canonicalize --debug-only=lower-arc-to-llvm
llvm/build/bin/mlir-translate --mlir-to-llvmir ~/sifive/arcs/x280.llvm.mlir -o ~/sifive/arcs/x280.ll
llvm/build/bin/opt ~/sifive/arcs/x280.ll -o ~/sifive/arcs/x280.opt.ll -O3 -S -stats -time-passes -strip-debug
llvm/build/bin/llc ~/sifive/arcs/x280.ll -o ~/sifive/arcs/x280.o -filetype=obj -O3 -stats -time-passes -mtriple=x86_64-unknown-linux-elf -mattr=+fxsr,+sse,+sse2
```
