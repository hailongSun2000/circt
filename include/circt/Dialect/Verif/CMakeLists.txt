add_circt_dialect(Verif verif)
add_circt_doc(Verif -gen-dialect-doc Verif Dialects/)

set(LLVM_TARGET_DEFINITIONS Verif.td)
mlir_tablegen(VerifEnums.h.inc -gen-enum-decls)
mlir_tablegen(VerifEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRVerifEnumsIncGen)
