set(LLVM_LINK_COMPONENTS Support)
add_llvm_tool(circt-verify
  circt-verify.cpp
)
llvm_update_compile_flags(circt-verify)
target_link_libraries(circt-verify PRIVATE
  CIRCTVerif
  CIRCTTransforms

  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRStandard
  MLIRTransforms
  MLIRTranslation
)
