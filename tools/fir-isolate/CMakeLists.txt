set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(fir-isolate
  fir-isolate.cpp
)
llvm_update_compile_flags(fir-isolate)
target_link_libraries(fir-isolate PRIVATE
  CIRCTImportFIRRTL
  CIRCTExportFIRRTL

  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRStandard
  MLIRTranslation
)
