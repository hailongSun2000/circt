add_circt_conversion_library(CIRCTMooreToCore
  MooreToCore.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTMoore
  CIRCTSeq
  CIRCTHW
  CIRCTComb
  MLIRControlFlowDialect
  MLIRTransforms
)
