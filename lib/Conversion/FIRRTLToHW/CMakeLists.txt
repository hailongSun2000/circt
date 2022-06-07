add_circt_conversion_library(CIRCTFIRRTLToHW
  LowerToHW.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTArc
  CIRCTFIRRTL
  CIRCTHW
  CIRCTSV
  MLIRTransforms
)
