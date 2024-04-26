add_circt_dialect_library(CIRCTMooreTransforms
InfoCollection.cpp
DeleteLocalVar.cpp

  DEPENDS
  CIRCTMooreTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTMoore
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
