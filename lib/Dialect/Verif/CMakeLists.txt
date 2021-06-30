file(GLOB globbed *.cpp)
add_circt_dialect_library(CIRCTVerif
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Verif

  DEPENDS
  CIRCTHW
  CIRCTComb
  MLIRVerifIncGen
  MLIRVerifEnumsIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTComb
  MLIRIR
)

add_dependencies(circt-headers MLIRVerifIncGen MLIRVerifEnumsIncGen)
