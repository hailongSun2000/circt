add_circt_dialect_library(CIRCTLTL
  LTLDialect.cpp
  LTLOps.cpp
  LTLTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/LTL

  DEPENDS
  MLIRLTLIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
)

add_dependencies(circt-headers MLIRLTLIncGen)
