add_circt_dialect_library(CIRCTArc
  ConvertToArcs.cpp
  Dedup.cpp
  Dialect.cpp
  DumpArcGraph.cpp
  InferMemories.cpp
  InlineModules.cpp
  LowerState.cpp
  LowerToLLVM.cpp
  Ops.cpp
  PrintArcInfo.cpp
  PullInputLogic.cpp
  PushOutputLogic.cpp
  RegToArc.cpp
  ScheduleModels.cpp
  SplitLoops.cpp
  StripSV.cpp
  Support.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Arc

  DEPENDS
  MLIRArcIncGen
  CIRCTArcPassesIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSCFToControlFlow
)

add_dependencies(circt-headers MLIRArcIncGen)
