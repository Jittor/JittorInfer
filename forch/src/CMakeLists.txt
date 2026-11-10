add_subdirectory(device)
add_subdirectory(pass)
add_subdirectory(ops)
add_subdirectory(huggingface)

if(FORCH_ENABLE_ASCEND)
  add_subdirectory(forch-ascend)
endif()

if(FORCH_ENABLE_GE)
  add_subdirectory(forch-ge)
endif()

target_sources(forch PRIVATE
  distributed.cpp
  exception.cpp
  forch.cpp
  graph.cpp
  utils.cpp
)
