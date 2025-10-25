# server/CMakeLists.txt

set(TARGET server-neo)

# compile main.cpp to server-neo
add_executable(${TARGET} 
  main.cpp
)

target_include_directories(${TARGET} PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR} 
  ${CMAKE_SOURCE_DIR}/third_party 
  ${CMAKE_SOURCE_DIR}/third_party/yaml-cpp/include
  ${CMAKE_SOURCE_DIR}/third_party/libzmq/include
  ${CMAKE_SOURCE_DIR}/third_party/cppzmq
)

target_link_libraries(${TARGET} PRIVATE cppzmq yaml-cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_20)

set(TARGET server-neo-demo-decoder)

# compile main.cpp to server-neo
add_executable(${TARGET} 
  demo-decoder.cpp
)

target_include_directories(${TARGET} PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR} 
  ${CMAKE_SOURCE_DIR}/third_party 
  ${CMAKE_SOURCE_DIR}/third_party/yaml-cpp/include
  ${CMAKE_SOURCE_DIR}/third_party/libzmq/include
  ${CMAKE_SOURCE_DIR}/third_party/cppzmq
  ${MPI_C_INCLUDE_PATH}
)

target_link_libraries(${TARGET} PRIVATE 
  cppzmq
  yaml-cpp
  ${MPI_C_LIBRARIES}
)
target_compile_features(${TARGET} PRIVATE cxx_std_20)