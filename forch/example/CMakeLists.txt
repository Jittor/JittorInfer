add_executable(forch-example-graph graph.cpp)
target_link_libraries(forch-example-graph PRIVATE forch)

if(FORCH_ENABLE_GE)
add_executable(forch-example-ge-graph ge-graph.cpp)
target_link_libraries(forch-example-ge-graph PRIVATE forch forch-ge)

add_executable(forch-example-ge-collective ge-collective.cpp)
target_link_libraries(forch-example-ge-collective PRIVATE forch forch-ge)

add_executable(forch-example-ge-mlp ge-mlp.cpp)
target_link_libraries(forch-example-ge-mlp PRIVATE forch forch-ge)

add_executable(forch-example-ge-explore ge-explore.cpp)
target_link_libraries(forch-example-ge-explore PRIVATE forch forch-ge)

add_executable(forch-example-module-mlp module-mlp.cpp)
target_link_libraries(forch-example-module-mlp PRIVATE forch forch-ge)

endif()
