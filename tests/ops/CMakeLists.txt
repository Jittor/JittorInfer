set(TARGET llama-test-flash-attn)
add_executable(${TARGET} 
    flash_attn_aclnn.cpp
    flash_attn_cpu.cpp
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET test_ops_speed)
add_executable(${TARGET} 
    test_ops_speed.cpp
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PRIVATE cxx_std_17)