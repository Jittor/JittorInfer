set(TARGET llama-test-flash-attn)
add_executable(${TARGET} 
    flash_attn_aclnn.cpp
    flash_attn_cpu.cpp
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET_MLA llama-test-mla)
add_executable(${TARGET_MLA}
    mla_aclnn.cpp
    flash_attn_cpu.cpp
    cached_attn_cpu.cpp
)
install(TARGETS ${TARGET_MLA} RUNTIME)
target_link_libraries(${TARGET_MLA} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET_MLA} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET_MLA} PRIVATE cxx_std_17)

set(TARGET_MLA_PREFILL llama-test-mla-prefill)
add_executable(${TARGET_MLA_PREFILL}
    mla_prefill_aclnn.cpp
    mla_prefill_attn_cpu.cpp
)
install(TARGETS ${TARGET_MLA_PREFILL} RUNTIME)
target_link_libraries(${TARGET_MLA_PREFILL} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET_MLA_PREFILL} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET_MLA_PREFILL} PRIVATE cxx_std_17)

set(TARGET test_backend_ops)
add_executable(${TARGET} 
    test_backend_ops.cpp
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-test-ops)
add_executable(${TARGET} 
    ops_test_case.cpp
    ops_test.cpp
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
