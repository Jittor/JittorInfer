set(TARGET llama-test-single-sequence)
add_executable(${TARGET}
    single_sequence.cpp
    common_local.cpp
    sampling_local.cpp
    chat_local.cpp
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-test-multi-sequence)
add_executable(${TARGET}
    multi_sequence.cpp
    common_local.cpp
    sampling_local.cpp
    chat_local.cpp
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_MPI)
    set(TARGET llama-test-mpi-dp-ep)
    add_executable(${TARGET}
        mpi_dp_ep.cpp
        common_local.cpp
        sampling_local.cpp
        chat_local.cpp
    )
    install(TARGETS ${TARGET} RUNTIME)
    target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()