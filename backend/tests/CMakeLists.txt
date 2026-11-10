# Every source is a standalone test or manual NPU harness. CONFIGURE_DEPENDS
# makes the next build discover newly added files without editing this list.
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
set(HOST_TESTS test_runner_contract test_model_tp test_prefix_cache test_scheduler_preemption test_attention_contract)

foreach(SRC_FILE IN LISTS TEST_SOURCES)
  get_filename_component(TARGET_NAME "${SRC_FILE}" NAME_WE)
  add_executable(${TARGET_NAME} "${SRC_FILE}")
  if(TARGET_NAME IN_LIST HOST_TESTS)
    target_link_libraries(${TARGET_NAME} PRIVATE forch::forch)
    string(REGEX REPLACE "^test_" "" TEST_NAME "${TARGET_NAME}")
    add_test(NAME ${TEST_NAME} COMMAND ${TARGET_NAME})
  else()
    # NPU harnesses require allocated devices and explicit runtime arguments.
    target_link_libraries(${TARGET_NAME} PRIVATE forch_infer)
  endif()
endforeach()

target_link_libraries(test_model_serving PRIVATE MPI::MPI_CXX)
