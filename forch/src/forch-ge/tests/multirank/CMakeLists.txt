# Multi-rank (collective) tests, deliberately under a subdirectory: the parent
# tests/CMakeLists.txt glob is non-recursive, so script/run-test.py (one
# process per card) never picks these up. Launch them with
# script/forch-ge-run.py instead.
#
# These link only forch and forch-ge: rank, device and rank table reach the
# backend through the environment, so no test needs ACL or the GE shim.

file(GLOB MR_SOURCES CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
foreach(src IN LISTS MR_SOURCES)
  get_filename_component(name ${src} NAME_WE)
  set(name "forch-ge-mr-${name}")
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE forch forch-ge)
endforeach()
