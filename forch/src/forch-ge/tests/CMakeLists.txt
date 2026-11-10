# For each <name>.cpp, add target and link forch forch-ge

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(src IN LISTS TEST_SOURCES)
	get_filename_component(name ${src} NAME_WE)
	set(name "forch-ge-test-${name}")
	add_executable(${name} ${src})
	target_link_libraries(${name} PRIVATE forch forch-ge)
  target_include_directories(${name} PRIVATE ..)
endforeach()

# Multi-rank tests need one process per card; the glob above is non-recursive,
# so they live in a subdirectory and run via script/forch-ge-run.py.
add_subdirectory(multirank)

