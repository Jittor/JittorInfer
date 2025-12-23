if ("cann${CANN_INSTALL_DIR}" STREQUAL "cann" AND DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(CANN_INSTALL_DIR $ENV{ASCEND_TOOLKIT_HOME})
    message(STATUS "CANN: updated CANN_INSTALL_DIR from ASCEND_TOOLKIT_HOME=$ENV{ASCEND_TOOLKIT_HOME}")
endif()

# Set CANN-OPS-ADV directory
if(LLAMA_USE_JITTOR_OPS)
    # TODO: check if JITTOR_OPS_DIR exists
    set(JITTOR_OPS_DIR ${CANN_INSTALL_DIR}/opp/vendors/jittor_infer CACHE PATH "Path to Jittor OPS installation")
    set(JITTOR_OPS_INCLUDE_DIR ${JITTOR_OPS_DIR}/op_api/include)
    set(JITTOR_OPS_LIB_DIR ${JITTOR_OPS_DIR}/op_api/lib)
    add_compile_definitions(LLAMA_JITTOR_OPS_SUPPORT)
    message(STATUS "set JITTOR_OPS_DIR: ${JITTOR_OPS_DIR}")
endif()

# Auto-detech Soc type and Soc version, if detect failed, will abort build
# Try to get SOC_TYPE from cache first
if(NOT SOC_TYPE)
    # If not in cache, or cache is empty, then detect
    if(NOT DEFINED CACHED_SOC_TYPE OR CACHED_SOC_TYPE STREQUAL "")
        message(STATUS "CANN: Detecting SOC_VERSION...")
        execute_process(
            COMMAND bash -c "npu-smi info|awk -F\' \' \'NF > 0 && NR==7 {print $3}\'"
            OUTPUT_VARIABLE npu_info
            RESULT_VARIABLE npu_result
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if("${npu_info}" STREQUAL "" OR ${npu_result})
            message(FATAL_ERROR "Auto-detech ascend soc type failed, please specify manually or check ascend device working normally.")
        endif()
        set(DETECTED_SOC_VERSION "Ascend${npu_info}")
        set(CACHED_SOC_TYPE ${DETECTED_SOC_VERSION} CACHE STRING "Detected Ascend SOC Version. Cleared to re-detect.")
        message(STATUS "CANN: SOC_VERSION auto-detected as: ${DETECTED_SOC_VERSION}. Cached.")
    endif()
    set(SOC_TYPE "${CACHED_SOC_TYPE}")
else()
    message(STATUS "CANN: Using provided SOC_TYPE: ${SOC_TYPE}")
endif()

string(TOLOWER ${SOC_TYPE} SOC_VERSION) # SOC_VERSION need lower

# Construct Soc specify compile option: ASCEND_#Soc_Major_SN. Such as ASCEND_910B, ASCEND_310P.
string(REGEX MATCH "[0-9]+[a-zA-Z]" SOC_TYPE_MAJOR_SN "${SOC_VERSION}")
set(SOC_TYPE_COMPILE_OPTION "ASCEND_${SOC_TYPE_MAJOR_SN}")
string(TOUPPER ${SOC_TYPE_COMPILE_OPTION} SOC_TYPE_COMPILE_OPTION)

if (CANN_INSTALL_DIR)
    # Only Support Linux.
    if (NOT UNIX)
        message(FATAL_ERROR "CANN: CANN toolkit supports unix but not ${CMAKE_SYSTEM_NAME}")
    endif()

    # Supported platforms: x86-64, arm64
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
    else()
        message(FATAL_ERROR "CANN: CANN toolkit supports x86-64 and arm64 but not ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    # Set header and libs
    set(CANN_INCLUDE_DIRS
        ${CANN_INSTALL_DIR}/include
        ${CANN_INSTALL_DIR}/include/aclnn
        ${CANN_INSTALL_DIR}/acllib/include
        ${CANN_INSTALL_DIR}/compiler/include/graph
        ${CANN_INSTALL_DIR}/compiler/include/ge
        ${CANN_INSTALL_DIR}/compiler/include
        ${CANN_INSTALL_DIR}/opp/vendors/jittor_infer/op_proto/inc
        ${CANN_INSTALL_DIR}/opp/built-in/op_proto/inc
        # ${CANN_INSTALL_DIR}/opp/vendors/customize/op_proto/inc
    )

    if (LLAMA_USE_JITTOR_OPS)
        list(APPEND CANN_INCLUDE_DIRS ${JITTOR_OPS_INCLUDE_DIR})
    endif()
    message(STATUS "CANN_INCLUDE_DIRS: ${CANN_INCLUDE_DIRS}")

    add_subdirectory(kernels)

    if (LLAMA_USE_JITTOR_OPS)
        file(MAKE_DIRECTORY ${JITTOR_OPS_LIB_DIR})
        
        if(LLAMA_INSTALL_JITTOR_OPS)
            message(STATUS "install customized operators...")
            execute_process(
                COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/kernels/op_project/install.sh
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/op_project
                RESULT_VARIABLE install_result
                OUTPUT_VARIABLE install_output
                ERROR_VARIABLE install_error
            )
            
            if(NOT install_result EQUAL "0")
                message(WARNING "customized operators install failed: ${install_error}")
            else()
                message(STATUS "customized operators installed")
            endif()
        
        endif()
        
        execute_process(
            COMMAND ls -la ${JITTOR_OPS_LIB_DIR}
            OUTPUT_VARIABLE ls_output
        )
        message(STATUS "custom ops ${JITTOR_OPS_LIB_DIR} : ${ls_output}")
        
        find_library(JITTOR_OPS_LIB
            NAMES cust_opapi
            PATHS ${JITTOR_OPS_LIB_DIR}
            NO_DEFAULT_PATH
        )
        
        if(NOT JITTOR_OPS_LIB)
            message(WARNING "cust_opapi not found")
        else()
            message(STATUS "found jittor_ops : ${JITTOR_OPS_LIB}")
            list(APPEND CANN_LIBRARIES ${JITTOR_OPS_LIB})
        endif()
    endif()
    
    # Add CANN-OPS-ADV library first to ensure its symbols take precedence
    list(APPEND CANN_LIBRARIES
        ascendcl
        nnopbase
        opapi
        acl_op_compiler
        ascendc_kernels
        hccl
        graph
        ge_runner
    )

    file(GLOB GGML_SOURCES_CANN "*.cpp")

    # 添加OpenMP支持
    find_package(OpenMP REQUIRED)

    ggml_add_backend_library(ggml-cann ${GGML_SOURCES_CANN})
    target_link_libraries(ggml-cann PRIVATE ${CANN_LIBRARIES} OpenMP::OpenMP_CXX)
    target_include_directories(ggml-cann PRIVATE ${CANN_INCLUDE_DIRS})
    target_link_directories(ggml-cann PRIVATE 
        ${CANN_INSTALL_DIR}/lib64
        ${CANN_INSTALL_DIR}/compiler/lib64/stub
    )

    target_compile_definitions(ggml-cann PRIVATE "-D${SOC_TYPE_COMPILE_OPTION}")
    target_compile_definitions(ggml-cann PRIVATE "-D_GLIBCXX_USE_CXX11_ABI=0")

    message(STATUS "CANN: CANN_INCLUDE_DIRS =  ${CANN_INCLUDE_DIRS}")
    message(STATUS "CANN: CANN_LIBRARIES =  ${CANN_LIBRARIES}")

else()
    message(FATAL_ERROR "CANN: Can't find CANN_INSTALL_DIR, did you forget to source set_var.sh?")
endif()
