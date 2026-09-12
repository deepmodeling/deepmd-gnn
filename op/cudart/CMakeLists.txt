enable_language(C ASM)

find_package(
  Python3
  COMPONENTS Interpreter
  REQUIRED)

get_property(
  CUDART_LOCATION
  TARGET CUDA::cudart
  PROPERTY IMPORTED_LOCATION)
execute_process(
  COMMAND
    ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/third_party/implib/implib-gen.py
    ${CUDART_LOCATION} --target ${CMAKE_SYSTEM_PROCESSOR} --dlopen-callback
    DPGNN_cudart_dlopen --dlsym-callback DPGNN_cudart_dlsym
    COMMAND_ERROR_IS_FATAL ANY
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB DEEPMD_GNN_CUDART_STUB_SRC ${CMAKE_CURRENT_BINARY_DIR}/*.tramp.S
     ${CMAKE_CURRENT_BINARY_DIR}/*.init.c)

add_library(deepmd_gnn_cudart OBJECT cudart_stub.cc
                                     ${DEEPMD_GNN_CUDART_STUB_SRC})
target_include_directories(deepmd_gnn_cudart PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(deepmd_gnn_cudart PRIVATE IMPLIB_EXPORT_SHIMS)
set_target_properties(deepmd_gnn_cudart PROPERTIES POSITION_INDEPENDENT_CODE ON)
