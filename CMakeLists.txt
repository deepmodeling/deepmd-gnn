cmake_minimum_required(VERSION 3.15)
project(deepmd-gnn CXX)

set(CMAKE_CXX_STANDARD 14)
macro(set_if_higher VARIABLE VALUE)
  # ${VARIABLE} is a variable name, not a string
  if(${VARIABLE} LESS "${VALUE}")
    set(${VARIABLE} ${VALUE})
  endif()
endmacro()

# build cpp or python interfaces
option(BUILD_CPP_IF "Build C++ interfaces" ON)
option(BUILD_PY_IF "Build Python interfaces" OFF)
option(USE_PT_PYTHON_LIBS "Use PyTorch Python libraries" OFF)

if((NOT BUILD_PY_IF) AND (NOT BUILD_CPP_IF))
  # nothing to do
  message(FATAL_ERROR "Nothing to build.")
endif()

if(BUILD_CPP_IF
   AND USE_PT_PYTHON_LIBS
   AND NOT CMAKE_CROSSCOMPILING
   AND NOT SKBUILD
   OR "$ENV{CIBUILDWHEEL}" STREQUAL "1")
  find_package(
    Python
    COMPONENTS Interpreter
    REQUIRED)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c
            "import torch;print(torch.utils.cmake_prefix_path)"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH
    RESULT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH_RESULT_VAR
    ERROR_VARIABLE PYTORCH_CMAKE_PREFIX_PATH_ERROR_VAR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT ${PYTORCH_CMAKE_PREFIX_PATH_RESULT_VAR} EQUAL 0)
    message(
      FATAL_ERROR
        "Cannot determine PyTorch CMake prefix path, error code: $PYTORCH_CMAKE_PREFIX_PATH_RESULT_VAR}, error message: ${PYTORCH_CMAKE_PREFIX_PATH_ERROR_VAR}"
    )
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH})
endif()
find_package(Torch REQUIRED)
if(NOT Torch_VERSION VERSION_LESS "2.1.0")
  set_if_higher(CMAKE_CXX_STANDARD 17)
elseif(NOT Torch_VERSION VERSION_LESS "1.5.0")
  set_if_higher(CMAKE_CXX_STANDARD 14)
endif()
string(REGEX MATCH "_GLIBCXX_USE_CXX11_ABI=([0-9]+)" CXXABI_PT_MATCH
             "${TORCH_CXX_FLAGS}")
if(CXXABI_PT_MATCH)
  set(OP_CXX_ABI_PT ${CMAKE_MATCH_1})
  message(STATUS "PyTorch CXX11 ABI: ${CMAKE_MATCH_1}")
else()
  # Maybe in macos/windows
  if(UNIX
     AND NOT APPLE
     AND Torch_VERSION VERSION_GREATER_EQUAL "2.8.0")
    # https://github.com/deepmodeling/deepmd-kit/issues/4877
    # torch.compiled_with_cxx11_abi in PyTorch 2.8 always return True
    set(OP_CXX_ABI 1)
  else()
    set(OP_CXX_ABI 0)
  endif()
endif()

# define build type
if((NOT DEFINED CMAKE_BUILD_TYPE) OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE release)
endif()

add_subdirectory(op)
