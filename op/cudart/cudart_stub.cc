// SPDX-License-Identifier: LGPL-3.0-or-later
// Dynamically load the CUDA runtime library.

#include <dlfcn.h>

#include <string>

#include "cuda_runtime_api.h"

namespace {

bool cudart_missing = false;

cudaError_t DPGNN_CudartGetSymbolNotFoundError(...) {
  return cudaErrorSharedObjectSymbolNotFound;
}

const char* DPGNN_CudartGetSymbolNotFoundString(cudaError_t) {
  return "CUDA runtime library or symbol not found";
}

const char* DPGNN_CudartGetSymbolNotFoundName(cudaError_t) {
  return "cudaErrorSharedObjectSymbolNotFound";
}

void** DPGNN_CudartRegisterFatBinary(...) { return nullptr; }

void DPGNN_CudartNoOp(...) {}

void* DPGNN_CudartFallback(const char* sym_name) {
  const std::string symbol(sym_name);
  if (symbol == "cudaGetErrorString") {
    return reinterpret_cast<void*>(&DPGNN_CudartGetSymbolNotFoundString);
  }
  if (symbol == "cudaGetErrorName") {
    return reinterpret_cast<void*>(&DPGNN_CudartGetSymbolNotFoundName);
  }
  if (symbol == "__cudaRegisterFatBinary") {
    return reinterpret_cast<void*>(&DPGNN_CudartRegisterFatBinary);
  }
  if (symbol == "__cudaRegisterFatBinaryEnd" ||
      symbol == "__cudaUnregisterFatBinary" ||
      symbol == "__cudaRegisterFunction" || symbol == "__cudaRegisterVar") {
    return reinterpret_cast<void*>(&DPGNN_CudartNoOp);
  }
  return reinterpret_cast<void*>(&DPGNN_CudartGetSymbolNotFoundError);
}

}  // namespace

extern "C" {

void* DPGNN_cudart_dlopen(const char* libname) {
  static void* handle = [](const std::string& name) -> void* {
    void* dso_handle = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!dso_handle) {
      cudart_missing = true;
      return dlopen(nullptr, RTLD_NOW | RTLD_LOCAL);
    }
    return dso_handle;
  }(std::string(libname));
  return handle;
}

void* DPGNN_cudart_dlsym(void* handle, const char* sym_name) {
  if (!handle || cudart_missing) {
    return DPGNN_CudartFallback(sym_name);
  }
  void* symbol = dlsym(handle, sym_name);
  if (!symbol) {
    return DPGNN_CudartFallback(sym_name);
  }
  return symbol;
}

}  // extern "C"
