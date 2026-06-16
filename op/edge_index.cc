// SPDX-License-Identifier: LGPL-3.0-or-later
// Target the oldest stable ABI version used by this file's helper APIs.
#ifndef TORCH_TARGET_VERSION
#define TORCH_TARGET_VERSION (((0ULL + 2) << 56) | ((0ULL + 10) << 48))
#endif

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
namespace th = torch::headeronly;
namespace ts = torch::stable;

ts::Tensor to_cpu_contiguous(const ts::Tensor& tensor) {
  return ts::contiguous(ts::to(tensor, ts::Device(th::DeviceType::CPU)));
}

ts::Tensor edge_index_kernel(const ts::Tensor& nlist_tensor,
                             const ts::Tensor& atype_tensor,
                             const ts::Tensor& mm_tensor) {
  ts::Tensor nlist_tensor_ = to_cpu_contiguous(nlist_tensor);
  ts::Tensor atype_tensor_ = to_cpu_contiguous(atype_tensor);
  ts::Tensor mm_tensor_ = to_cpu_contiguous(mm_tensor);

  int64_t nf = 0;
  int64_t nloc = 0;
  int64_t nnei = 0;
  int64_t nall = 0;
  if (nlist_tensor_.dim() == 2) {
    if (atype_tensor_.dim() != 1) {
      throw std::invalid_argument("atype_tensor must be 1D");
    }
    nf = 1;
    nloc = nlist_tensor_.size(0);
    nnei = nlist_tensor_.size(1);
    nall = atype_tensor_.size(0);
  } else if (nlist_tensor_.dim() == 3) {
    if (atype_tensor_.dim() != 2) {
      throw std::invalid_argument("atype_tensor must be 2D");
    }
    nf = nlist_tensor_.size(0);
    nloc = nlist_tensor_.size(1);
    nnei = nlist_tensor_.size(2);
    if (atype_tensor_.size(0) != nf) {
      throw std::invalid_argument(
          "atype_tensor must have the same size as nlist_tensor");
    }
    nall = atype_tensor_.size(1);
  } else {
    throw std::invalid_argument("nlist_tensor must be 2D or 3D");
  }

  const int64_t nmm = mm_tensor_.size(0);
  const int64_t* nlist = nlist_tensor_.const_data_ptr<int64_t>();
  const int64_t* atype = atype_tensor_.const_data_ptr<int64_t>();
  const int64_t* mm = mm_tensor_.const_data_ptr<int64_t>();

  std::vector<int64_t> edge_index;
  edge_index.reserve(nf * nloc * nnei * 2);

  for (int64_t ff = 0; ff < nf; ff++) {
    for (int64_t ii = 0; ii < nloc; ii++) {
      for (int64_t jj = 0; jj < nnei; jj++) {
        const int64_t idx = ff * nloc * nnei + ii * nnei + jj;
        const int64_t kk = nlist[idx];
        if (kk < 0) {
          continue;
        }
        const int64_t global_kk = ff * nall + kk;
        const int64_t global_ii = ff * nall + ii;
        bool in_mm1 = false;
        for (int64_t mm_idx = 0; mm_idx < nmm; mm_idx++) {
          if (atype[global_ii] == mm[mm_idx]) {
            in_mm1 = true;
            break;
          }
        }
        bool in_mm2 = false;
        for (int64_t mm_idx = 0; mm_idx < nmm; mm_idx++) {
          if (atype[global_kk] == mm[mm_idx]) {
            in_mm2 = true;
            break;
          }
        }
        if (in_mm1 && in_mm2) {
          continue;
        }
        edge_index.push_back(global_kk);
        edge_index.push_back(global_ii);
      }
    }
  }

  const int64_t edge_size = static_cast<int64_t>(edge_index.size() / 2);
  ts::Tensor edge_index_tensor = ts::empty(
      {edge_size, 2},
      th::ScalarType::Long,
      th::Layout::Strided,
      ts::Device(th::DeviceType::CPU));
  int64_t* edge_index_data = edge_index_tensor.mutable_data_ptr<int64_t>();
  std::copy(edge_index.begin(), edge_index.end(), edge_index_data);
  return ts::to(edge_index_tensor, nlist_tensor.device());
}
}  // namespace

STABLE_TORCH_LIBRARY(deepmd_gnn, m) {
  m.def("edge_index(Tensor nlist, Tensor atype, Tensor mm) -> Tensor");
  m.impl("edge_index", TORCH_BOX(edge_index_kernel));
}

// Compatibility with old models frozen by deepmd_mace package.
STABLE_TORCH_LIBRARY(deepmd_mace, m) {
  m.def("mace_edge_index(Tensor nlist, Tensor atype, Tensor mm) -> Tensor");
  m.impl("mace_edge_index", TORCH_BOX(edge_index_kernel));
}
