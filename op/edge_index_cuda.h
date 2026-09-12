// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstdint>

int64_t edge_index_cuda(const int64_t* nlist,
                        const int64_t* atype,
                        const int64_t* mm,
                        int64_t* edge_index,
                        int64_t nf,
                        int64_t nloc,
                        int64_t nnei,
                        int64_t nall,
                        int64_t nmm,
                        int device_index);
