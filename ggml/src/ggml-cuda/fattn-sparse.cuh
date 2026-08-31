#pragma once
#include "common.cuh"

// DSV4-Flash gather-sparse flash attention (see fattn-sparse.cu).
void ggml_cuda_flash_attn_ext_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_flash_attn_ext_sparse_supported(const ggml_tensor * dst);
