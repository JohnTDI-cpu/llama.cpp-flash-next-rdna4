#include "common.cuh"
#include "ggml.h"

#define CUDA_CONV_SNAP_BLOCK_SIZE 256
#define CUDA_CONV_SNAP_MAX_SLOTS 8

// [johnv8] E4: migawki stanu conv dla slotow rollbacku MTP (qwen4exp build_layer_gdn):
//   n_slots kolejnych wezlow CPY (widok ogona conv_input -> widok wiersza conv_states_all)
//   -> jedno jadro kopiujace wszystkie sloty. Czysta kopia, bit-exact.
// GGML_JOHNV8_CONV_SNAP=0 wylacza.
bool ggml_cuda_conv_snap_enabled();
bool ggml_cuda_conv_snap_ok(const ggml_cgraph * cgraph, int node_idx, int * n_nodes);
void ggml_cuda_op_conv_snap(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx, int n_nodes);
