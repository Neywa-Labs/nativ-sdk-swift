// Exact affine verification kernels ported from mlx-vlm at
// 06ff63d6e9c8e14649ed3e5cb380810d27c790b8, models/quantized_verifier.py.
// The singleton reduction order is intentional; do not reassociate arithmetic.
/*
MIT License

Copyright © 2025 Prince Canuma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

enum ExactQuantizedKernelSource {
    static let header = #"""
        using namespace metal;

        constant constexpr int SIMD_SIZE = 32;
        constant constexpr int BITS = __BITS__;
        constant constexpr int GS = __GS__;
        constant constexpr int PACK_FACTOR = (BITS == 5 ? 8 : 32 / BITS);
        constant constexpr int BYTES_PER_PACK = (BITS == 5 ? 5 : 32 / 8);
        constant constexpr int PACKS_PER_THREAD = 2;
        constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
        constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
        constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
        constant constexpr int RESULTS_PER_SIMDGROUP = __RESULTS_PER_SIMDGROUP__;
        constant constexpr int NUM_SIMDGROUPS = 2;
        constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

        template <typename T>
        inline float load_vector_exact(const device T* x, thread float* x_thread) {
          float sum = 0.0f;
          if (BITS == 4) {
            for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
              sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1];
              x_thread[i + 2] = x[i + 2];
              x_thread[i + 3] = x[i + 3];
            }
          } else if (BITS == 5) {
            for (int i = 0; i < VALUES_PER_THREAD; i += 8) {
              sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) +
                  float(x[i + 4]) + float(x[i + 5]) + float(x[i + 6]) + float(x[i + 7]);
              x_thread[i] = x[i];
              x_thread[i + 1] = x[i + 1] / 32.0f;
              x_thread[i + 2] = x[i + 2] / 4.0f;
              x_thread[i + 3] = x[i + 3] / 128.0f;
              x_thread[i + 4] = x[i + 4] / 16.0f;
              x_thread[i + 5] = x[i + 5] / 2.0f;
              x_thread[i + 6] = x[i + 6] / 64.0f;
              x_thread[i + 7] = x[i + 7] / 8.0f;
            }
          } else if (BITS == 8) {
            for (int i = 0; i < VALUES_PER_THREAD; i++) {
              sum += x[i];
              x_thread[i] = x[i];
            }
          }
          return sum;
        }

        inline float qdot_exact(
            const device uint8_t* w,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
          float accum = 0.0f;
          if (BITS == 4) {
            const device uint16_t* ws = (const device uint16_t*)w;
            for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
              uint packed = ws[i];
              accum +=
                  (x_thread[4 * i] * (packed & 0x000f) +
                   x_thread[4 * i + 1] * ((packed >> 4) & 0x000f) +
                   x_thread[4 * i + 2] * ((packed >> 8) & 0x000f) +
                   x_thread[4 * i + 3] * ((packed >> 12) & 0x000f));
            }
          } else if (BITS == 5) {
            for (int i = 0; i < (VALUES_PER_THREAD / 8); i++) {
              const thread float* xt = x_thread + 8 * i;
              const device uint8_t* wb = w + 5 * i;

              accum += (wb[0] & 0x1f) * xt[0];
              accum += (wb[0] & 0xe0) * xt[1];
              accum += (wb[1] & 0x3) * (xt[1] * 256.0f);
              accum += (wb[1] & 0x7c) * xt[2];
              accum += (wb[1] & 0x80) * xt[3];
              accum += (wb[2] & 0xf) * (xt[3] * 256.0f);
              accum += (wb[2] & 0xf0) * xt[4];
              accum += (wb[3] & 0x1) * (xt[4] * 256.0f);
              accum += (wb[3] & 0x3e) * xt[5];
              accum += (wb[3] & 0xc0) * xt[6];
              accum += (wb[4] & 0x7) * (xt[6] * 256.0f);
              accum += (wb[4] & 0xf8) * xt[7];
            }
          } else if (BITS == 8) {
            for (int i = 0; i < VALUES_PER_THREAD; i++) {
              accum += x_thread[i] * w[i];
            }
          }
          return scale * accum + sum * bias;
        }

        inline float qdot_exact(
            const thread uint16_t* ws,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
          float accum = 0.0f;
          if (BITS == 4) {
            for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
              uint packed = ws[i];
              accum +=
                  (x_thread[4 * i] * (packed & 0x000f) +
                   x_thread[4 * i + 1] * ((packed >> 4) & 0x000f) +
                   x_thread[4 * i + 2] * ((packed >> 8) & 0x000f) +
                   x_thread[4 * i + 3] * ((packed >> 12) & 0x000f));
            }
          }
          return scale * accum + sum * bias;
        }
        """#

    static let standard = #"""
        uint n_tile = threadgroup_position_in_grid.y;
            uint b_idx = threadgroup_position_in_grid.z;
            uint simd_gid = simdgroup_index_in_threadgroup;
            uint simd_lid = thread_index_in_simdgroup;

            int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
            int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
            int in_vec_size_g = K_SIZE / GS;

            const device uint8_t* ws_base =
                (const device uint8_t*)w + out_row * in_vec_size_w +
                int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
            const device T* scales_base =
                scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
            const device T* biases_base =
                biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
            const device T* x_base =
                x + int(b_idx) * VERIFY_T * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

            float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
            float x_thread[VERIFY_T][VALUES_PER_THREAD];
            for (int t = 0; t < VERIFY_T; ++t) {
              for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                result[t][row] = 0.0f;
              }
            }

            const device uint8_t* ws = ws_base;
            const device T* sc = scales_base;
            const device T* bs = biases_base;
            const device T* xk = x_base;

            for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
              float sums[VERIFY_T];
              for (int t = 0; t < VERIFY_T; ++t) {
                sums[t] = load_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
              }

        #pragma clang loop unroll_count(2)
              for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                const device uint8_t* wl = ws + row * in_vec_size_w;
                const device T* sl = sc + row * in_vec_size_g;
                const device T* bl = bs + row * in_vec_size_g;
                float s = float(sl[0]);
                float b = float(bl[0]);
                for (int t = 0; t < VERIFY_T; ++t) {
                  result[t][row] += qdot_exact(wl, x_thread[t], s, b, sums[t]);
                }
              }

              ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
              sc += BLOCK_SIZE / GS;
              bs += BLOCK_SIZE / GS;
              xk += BLOCK_SIZE;
            }

            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
              int n = out_row + row;
              for (int t = 0; t < VERIFY_T; ++t) {
                float r = simd_sum(result[t][row]);
                if (simd_lid == 0) {
                  y[(int(b_idx) * VERIFY_T + t) * N_SIZE + n] = T(r);
                }
              }
            }
        """#

    static let tiled = #"""
        uint n_tile = threadgroup_position_in_grid.y;
        uint token_tile = threadgroup_position_in_grid.z;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        constexpr int TOKEN_TILE = 2;
        constexpr int NUM_TOKEN_TILES = (VERIFY_T + TOKEN_TILE - 1) / TOKEN_TILE;
        int b_idx = int(token_tile) / NUM_TOKEN_TILES;
        int token_tile_idx = int(token_tile) - b_idx * NUM_TOKEN_TILES;
        int token_start = token_tile_idx * TOKEN_TILE;
        int token_count = min(TOKEN_TILE, VERIFY_T - token_start);
        int flat_token_start = b_idx * VERIFY_T + token_start;
        int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K_SIZE / GS;

        const device uint8_t* ws =
            (const device uint8_t*)w + out_row * in_vec_size_w +
            int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* sc =
            scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* bs =
            biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* xk =
            x + flat_token_start * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

        float result[TOKEN_TILE][RESULTS_PER_SIMDGROUP];
        float x_thread[TOKEN_TILE][VALUES_PER_THREAD];
        for (int t = 0; t < TOKEN_TILE; ++t) {
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            result[t][row] = 0.0f;
          }
        }

        for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
          float sums[TOKEN_TILE];
          for (int t = 0; t < token_count; ++t) {
            sums[t] = load_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
          }
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            const device uint8_t* wl = ws + row * in_vec_size_w;
            const device T* sl = sc + row * in_vec_size_g;
            const device T* bl = bs + row * in_vec_size_g;
            float s = float(sl[0]);
            float b = float(bl[0]);
            for (int t = 0; t < token_count; ++t) {
              result[t][row] += qdot_exact(wl, x_thread[t], s, b, sums[t]);
            }
          }

          ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          sc += BLOCK_SIZE / GS;
          bs += BLOCK_SIZE / GS;
          xk += BLOCK_SIZE;
        }

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
          int n = out_row + row;
          for (int t = 0; t < token_count; ++t) {
            float r = simd_sum(result[t][row]);
            if (simd_lid == 0) {
              y[(flat_token_start + t) * N_SIZE + n] = T(r);
            }
          }
        }
        """#

    static let streamed = #"""
        uint n_tile = threadgroup_position_in_grid.y;
        uint b_idx = threadgroup_position_in_grid.z;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;

        int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
        int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K_SIZE / GS;

        const device uint8_t* ws =
            (const device uint8_t*)w + out_row * in_vec_size_w +
            int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
        const device T* sc =
            scales + out_row * in_vec_size_g +
            int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* bs =
            biases + out_row * in_vec_size_g +
            int(simd_lid) / SCALE_STEP_PER_THREAD;
        const device T* xk =
            x + int(b_idx) * VERIFY_T * K_SIZE +
            int(simd_lid) * VALUES_PER_THREAD;

        float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
        for (int t = 0; t < VERIFY_T; ++t) {
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            result[t][row] = 0.0f;
          }
        }

        for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            const device uint16_t* source =
                (const device uint16_t*)(ws + row * in_vec_size_w);
            uint16_t packed[VALUES_PER_THREAD / 4];
            for (int index = 0; index < VALUES_PER_THREAD / 4; ++index) {
              packed[index] = source[index];
            }
            float scale = float(sc[row * in_vec_size_g]);
            float bias = float(bs[row * in_vec_size_g]);
            for (int t = 0; t < VERIFY_T; ++t) {
              float x_thread[VALUES_PER_THREAD];
              float sum = load_vector_exact<T>(
                  xk + t * K_SIZE, x_thread);
              result[t][row] += qdot_exact(
                  packed, x_thread, scale, bias, sum);
            }
          }

          ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
          sc += BLOCK_SIZE / GS;
          bs += BLOCK_SIZE / GS;
          xk += BLOCK_SIZE;
        }

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
          int n = out_row + row;
          for (int t = 0; t < VERIFY_T; ++t) {
            float value = simd_sum(result[t][row]);
            if (simd_lid == 0) {
              y[(int(b_idx) * VERIFY_T + t) * N_SIZE + n] = T(value);
            }
          }
        }
        """#

}
