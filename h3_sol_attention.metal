/* SPDX-License-Identifier: MIT
 *
 * Native Apple-Silicon Sol-Attn kernel, adapted from yshenaw's MPS backend at
 * commit 45071126b0c1ee30b0e6b7103fa9d70924828ba5 and NVIDIA's merge at
 * 5cebf61bd70b3d164c3d26ca858f0044ec6bd3ee. The simdgroup tile helpers are
 * adapted from PyTorch PrefillAttention.h. See THIRD_PARTY_NOTICES.md.
 */
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define PREFILL_CONST static constant constexpr const
#define PREFILL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

template <typename T>
struct pointer_element {};
template <typename T>
struct pointer_element<thread T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<device T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<constant T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<threadgroup T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;

template <int val>
using Int = integral_constant<int, val>;

template <
    typename T,
    short BROWS,
    short BCOLS,
    short kDstStrRow,
    short kDstStrCol,
    short reduction_dim,
    short tgp_size,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BlockLoaderT {
  PREFILL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  PREFILL_CONST short vec_size = n_reads;

  const int src_ld;
  const int tile_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device T* src;

  METAL_FUNC BlockLoaderT(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * kDstStrRow + bj * kDstStrCol),
        src(src_ + bi * src_ld + bj) {}

  METAL_FUNC void load_unsafe() const {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = src[i * src_ld + j];
      }
    }
  }

  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      PREFILL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        PREFILL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * kDstStrRow + j * kDstStrCol] = T(0);
        }
      }
      return;
    }

    bool tmp_idx[vec_size];
    T tmp_val[vec_size];

    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = tmp_val[j];
      }
    }
  }

  METAL_FUNC void next() {
    src += tile_stride;
  }
};

struct MaxOp {
  template <typename T>
  METAL_FUNC static T apply(T x, T y) {
    // metal::max propagates NaN (plain max drops it -> row max -inf -> 0).
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct ExpSubOp {
  // If y (the row max) is -inf, every score in this row was masked out
  // (e.g. an explicit additive mask of all -inf, or a fully causally-masked
  // row). exp(-inf - -inf) = exp(NaN) = NaN, which then poisons the running
  // sum and output. Returning 0 is the mathematically correct limit
  // (exp(-inf) = 0) and is what flash-attention implementations do.
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return (y == -INFINITY) ? T(0) : fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

template <typename T, int kFragRows_, int kFragCols_>
struct BaseMMAFrag {
  static_assert(kFragRows_ == 8, "Only 8x8 fragments are supported");
  static_assert(kFragCols_ == 8, "Only 8x8 fragments are supported");
};

template <typename T>
struct BaseMMAFrag<T, 8, 8> {
  PREFILL_CONST int kFragRows = 8;
  PREFILL_CONST int kFragCols = 8;

  PREFILL_CONST int kElemsPerFrag = (kFragRows * kFragCols) / 32;

  PREFILL_CONST int kElemRows = 1;
  PREFILL_CONST int kElemCols = 2;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MMAFrag shape is not consistent with MMAFrag size");

  typedef metal::simdgroup_matrix<T, kFragRows, kFragCols> mat_type;
  typedef metal::vec<T, kElemsPerFrag> frag_type;
  typedef metal::vec<T, kElemRows> row_frag_type;
  typedef metal::vec<T, kElemCols> col_frag_type;

  template <typename U>
  using dtype_mat_t = typename metal::simdgroup_matrix<U, kFragRows, kFragCols>;

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static constexpr short2 get_coord(ushort simd_lane_id
                                               [[thread_index_in_simdgroup]]) {
    const short qid = simd_lane_id / 4;
    const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
    const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    return short2{fn, fm};
  }

  template <typename SrcPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void load(
      thread frag_type& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y) {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * kElemCols + j] =
            static_cast<T>(src[i * str_x.value + j * str_y.value]);
      }
    }
  }

  template <
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX,
      typename OffY>
  METAL_FUNC static constexpr void load_safe(
      thread frag_type& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[i * kElemCols + j] = static_cast<T>(
              src[(off_x + i) * str_x + (off_y + j) * str_y.value]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <typename DstPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void store(
      const thread frag_type& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y) {
    using U = pointer_element_t<DstPtrType>;
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * str_x + j * str_y.value] =
            static_cast<U>(src[i * kElemCols + j]);
      }
    }
  }

  template <
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX,
      typename OffY>
  METAL_FUNC static constexpr void store_safe(
      const thread frag_type& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[(off_x + i) * str_x + (off_y + j) * str_y.value] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <typename Atype, typename Btype, typename Ctype>
  METAL_FUNC static constexpr void mma(
      thread frag_type& D,
      thread dtype_frag_t<Atype>& A,
      thread dtype_frag_t<Btype>& B,
      thread dtype_frag_t<Ctype>& C) {
    mat_type D_mat;
    dtype_mat_t<Atype> A_mat;
    dtype_mat_t<Btype> B_mat;
    dtype_mat_t<Ctype> C_mat;

    reinterpret_cast<thread dtype_frag_t<Atype>&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread dtype_frag_t<Btype>&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread dtype_frag_t<Ctype>&>(C_mat.thread_elements()) = C;

    simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);

    D = reinterpret_cast<thread frag_type&>(D_mat.thread_elements());
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_reduce(
      thread const frag_type& inp_vals,
      thread T* reduced_vals) {
    T thr_reduce = Op::apply(inp_vals.x, inp_vals.y);

    T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
    qgr_reduce = Op::apply(thr_reduce, qgr_reduce);

    T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
    sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);

    reduced_vals[0] = Op::apply(reduced_vals[0], sgr_reduce);
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_bin_op(
      thread frag_type& inp_vals,
      thread T* row_vals) {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }
};

template <
    typename T,
    int kTileRows_,
    int kTileCols_,
    class MMAFrag_ = BaseMMAFrag<T, 8, 8>>
struct MMATile {
  using MMAFrag_t = MMAFrag_;
  using elem_type = T;
  PREFILL_CONST int kFragRows = MMAFrag_t::kFragRows;
  PREFILL_CONST int kFragCols = MMAFrag_t::kFragCols;
  PREFILL_CONST int kElemsPerFrag = MMAFrag_t::kElemsPerFrag;

  PREFILL_CONST int kTileRows = kTileRows_;
  PREFILL_CONST int kTileCols = kTileCols_;

  PREFILL_CONST int kRows = kTileRows * kFragRows;
  PREFILL_CONST int kCols = kTileCols * kFragCols;

  PREFILL_CONST int kNumFrags = kTileRows * kTileCols;
  PREFILL_CONST int kElemsPerTile = kNumFrags * kElemsPerFrag;

  PREFILL_CONST int kRowsPerThread = kTileRows * MMAFrag_t::kElemRows;
  PREFILL_CONST int kColsPerThread = kTileCols * MMAFrag_t::kElemCols;

  typedef typename MMAFrag_t::mat_type mat_type;
  typedef typename MMAFrag_t::frag_type frag_type;

  frag_type val_frags[kNumFrags];

  METAL_FUNC MMATile() thread {}

  METAL_FUNC constexpr void clear() thread {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) thread {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const thread {
    return val_frags[i * kTileCols + j];
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread T vals[kRowsPerThread]) const thread {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vals[i * MMAFrag_t::kElemRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread T vals[kRowsPerThread]) thread {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vals[i * MMAFrag_t::kElemRows]);
      }
    }
  }

  template <typename U, int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) thread {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load(
            frag_at(i, j),
            &(
                src[(i * kFragRows) * w_x * str_x +
                    (j * kFragCols) * w_y * str_y]),
            Int<str_x>{},
            Int<str_y>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store(device U* dst, const int ld) const thread {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store(
            frag_at(i, j),
            &(dst[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]),
            ld,
            Int<1>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store_safe(
      device U* dst,
      const int ld,
      const short2 dst_tile_dims) const thread {
    PREFILL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      PREFILL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store_safe(
            frag_at(i, j),
            dst,
            ld,
            Int<1>{},
            dst_tile_dims.y,
            dst_tile_dims.x,
            (i * kFragRows) * w_x,
            (j * kFragCols) * w_y);
      }
    }
  }
};

template <
    typename Dtype,
    typename Atype,
    typename Btype,
    typename Ctype,
    int M,
    int N,
    int K,
    class MMAFragD,
    class MMAFragA,
    class MMAFragB,
    class MMAFragC>
METAL_FUNC void tile_matmad(
    thread MMATile<Dtype, M, N, MMAFragD>& D,
    thread MMATile<Atype, M, K, MMAFragA>& A,
    thread MMATile<Btype, K, N, MMAFragB>& B,
    thread MMATile<Ctype, M, N, MMAFragC>& C) {
  PREFILL_PRAGMA_UNROLL
  for (short m = 0; m < M; ++m) {
    PREFILL_PRAGMA_UNROLL
    for (short n = 0; n < N; ++n) {
      short m_serp = m;
      short n_serp = (m % 2) ? (N - 1 - n) : n;

      PREFILL_PRAGMA_UNROLL
      for (short k = 0; k < K; ++k) {
        MMAFragD::mma(
            D.frag_at(m_serp, n_serp),
            A.frag_at(m_serp, k),
            B.frag_at(k, n_serp),
            C.frag_at(m_serp, n_serp));
      }
    }
  }
}


/* PyTorch computes these diagonal routing statistics with tensor operations.
 * Keeping them on Metal avoids any readback between H3's QKV projection and
 * attention. K statistics are shared by every query block in one head. */
kernel void h3_sol_k_stats_bf16(
    device const bfloat *centroids [[buffer(0)]],
    device float *means [[buffer(1)]],
    device float *variances [[buffer(2)]],
    constant uint &blocks [[buffer(3)]],
    constant uint &heads [[buffer(4)]],
    uint dimension [[thread_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]]) {
    if (head >= heads || dimension >= 128) return;
    float total = 0.0f;
    float total_squared = 0.0f;
    for (uint block = 0; block < blocks; block++) {
        float value = float(centroids[(head * blocks + block) * 128 + dimension]);
        total += value;
        total_squared = fma(value, value, total_squared);
    }
    float mean = total / float(blocks);
    means[head * 128 + dimension] = mean;
    variances[head * 128 + dimension] =
        max(total_squared / float(blocks) - mean * mean, 0.0f);
}

kernel void h3_sol_thresholds_bf16(
    device const float *query_centroids [[buffer(0)]],
    device const float *key_means [[buffer(1)]],
    device const float *key_variances [[buffer(2)]],
    device float *thresholds [[buffer(3)]],
    constant uint &blocks [[buffer(4)]],
    constant uint &heads [[buffer(5)]],
    constant float &scale [[buffer(6)]],
    constant float &tau [[buffer(7)]],
    uint dimension [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    uint query_block = group % blocks;
    uint head = group / blocks;
    if (head >= heads || dimension >= 128) return;
    threadgroup float partial_mean[128];
    threadgroup float partial_variance[128];
    float query = query_centroids[(head * blocks + query_block) * 128 + dimension];
    partial_mean[dimension] = query * key_means[head * 128 + dimension];
    partial_variance[dimension] =
        query * query * key_variances[head * 128 + dimension];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64; stride; stride >>= 1) {
        if (dimension < stride) {
            partial_mean[dimension] += partial_mean[dimension + stride];
            partial_variance[dimension] += partial_variance[dimension + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (dimension == 0) {
        float log2_scale = scale * 1.4426950408889634f;
        float variance = max(partial_variance[0], 0.0f) *
                         log2_scale * log2_scale;
        thresholds[head * blocks + query_block] =
            partial_mean[0] * log2_scale +
            tau * sqrt(variance + 1.0e-6f);
    }
}

#define instantiate_kernel(name, function, ...) \
    template [[host_name(name)]] [[kernel]] \
    decltype(function<__VA_ARGS__>) function<__VA_ARGS__>;


template <typename T, int BD>
[[kernel, max_total_threads_per_threadgroup(128)]] void sol_reduce_summaries(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device float* QC [[buffer(3)]],
    device T* KC [[buffer(4)]],
    device T* VC [[buffer(5)]],
    constant uint& tokens [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& blocks [[buffer(8)]],
    constant ulong& sq_b [[buffer(9)]],
    constant ulong& sq_t [[buffer(10)]],
    constant ulong& sq_h [[buffer(11)]],
    constant ulong& sk_b [[buffer(12)]],
    constant ulong& sk_t [[buffer(13)]],
    constant ulong& sk_h [[buffer(14)]],
    constant ulong& sv_b [[buffer(15)]],
    constant ulong& sv_t [[buffer(16)]],
    constant ulong& sv_h [[buffer(17)]],
    uint dim [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  const uint block = group % blocks;
  const uint batch_head = group / blocks;
  const uint head = batch_head % heads;
  const uint batch = batch_head / heads;
  const uint token_start = block * 64;
  const uint block_length = min(64u, tokens - token_start);

  if (dim >= BD) {
    return;
  }

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  float v_sum = 0.0f;
  for (uint offset = 0; offset < block_length; ++offset) {
    const uint token = token_start + offset;
    q_sum += float(Q[batch * sq_b + token * sq_t + head * sq_h + dim]);
    k_sum += float(K[batch * sk_b + token * sk_t + head * sk_h + dim]);
    v_sum += float(V[batch * sv_b + token * sv_t + head * sv_h + dim]);
  }

  const ulong summary_offset =
      ((ulong(batch) * heads + head) * blocks + block) * BD + dim;
  QC[summary_offset] = q_sum / float(block_length);
  KC[summary_offset] = T(k_sum / float(block_length));
  VC[summary_offset] = T(v_sum);
}

template <bool WEIGHT_DENOMINATOR, typename T, typename AccumType,
      typename STile, typename OTile, int LDV>
METAL_FUNC void sol_accumulate_tile(
    thread STile& scores,
    thread OTile& output,
    threadgroup T* values,
    short values_offset,
    thread AccumType* maximum,
  thread AccumType* denominator,
  uint key_start,
  short key_column,
  uint blocks,
  uint tokens) {
  constexpr short rows_per_thread = STile::kRowsPerThread;
  constexpr short value_tiles = OTile::kTileCols;
  constexpr short key_tiles = STile::kTileCols;
  using MMAFrag = typename STile::MMAFrag_t;
  MMATile<AccumType, 1, 1, MMAFrag> value_tile;

  AccumType next_maximum[rows_per_thread];
  AccumType correction[rows_per_thread];
  PREFILL_PRAGMA_UNROLL
  for (short row = 0; row < rows_per_thread; ++row) {
    next_maximum[row] = maximum[row];
  }
  scores.template row_reduce<MaxOp>(next_maximum);
  scores.template row_bin_op<ExpSubOp>(next_maximum);
  PREFILL_PRAGMA_UNROLL
  for (short row = 0; row < rows_per_thread; ++row) {
    correction[row] = next_maximum[row] == -INFINITY
        ? AccumType(1)
        : fast::exp2(maximum[row] - next_maximum[row]);
    maximum[row] = next_maximum[row];
  }

  output.template row_bin_op<MulOp>(correction);

  threadgroup_barrier(mem_flags::mem_threadgroup);
  PREFILL_PRAGMA_UNROLL
  for (short value_tile_index = 0; value_tile_index < value_tiles; ++value_tile_index) {
    PREFILL_PRAGMA_UNROLL
    for (short key_tile_index = 0; key_tile_index < key_tiles; ++key_tile_index) {
      const short key_offset = key_tile_index * 8;
      const short dim_offset = value_tile_index * 8;
      value_tile.template load<T, 1, 1, LDV, 1>(
          &values[values_offset + key_offset * LDV + dim_offset]);
      simdgroup_barrier(mem_flags::mem_none);
      MMAFrag::mma(
          output.frag_at(0, value_tile_index),
          scores.frag_at(0, key_tile_index),
          value_tile.frag_at(0, 0),
          output.frag_at(0, value_tile_index));
    }
  }

  if (WEIGHT_DENOMINATOR) {
    PREFILL_PRAGMA_UNROLL
    for (short key_tile_index = 0; key_tile_index < key_tiles; ++key_tile_index) {
      PREFILL_PRAGMA_UNROLL
      for (short element = 0; element < MMAFrag::kElemCols; ++element) {
        const uint block = key_start + key_column + key_tile_index * 8 + element;
        const uint block_length = block < blocks
            ? min(64u, tokens - block * 64)
            : 0u;
        scores.frag_at(0, key_tile_index)[element] *= AccumType(block_length);
      }
    }
  }

  AccumType tile_sum[rows_per_thread] = {0};
  scores.template row_reduce<SumOp>(tile_sum);
  PREFILL_PRAGMA_UNROLL
  for (short row = 0; row < rows_per_thread; ++row) {
    denominator[row] = denominator[row] * correction[row] + tile_sum[row];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <typename T, int BD, int LDK>
METAL_FUNC void sol_route_tile(
  const device float* route_query,
    threadgroup T* keys,
    threadgroup uchar* route_flags,
    uint summary_start,
    uint blocks,
    uint route_query_block,
    float route_threshold,
    float score_scale,
    uint sink_start,
    uint sink_end,
    uint sink_q_start,
    uint sink_q_end,
    uint lane,
    uint simdgroup) {
  constexpr short TD = BD / 8;
  using AccumType = float;
  using MMAFrag = BaseMMAFrag<AccumType, 8, 8>;
  MMATile<AccumType, 1, 1, MMAFrag> route_query_fragment;
  MMATile<AccumType, 1, 2, MMAFrag> route_key_fragment;
  MMATile<AccumType, 1, 2, MMAFrag> route_score_tile;
  route_score_tile.clear();

  const short2 coordinate = MMAFrag::get_coord(lane);
  const short row = coordinate.y;
  const short column = coordinate.x;
  const short key_offset = row * LDK + column;
  if (simdgroup < 4) {
    PREFILL_PRAGMA_UNROLL
    for (short dim_tile = 0; dim_tile < TD; ++dim_tile) {
        MMAFrag::load(
          route_query_fragment.frag_at(0, 0),
          &route_query[column + dim_tile * 8], Int<0>{}, Int<1>{});
      route_key_fragment.template load<T, 1, 1, LDK, 1>(
          &keys[key_offset + simdgroup * 16 + dim_tile * 8 * LDK]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(
          route_score_tile, route_query_fragment, route_key_fragment,
          route_score_tile);
    }
    PREFILL_PRAGMA_UNROLL
    for (short key_tile_index = 0; key_tile_index < 2; ++key_tile_index) {
      PREFILL_PRAGMA_UNROLL
      for (short element = 0; element < MMAFrag::kElemCols; ++element) {
        route_score_tile.frag_at(0, key_tile_index)[element] *= score_scale;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool query_in_sink =
      route_query_block >= sink_q_start && route_query_block < sink_q_end;
  if (simdgroup < 4 && row == 0) {
    PREFILL_PRAGMA_UNROLL
    for (short key_tile_index = 0; key_tile_index < 2; ++key_tile_index) {
      PREFILL_PRAGMA_UNROLL
      for (short element = 0; element < MMAFrag::kElemCols; ++element) {
        const uint local_block =
            simdgroup * 16 + column + key_tile_index * 8 + element;
        const uint block = summary_start + local_block;
        const bool neighbor =
            abs(int(route_query_block) - int(block)) <= 1;
        const bool sink = block >= sink_start && block < sink_end;
        route_flags[local_block] = block < blocks && (
            query_in_sink || neighbor || sink ||
            route_score_tile.frag_at(0, key_tile_index)[element] > route_threshold
        );
      }
    }
  }
}

template <typename T, int BD>
[[kernel, max_total_threads_per_threadgroup(128)]] void sol_route_mask_debug(
    const device float* QC [[buffer(0)]],
    const device T* KC [[buffer(1)]],
    const device float* thresholds [[buffer(2)]],
    device uchar* routes [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& blocks [[buffer(6)]],
    constant uint& sink_start [[buffer(7)]],
    constant uint& sink_end [[buffer(8)]],
    constant uint& sink_q_start [[buffer(9)]],
    constant uint& sink_q_end [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  constexpr short BK = 64;
  constexpr short LDK = BK + 8;
  using KLoader = BlockLoaderT<T, BK, BD, 1, LDK, 0, 128>;

  const uint route_query_block = group % blocks;
  const uint batch_head = group / blocks;
  const uint head = batch_head % heads;
  const uint batch = batch_head / heads;
  QC += ((ulong(batch) * heads + head) * blocks + route_query_block) * BD;
  KC += (ulong(batch) * heads + head) * blocks * BD;
  thresholds += (ulong(batch) * heads + head) * blocks + route_query_block;
  routes += ((ulong(batch) * heads + head) * blocks + route_query_block) * blocks;

  threadgroup T keys[BK * LDK];
  threadgroup uchar route_flags[BK];
  const uint thread_index = simdgroup * 32 + lane;

  for (uint summary_start = 0; summary_start < blocks; summary_start += BK) {
    const uint summary_count = min(uint(BK), blocks - summary_start);
    KLoader key_loader(KC + ulong(summary_start) * BD, BD, keys, simdgroup, lane);
    key_loader.load_safe(short2(BD, summary_count));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sol_route_tile<T, BD, LDK>(
      QC, keys, route_flags, summary_start, blocks,
        route_query_block, thresholds[0], scale * 1.44269504089f,
        sink_start, sink_end, sink_q_start, sink_q_end, lane, simdgroup);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < summary_count) {
      routes[summary_start + thread_index] = route_flags[thread_index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename T, int BD, int BQ, int TGP_SIZE>
[[kernel, max_total_threads_per_threadgroup(256)]] void sol_attn_tiled(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
  const device float* QC [[buffer(3)]],
  const device T* KC [[buffer(4)]],
  const device T* VC [[buffer(5)]],
  const device float* thresholds [[buffer(6)]],
  device T* O [[buffer(7)]],
  constant float& scale [[buffer(8)]],
  constant uint& tokens [[buffer(9)]],
  constant uint& heads [[buffer(10)]],
  constant uint& blocks [[buffer(11)]],
  constant uint& sink_start [[buffer(12)]],
  constant uint& sink_end [[buffer(13)]],
  constant uint& sink_q_start [[buffer(14)]],
  constant uint& sink_q_end [[buffer(15)]],
  constant ulong& sq_b [[buffer(16)]],
  constant ulong& sq_t [[buffer(17)]],
  constant ulong& sq_h [[buffer(18)]],
  constant ulong& sk_b [[buffer(19)]],
  constant ulong& sk_t [[buffer(20)]],
  constant ulong& sk_h [[buffer(21)]],
  constant ulong& sv_b [[buffer(22)]],
  constant ulong& sv_t [[buffer(23)]],
  constant ulong& sv_h [[buffer(24)]],
  constant ulong& so_b [[buffer(25)]],
  constant ulong& so_t [[buffer(26)]],
  constant ulong& so_h [[buffer(27)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  constexpr short BK = 64;
  constexpr short LDQ = BQ == 64 ? BD : BD + 8;
  constexpr short LDK = BQ == 64 ? BK : BK + 8;
  constexpr short LDV = BQ == 64 ? BD : BD + 8;
    constexpr short shared_count = LDK * BD > BK * LDV
      ? LDK * BD
      : BK * LDV;
  constexpr short TD = BD / 8;
  constexpr short TK = BK / 8;
  using AccumType = float;
  using MMAFrag = BaseMMAFrag<AccumType, 8, 8>;
  using QLoader = BlockLoaderT<T, BQ, BD, LDQ, 1, 1, TGP_SIZE>;
  using KLoader = BlockLoaderT<T, BK, BD, 1, LDK, 0, TGP_SIZE>;
  using VLoader = BlockLoaderT<T, BK, BD, LDV, 1, 0, TGP_SIZE>;

  const uint query_tiles = (tokens + BQ - 1) / BQ;
  const uint query_tile = group % query_tiles;
  const uint batch_head = group / query_tiles;
  const uint head = batch_head % heads;
  const uint batch = batch_head / heads;
  const uint query_start = query_tile * BQ;
  const uint query_size = min(uint(BQ), tokens - query_start);
  const uint route_query_block = query_start / 64;

  Q += batch * sq_b + query_start * sq_t + head * sq_h;
  K += batch * sk_b + head * sk_h;
  V += batch * sv_b + head * sv_h;
  QC += ((ulong(batch) * heads + head) * blocks + route_query_block) * BD;
  KC += (ulong(batch) * heads + head) * blocks * BD;
  VC += (ulong(batch) * heads + head) * blocks * BD;
  thresholds += (ulong(batch) * heads + head) * blocks + route_query_block;
  O += ulong(batch) * so_b + ulong(query_start) * so_t + ulong(head) * so_h;

  threadgroup T query_shared[BQ * LDQ];
  threadgroup T key_value_shared[shared_count];
  threadgroup T* keys = key_value_shared;
  threadgroup T* values = key_value_shared;
  threadgroup uchar* route_flags =
      reinterpret_cast<threadgroup uchar*>(key_value_shared);

  QLoader query_loader(Q, int(sq_t), query_shared, simdgroup, lane);
  if (query_size < BQ) {
    query_loader.load_safe(short2(BD, query_size));
  } else {
    query_loader.load_unsafe();
  }

  MMATile<AccumType, 1, 1, MMAFrag> query_tile_fragment;
  MMATile<AccumType, 1, TK, MMAFrag> key_tile_fragment;
  MMATile<AccumType, 1, TK, MMAFrag> score_tile;
  MMATile<AccumType, 1, TD, MMAFrag> output_tile;
  output_tile.clear();

  const short2 coordinate = MMAFrag::get_coord(lane);
  const short row = coordinate.y;
  const short column = coordinate.x;
  const short query_row = 8 * simdgroup;
  const short query_offset = (query_row + row) * LDQ + column;
  const short key_offset = row * LDK + column;
  const short value_offset = row * LDV + column;
  constexpr short rows_per_thread = decltype(score_tile)::kRowsPerThread;
  const AccumType score_scale = AccumType(scale * 1.44269504089f);
  const uint thread_index = simdgroup * 32 + lane;
  const float route_threshold = thresholds[0];
  AccumType maximum[rows_per_thread];
  AccumType denominator[rows_per_thread] = {0};
  PREFILL_PRAGMA_UNROLL
  for (short index = 0; index < rows_per_thread; ++index) {
    maximum[index] = -INFINITY;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Unrouted blocks use K centroids and V sums, matching the Triton forward.
  for (uint summary_start = 0; summary_start < blocks; summary_start += BK) {
    const uint summary_count = min(uint(BK), blocks - summary_start);
    KLoader key_loader(KC + ulong(summary_start) * BD, BD, keys, simdgroup, lane);
    VLoader value_loader(VC + ulong(summary_start) * BD, BD, values, simdgroup, lane);
    key_loader.load_safe(short2(BD, summary_count));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    score_tile.clear();
    PREFILL_PRAGMA_UNROLL
    for (short dim_tile = 0; dim_tile < TD; ++dim_tile) {
      query_tile_fragment.template load<T, 1, 1, LDQ, 1>(
          &query_shared[query_offset + dim_tile * 8]);
      key_tile_fragment.template load<T, 1, 1, LDK, 1>(
          &keys[key_offset + dim_tile * 8 * LDK]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(score_tile, query_tile_fragment, key_tile_fragment, score_tile);
    }
    PREFILL_PRAGMA_UNROLL
    for (short key_tile_index = 0; key_tile_index < TK; ++key_tile_index) {
      PREFILL_PRAGMA_UNROLL
      for (short element = 0; element < MMAFrag::kElemsPerFrag; ++element) {
        score_tile.frag_at(0, key_tile_index)[element] *= score_scale;
      }
    }

    // Route scores and summary QK both consume KC. Reuse KC storage for flags
    // only after both calculations have finished, then retain flags in a
    // register mask before V/exact loads overwrite the shared buffer.
    sol_route_tile<T, BD, LDK>(
      QC, keys, route_flags, summary_start, blocks,
        route_query_block, route_threshold, score_scale,
        sink_start, sink_end, sink_q_start, sink_q_end, lane, simdgroup);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong route_mask = 0;
    for (uint local_block = 0; local_block < summary_count; ++local_block) {
      route_mask |= ulong(route_flags[local_block] != 0) << local_block;
    }

    PREFILL_PRAGMA_UNROLL
    for (short key_tile_index = 0; key_tile_index < TK; ++key_tile_index) {
      const uint block_column = summary_start + column + key_tile_index * 8;
      PREFILL_PRAGMA_UNROLL
      for (short element = 0; element < MMAFrag::kElemCols; ++element) {
        const uint block = block_column + element;
        if (block >= blocks || ((route_mask >> (block - summary_start)) & 1ul)) {
          score_tile.frag_at(0, key_tile_index)[element] = -INFINITY;
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    value_loader.load_safe(short2(BD, summary_count));
    sol_accumulate_tile<true, T, AccumType,
              decltype(score_tile), decltype(output_tile), LDV>(
      score_tile, output_tile, values, value_offset, maximum, denominator,
      summary_start, column, blocks, tokens);

    // Consume exact blocks while this summary group's route mask is in registers.
    for (uint local_key_block = 0; local_key_block < summary_count; ++local_key_block) {
      if (((route_mask >> local_key_block) & 1ul) == 0) {
        continue;
      }
      const uint key_start = (summary_start + local_key_block) * BK;
      const uint key_count = min(uint(BK), tokens - key_start);
      KLoader exact_key_loader(
          K + ulong(key_start) * sk_t, int(sk_t), keys, simdgroup, lane);
      VLoader exact_value_loader(
          V + ulong(key_start) * sv_t, int(sv_t), values, simdgroup, lane);
      if (key_count < BK) {
        exact_key_loader.load_safe(short2(BD, key_count));
      } else {
        exact_key_loader.load_unsafe();
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      score_tile.clear();
      PREFILL_PRAGMA_UNROLL
      for (short dim_tile = 0; dim_tile < TD; ++dim_tile) {
        query_tile_fragment.template load<T, 1, 1, LDQ, 1>(
            &query_shared[query_offset + dim_tile * 8]);
        key_tile_fragment.template load<T, 1, 1, LDK, 1>(
            &keys[key_offset + dim_tile * 8 * LDK]);
        simdgroup_barrier(mem_flags::mem_none);
        tile_matmad(score_tile, query_tile_fragment, key_tile_fragment, score_tile);
      }
      PREFILL_PRAGMA_UNROLL
      for (short key_tile_index = 0; key_tile_index < TK; ++key_tile_index) {
        PREFILL_PRAGMA_UNROLL
        for (short element = 0; element < MMAFrag::kElemCols; ++element) {
          score_tile.frag_at(0, key_tile_index)[element] *= score_scale;
        }
      }
      if (key_count < BK) {
        PREFILL_PRAGMA_UNROLL
        for (short key_tile_index = 0; key_tile_index < TK; ++key_tile_index) {
          const short key_column = column + key_tile_index * 8;
          PREFILL_PRAGMA_UNROLL
          for (short element = 0; element < MMAFrag::kElemCols; ++element) {
            if (key_column + element >= key_count) {
              score_tile.frag_at(0, key_tile_index)[element] = -INFINITY;
            }
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (key_count < BK) {
        exact_value_loader.load_safe(short2(BD, key_count));
      } else {
        exact_value_loader.load_unsafe();
      }
      sol_accumulate_tile<false, T, AccumType,
                          decltype(score_tile), decltype(output_tile), LDV>(
          score_tile, output_tile, values, value_offset, maximum, denominator,
          0, 0, blocks, tokens);
    }
  }

  PREFILL_PRAGMA_UNROLL
  for (short index = 0; index < rows_per_thread; ++index) {
    if (maximum[index] == -INFINITY) {
      denominator[index] = AccumType(1);
    }
  }
  output_tile.template row_bin_op<DivOp>(denominator);
  device T* output_ptr = O + ulong(query_row + row) * so_t + column;
  if (query_row + row < query_size) {
    output_tile.template store_safe<T, 1, 1>(
        output_ptr, so_t, short2(BD - column, query_size - (query_row + row)));
  }
}



instantiate_kernel("h3_sol_attn_tiled_bf16_d128_bq64",
                   sol_attn_tiled, bfloat, 128, 64, 256)
instantiate_kernel("h3_sol_reduce_summaries_bf16_d128",
                   sol_reduce_summaries, bfloat, 128)
instantiate_kernel("h3_sol_route_mask_debug_bf16_d128",
                   sol_route_mask_debug, bfloat, 128)
