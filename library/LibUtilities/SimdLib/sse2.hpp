///////////////////////////////////////////////////////////////////////////////
//
// File: sse2.hpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description: Vector type using sse2 extension.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_LIBUTILITES_SIMDLIB_SSE2_H
#define NEKTAR_LIB_LIBUTILITES_SIMDLIB_SSE2_H

#if defined(__x86_64__)
#include <immintrin.h>
#if defined(__INTEL_COMPILER) && !defined(TINYSIMD_HAS_SVML)
#define TINYSIMD_HAS_SVML
#endif
#endif
#include "allocator.hpp"
#include "traits.hpp"
#include <cmath>
#include <cstdint>
#include <vector>

namespace tinysimd::abi
{

template <typename scalarType> struct simd64
{
    using type = void;
};

} // namespace tinysimd::abi

#if defined(__SSE2__) && defined(NEKTAR_ENABLE_SIMD_SSE2)

namespace tinysimd
{

// forward declaration of concrete types
template <typename T> struct simd64Int2;

namespace abi
{

template <> struct simd64<std::int32_t>
{
    using type = simd64Int2<std::int32_t>;
};
template <> struct simd64<std::uint32_t>
{
    using type = simd64Int2<std::uint32_t>;
};

} // namespace abi

// concrete types
template <typename T> struct simd64Int2
{
    static_assert(std::is_integral_v<T> && sizeof(T) == 4,
                  "4 bytes Integral required.");

    static constexpr unsigned int width     = 2;
    static constexpr unsigned int alignment = 8;

    using scalarType  = T;
    using vectorType  = scalarType[width];
    using scalarArray = scalarType[width];

    // storage
    vectorType _data;

    // ctors
    inline simd64Int2()                      = default;
    inline simd64Int2(const simd64Int2 &rhs) = default;
    inline simd64Int2(const vectorType &rhs) : _data(rhs)
    {
    }
    inline simd64Int2(const scalarType rhs)
    {
        _data[0] = rhs;
        _data[1] = rhs;
    }

    // store
    inline void store(scalarType *p) const
    {
        p[0] = _data[0];
        p[1] = _data[1];
    }

    template <class flag,
              typename std::enable_if<is_requiring_alignment_v<flag> &&
                                          !is_streaming_v<flag>,
                                      bool>::type = 0>
    inline void store(scalarType *p, flag) const
    {
        p[0] = _data[0];
        p[1] = _data[1];
    }

    template <class flag, typename std::enable_if<
                              !is_requiring_alignment_v<flag>, bool>::type = 0>
    inline void store(scalarType *p, flag) const
    {
        p[0] = _data[0];
        p[1] = _data[1];
    }

    inline void load(const scalarType *p)
    {
        _data[0] = p[0];
        _data[1] = p[1];
    }

    template <class flag,
              typename std::enable_if<is_requiring_alignment_v<flag> &&
                                          !is_streaming_v<flag>,
                                      bool>::type = 0>
    inline void load(const scalarType *p, flag)
    {
        _data[0] = p[0];
        _data[1] = p[1];
    }

    template <class flag, typename std::enable_if<
                              !is_requiring_alignment_v<flag>, bool>::type = 0>
    inline void load(const scalarType *p, flag)
    {
        _data[0] = p[0];
        _data[1] = p[1];
    }

    // gather/scatter with sse2
    inline void gather(scalarType const *p, const simd64Int2<T> &indices)
    {
        _data[0] = p[indices[0]];
        _data[1] = p[indices[1]];
    }

    inline void scatter(scalarType *out, const simd64Int2<T> &indices) const
    {
        out[indices[0]] = _data[0];
        out[indices[1]] = _data[1];
    }

    inline void broadcast(const scalarType rhs)
    {
        _data[0] = rhs[0];
        _data[1] = rhs[0];
    }

    // subscript
    // subscript operators are convienient but expensive
    // should not be used in optimized kernels
    inline scalarType operator[](size_t i) const
    {
        return _data[i];
    }
};
} // namespace tinysimd

#endif

namespace tinysimd::abi
{

template <typename scalarType, int width = 0> struct sse2
{
    using type = void;
};

} // namespace tinysimd::abi

#if defined(__SSE2__) && defined(NEKTAR_ENABLE_SIMD_SSE2)

namespace tinysimd
{

// forward declaration of concrete types
template <typename T> struct sse2Long2;
template <typename T> struct sse2Int4;
struct sse2Double2;
struct sse2Float4;
struct sse2Mask2;
struct sse2Mask4;

namespace abi
{

// mapping between abstract types and concrete floating point types
template <> struct sse2<double>
{
    using type = sse2Double2;
};
template <> struct sse2<float>
{
    using type = sse2Float4;
};
// generic index mapping
// assumes index type width same as floating point type
template <> struct sse2<std::int64_t>
{
    using type = sse2Long2<std::int64_t>;
};
template <> struct sse2<std::uint64_t>
{
    using type = sse2Long2<std::uint64_t>;
};
#if defined(__APPLE__)
template <> struct sse2<std::size_t>
{
    using type = sse2Long2<std::size_t>;
};
#endif
template <> struct sse2<std::int32_t>
{
    using type = sse2Int4<std::int32_t>;
};
template <> struct sse2<std::uint32_t>
{
    using type = sse2Int4<std::uint32_t>;
};
// specialized index mapping
template <> struct sse2<std::int64_t, 2>
{
    using type = sse2Long2<std::int64_t>;
};
template <> struct sse2<std::uint64_t, 2>
{
    using type = sse2Long2<std::uint64_t>;
};
#if defined(__APPLE__)
template <> struct sse2<std::size_t, 2>
{
    using type = sse2Long2<std::size_t>;
};
#endif
template <> struct sse2<std::int32_t, 2>
{
    using type = simd64Int2<std::int32_t>;
};
template <> struct sse2<std::uint32_t, 2>
{
    using type = simd64Int2<std::uint32_t>;
};
template <> struct sse2<std::int32_t, 4>
{
    using type = sse2Int4<std::int32_t>;
};
template <> struct sse2<std::uint32_t, 4>
{
    using type = sse2Int4<std::uint32_t>;
};
// bool mapping
template <> struct sse2<bool, 2>
{
    using type = sse2Mask2;
};
template <> struct sse2<bool, 4>
{
    using type = sse2Mask4;
};

} // namespace abi

// concrete types
template <typename T> struct sse2Int4
{
    static_assert(std::is_integral_v<T> && sizeof(T) == 4,
                  "4 bytes Integral required.");

    static constexpr unsigned int width     = 4;
    static constexpr unsigned int alignment = 16;

    using scalarType  = T;
    using vectorType  = __m128i;
    using scalarArray = scalarType[width];

    // storage
    vectorType _data;

    // ctors
    inline sse2Int4()                    = default;
    inline sse2Int4(const sse2Int4 &rhs) = default;
    inline sse2Int4(const vectorType &rhs) : _data(rhs)
    {
    }
    inline sse2Int4(const scalarType rhs)
    {
        _data = _mm_set1_epi32(rhs);
    }
    explicit inline sse2Int4(scalarArray &rhs)
    {
        _data = _mm_load_si128(reinterpret_cast<vectorType *>(rhs));
    }

    // copy assignment
    inline sse2Int4 &operator=(const sse2Int4 &) = default;

    // store
    inline void store(scalarType *p) const
    {
        _mm_store_si128(reinterpret_cast<vectorType *>(p), _data);
    }

    template <class flag,
              typename std::enable_if<is_requiring_alignment_v<flag> &&
                                          !is_streaming_v<flag>,
                                      bool>::type = 0>
    inline void store(scalarType *p, flag) const
    {
        _mm_store_si128(reinterpret_cast<vectorType *>(p), _data);
    }

    template <class flag, typename std::enable_if<
                              !is_requiring_alignment_v<flag>, bool>::type = 0>
    inline void store(scalarType *p, flag) const
    {
        _mm_storeu_si128(reinterpret_cast<vectorType *>(p), _data);
    }

    inline void load(const scalarType *p)
    {
        _data = _mm_load_si128(reinterpret_cast<const vectorType *>(p));
    }

    template <class flag,
              typename std::enable_if<is_requiring_alignment_v<flag> &&
                                          !is_streaming_v<flag>,
                                      bool>::type = 0>
    inline void load(const scalarType *p, flag)
    {
        _data = _mm_load_si128(reinterpret_cast<const vectorType *>(p));
    }

    template <class flag, typename std::enable_if<
                              !is_requiring_alignment_v<flag>, bool>::type = 0>
    inline void load(const scalarType *p, flag)
    {
        _data = _mm_loadu_si128(reinterpret_cast<const vectorType *>(p));
    }

    inline void broadcast(const scalarType rhs)
    {
        _data = _mm_set1_epi32(rhs);
    }

    /*// gather/scatter with sse2
    inline void gather(scalarType const *p, const sse2Int4<T> &indices)
    {
        _data = _mm_i32gather_epi32(p, indices._data, 8);
    }

    inline void scatter(scalarType *out, const sse2Int4<T> &indices) const
    {
        // no scatter intrinsics for sse2
        alignas(alignment) scalarArray tmp;
        _mm_store_epi32(tmp, _data);

        out[_mm_extract_epi32(indices._data, 0)] = tmp[0]; // SSE4.1
        out[_mm_extract_epi32(indices._data, 1)] = tmp[1];
    }*/

    // subscript
    // subscript operators are convienient but expensive
    // should not be used in optimized kernels
    inline scalarType operator[](size_t i) const
    {
        alignas(alignment) scalarArray tmp;
        store(tmp, is_aligned);
        return tmp[i];
    }

    inline scalarType &operator[](size_t i)
    {
        scalarType *tmp = reinterpret_cast<scalarType *>(&_data);
        return tmp[i];
    }
};

template <typename T>
inline sse2Int4<T> operator+(sse2Int4<T> lhs, sse2Int4<T> rhs)
{
    return _mm_add_epi32(lhs._data, rhs._data);
}

template <typename T, typename U,
          typename = typename std::enable_if<std::is_arithmetic_v<U>>::type>
inline sse2Int4<T> operator+(sse2Int4<T> lhs, U rhs)
{
    return _mm_add_epi32(lhs._data, _mm_set1_epi32(rhs));
}

////////////////////////////////////////////////////////////////////////////////

template <typename T> struct sse2Long2
{
    static_assert(std::is_integral_v<T> && sizeof(T) == 8,
                  "8 bytes Integral required.");

    static constexpr unsigned int width     = 2;
    static constexpr unsigned int alignment = 16;

    using scalarType  = T;
    using vectorType  = __m128i;
    using scalarArray = scalarType[width];

    // storage
    vectorType _data;

    // ctors
    inline sse2Long2()                     = default;
    inline sse2Long2(const sse2Long2 &rhs) = default;
    inline sse2Long2(const vectorType &rhs) : _data(rhs)
    {
    }
    inline sse2Long2(const scalarType rhs)
    {
        _data = _mm_set1_epi64x(rhs);
    }
    explicit inline sse2Long2(scalarArray &rhs)
    {
        _data = _mm_load_si128(reinterpret_cast<vectorType *>(rhs));
    }

    // copy assignment
    inline sse2Long2 &operator=(const sse2Long2 &) = default;

    // store
    inline void store(scalarType *p) const
    {
        _mm_store_si128(reinterpret_cast<vectorType *>(p), _data);
    }

    template <class flag,
              typename std::enable_if<is_requiring_alignment_v<flag> &&
                                          !is_streaming_v<flag>,
                                      bool>::type = 0>
    inline void store(scalarType *p, flag) const
    {
        _mm_store_si128(reinterpret_cast<vectorType *>(p), _data);
    }

    template <class flag, typename std::enable_if<
                              !is_requiring_alignment_v<flag>, bool>::type = 0>
    inline void store(scalarType *p, flag) const
    {
        _mm_storeu_si128(reinterpret_cast<vectorType *>(p), _data);
    }

    inline void load(const scalarType *p)
    {
        _data = _mm_load_si128(reinterpret_cast<const vectorType *>(p));
    }

    template <class flag,
              typename std::enable_if<is_requiring_alignment_v<flag> &&
                                          !is_streaming_v<flag>,
                                      bool>::type = 0>
    inline void load(const scalarType *p, flag)
    {
        _data = _mm_load_si128(reinterpret_cast<const vectorType *>(p));
    }

    template <class flag, typename std::enable_if<
                              !is_requiring_alignment_v<flag>, bool>::type = 0>
    inline void load(const scalarType *p, flag)
    {
        _data = _mm_loadu_si128(reinterpret_cast<const vectorType *>(p));
    }

    inline void broadcast(const scalarType rhs)
    {
        _data = _mm_set1_epi64x(rhs);
    }

    // subscript
    // subscript operators are convienient but expensive
    // should not be used in optimized kernels
    inline scalarType operator[](size_t i) const
    {
        alignas(alignment) scalarArray tmp;
        store(tmp, is_aligned);
        return tmp[i];
    }

    inline scalarType &operator[](size_t i)
    {
        scalarType *tmp = reinterpret_cast<scalarType *>(&_data);
        return tmp[i];
    }
};

template <typename T>
inline sse2Long2<T> operator+(sse2Long2<T> lhs, sse2Long2<T> rhs)
{
    return _mm_add_epi64(lhs._data, rhs._data);
}

template <typename T, typename U,
          typename = typename std::enable_if<std::is_arithmetic_v<U>>::type>
inline sse2Long2<T> operator+(sse2Long2<T> lhs, U rhs)
{
    return _mm_add_epi64(lhs._data, _mm_set1_epi64x(rhs));
}

////////////////////////////////////////////////////////////////////////////////

struct sse2Double2
{
    static constexpr unsigned width     = 2;
    static constexpr unsigned alignment = 16;

    using scalarType      = double;
    using scalarIndexType = std::uint64_t;
    using vectorType      = __m128d;
    using scalarArray     = scalarType[width];

    // storage
    vectorType _data;

    // ctors
    inline sse2Double2()                       = default;
    inline sse2Double2(const sse2Double2 &rhs) = default;
    inline sse2Double2(const vectorType &rhs) : _data(rhs)
    {
    }
    inline sse2Double2(const scalarType rhs)
    {
        _data = _mm_set1_pd(rhs);
    }

    // copy assignment
    inline sse2Double2 &operator=(const sse2Double2 &) = default;

    // store
    inline void store(scalarType *p) const
    {
        _mm_store_pd(p, _data);
    }

    template <class flag,
              typename std::enable_if<is_requiring_alignment_v<flag> &&
                                          !is_streaming_v<flag>,
                                      bool>::type = 0>
    inline void store(scalarType *p, flag) const
    {
        _mm_store_pd(p, _data);
    }

    template <class flag, typename std::enable_if<
                              !is_requiring_alignment_v<flag>, bool>::type = 0>
    inline void store(scalarType *p, flag) const
    {
        _mm_storeu_pd(p, _data);
    }

    template <class flag,
              typename std::enable_if<is_streaming_v<flag>, bool>::type = 0>
    inline void store(scalarType *p, flag) const
    {
        _mm_stream_pd(p, _data);
    }

    // load packed
    inline void load(const scalarType *p)
    {
        _data = _mm_load_pd(p);
    }

    template <class flag, typename std::enable_if<
                              is_requiring_alignment_v<flag>, bool>::type = 0>
    inline void load(const scalarType *p, flag)
    {
        _data = _mm_load_pd(p);
    }

    template <class flag, typename std::enable_if<
                              !is_requiring_alignment_v<flag>, bool>::type = 0>
    inline void load(const scalarType *p, flag)
    {
        _data = _mm_loadu_pd(p);
    }

    // broadcast
    inline void broadcast(const scalarType rhs)
    {
        _data = _mm_set1_pd(rhs);
    }

    // gather/scatter with simd64
    template <typename T>
    inline void gather(scalarType const *p, const simd64Int2<T> &indices)
    {
        // no gather intrinsics for SSE2
        alignas(alignment) scalarArray tmp;
        tmp[0] = p[indices[0]];
        tmp[1] = p[indices[1]];
        _data  = _mm_load_pd(&tmp[0]);
    }

    template <typename T>
    inline void scatter(scalarType *out, const simd64Int2<T> &indices) const
    {
        // no scatter intrinsics for SSE2
        alignas(alignment) scalarArray tmp;
        _mm_store_pd(tmp, _data);

        out[indices[0]] = tmp[0]; // SSE4.1
        out[indices[1]] = tmp[1];
    }

    template <typename T>
    inline void gather(scalarType const *p, const sse2Long2<T> &indices)
    {
        // no gather intrinsics for SSE2
        alignas(alignment) scalarArray tmp;
        tmp[0] = p[_mm_extract_epi64(indices._data, 0)];
        tmp[1] = p[_mm_extract_epi64(indices._data, 1)];
        _data  = _mm_load_pd(&tmp[0]);
    }

    template <typename T>
    inline void scatter(scalarType *out, const sse2Long2<T> &indices) const
    {
        // no scatter intrinsics for SSE2
        alignas(alignment) scalarArray tmp;
        _mm_store_pd(tmp, _data);

        out[_mm_extract_epi64(indices._data, 0)] = tmp[0];
        out[_mm_extract_epi64(indices._data, 1)] = tmp[1];
    }

    // fma
    // this = this + a * b
    inline void fma(const sse2Double2 &a, const sse2Double2 &b)
    {
        _data = _mm_fmadd_pd(a._data, b._data, _data);
    }

    // subscript
    // subscript operators are convienient but expensive
    // should not be used in optimized kernels
    inline scalarType operator[](size_t i) const
    {
        alignas(alignment) scalarArray tmp;
        store(tmp, is_aligned);
        return tmp[i];
    }

    inline scalarType &operator[](size_t i)
    {
        scalarType *tmp = reinterpret_cast<scalarType *>(&_data);
        return tmp[i];
    }

    // unary ops
    inline void operator+=(sse2Double2 rhs)
    {
        _data = _mm_add_pd(_data, rhs._data);
    }

    inline void operator-=(sse2Double2 rhs)
    {
        _data = _mm_sub_pd(_data, rhs._data);
    }

    inline void operator*=(sse2Double2 rhs)
    {
        _data = _mm_mul_pd(_data, rhs._data);
    }

    inline void operator/=(sse2Double2 rhs)
    {
        _data = _mm_div_pd(_data, rhs._data);
    }
};

inline sse2Double2 operator+(sse2Double2 lhs, sse2Double2 rhs)
{
    return _mm_add_pd(lhs._data, rhs._data);
}

inline sse2Double2 operator-(sse2Double2 lhs, sse2Double2 rhs)
{
    return _mm_sub_pd(lhs._data, rhs._data);
}

inline sse2Double2 operator-(sse2Double2 in)
{
    return _mm_xor_pd(in._data, _mm_set1_pd(-0.0));
}

inline sse2Double2 operator*(sse2Double2 lhs, sse2Double2 rhs)
{
    return _mm_mul_pd(lhs._data, rhs._data);
}

inline sse2Double2 operator/(sse2Double2 lhs, sse2Double2 rhs)
{
    return _mm_div_pd(lhs._data, rhs._data);
}

inline sse2Double2 sqrt(sse2Double2 in)
{
    return _mm_sqrt_pd(in._data);
}

inline sse2Double2 abs(sse2Double2 in)
{
    // there is no sse2 _mm_abs_pd intrinsic
    static const __m128d sign_mask = _mm_set1_pd(-0.); // -0. = 1 << 63
    return _mm_andnot_pd(sign_mask, in._data);         // !sign_mask & x
}

inline sse2Double2 min(sse2Double2 lhs, sse2Double2 rhs)
{
    return _mm_min_pd(lhs._data, rhs._data);
}

inline sse2Double2 max(sse2Double2 lhs, sse2Double2 rhs)
{
    return _mm_max_pd(lhs._data, rhs._data);
}

inline sse2Double2 log(sse2Double2 in)
{
#if defined(TINYSIMD_HAS_SVML)
    return _mm_log_pd(in._data);
#else
    // there is no sse2 log intrinsic
    // this is a dreadful implementation and is simply a stop gap measure
    alignas(sse2Double2::alignment) sse2Double2::scalarArray tmp;
    in.store(tmp);
    tmp[0] = std::log(tmp[0]);
    tmp[1] = std::log(tmp[1]);
    sse2Double2 ret;
    ret.load(tmp);
    return ret;
#endif
}

inline void load_unalign_interleave(
    const double *in, const std::uint32_t dataLen,
    std::vector<sse2Double2, allocator<sse2Double2>> &out)
{
    alignas(sse2Double2::alignment) sse2Double2::scalarArray tmp;
    for (size_t i = 0; i < dataLen; ++i)
    {
        tmp[0] = in[i];
        tmp[1] = in[i + dataLen];
        out[i].load(tmp);
    }
}

inline void load_interleave(
    const double *in, std::uint32_t dataLen,
    std::vector<sse2Double2, allocator<sse2Double2>> &out)
{
    alignas(sse2Double2::alignment)
        size_t tmp[sse2Double2::width] = {0, dataLen};
    using index_t                      = sse2Long2<size_t>;
    index_t index0(tmp);
    index_t index1 = index0 + 1;

    // 4x unrolled loop
    constexpr uint16_t unrl = 2;
    size_t nBlocks          = dataLen / unrl;
    for (size_t i = 0; i < nBlocks; ++i)
    {
        out[unrl * i + 0].gather(in, index0);
        out[unrl * i + 1].gather(in, index1);
        index0 = index0 + unrl;
        index1 = index1 + unrl;
    }

    // spillover loop
    for (size_t i = unrl * nBlocks; i < dataLen; ++i)
    {
        out[i].gather(in, index0);
        index0 = index0 + 1;
    }
}

inline void deinterleave_unalign_store(
    const std::vector<sse2Double2, allocator<sse2Double2>> &in,
    const std::uint32_t dataLen, double *out)
{
    alignas(sse2Double2::alignment) sse2Double2::scalarArray tmp;
    for (size_t i = 0; i < dataLen; ++i)
    {
        in[i].store(tmp);
        out[i]           = tmp[0];
        out[i + dataLen] = tmp[1];
    }
}

inline void deinterleave_store(
    const std::vector<sse2Double2, allocator<sse2Double2>> &in,
    std::uint32_t dataLen, double *out)
{
    alignas(sse2Double2::alignment)
        size_t tmp[sse2Double2::width] = {0, dataLen};
    using index_t                      = sse2Long2<size_t>;
    index_t index0(tmp);

    for (size_t i = 0; i < dataLen; ++i)
    {
        in[i].scatter(out, index0);
        index0 = index0 + 1;
    }
}

//////////////////////////////////////////////////////////////////////////////

struct sse2Float4
{
    static constexpr unsigned width     = 4;
    static constexpr unsigned alignment = 16;

    using scalarType      = float;
    using scalarIndexType = std::uint32_t;
    using vectorType      = __m128;
    using scalarArray     = scalarType[width];

    // storage
    vectorType _data;

    // ctors
    inline sse2Float4()                      = default;
    inline sse2Float4(const sse2Float4 &rhs) = default;
    inline sse2Float4(const vectorType &rhs) : _data(rhs)
    {
    }
    inline sse2Float4(const scalarType rhs)
    {
        _data = _mm_set1_ps(rhs);
    }

    // copy assignment
    inline sse2Float4 &operator=(const sse2Float4 &) = default;

    // store
    inline void store(scalarType *p) const
    {
        _mm_store_ps(p, _data);
    }

    template <class flag,
              typename std::enable_if<is_requiring_alignment_v<flag> &&
                                          !is_streaming_v<flag>,
                                      bool>::type = 0>
    inline void store(scalarType *p, flag) const
    {
        _mm_store_ps(p, _data);
    }

    template <class flag, typename std::enable_if<
                              !is_requiring_alignment_v<flag>, bool>::type = 0>
    inline void store(scalarType *p, flag) const
    {
        _mm_storeu_ps(p, _data);
    }

    template <class flag,
              typename std::enable_if<is_streaming_v<flag>, bool>::type = 0>
    inline void store(scalarType *p, flag) const
    {
        _mm_stream_ps(p, _data);
    }

    // load packed
    inline void load(const scalarType *p)
    {
        _data = _mm_load_ps(p);
    }

    template <class flag, typename std::enable_if<
                              is_requiring_alignment_v<flag>, bool>::type = 0>
    inline void load(const scalarType *p, flag)
    {
        _data = _mm_load_ps(p);
    }

    template <class flag, typename std::enable_if<
                              !is_requiring_alignment_v<flag>, bool>::type = 0>
    inline void load(const scalarType *p, flag)
    {
        _data = _mm_loadu_ps(p);
    }

    // broadcast
    inline void broadcast(const scalarType rhs)
    {
        _data = _mm_set1_ps(rhs);
    }

    // gather scatter with sse2
    template <typename T>
    inline void gather(scalarType const *p, const sse2Int4<T> &indices)
    {
        // no gather intrinsics for SSE2
        alignas(alignment) scalarArray tmp;
        tmp[0] = p[_mm_extract_epi32(indices._data, 0)];
        tmp[1] = p[_mm_extract_epi32(indices._data, 1)];
        tmp[2] = p[_mm_extract_epi32(indices._data, 2)];
        tmp[3] = p[_mm_extract_epi32(indices._data, 3)];
        _data  = _mm_load_ps(&tmp[0]);
    }

    template <typename T>
    inline void scatter(scalarType *out, const sse2Int4<T> &indices) const
    {
        // no scatter intrinsics for SSE2
        alignas(alignment) scalarArray tmp;
        _mm_store_ps(tmp, _data);

        out[_mm_extract_epi32(indices._data, 0)] = tmp[0];
        out[_mm_extract_epi32(indices._data, 1)] = tmp[1];
        out[_mm_extract_epi32(indices._data, 2)] = tmp[2];
        out[_mm_extract_epi32(indices._data, 3)] = tmp[3];
    }

    // fma
    // this = this + a * b
    inline void fma(const sse2Float4 &a, const sse2Float4 &b)
    {
        _data = _mm_fmadd_ps(a._data, b._data, _data);
    }

    // subscript
    // subscript operators are convienient but expensive
    // should not be used in optimized kernels
    inline scalarType operator[](size_t i) const
    {
        alignas(alignment) scalarArray tmp;
        store(tmp, is_aligned);
        return tmp[i];
    }

    inline scalarType &operator[](size_t i)
    {
        scalarType *tmp = reinterpret_cast<scalarType *>(&_data);
        return tmp[i];
    }

    inline void operator+=(sse2Float4 rhs)
    {
        _data = _mm_add_ps(_data, rhs._data);
    }

    inline void operator-=(sse2Float4 rhs)
    {
        _data = _mm_sub_ps(_data, rhs._data);
    }

    inline void operator*=(sse2Float4 rhs)
    {
        _data = _mm_mul_ps(_data, rhs._data);
    }

    inline void operator/=(sse2Float4 rhs)
    {
        _data = _mm_div_ps(_data, rhs._data);
    }
};

inline sse2Float4 operator+(sse2Float4 lhs, sse2Float4 rhs)
{
    return _mm_add_ps(lhs._data, rhs._data);
}

inline sse2Float4 operator-(sse2Float4 lhs, sse2Float4 rhs)
{
    return _mm_sub_ps(lhs._data, rhs._data);
}

inline sse2Float4 operator-(sse2Float4 in)
{
    return _mm_xor_ps(in._data, _mm_set1_ps(-0.0));
}

inline sse2Float4 operator*(sse2Float4 lhs, sse2Float4 rhs)
{
    return _mm_mul_ps(lhs._data, rhs._data);
}

inline sse2Float4 operator/(sse2Float4 lhs, sse2Float4 rhs)
{
    return _mm_div_ps(lhs._data, rhs._data);
}

inline sse2Float4 sqrt(sse2Float4 in)
{
    return _mm_sqrt_ps(in._data);
}

inline sse2Float4 abs(sse2Float4 in)
{
    // there is no sse2 _mm_abs_ps intrinsic
    static const __m128 sign_mask = _mm_set1_ps(-0.); // -0. = 1 << 63
    return _mm_andnot_ps(sign_mask, in._data);        // !sign_mask & x
}

inline sse2Float4 min(sse2Float4 lhs, sse2Float4 rhs)
{
    return _mm_min_ps(lhs._data, rhs._data);
}

inline sse2Float4 max(sse2Float4 lhs, sse2Float4 rhs)
{
    return _mm_max_ps(lhs._data, rhs._data);
}

inline sse2Float4 log(sse2Float4 in)
{
    // there is no sse2 log intrinsic
    // this is a dreadful implementation and is simply a stop gap measure
    alignas(sse2Float4::alignment) sse2Float4::scalarArray tmp;
    in.store(tmp);
    tmp[0] = std::log(tmp[0]);
    tmp[1] = std::log(tmp[1]);
    tmp[2] = std::log(tmp[2]);
    tmp[3] = std::log(tmp[3]);
    sse2Float4 ret;
    ret.load(tmp);
    return ret;
}

inline void load_unalign_interleave(
    const double *in, const std::uint32_t dataLen,
    std::vector<sse2Float4, allocator<sse2Float4>> &out)
{
    alignas(sse2Float4::alignment) sse2Float4::scalarArray tmp;
    for (size_t i = 0; i < dataLen; ++i)
    {
        tmp[0] = in[i];
        tmp[1] = in[i + dataLen];
        tmp[2] = in[i + 2 * dataLen];
        tmp[3] = in[i + 3 * dataLen];
        out[i].load(tmp);
    }
}

inline void load_interleave(const float *in, std::uint32_t dataLen,
                            std::vector<sse2Float4, allocator<sse2Float4>> &out)
{

    alignas(sse2Float4::alignment) sse2Float4::scalarIndexType tmp[4] = {
        0, dataLen, 2 * dataLen, 3 * dataLen};

    using index_t = sse2Int4<sse2Float4::scalarIndexType>;
    index_t index0(tmp);
    index_t index1 = index0 + 1;

    // 4x unrolled loop
    size_t nBlocks = dataLen / 2;
    for (size_t i = 0; i < nBlocks; ++i)
    {
        out[2 * i + 0].gather(in, index0);
        out[2 * i + 1].gather(in, index1);
        index0 = index0 + 2;
        index1 = index1 + 2;
    }

    // spillover loop
    for (size_t i = 2 * nBlocks; i < dataLen; ++i)
    {
        out[i].gather(in, index0);
        index0 = index0 + 1;
    }
}

inline void deinterleave_unalign_store(
    const std::vector<sse2Float4, allocator<sse2Float4>> &in,
    const std::uint32_t dataLen, double *out)
{
    alignas(sse2Float4::alignment) sse2Float4::scalarArray tmp;
    for (size_t i = 0; i < dataLen; ++i)
    {
        in[i].store(tmp);
        out[i]               = tmp[0];
        out[i + dataLen]     = tmp[1];
        out[i + 2 * dataLen] = tmp[2];
        out[i + 3 * dataLen] = tmp[3];
    }
}

inline void deinterleave_store(
    const std::vector<sse2Float4, allocator<sse2Float4>> &in,
    std::uint32_t dataLen, float *out)
{
    alignas(sse2Float4::alignment) sse2Float4::scalarIndexType tmp[4] = {
        0, dataLen, 2 * dataLen, 3 * dataLen};
    using index_t = sse2Int4<sse2Float4::scalarIndexType>;
    index_t index0(tmp);

    for (size_t i = 0; i < dataLen; ++i)
    {
        in[i].scatter(out, index0);
        index0 = index0 + 1;
    }
}

////////////////////////////////////////////////////////////////////////////////

// mask type
// mask is a int type with special properties (broad boolean vector)
// broad boolean vectors defined and allowed values are:
// false=0x0 and true=0xFFFFFFFF
//
// VERY LIMITED SUPPORT...just enough to make cubic eos work...
//
struct sse2Mask2 : sse2Long2<std::uint64_t>
{
    // bring in ctors
    using sse2Long2::sse2Long2;

    static constexpr scalarType true_v  = -1;
    static constexpr scalarType false_v = 0;
};

inline sse2Mask2 operator>(sse2Double2 lhs, sse2Double2 rhs)
{
    return reinterpret_cast<__m128i>(
        _mm_cmp_pd(lhs._data, rhs._data, _CMP_GT_OQ));
}

inline bool operator&&(sse2Mask2 lhs, bool rhs)
{
    bool tmp = _mm_testc_si128(lhs._data, _mm_set1_epi64x(sse2Mask2::true_v));

    return tmp && rhs;
}

struct sse2Mask4 : sse2Int4<std::uint32_t>
{
    // bring in ctors
    using sse2Int4::sse2Int4;

    static constexpr scalarType true_v  = -1;
    static constexpr scalarType false_v = 0;
};

inline sse2Mask4 operator>(sse2Float4 lhs, sse2Float4 rhs)
{
    return reinterpret_cast<__m128i>(_mm_cmp_ps(rhs._data, lhs._data, 1));
}

inline bool operator&&(sse2Mask4 lhs, bool rhs)
{
    bool tmp = _mm_testc_si128(lhs._data, _mm_set1_epi64x(sse2Mask4::true_v));

    return tmp && rhs;
}

} // namespace tinysimd

#endif // defined(__SSE2__) && defined(NEKTAR_ENABLE_SIMD_SSE2)
#endif
