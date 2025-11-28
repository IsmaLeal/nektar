///////////////////////////////////////////////////////////////////////////////
//
// File: sve.hpp
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
// Description: Vector type using Armv8 Scalable Vector Extension (SVE).
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_LIBUTILITES_SIMDLIB_SVE_H
#define NEKTAR_LIB_LIBUTILITES_SIMDLIB_SVE_H

#if defined(__ARM_FEATURE_SVE)
#include <arm_acle.h>
#include <arm_sve.h>
#endif

#include "allocator.hpp"
#include "traits.hpp"
#include <vector>

namespace tinysimd::abi
{
template <typename scalarType, int width = 0> struct sve
{
    using type = void;
};

} // namespace tinysimd::abi

// requires clang >= 12.0.0 or gcc >= 10
// requires -msve-vector-bits=<length>
#if __ARM_FEATURE_SVE_BITS > 0 && defined(NEKTAR_ENABLE_SIMD_SVE)

namespace tinysimd
{

// from VLA to VLST
// C++ does not allow for incomplete class member types
// to get around that we force a known size at compile time
typedef svfloat64_t svfloat64_vlst_t
    __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
typedef svint64_t svint64_vlst_t
    __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
typedef svuint64_t svuint64_vlst_t
    __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
typedef svfloat32_t svfloat32_vlst_t
    __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
typedef svint32_t svint32_vlst_t
    __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
typedef svuint32_t svuint32_vlst_t
    __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
typedef svbool_t svbool_vlst_t
    __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));

// forward declaration of concrete types
template <typename T> struct sveInt64;
template <typename T> struct sveInt32;
struct sveFloat64;
struct sveFloat32;
struct sveMask64;
struct sveMask32;

namespace abi
{

// mapping between abstract types and concrete floating point types
template <> struct sve<double>
{
    using type = sveFloat64;
};
template <> struct sve<float>
{
    using type = sveFloat32;
};
// generic index mapping
// assumes index type width same as floating point type
template <> struct sve<std::int64_t>
{
    using type = sveInt64<std::int64_t>;
};
template <> struct sve<std::uint64_t>
{
    using type = sveInt64<std::uint64_t>;
};
template <> struct sve<std::int32_t>
{
    using type = sveInt32<std::int32_t>;
};
template <> struct sve<std::uint32_t>
{
    using type = sveInt32<std::uint32_t>;
};
// specialized index mapping
template <> struct sve<std::int64_t, __ARM_FEATURE_SVE_BITS / 64>
{
    using type = sveInt64<std::int64_t>;
};
template <> struct sve<std::uint64_t, __ARM_FEATURE_SVE_BITS / 64>
{
    using type = sveInt64<std::uint64_t>;
};
// the number of lanes dictate the simd type
// then we need to make sure we can load properly
// a 32 bit pointer to a 64 bit vector (zero or sign extend)
template <> struct sve<std::int32_t, __ARM_FEATURE_SVE_BITS / 64>
{
    using type = sveInt64<std::int64_t>;
};
template <> struct sve<std::uint32_t, __ARM_FEATURE_SVE_BITS / 64>
{
    using type = sveInt64<std::uint64_t>;
};
template <> struct sve<std::int32_t, __ARM_FEATURE_SVE_BITS / 32>
{
    using type = sveInt32<std::int32_t>;
};
template <> struct sve<std::uint32_t, __ARM_FEATURE_SVE_BITS / 32>
{
    using type = sveInt32<std::uint32_t>;
};
// bool mapping
template <> struct sve<bool, __ARM_FEATURE_SVE_BITS / 64>
{
    using type = sveMask64;
};
template <> struct sve<bool, __ARM_FEATURE_SVE_BITS / 32>
{
    using type = sveMask32;
};

} // namespace abi

// concrete types, could add enable if to allow only unsigned long and long...
template <typename T> struct sveInt32
{
    static_assert(std::is_integral_v<T> && sizeof(T) == 4,
                  "4 bytes Integral required.");

    static constexpr unsigned int alignment =
        __ARM_FEATURE_SVE_BITS / sizeof(T);
    static constexpr unsigned int width = alignment / 8;

    using scalarType = T;
    using vectorType =
        typename std::conditional<std::is_signed_v<T>, svint32_vlst_t,
                                  svuint32_vlst_t>::type;
    using scalarArray = scalarType[width];

    // storage
    vectorType _data;

    // ctors
    inline sveInt32()                    = default;
    inline sveInt32(const sveInt32 &rhs) = default;
    inline sveInt32(const vectorType &rhs) : _data(rhs)
    {
    }
    inline sveInt32(const scalarType rhs)
    {
        _data = svdup_s32(rhs);
    }
    explicit inline sveInt32(scalarArray &rhs)
    {
        _data = svld1(svptrue_b32(), rhs);
    }

    // store packed
    inline void store(scalarType *p) const
    {
        svst1(svptrue_b32(), p, _data);
    }
    // refer to x86_64 implementations
    // sve has no requirements on alignment
    // nevertheless we should accept valid tags for compatibility
    template <typename TAG,
              typename std::enable_if<is_load_tag_v<TAG>, bool>::type = 0>
    inline void store(scalarType *p, TAG) const
    {
        svst1(svptrue_b32(), p, _data);
    }

    // load packed
    inline void load(const scalarType *p)
    {
        _data = svld1(svptrue_b32(), p);
    }
    // refer to x86_64 implementations
    // sve has no requirements on alignment
    // nevertheless we should accept valid tags for compatibility
    template <typename TAG,
              typename std::enable_if<is_load_tag_v<TAG>, bool>::type = 0>
    inline void load(const scalarType *p, TAG)
    {
        _data = svld1(svptrue_b32(), p);
    }

    // broadcast
    inline void broadcast(const scalarType rhs)
    {
        _data = svdup(rhs);
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
    inline void operator+=(sveInt32 rhs)
    {
        if constexpr (std::is_signed_v<T>)
        {
            _data = svadd_s32_x(svptrue_b32(), _data, rhs._data);
        }
        else
        {
            _data = svadd_u32_x(svptrue_b32(), _data, rhs._data);
        }
    }

    inline void operator-=(sveInt32 rhs)
    {
        _data = svsub_x(svptrue_b32(), _data, rhs._data);
    }

    inline void operator*=(sveInt32 rhs)
    {
        _data = svmul_x(svptrue_b32(), _data, rhs._data);
    }

    inline void operator/=(sveInt32 rhs)
    {
        _data = svdiv_x(svptrue_b32(), _data, rhs._data);
    }
};

template <typename T>
inline sveInt32<T> operator+(sveInt32<T> lhs, sveInt32<T> rhs)
{
    if constexpr (std::is_signed_v<T>)
    {
        return svadd_s32_x(svptrue_b32(), lhs._data, rhs._data);
    }
    else
    {
        return svadd_u32_x(svptrue_b32(), lhs._data, rhs._data);
    }
}

template <typename T> inline sveInt32<T> operator+(sveInt32<T> lhs, T rhs)
{
    if constexpr (std::is_signed_v<T>)
    {
        return svadd_s32_x(svptrue_b32(), lhs._data, sveInt32<T>(rhs)._data);
    }
    else
    {
        return svadd_u32_x(svptrue_b32(), lhs._data, sveInt32<T>(rhs)._data);
    }
}

template <typename T>
inline sveInt32<T> operator-(sveInt32<T> lhs, sveInt32<T> rhs)
{
    return svsub_x(svptrue_b32(), lhs._data, rhs._data);
}

template <typename T>
inline sveInt32<T> operator*(sveInt32<T> lhs, sveInt32<T> rhs)
{
    return svmul_x(svptrue_b32(), lhs._data, rhs._data);
}

template <typename T>
inline sveInt32<T> operator/(sveInt32<T> lhs, sveInt32<T> rhs)
{
    return svdiv_x(svptrue_b32(), lhs._data, rhs._data);
}

template <typename T> inline sveInt32<T> abs(sveInt32<T> in)
{
    return svabs_x(svptrue_b32(), in._data);
}

////////////////////////////////////////////////////////////////////////////////

template <typename T> struct sveInt64
{
    static_assert(std::is_integral_v<T> && sizeof(T) == 8,
                  "8 bytes Integral required.");

    static constexpr unsigned int alignment =
        __ARM_FEATURE_SVE_BITS / sizeof(T);
    static constexpr unsigned int width = alignment / 8;

    using scalarType = T;
    using vectorType =
        typename std::conditional<std::is_signed_v<T>, svint64_vlst_t,
                                  svuint64_vlst_t>::type;
    using scalarArray = scalarType[width];

    // storage
    vectorType _data;

    // ctors
    inline sveInt64()                    = default;
    inline sveInt64(const sveInt64 &rhs) = default;
    inline sveInt64(const vectorType &rhs) : _data(rhs)
    {
    }
    inline sveInt64(const scalarType rhs)
    {
        _data = svdup_s64(rhs);
    }
    explicit inline sveInt64(scalarArray &rhs)
    {
        _data = svld1(svptrue_b64(), rhs);
    }

    // store packed
    inline void store(scalarType *p) const
    {
        svst1(svptrue_b64(), p, _data);
    }
    // refer to x86_64 implementations
    // sve has no requirements on alignment
    // nevertheless we should accept valid tags for compatibility
    template <typename TAG,
              typename std::enable_if<is_load_tag_v<TAG>, bool>::type = 0>
    inline void store(scalarType *p, TAG) const
    {
        svst1(svptrue_b64(), p, _data);
    }

    // load packed
    inline void load(const scalarType *p)
    {
        _data = svld1(svptrue_b64(), p);
    }
    // refer to x86_64 implementations
    // sve has no requirements on alignment
    // nevertheless we should accept valid tags for compatibility
    template <typename TAG,
              typename std::enable_if<is_load_tag_v<TAG>, bool>::type = 0>
    inline void load(const scalarType *p, TAG)
    {
        _data = svld1(svptrue_b64(), p);
    }

    // load packed from 32 bit
    template <typename I32,
              typename std::enable_if<std::is_integral_v<I32> &&
                                          std::is_signed_v<scalarType> &&
                                          sizeof(I32) == 4,
                                      bool>::type = 0>
    inline void load(const I32 *p)
    {
        _data = svld1sw_s64(svptrue_b64(), p);
    }
    template <typename I32,
              typename std::enable_if<std::is_integral_v<I32> &&
                                          !std::is_signed_v<scalarType> &&
                                          sizeof(I32) == 4,
                                      bool>::type = 0>
    inline void load(const I32 *p)
    {
        _data = svld1uw_s64(svptrue_b64(), p);
    }
    template <typename I32, typename TAG,
              typename std::enable_if<
                  is_load_tag_v<TAG> && std::is_integral_v<I32> &&
                      std::is_signed_v<scalarType> && sizeof(I32) == 4,
                  bool>::type = 0>
    inline void load(const I32 *p, TAG)
    {
        _data = svld1sw_s64(svptrue_b64(), p);
    }
    template <typename I32, typename TAG,
              typename std::enable_if<
                  is_load_tag_v<TAG> && std::is_integral_v<I32> &&
                      !std::is_signed_v<scalarType> && sizeof(I32) == 4,
                  bool>::type = 0>
    inline void load(const I32 *p, TAG)
    {
        _data = svld1uw_s64(svptrue_b64(), p);
    }

    // broadcast
    inline void broadcast(const scalarType rhs)
    {
        _data = svdup(rhs);
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

    // unary ops
    inline void operator+=(sveInt64 rhs)
    {
        if constexpr (std::is_signed_v<T>)
        {
            _data = svadd_s64_x(svptrue_b64(), _data, rhs._data);
        }
        else
        {
            _data = svadd_u64_x(svptrue_b64(), _data, rhs._data);
        }
    }

    inline void operator-=(sveInt64 rhs)
    {
        if constexpr (std::is_signed_v<T>)
        {
            _data = svsub_s64_x(svptrue_b64(), _data, rhs._data);
        }
        else
        {
            _data = svsub_u64_x(svptrue_b64(), _data, rhs._data);
        }
    }

    inline void operator*=(sveInt64 rhs)
    {
        if constexpr (std::is_signed_v<T>)
        {
            _data = svmul_s64_x(svptrue_b64(), _data, rhs._data);
        }
        else
        {
            _data = svmul_u64_x(svptrue_b64(), _data, rhs._data);
        }
    }

    inline void operator/=(sveInt64 rhs)
    {
        if constexpr (std::is_signed_v<T>)
        {
            _data = svdiv_s64_x(svptrue_b64(), _data, rhs._data);
        }
        else
        {
            _data = svdiv_u64_x(svptrue_b64(), _data, rhs._data);
        }
    }
};

template <typename T>
inline sveInt64<T> operator+(sveInt64<T> lhs, sveInt64<T> rhs)
{
    if constexpr (std::is_signed_v<T>)
    {
        return svadd_s64_x(svptrue_b64(), lhs._data, rhs._data);
    }
    else
    {
        return svadd_u64_x(svptrue_b64(), lhs._data, rhs._data);
    }
}

template <typename T> inline sveInt64<T> operator+(sveInt64<T> lhs, T rhs)
{
    if constexpr (std::is_signed_v<T>)
    {
        return svadd_s64_x(svptrue_b64(), lhs._data, sveInt64<T>(rhs)._data);
    }
    else
    {
        return svadd_u64_x(svptrue_b64(), lhs._data, sveInt64<T>(rhs)._data);
    }
}

template <typename T>
inline sveInt64<T> operator-(sveInt64<T> lhs, sveInt64<T> rhs)
{
    if constexpr (std::is_signed_v<T>)
    {
        return svsub_s64_x(svptrue_b64(), lhs._data, rhs._data);
    }
    else
    {
        return svsub_u64_x(svptrue_b64(), lhs._data, rhs._data);
    }
}

template <typename T>
inline sveInt64<T> operator*(sveInt64<T> lhs, sveInt64<T> rhs)
{
    if constexpr (std::is_signed_v<T>)
    {
        return svmul_s64_x(svptrue_b64(), lhs._data, rhs._data);
    }
    else
    {
        return svmul_u64_x(svptrue_b64(), lhs._data, rhs._data);
    }
}

template <typename T>
inline sveInt64<T> operator/(sveInt64<T> lhs, sveInt64<T> rhs)
{
    if constexpr (std::is_signed_v<T>)
    {
        return svdiv_s64_x(svptrue_b64(), lhs._data, rhs._data);
    }
    else
    {
        return svdiv_u64_x(svptrue_b64(), lhs._data, rhs._data);
    }
}

template <typename T> inline sveInt64<T> abs(sveInt64<T> in)
{
    if constexpr (std::is_signed_v<T>)
    {
        return svabs_s64_x(svptrue_b64(), in._data);
    }
    else
    {
        return svabs_u64_x(svptrue_b64(), in._data);
    }
}

////////////////////////////////////////////////////////////////////////////////

struct sveFloat32
{
    static constexpr unsigned int alignment =
        __ARM_FEATURE_SVE_BITS / sizeof(float);
    static constexpr unsigned int width = alignment / 8;

    using scalarType      = float;
    using scalarIndexType = std::uint32_t;
    using vectorType      = svfloat32_vlst_t;
    using scalarArray     = scalarType[width];

    // storage
    vectorType _data;

    // ctors
    inline sveFloat32()                      = default;
    inline sveFloat32(const sveFloat32 &rhs) = default;
    inline sveFloat32(const vectorType &rhs) : _data(rhs)
    {
    }
    inline sveFloat32(const scalarType rhs)
    {
        _data = svdup_f32(rhs);
    }

    // store packed
    inline void store(scalarType *p) const
    {
        svst1_f32(svptrue_b32(), p, _data);
    }
    // refer to x86_64 implementations
    // sve has no requirements on alignment
    // nevertheless we should accept valid tags for compatibility
    template <typename T,
              typename std::enable_if<is_load_tag_v<T>, bool>::type = 0>
    inline void store(scalarType *p, T) const
    {
        svst1_f32(svptrue_b32(), p, _data);
    }

    // load packed
    inline void load(const scalarType *p)
    {
        _data = svld1_f32(svptrue_b32(), p);
    }
    // refer to x86_64 implementations
    // sve has no requirements on alignment
    // nevertheless we should accept valid tags for compatibility
    template <typename T,
              typename std::enable_if<is_load_tag_v<T>, bool>::type = 0>
    inline void load(const scalarType *p, T)
    {
        _data = svld1_f32(svptrue_b32(), p);
    }

    // broadcast
    inline void broadcast(const scalarType rhs)
    {
        _data = svdup_f32(rhs);
    }

    // gather/scatter
    template <typename T>
    inline void gather(scalarType const *p, const sveInt32<T> &indices)
    {
        if constexpr (std::is_signed_v<T>)
        {
            _data = svld1_gather_s32index_f32(svptrue_b32(), p, indices._data);
        }
        else
        {
            _data = svld1_gather_u32index_f32(svptrue_b32(), p, indices._data);
        }
    }

    template <typename T>
    inline void scatter(scalarType *out, const sveInt32<T> &indices) const
    {
        if constexpr (std::is_signed_v<T>)
        {
            svst1_scatter_s32index_f32(svptrue_b32(), out, indices._data,
                                       _data);
        }
        else
        {
            svst1_scatter_u32index_f32(svptrue_b32(), out, indices._data,
                                       _data);
        }
    }

    // fma
    // this = this + a * b
    inline void fma(const sveFloat32 &a, const sveFloat32 &b)
    {
        _data = svmad_f32_x(svptrue_b32(), a._data, b._data, _data);
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
    inline void operator+=(sveFloat32 rhs)
    {
        _data = svadd_f32_x(svptrue_b32(), _data, rhs._data);
    }

    inline void operator-=(sveFloat32 rhs)
    {
        _data = svsub_f32_x(svptrue_b32(), _data, rhs._data);
    }

    inline void operator*=(sveFloat32 rhs)
    {
        _data = svmul_f32_x(svptrue_b32(), _data, rhs._data);
    }

    inline void operator/=(sveFloat32 rhs)
    {
        _data = svdiv_f32_x(svptrue_b32(), _data, rhs._data);
    }
};

inline sveFloat32 operator+(sveFloat32 lhs, sveFloat32 rhs)
{
    return svadd_f32_x(svptrue_b32(), lhs._data, rhs._data);
}

inline sveFloat32 operator-(sveFloat32 lhs, sveFloat32 rhs)
{
    return svsub_f32_x(svptrue_b32(), lhs._data, rhs._data);
}

inline sveFloat32 operator-(sveFloat32 in)
{
    return svsub_f32_x(svptrue_b32(), svdup_f32(-0.0), in._data);
    // return sveor_f32_x(svptrue_b32(), in._data, svdup_f32(-0.0));
}

inline sveFloat32 operator*(sveFloat32 lhs, sveFloat32 rhs)
{
    return svmul_f32_x(svptrue_b32(), lhs._data, rhs._data);
}

inline sveFloat32 operator/(sveFloat32 lhs, sveFloat32 rhs)
{
    return svdiv_f32_x(svptrue_b32(), lhs._data, rhs._data);
}

inline sveFloat32 sqrt(sveFloat32 in)
{
    return svsqrt_f32_x(svptrue_b32(), in._data);
}

inline sveFloat32 abs(sveFloat32 in)
{
    return svabs_f32_x(svptrue_b32(), in._data);
}

inline sveFloat32 min(sveFloat32 lhs, sveFloat32 rhs)
{
    return svmin_f32_x(svptrue_b32(), lhs._data, rhs._data);
}

inline sveFloat32 max(sveFloat32 lhs, sveFloat32 rhs)
{
    return svmax_f32_x(svptrue_b32(), lhs._data, rhs._data);
}

inline sveFloat32 log(sveFloat32 in)
{
    // there is no sve log intrinsic
    // this is a dreadful implementation and is simply a stop gap measure
    alignas(sveFloat32::alignment) sveFloat32::scalarArray tmp;
    in.store(tmp);
    for (size_t i = 0; i < sveFloat32::width; ++i)
    {
        tmp[i] = std::log(tmp[i]);
    }
    sveFloat32 ret;
    ret.load(tmp);
    return ret;
}

inline void load_unalign_interleave(
    const double *in, const std::uint32_t dataLen,
    std::vector<sveFloat32, allocator<sveFloat32>> &out)
{
    alignas(sveFloat32::alignment) sveFloat32::scalarArray tmp;
    for (size_t i = 0; i < dataLen; ++i)
    {
        for (size_t j = 0; j < sveFloat32::width; ++j)
        {
            tmp[j] = in[i + j * dataLen];
        }
        out[i].load(tmp);
    }
}

inline void load_interleave(const float *in, std::uint32_t dataLen,
                            std::vector<sveFloat32, allocator<sveFloat32>> &out)
{

    alignas(sveFloat32::alignment)
        sveFloat32::scalarIndexType tmp[sveFloat32::width] = {};

    // populate scalar index of unknown size
    // (known at compile time)
    for (size_t i = 0; i < sveFloat32::width; ++i)
    {
        tmp[i] = i * dataLen;
    }

    using index_t = sveInt32<sveFloat32::scalarIndexType>;
    index_t index0(tmp);
    index_t index1 = index0 + 1u;

    // 2x unrolled loop -- minimun width is 2
    size_t nBlocks = dataLen / 2;
    for (size_t i = 0; i < nBlocks; ++i)
    {
        out[2 * i + 0].gather(in, index0);
        out[2 * i + 1].gather(in, index1);
        index0 = index0 + 2u;
        index1 = index1 + 2u;
    }

    // spillover loop
    for (size_t i = 2 * nBlocks; i < dataLen; ++i)
    {
        out[i].gather(in, index0);
        index0 = index0 + 1u;
    }
}

inline void deinterleave_unalign_store(
    const std::vector<sveFloat32, allocator<sveFloat32>> &in,
    const std::uint32_t dataLen, double *out)
{
    alignas(sveFloat32::alignment) sveFloat32::scalarArray tmp;
    for (size_t i = 0; i < dataLen; ++i)
    {
        in[i].store(tmp);
        for (size_t j = 0; j < sveFloat32::width; ++j)
        {
            out[i + j * dataLen] = tmp[j];
        }
    }
}

inline void deinterleave_store(
    const std::vector<sveFloat32, allocator<sveFloat32>> &in,
    std::uint32_t dataLen, float *out)
{
    alignas(sveFloat32::alignment)
        sveFloat32::scalarIndexType tmp[sveFloat32::width] = {};

    // populate scalar index of unknown size
    // (known at compile time)
    for (size_t i = 0; i < sveFloat32::width; ++i)
    {
        tmp[i] = i * dataLen;
    }

    using index_t = sveInt32<sveFloat32::scalarIndexType>;
    index_t index0(tmp);

    for (size_t i = 0; i < dataLen; ++i)
    {
        in[i].scatter(out, index0);
        index0 = index0 + 1u;
    }
}

////////////////////////////////////////////////////////////////////////////////

struct sveFloat64
{
    static constexpr unsigned int alignment =
        __ARM_FEATURE_SVE_BITS / sizeof(double);
    static constexpr unsigned int width = alignment / 8;

    using scalarType      = double;
    using scalarIndexType = std::uint64_t;
    using vectorType      = svfloat64_vlst_t;
    using scalarArray     = scalarType[width];

    // storage
    vectorType _data;

    // ctors
    inline sveFloat64()                      = default;
    inline sveFloat64(const sveFloat64 &rhs) = default;
    inline sveFloat64(const vectorType &rhs) : _data(rhs)
    {
    }
    inline sveFloat64(const scalarType rhs)
    {
        _data = svdup_f64(rhs);
    }

    // store packed
    inline void store(scalarType *p) const
    {
        svst1_f64(svptrue_b64(), p, _data);
    }
    // refer to x86_64 implementations
    // sve has no requirements on alignment
    // nevertheless we should accept valid tags for compatibility
    template <typename T,
              typename std::enable_if<is_load_tag_v<T>, bool>::type = 0>
    inline void store(scalarType *p, T) const
    {
        svst1_f64(svptrue_b64(), p, _data);
    }

    // load packed
    inline void load(const scalarType *p)
    {
        _data = svld1_f64(svptrue_b64(), p);
    }
    // refer to x86_64 implementations
    // sve has no requirements on alignment
    // nevertheless we should accept valid tags for compatibility
    template <typename T,
              typename std::enable_if<is_load_tag_v<T>, bool>::type = 0>
    inline void load(const scalarType *p, T)
    {
        _data = svld1_f64(svptrue_b64(), p);
    }

    // broadcast
    inline void broadcast(const scalarType rhs)
    {
        _data = svdup_f64(rhs);
    }

    // gather/scatter
    template <typename T>
    inline void gather(scalarType const *p, const sveInt64<T> &indices)
    {
        if constexpr (std::is_signed_v<T>)
        {
            _data = svld1_gather_s64index_f64(svptrue_b64(), p, indices._data);
        }
        else
        {
            _data = svld1_gather_u64index_f64(svptrue_b64(), p, indices._data);
        }
    }

    template <typename T>
    inline void scatter(scalarType *out, const sveInt64<T> &indices) const
    {
        if constexpr (std::is_signed_v<T>)
        {
            svst1_scatter_s64index_f64(svptrue_b64(), out, indices._data,
                                       _data);
        }
        else
        {
            svst1_scatter_u64index_f64(svptrue_b64(), out, indices._data,
                                       _data);
        }
    }

    // fma
    // this = this + a * b
    inline void fma(const sveFloat64 &a, const sveFloat64 &b)
    {
        _data = svmad_f64_x(svptrue_b64(), a._data, b._data, _data);
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
    inline void operator+=(sveFloat64 rhs)
    {
        _data = svadd_f64_x(svptrue_b64(), _data, rhs._data);
    }

    inline void operator-=(sveFloat64 rhs)
    {
        _data = svsub_f64_x(svptrue_b64(), _data, rhs._data);
    }

    inline void operator*=(sveFloat64 rhs)
    {
        _data = svmul_f64_x(svptrue_b64(), _data, rhs._data);
    }

    inline void operator/=(sveFloat64 rhs)
    {
        _data = svdiv_f64_x(svptrue_b64(), _data, rhs._data);
    }
};

inline sveFloat64 operator+(sveFloat64 lhs, sveFloat64 rhs)
{
    return svadd_f64_x(svptrue_b64(), lhs._data, rhs._data);
}

inline sveFloat64 operator-(sveFloat64 lhs, sveFloat64 rhs)
{
    return svsub_f64_x(svptrue_b64(), lhs._data, rhs._data);
}

inline sveFloat64 operator-(sveFloat64 in)
{
    return svsub_f64_x(svptrue_b64(), svdup_f64(-0.0), in._data);
    // return sveor_f64_x(svptrue_b64(), in._data, svdup_f64(-0.0));
}

inline sveFloat64 operator*(sveFloat64 lhs, sveFloat64 rhs)
{
    return svmul_f64_x(svptrue_b64(), lhs._data, rhs._data);
}

inline sveFloat64 operator/(sveFloat64 lhs, sveFloat64 rhs)
{
    return svdiv_f64_x(svptrue_b64(), lhs._data, rhs._data);
}

inline sveFloat64 sqrt(sveFloat64 in)
{
    return svsqrt_f64_x(svptrue_b64(), in._data);
}

inline sveFloat64 abs(sveFloat64 in)
{
    return svabs_f64_x(svptrue_b64(), in._data);
}

inline sveFloat64 min(sveFloat64 lhs, sveFloat64 rhs)
{
    return svmin_f64_x(svptrue_b64(), lhs._data, rhs._data);
}

inline sveFloat64 max(sveFloat64 lhs, sveFloat64 rhs)
{
    return svmax_f64_x(svptrue_b64(), lhs._data, rhs._data);
}

inline sveFloat64 log(sveFloat64 in)
{
    // there is no sve log intrinsic
    // this is a dreadful implementation and is simply a stop gap measure
    alignas(sveFloat64::alignment) sveFloat64::scalarArray tmp;
    in.store(tmp);
    for (size_t i = 0; i < sveFloat64::width; ++i)
    {
        tmp[i] = std::log(tmp[i]);
    }
    sveFloat64 ret;
    ret.load(tmp);
    return ret;
}

inline void load_unalign_interleave(
    const double *in, const std::uint32_t dataLen,
    std::vector<sveFloat64, allocator<sveFloat64>> &out)
{
    alignas(sveFloat64::alignment) sveFloat64::scalarArray tmp;
    for (size_t i = 0; i < dataLen; ++i)
    {
        for (size_t j = 0; j < sveFloat64::width; ++j)
        {
            tmp[j] = in[i + j * dataLen];
        }
        out[i].load(tmp);
    }
}

inline void load_interleave(const double *in, std::uint32_t dataLen,
                            std::vector<sveFloat64, allocator<sveFloat64>> &out)
{

    alignas(sveFloat64::alignment) size_t tmp[sveFloat64::width] = {};

    // populate scalar index of unknown size
    // (known at compile time)
    for (size_t i = 0; i < sveFloat64::width; ++i)
    {
        tmp[i] = i * dataLen;
    }

    using index_t = sveInt64<size_t>;
    index_t index0(tmp);
    index_t index1 = index0 + 1ul;

    // 2x unrolled loop -- minimun width is 2
    size_t nBlocks = dataLen / 2;
    for (size_t i = 0; i < nBlocks; ++i)
    {
        out[2 * i + 0].gather(in, index0);
        out[2 * i + 1].gather(in, index1);
        index0 = index0 + 2ul;
        index1 = index1 + 2ul;
    }

    // spillover loop
    for (size_t i = 2 * nBlocks; i < dataLen; ++i)
    {
        out[i].gather(in, index0);
        index0 = index0 + 1ul;
    }
}

inline void deinterleave_unalign_store(
    const std::vector<sveFloat64, allocator<sveFloat64>> &in,
    const std::uint32_t dataLen, double *out)
{
    alignas(sveFloat64::alignment) sveFloat64::scalarArray tmp;
    for (size_t i = 0; i < dataLen; ++i)
    {
        in[i].store(tmp);
        for (size_t j = 0; j < sveFloat64::width; ++j)
        {
            out[i + j * dataLen] = tmp[j];
        }
    }
}

inline void deinterleave_store(
    const std::vector<sveFloat64, allocator<sveFloat64>> &in,
    std::uint32_t dataLen, double *out)
{
    alignas(sveFloat64::alignment) size_t tmp[sveFloat64::width] = {};

    // populate scalar index of unknown size
    // (known at compile time)
    for (size_t i = 0; i < sveFloat64::width; ++i)
    {
        tmp[i] = i * dataLen;
    }

    using index_t = sveInt64<size_t>;
    index_t index0(tmp);

    for (size_t i = 0; i < dataLen; ++i)
    {
        in[i].scatter(out, index0);
        index0 = index0 + 1ul;
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
struct sveMask64 : sveInt64<std::uint64_t>
{
    // bring in ctors
    using sveInt64::sveInt64;

    static constexpr scalarType true_v  = -1;
    static constexpr scalarType false_v = 0;
};

inline sveMask64 operator>(sveFloat64 lhs, sveFloat64 rhs)
{
    // set mask
    svbool_vlst_t mask = svcmpgt(svptrue_b64(), lhs._data, rhs._data);
    // abuse set inactive to zero to convert
    sveMask64::vectorType sveTrue_v = svdup_u64(sveMask64::true_v);
    return svand_u64_z(mask, sveTrue_v, sveTrue_v);
}

// logical and
inline bool operator&&(sveMask64 lhs, bool rhs)
{
    // set mask
    sveMask64::vectorType sveFalse_v = svdup_u64(sveMask64::false_v);
    svbool_vlst_t mask = svcmpne_u64(svptrue_b64(), lhs._data, sveFalse_v);
    // is any equal to false (zero)?
    bool tmp = svptest_any(svptrue_b64(), mask);
    return tmp && rhs;
}

////////////////////////////////////////////////////////////////////////////////

struct sveMask32 : sveInt32<std::uint32_t>
{
    // bring in ctors
    using sveInt32::sveInt32;

    static constexpr scalarType true_v  = -1;
    static constexpr scalarType false_v = 0;
};

inline sveMask32 operator>(sveFloat32 lhs, sveFloat32 rhs)
{
    // set mask
    svbool_vlst_t mask = svcmpgt(svptrue_b32(), lhs._data, rhs._data);
    // abuse set inactive to zero to convert
    sveMask32::vectorType sveTrue_v = svdup_u32(sveMask32::true_v);
    return svand_u32_z(mask, sveTrue_v, sveTrue_v);
}

// logical and
inline bool operator&&(sveMask32 lhs, bool rhs)
{
    // set mask
    sveMask32::vectorType sveFalse_v = svdup_u32(sveMask32::false_v);
    svbool_vlst_t mask = svcmpne_u32(svptrue_b32(), lhs._data, sveFalse_v);
    // is any equal to false (zero)?
    bool tmp = svptest_any(svptrue_b32(), mask);
    return tmp && rhs;
}

} // namespace tinysimd

#endif // defined(__ARM_FEATURE_SVE_BITS)

#endif
