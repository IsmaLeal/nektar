#
# NektarSIMD.cmake
#
# Sets up cmake variables needed for the SIMD library in Nektar++
#

# CMAKE_SYSTEM_PROCESSOR is not available at this point because we want to
# initialize the flags before PROJECT()
EXECUTE_PROCESS(COMMAND uname -m OUTPUT_VARIABLE _SYSTEM_PROCESSOR)
# guard in case uname doesn't work, for instance on win..
IF(_SYSTEM_PROCESSOR)
    STRING(STRIP ${_SYSTEM_PROCESSOR} _SYSTEM_PROCESSOR)
ENDIF()

SET(NEKTAR_ENABLE_SIMD OFF CACHE STRING "")

IF(_SYSTEM_PROCESSOR STREQUAL "x86_64")
    SET_PROPERTY(CACHE NEKTAR_ENABLE_SIMD PROPERTY STRINGS OFF SSE2 AVX2 AVX512)
    IF (NEKTAR_ENABLE_SIMD STREQUAL "AVX512")
        MESSAGE(STATUS "Enabling avx512, you might need to clear CMAKE_CXX_FLAGS or add the appriopriate flags")
        ADD_DEFINITIONS(-DNEKTAR_ENABLE_SIMD_AVX512)
        ADD_DEFINITIONS(-DNEKTAR_ENABLE_SIMD_AVX2)
        ADD_DEFINITIONS(-DNEKTAR_ENABLE_SIMD_SSE2)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mavx2 -mavx512f -mfma")
    ELSEIF (NEKTAR_ENABLE_SIMD STREQUAL "AVX2")
        MESSAGE(STATUS "Enabling avx2, you might need to clear CMAKE_CXX_FLAGS or add the appriopriate flags")
        ADD_DEFINITIONS(-DNEKTAR_ENABLE_SIMD_AVX2)
        ADD_DEFINITIONS(-DNEKTAR_ENABLE_SIMD_SSE2)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mavx2 -mfma")
    ELSEIF (NEKTAR_ENABLE_SIMD STREQUAL "SSE2")
        MESSAGE(STATUS "Enabling sse, you might need to clear CMAKE_CXX_FLAGS or add the appriopriate flags")
        ADD_DEFINITIONS(-DNEKTAR_ENABLE_SIMD_SSE2)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfma")
    ELSEIF (NEKTAR_ENABLE_SIMD)
        MESSAGE(FATAL_ERROR "Unknown NEKTAR_ENABLE_SIMD value:  ${NEKTAR_ENABLE_SIMD}, available options: OFF, SSE2, AVX2, and AVX512")
    ENDIF()
ELSEIF(_SYSTEM_PROCESSOR STREQUAL "aarch64")
    SET(NEKTAR_SVE_BITS scalable CACHE STRING "sve vector bits")
    SET_PROPERTY(CACHE NEKTAR_ENABLE_SIMD PROPERTY STRINGS OFF SVE SVE2)
    SET_PROPERTY(CACHE NEKTAR_SVE_BITS PROPERTY STRINGS "scalable;128;256;512;1024;2048")
    IF (NEKTAR_ENABLE_SIMD STREQUAL "SVE")
        MESSAGE(STATUS "Enabling sve, you might need to clear CMAKE_CXX_FLAGS or add the appriopriate flags")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+sve -flax-vector-conversions -msve-vector-bits=${NEKTAR_SVE_BITS}")
        ADD_DEFINITIONS(-DNEKTAR_ENABLE_SIMD_SVE)
    ELSEIF (NEKTAR_ENABLE_SIMD STREQUAL "SVE2")
        MESSAGE(STATUS "Enabling sve2, you might need to clear CMAKE_CXX_FLAGS or add the appriopriate flags")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+sve2 -flax-vector-conversions -msve-vector-bits=${NEKTAR_SVE_BITS}")
        ADD_DEFINITIONS(-DNEKTAR_ENABLE_SIMD_SVE)
    ELSEIF (NEKTAR_ENABLE_SIMD)
        MESSAGE(FATAL_ERROR "Unknown NEKTAR_ENABLE_SIMD value:  ${NEKTAR_ENABLE_SIMD}, available options: OFF, SVE, and SVE2")
    ENDIF()
ENDIF()

# Vmath Simd
OPTION(NEKTAR_ENABLE_SIMD_VMATH "Enable vector types in vmath" OFF)
IF (NEKTAR_ENABLE_SIMD_VMATH)
    ADD_DEFINITIONS(-DNEKTAR_ENABLE_SIMD_VMATH)
ENDIF()
MARK_AS_ADVANCED(FORCE NEKTAR_ENABLE_SIMD_VMATH)
