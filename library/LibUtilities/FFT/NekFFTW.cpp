///////////////////////////////////////////////////////////////////////////////
//
// File: NekFFTW.cpp
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
// Description: Wrapper around FFTW library
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/BasicUtils/VmathArray.hpp>
#include <LibUtilities/FFT/NekFFTW.h>

namespace Nektar::LibUtilities
{
std::string NekFFTW::className =
    GetNektarFFTFactory().RegisterCreatorFunction("NekFFTW", NekFFTW::create);

NekFFTW::NekFFTW(int N) : NektarFFT(N)
{
    m_wsp = Array<OneD, NekDouble>(m_N);

    m_plan_forward =
        fftw_plan_r2r_1d(m_N, nullptr, nullptr, FFTW_R2HC, FFTW_ESTIMATE);
    m_plan_backward =
        fftw_plan_r2r_1d(m_N, nullptr, nullptr, FFTW_HC2R, FFTW_ESTIMATE);

    m_FFTW_w     = Array<OneD, NekDouble>(m_N);
    m_FFTW_w_inv = Array<OneD, NekDouble>(m_N);

    m_FFTW_w[0] = 1.0 / (NekDouble)m_N;
    m_FFTW_w[1] = 0.0;

    m_FFTW_w_inv[0] = 1.0;
    m_FFTW_w_inv[1] = 0.0;

    for (int i = 2; i < m_N; i++)
    {
        m_FFTW_w[i]     = m_FFTW_w[0] * 2;
        m_FFTW_w_inv[i] = m_FFTW_w_inv[0] / 2;
    }
}

// Destructor
NekFFTW::~NekFFTW()
{
    fftw_destroy_plan(m_plan_forward);
    fftw_destroy_plan(m_plan_backward);
}

// Forward transformation
void NekFFTW::v_FFTFwdTrans(Array<OneD, NekDouble> &inarray,
                            Array<OneD, NekDouble> &outarray)
{
    // FFTW_R2HC
    fftw_execute_r2r(m_plan_forward, inarray.data(), m_wsp.data());

    // Reshuffle
    int halfN = m_N / 2;

    outarray[1] = m_FFTW_w[1] * m_wsp[halfN];

    Vmath::Vmul(halfN, m_wsp, 1, m_FFTW_w, 2, outarray, 2);

    for (int i = 0; i < halfN - 1; i++)
    {
        outarray[(m_N - 1) - 2 * i] =
            m_FFTW_w[(m_N - 1) - 2 * i] * m_wsp[halfN + 1 + i];
    }
}

// Backward transformation
void NekFFTW::v_FFTBwdTrans(Array<OneD, NekDouble> &inarray,
                            Array<OneD, NekDouble> &outarray)
{
    // Reshuffle
    int halfN = m_N / 2;

    m_wsp[halfN] = m_FFTW_w_inv[1] * inarray[1];

    Vmath::Vmul(halfN, inarray, 2, m_FFTW_w_inv, 2, m_wsp, 1);

    for (int i = 0; i < (halfN - 1); i++)
    {
        m_wsp[halfN + 1 + i] =
            m_FFTW_w_inv[(m_N - 1) - 2 * i] * inarray[(m_N - 1) - 2 * i];
    }

    // FFTW_HC2R
    fftw_execute_r2r(m_plan_backward, m_wsp.data(), outarray.data());
}

} // namespace Nektar::LibUtilities
