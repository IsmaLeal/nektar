///////////////////////////////////////////////////////////////////////////////
//
// File: TimeIntegrationSolutionGLM.cpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// License for the specific language governing rights and limitations under
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
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/VmathArray.hpp>

#include <LibUtilities/TimeIntegration/TimeIntegrationSchemeGLM.h>

namespace Nektar::LibUtilities
{

namespace
{

Array<OneD, size_t> BuildVarSizesFromState(ConstDoubleArray &y)
{
    Array<OneD, size_t> varSizes(y.size(), size_t(0));
    for (size_t i = 0; i < y.size(); ++i)
    {
        varSizes[i] = y[i].size();
    }

    return varSizes;
}

Array<OneD, size_t> GetUniformVarSizes(const size_t nvar, const size_t npoints)
{
    return Array<OneD, size_t>(nvar, npoints);
}

} // namespace

TimeIntegrationSolutionGLM::TimeIntegrationSolutionGLM(
    const TimeIntegrationAlgorithmGLM *schemeAlgorithm, const DoubleArray &y,
    const NekDouble time, const NekDouble timestep)
    : m_schemeAlgorithm(schemeAlgorithm),
      m_solVector(m_schemeAlgorithm->m_numsteps),
      m_t(m_schemeAlgorithm->m_numsteps),
      m_setflag(m_schemeAlgorithm->m_numsteps)
{
    size_t nsteps         = m_schemeAlgorithm->m_numsteps;
    size_t nvar           = y.size();
    size_t nMultiStepVals = m_schemeAlgorithm->GetNmultiStepValues();
    SetVarSizes(BuildVarSizesFromState(y));

    for (size_t i = 0; i < nsteps; i++)
    {
        m_solVector[i] = Array<OneD, Array<OneD, NekDouble>>(nvar);
        for (size_t j = 0; j < nvar; j++)
        {
            const size_t npoints = GetVarSize(j);
            m_solVector[i][j]    = Array<OneD, NekDouble>(npoints, 0.0);
            if (i == 0)
            {
                Vmath::Vcopy(npoints, y[j].data(), 1, m_solVector[i][j].data(),
                             1);
            }
        }

        if (i < nMultiStepVals)
        {
            m_t[i] = time - i * timestep;
        }
        else
        {
            m_t[i] = timestep;
        }

        m_setflag[i] = (i == 0);
    }
}

TimeIntegrationSolutionGLM::TimeIntegrationSolutionGLM(
    const TimeIntegrationAlgorithmGLM *schemeAlgorithm, const size_t nvar,
    const size_t npoints)
    : TimeIntegrationSolutionGLM(schemeAlgorithm, GetUniformVarSizes(nvar,
                                                                     npoints))
{
}

TimeIntegrationSolutionGLM::TimeIntegrationSolutionGLM(
    const TimeIntegrationAlgorithmGLM *schemeAlgorithm,
    const Array<OneD, size_t> &varSizes)
    : m_schemeAlgorithm(schemeAlgorithm),
      m_solVector(schemeAlgorithm->m_numsteps),
      m_t(schemeAlgorithm->m_numsteps),
      m_setflag(m_schemeAlgorithm->m_numsteps, true)
{
    SetVarSizes(varSizes);

    for (size_t i = 0; i < m_schemeAlgorithm->m_numsteps; i++)
    {
        m_solVector[i] = Array<OneD, Array<OneD, NekDouble>>(m_varSizes.size());
        for (size_t j = 0; j < m_varSizes.size(); j++)
        {
            m_solVector[i][j] = Array<OneD, NekDouble>(GetVarSize(j));
        }
    }
}

TimeIntegrationSolutionGLM::TimeIntegrationSolutionGLM(
    const TimeIntegrationAlgorithmGLM *schemeAlgorithm)
    : m_schemeAlgorithm(schemeAlgorithm),
      m_solVector(m_schemeAlgorithm->m_numsteps),
      m_t(m_schemeAlgorithm->m_numsteps),
      m_setflag(m_schemeAlgorithm->m_numsteps, false)
{
    SetVarSizes(Array<OneD, size_t>());
}

} // namespace Nektar::LibUtilities
