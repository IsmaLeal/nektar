///////////////////////////////////////////////////////////////////////////////
//
// File: NekLinSysIter.cpp
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
// Description: NekLinSysIter definition
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/LinearAlgebra/NekLinSysIter.h>

namespace Nektar::LibUtilities
{
/**
 * @class  NekLinSysIter
 *
 * Solves a linear system using iterative methods.
 */
NekLinSysIterFactory &GetNekLinSysIterFactory()
{
    static NekLinSysIterFactory instance;
    return instance;
}

NekLinSysIter::NekLinSysIter(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const LibUtilities::CommSharedPtr &vRowComm, const int nDimen,
    const NekSysKey &pKey)
    : NekSys(pSession, vRowComm, nDimen, pKey)
{
    m_NekLinSysTolerance     = fmax(pKey.m_NekLinSysTolerance, 1.0E-16);
    m_NekLinSysMaxIterations = pKey.m_NekLinSysMaxIterations;
    m_isLocal                = false;
}

void NekLinSysIter::v_InitObject()
{
    NekSys::v_InitObject();
    SetUniversalUniqueMap();
}

void NekLinSysIter::SetUniversalUniqueMap(const Array<OneD, const int> &map,
                                          const int nDir)
{
    int nmap = map.size();
    if (m_map.size() != nmap)
    {
        m_map = Array<OneD, int>(nmap, 0);
    }
    Vmath::Vcopy(nmap, map, 1, m_map, 1);
    // check if map is all ones, ignore Dirichlet BCs
    m_mapIsOnes = true;
    for (int i = nDir; i < nmap; ++i)
    {
        if (map[i] != 1)
        {
            m_mapIsOnes = false;
            break;
        }
    }
}

void NekLinSysIter::SetUniversalUniqueMap()
{
    m_map       = Array<OneD, int>(m_SysDimen, 1);
    m_mapIsOnes = true;
}

void NekLinSysIter::Set_Rhs_Magnitude(const Array<OneD, NekDouble> &pIn)
{
    NekDouble vExchange(0.0);

    if (m_isLocal)
    {
        Array<OneD, NekDouble> wk(pIn.size());
        m_operator.DoAssembleLoc(pIn, wk);
        vExchange = Vmath::Dot(pIn.size(), wk, pIn);
    }
    else
    {
        vExchange = Vmath::Dot2(pIn.size(), pIn, pIn, m_map);
    }

    m_rowComm->AllReduce(vExchange, LibUtilities::ReduceSum);
    m_rhs_magnitude = (vExchange > 1.0e-6) ? vExchange : 1.0;
}

void NekLinSysIter::ConvergenceCheck(
    const Array<OneD, const NekDouble> &Residual)
{
    m_finalError = Vmath::Dot(Residual.size(), Residual, Residual);
    m_rowComm->AllReduce(m_finalError, Nektar::LibUtilities::ReduceSum);

    m_converged = m_finalError <
                  m_NekLinSysTolerance * m_NekLinSysTolerance * m_rhs_magnitude;
}

} // namespace Nektar::LibUtilities
