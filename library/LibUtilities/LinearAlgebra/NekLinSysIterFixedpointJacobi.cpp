///////////////////////////////////////////////////////////////////////////////
//
// File: NekLinSysIterFixedpointJacobi.cpp
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
// Description: NekLinSysIterFixedpointJacobi definition
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/LinearAlgebra/NekLinSysIterFixedpointJacobi.h>

using namespace std;

namespace Nektar::LibUtilities
{
/**
 * @class  NekLinSysIterFixedpointJacobi
 *
 * Solves a linear system using iterative methods.
 */
string NekLinSysIterFixedpointJacobi::className =
    LibUtilities::GetNekLinSysIterFactory().RegisterCreatorFunction(
        "FixedpointJacobi", NekLinSysIterFixedpointJacobi::create,
        "NekLinSysIterFixedpointJacobi solver.");

NekLinSysIterFixedpointJacobi::NekLinSysIterFixedpointJacobi(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const LibUtilities::CommSharedPtr &vRowComm, const int nDimen,
    const NekSysKey &pKey)
    : NekLinSysIter(pSession, vRowComm, nDimen, pKey)
{
}

/**
 *
 */
void NekLinSysIterFixedpointJacobi::v_InitObject()
{
    NekLinSysIter::v_InitObject();
}

/**
 *
 */
int NekLinSysIterFixedpointJacobi::v_SolveSystem(
    const int nGlobal, const Array<OneD, const NekDouble> &rhs,
    Array<OneD, NekDouble> &x, [[maybe_unused]] const int nDir)
{
    // store the residual
    Array<OneD, NekDouble> r(nGlobal);

    int m_totalIterations = 0;
    for (int i = 0; i < m_NekLinSysMaxIterations; ++i)
    {
        m_operator.DoNekSysFixPointIte(rhs, x, r);
        m_totalIterations++;
        if (m_totalIterations % m_errorCheckInterval == 0)
        {
            ConvergenceCheck(r);
            if (m_converged)
            {
                break;
            }
        }
    }

    return m_totalIterations;
}

void NekLinSysIterFixedpointJacobi::v_DoIterate(
    const int nGlobal, const Array<OneD, NekDouble> &rhs,
    Array<OneD, NekDouble> &x, [[maybe_unused]] const int nDir, NekDouble &err,
    int &iter)
{
    v_SolveSystem(nGlobal, rhs, x, nDir);

    iter = m_totalIterations;
    err  = sqrt(m_finalError / m_rhs_magnitude);
}

} // namespace Nektar::LibUtilities
