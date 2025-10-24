///////////////////////////////////////////////////////////////////////////////
//
// File: NekLinSysIterGMRES.h
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
// Description: NekLinSysIterGMRES header
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_UTILITIES_LINEAR_ALGEBRA_NEK_LINSYS_ITERAT_GMRES_H
#define NEKTAR_LIB_UTILITIES_LINEAR_ALGEBRA_NEK_LINSYS_ITERAT_GMRES_H

#include <LibUtilities/LinearAlgebra/NekLinSysIter.h>

namespace Nektar::LibUtilities
{
/// A global linear system.
class NekLinSysIterGMRES;

class NekLinSysIterGMRES : public NekLinSysIter
{
public:
    /// Support creation through MemoryManager.
    friend class MemoryManager<NekLinSysIterGMRES>;

    LIB_UTILITIES_EXPORT static NekLinSysIterSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const LibUtilities::CommSharedPtr &vRowComm, const int nDimen,
        const NekSysKey &pKey)
    {
        NekLinSysIterSharedPtr p =
            MemoryManager<NekLinSysIterGMRES>::AllocateSharedPtr(
                pSession, vRowComm, nDimen, pKey);
        p->InitObject();
        return p;
    }

    static std::string className;

    LIB_UTILITIES_EXPORT NekLinSysIterGMRES(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const LibUtilities::CommSharedPtr &vRowComm, const int nDimen,
        const NekSysKey &pKey = NekSysKey());
    LIB_UTILITIES_EXPORT ~NekLinSysIterGMRES() override = default;

    LIB_UTILITIES_EXPORT int GetMaxLinIte()
    {
        return (m_maxrestart * m_LinSysMaxStorage);
    }

protected:
    // This is maximum gmres restart iteration
    int m_maxrestart;

    // This is maximum bandwidth of Hessenburg matrix
    // if use truncted Gmres(m)
    int m_KrylovMaxHessMatBand;

    // This is the maximum number of solution vectors that can be stored
    // For example, in gmres, it is the max number of Krylov space
    // search directions can be stored
    // It determines the max storage usage
    int m_LinSysMaxStorage;

    NekDouble m_prec_factor = 1.0;

    bool m_NekLinSysLeftPrecon    = false;
    bool m_NekLinSysRightPrecon   = true;
    bool m_GMRESCentralDifference = false;

    void v_InitObject() override;

    int v_SolveSystem(const int nGlobal,
                      const Array<OneD, const NekDouble> &pInput,
                      Array<OneD, NekDouble> &pOutput, const int nDir) override;

    void v_DoIterate(const int nGlobal, const Array<OneD, NekDouble> &rhs,
                     Array<OneD, NekDouble> &x, const int nDir, NekDouble &err,
                     int &iter) override;

private:
    /// Actual iterative solve-GMRES
    int DoGMRES(const int pNumRows, const Array<OneD, const NekDouble> &pInput,
                Array<OneD, NekDouble> &pOutput, const int pNumDir);

    /// Actual iterative gmres solver for one restart
    NekDouble DoGmresRestart(const bool restarted, const bool truncted,
                             const int nGlobal,
                             const Array<OneD, const NekDouble> &pInput,
                             Array<OneD, NekDouble> &pOutput, const int nDir);

    // Arnoldi process
    void DoArnoldi(const int starttem, const int endtem, const int nGlobal,
                   const int nDir, Array<OneD, NekDouble> &w,
                   // V[nd] current search direction
                   Array<OneD, NekDouble> &Vsingle1,
                   // V[nd+1] new search direction
                   Array<OneD, NekDouble> &Vsingle2,
                   // One line of Hessenburg matrix
                   Array<OneD, NekDouble> &hsingle);

    // QR fatorization through Givens rotation
    void DoGivensRotation(const int starttem, const int endtem,
                          const int nGlobal, const int nDir,
                          Array<OneD, NekDouble> &c, Array<OneD, NekDouble> &s,
                          Array<OneD, NekDouble> &hsingle,
                          Array<OneD, NekDouble> &eta);

    // Backward calculation to calculate coeficients
    // of least square problem
    // To notice, Hessenburg's columnns and rows are reverse
    void DoBackward(const int number, Array<OneD, Array<OneD, NekDouble>> &A,
                    const Array<OneD, const NekDouble> &b,
                    Array<OneD, NekDouble> &y);

    static std::string lookupIds[];
    static std::string def;

    // Hessenburg matrix
    Array<OneD, Array<OneD, NekDouble>> m_hes;
    // Hesseburg matrix after rotation
    Array<OneD, Array<OneD, NekDouble>> m_Upper;
    // Total search directions
    Array<OneD, Array<OneD, NekDouble>> m_V_total;
};
} // namespace Nektar::LibUtilities

#endif
