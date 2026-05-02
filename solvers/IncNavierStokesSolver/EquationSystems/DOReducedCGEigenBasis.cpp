///////////////////////////////////////////////////////////////////////////////
//
// File: DOReducedCGEigenBasis.cpp
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
// Description: Reduced homogeneous CG Laplacian eigenbasis helper for DOVelocityCorrectionScheme.
//
///////////////////////////////////////////////////////////////////////////////

#include <IncNavierStokesSolver/EquationSystems/DOReducedCGEigenBasis.h>

#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/LinearAlgebra/Arpack.hpp>
#include <LibUtilities/LinearAlgebra/MatrixOperations.hpp>
#include <LibUtilities/LinearAlgebra/NekVector.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace Nektar
{
class DOReducedCGEigenBasis::Impl
{
public:
    struct ReducedSpace
    {
        MultiRegions::ContFieldSharedPtr field;
        MultiRegions::AssemblyMapCGSharedPtr map;
        int nGlob;
        int nDir;
        int nHom;
        int nLocal;
        std::vector<int> localToGlobal;
        std::vector<int> globalMultiplicity;
        std::vector<NekDouble> localSign;

        explicit ReducedSpace(const MultiRegions::ContFieldSharedPtr &inField)
            : field(inField), map(inField->GetLocalToGlobalMap()),
              nGlob(map->GetNumGlobalCoeffs()),
              nDir(map->GetNumGlobalDirBndCoeffs()), nHom(nGlob - nDir),
              nLocal(map->GetNumLocalCoeffs()), localToGlobal(nLocal, -1),
              globalMultiplicity(nGlob, 0), localSign(nLocal, 0.0)
        {
            ASSERTL0(nHom > 0, "Reduced homogeneous CG space is empty.");

            for (int i = 0; i < nLocal; ++i)
            {
                const int gid = map->GetLocalToGlobalMap(i);
                localToGlobal[i] = gid;
                localSign[i] = map->GetLocalToGlobalSign(i);
                ++globalMultiplicity[gid];
            }
        }

        void ReducedToFull(const NekDouble *reduced,
                           Array<OneD, NekDouble> &full) const
        {
            Vmath::Zero(nGlob, full.data(), 1);
            Vmath::Vcopy(nHom, reduced, 1, full.data() + nDir, 1);
        }

        void FullToReduced(const Array<OneD, const NekDouble> &full,
                           NekDouble *reduced) const
        {
            Vmath::Vcopy(nHom, full.data() + nDir, 1, reduced, 1);
        }

        void ReducedToLocal(const NekDouble *reduced,
                            Array<OneD, NekDouble> &full,
                            Array<OneD, NekDouble> &local) const
        {
            ReducedToFull(reduced, full);
            map->GlobalToLocal(full, local);
        }

        void FullToLocalRhs(const Array<OneD, const NekDouble> &full,
                            Array<OneD, NekDouble> &local) const
        {
            for (int i = 0; i < nLocal; ++i)
            {
                const int gid = localToGlobal[i];
                local[i] = gid < nDir
                               ? 0.0
                               : localSign[i] * full[gid] /
                                     static_cast<NekDouble>(
                                         globalMultiplicity[gid]);
            }
        }
    };

    struct Workspace
    {
        Array<OneD, NekDouble> fullIn;
        Array<OneD, NekDouble> fullOut;
        Array<OneD, NekDouble> localIn;
        Array<OneD, NekDouble> localOut;
        std::vector<NekDouble> reducedTmp;
        std::vector<NekDouble> reducedTmp2;

        explicit Workspace(const ReducedSpace &space)
            : fullIn(space.nGlob, 0.0), fullOut(space.nGlob, 0.0),
              localIn(space.nLocal, 0.0), localOut(space.nLocal, 0.0),
              reducedTmp(space.nHom, 0.0), reducedTmp2(space.nHom, 0.0)
        {
        }
    };

    explicit Impl(const MultiRegions::ContFieldSharedPtr &field)
        : m_space(field), m_work(m_space),
          m_lapKey(StdRegions::eLaplacian, m_space.map)
    {
    }

    int GetNumHomCoeffs() const
    {
        return m_space.nHom;
    }

    std::vector<DOReducedCGEigenBasis::Eigenpair> ComputeSmallest(int nev)
    {
        ASSERTL0(nev > 0, "Need at least one eigenpair.");
        ASSERTL0(nev < m_space.nHom,
                 "Requested too many modes for reduced homogeneous CG space.");

        const int n = m_space.nHom;
        const int ncv = std::min(n, std::max(2 * nev + 8, 20));
        const int lworkl = ncv * (ncv + 8);
        const char bmat = 'G';
        const char *which = "LM";
        const NekDouble tol = 0.0;
        const NekDouble sigma = 0.0;
        int ido = 0;
        int info = 0;

        std::vector<NekDouble> resid(n, 0.0), v(n * ncv, 0.0), workd(3 * n, 0.0),
            workl(lworkl, 0.0), evals(nev, 0.0), evecs(n * nev, 0.0);
        std::vector<int> iparam(11, 0), ipntr(11, 0), select(ncv, 0);

        iparam[0] = 1;
        iparam[2] = 500;
        iparam[6] = 3;

        while (ido != 99)
        {
            Arpack::Dsaupd(ido, &bmat, n, which, nev, tol, resid.data(), ncv,
                           v.data(), n, iparam.data(), ipntr.data(),
                           workd.data(), workl.data(), lworkl, info);

            if (ido == -1)
            {
                ApplyM(workd.data() + ipntr[0] - 1, m_work.reducedTmp.data());
                SolveK(m_work.reducedTmp.data(), workd.data() + ipntr[1] - 1);
            }
            else if (ido == 1)
            {
                SolveK(workd.data() + ipntr[2] - 1,
                       workd.data() + ipntr[1] - 1);
            }
            else if (ido == 2)
            {
                ApplyM(workd.data() + ipntr[0] - 1,
                       workd.data() + ipntr[1] - 1);
            }
            else if (ido != 99)
            {
                ASSERTL0(false,
                         "Unexpected ARPACK reverse communication request.");
            }
        }

        ASSERTL0(info == 0, "ARPACK Dsaupd failed for DOVelocityCorrectionScheme mode init.");
        ASSERTL0(iparam[4] >= nev,
                 "ARPACK converged fewer modes than requested.");

        int rvec = 1;
        info = 0;
        Arpack::Dseupd(rvec, "A", select.data(), evals.data(), evecs.data(), n,
                       sigma, &bmat, n, which, nev, tol, resid.data(), ncv,
                       v.data(), n, iparam.data(), ipntr.data(), workd.data(),
                       workl.data(), lworkl, info);
        ASSERTL0(info == 0, "ARPACK Dseupd failed for DOVelocityCorrectionScheme mode init.");

        std::vector<int> order(nev);
        std::iota(order.begin(), order.end(), 0);
        for (int i = 0; i < nev; ++i)
        {
            evals[i] = RayleighQuotient(evecs.data() + i * n);
        }

        std::sort(order.begin(), order.end(),
                  [&evals](int lhs, int rhs) { return evals[lhs] < evals[rhs]; });

        std::vector<DOReducedCGEigenBasis::Eigenpair> out(nev);
        for (int i = 0; i < nev; ++i)
        {
            const int idx = order[i];
            out[i].lambda = evals[idx];
            out[i].reduced.assign(evecs.begin() + idx * n,
                                  evecs.begin() + (idx + 1) * n);
            out[i].residual =
                RelativeResidual(out[i].reduced.data(), out[i].lambda);
        }

        return out;
    }

    void ExportToLocalAndPhys(const std::vector<NekDouble> &reduced,
                              Array<OneD, NekDouble> &local,
                              Array<OneD, NekDouble> &phys)
    {
        ASSERTL0(static_cast<int>(reduced.size()) == m_space.nHom,
                 "Reduced mode has the wrong size.");
        m_space.ReducedToLocal(reduced.data(), m_work.fullIn, local);
        m_space.field->BwdTrans(local, phys);
    }

private:
    ReducedSpace m_space;
    Workspace m_work;
    MultiRegions::GlobalLinSysKey m_lapKey;

    void ApplyK(const NekDouble *x, NekDouble *y)
    {
        ApplyElementOperator(StdRegions::eLaplacian, x, y);
    }

    void ApplyM(const NekDouble *x, NekDouble *y)
    {
        ApplyElementOperator(StdRegions::eMass, x, y);
    }

    void SolveK(const NekDouble *rhs, NekDouble *sol)
    {
        auto &space = m_space;
        auto &work = m_work;

        space.ReducedToFull(rhs, work.fullIn);
        space.FullToLocalRhs(work.fullIn, work.localIn);
        Vmath::Zero(space.nLocal, work.localOut.data(), 1);
        space.field->GlobalSolve(m_lapKey, work.localIn, work.localOut);

        Vmath::Zero(space.nGlob, work.fullOut.data(), 1);
        space.field->LocalToGlobal(work.localOut, work.fullOut, false);
        space.FullToReduced(work.fullOut, sol);
    }

    NekDouble RayleighQuotient(const NekDouble *phi)
    {
        ApplyK(phi, m_work.reducedTmp.data());
        ApplyM(phi, m_work.reducedTmp2.data());

        const NekDouble num =
            Vmath::Dot(m_space.nHom, phi, 1, m_work.reducedTmp.data(), 1);
        const NekDouble den =
            Vmath::Dot(m_space.nHom, phi, 1, m_work.reducedTmp2.data(), 1);
        return num / den;
    }

    NekDouble RelativeResidual(const NekDouble *phi, NekDouble lambda)
    {
        ApplyK(phi, m_work.reducedTmp.data());
        ApplyM(phi, m_work.reducedTmp2.data());

        NekDouble rn = 0.0;
        NekDouble dn = 0.0;
        for (int i = 0; i < m_space.nHom; ++i)
        {
            const NekDouble ri =
                m_work.reducedTmp[i] - lambda * m_work.reducedTmp2[i];
            const NekDouble di = lambda * m_work.reducedTmp2[i];
            rn += ri * ri;
            dn += di * di;
        }

        return std::sqrt(rn / dn);
    }

    void ApplyElementOperator(const StdRegions::MatrixType mtype,
                              const NekDouble *x, NekDouble *y)
    {
        auto &space = m_space;
        auto &work = m_work;

        space.ReducedToLocal(x, work.fullIn, work.localIn);
        Vmath::Zero(space.nLocal, work.localOut.data(), 1);

        for (int e = 0; e < space.field->GetExpSize(); ++e)
        {
            auto exp = space.field->GetExp(e);
            const int nCoeffs = exp->GetNcoeffs();
            const int offset = space.field->GetCoeff_Offset(e);

            Array<OneD, NekDouble> elemIn(nCoeffs, work.localIn.data() + offset,
                                          eArrayWrapper);
            Array<OneD, NekDouble> elemOut(nCoeffs,
                                           work.localOut.data() + offset,
                                           eArrayWrapper);
            NekVector<NekDouble> inVec(nCoeffs, elemIn, eWrapper);
            NekVector<NekDouble> outVec(nCoeffs, elemOut, eWrapper);

            Multiply(outVec, *exp->GetLocMatrix(mtype), inVec);
        }

        Vmath::Zero(space.nGlob, work.fullOut.data(), 1);
        space.map->Assemble(work.localOut, work.fullOut);
        space.FullToReduced(work.fullOut, y);
    }
};

DOReducedCGEigenBasis::DOReducedCGEigenBasis(
    const MultiRegions::ContFieldSharedPtr &field)
    : m_impl(std::make_unique<Impl>(field))
{
}

DOReducedCGEigenBasis::~DOReducedCGEigenBasis() = default;

int DOReducedCGEigenBasis::GetNumHomCoeffs() const
{
    return m_impl->GetNumHomCoeffs();
}

std::vector<DOReducedCGEigenBasis::Eigenpair>
DOReducedCGEigenBasis::ComputeSmallest(int nev)
{
    return m_impl->ComputeSmallest(nev);
}

void DOReducedCGEigenBasis::ExportToLocalAndPhys(
    const std::vector<NekDouble> &reduced, Array<OneD, NekDouble> &local,
    Array<OneD, NekDouble> &phys)
{
    m_impl->ExportToLocalAndPhys(reduced, local, phys);
}
} // namespace Nektar
