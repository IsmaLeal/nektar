///////////////////////////////////////////////////////////////////////////////
//
// File: DOVelocityCorrectionScheme.cpp
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
// Description: DO (dynamically orthogonal) extension of the velocity-
// correction incompressible solver implementation.
//
///////////////////////////////////////////////////////////////////////////////

#include <IncNavierStokesSolver/EquationSystems/DOVelocityCorrectionScheme.h>
#include <IncNavierStokesSolver/EquationSystems/DOPODInitialiser.h>
#include <IncNavierStokesSolver/EquationSystems/DOReducedCGEigenBasis.h>

#include <LibUtilities/BasicUtils/FieldIO.h>
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/LinearAlgebra/Lapack.hpp>
#include <LibUtilities/LinearAlgebra/Blas.hpp>
#include <MultiRegions/ContField.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace Nektar
{
namespace
{

// ===========================================================================
// BC capture / homogenise / restore helpers
// ===========================================================================
// These helpers are used to enforce the homogeneous BCs for the modes.
// This is because some Nektar's built-in methods (FwdTrans, HelmSolve, etc)
// enforce the BCs on the fields stored in m_fields (the mean fields, with
// the inhomogeneous BCs). These helpers allow us to apply those methods to the
// modes, enforcing their homogeneous BCs, without permanently altering the BCs
// of the mean fields.

/**
 * Stores one BC state for one velocity component and one boundary region.
 * - fieldId: which component;
 * - region: which boundary region;
 * - phys and coeffs: the BC arrays for that region, which get overwritten
 *      during mode projection and need to be restored afterward.
 */
struct BcArrayState
{
    int fieldId = -1;
    int region  = -1;
    std::vector<NekDouble> phys;
    std::vector<NekDouble> coeffs;
};

/**
 * Collection of BcArrayState, representing the full BC state for
 * all velocity components in all regions
 */
struct VelocityBCState
{
    std::vector<BcArrayState> entries;
};

/**
 * Saves the current boundary data for all velocity components and BC regions.
 * - fields: all fields (velocity + pressure);
 * - velocity: indices picking velocity components in fields.
 */
VelocityBCState CaptureVelocityBCState(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    const Array<OneD, int> &velocity)
{
    VelocityBCState state;
    for (int c = 0; c < velocity.size(); ++c)
    {
        const int fieldId = velocity[c];
        auto bcs = fields[fieldId]->GetBndConditions();
        auto bnd = fields[fieldId]->GetBndCondExpansions();
        for (int region = 0; region < bcs.size(); ++region)
        {
            // keep only Dirichlet & Neumann
            const auto type = bcs[region]->GetBoundaryConditionType();
            if (type != SpatialDomains::eDirichlet &&
                type != SpatialDomains::eNeumann)
            {
                continue;
            }

            // copy BC values
            BcArrayState entry;
            entry.fieldId = fieldId;
            entry.region  = region;
            entry.phys.assign(bnd[region]->GetPhys().data(),
                              bnd[region]->GetPhys().data() +
                                  bnd[region]->GetPhys().size());
            entry.coeffs.assign(bnd[region]->GetCoeffs().data(),
                                bnd[region]->GetCoeffs().data() +
                                    bnd[region]->GetCoeffs().size());
            state.entries.push_back(std::move(entry));
        }
    }
    return state;
}

/**
 * Zeroes velocity BC data for modes.
 */
void HomogenizeVelocityBCsForModes(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    const Array<OneD, int> &velocity)
{
    for (int c = 0; c < velocity.size(); ++c)
    {
        const int fieldId = velocity[c];
        auto bcs = fields[fieldId]->GetBndConditions();
        auto bnd = fields[fieldId]->GetBndCondExpansions();
        for (int region = 0; region < bcs.size(); ++region)
        {
            // keep only Dirichlet & Neumann
            const auto type = bcs[region]->GetBoundaryConditionType();
            if (type != SpatialDomains::eDirichlet &&
                type != SpatialDomains::eNeumann)
            {
                continue;
            }

            // zero out BC values
            auto phys   = bnd[region]->UpdatePhys();
            auto coeffs = bnd[region]->UpdateCoeffs();
            Vmath::Zero(phys.size(),   phys.data(),   1);
            Vmath::Zero(coeffs.size(), coeffs.data(), 1);
        }
    }
}

/**
 * Restores velocity BC data for modes from a captured state.
 */
void RestoreVelocityBCState(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    const VelocityBCState &state)
{
    for (const auto &entry : state.entries) // loop over saved BC entries
    {
        auto exp = fields[entry.fieldId]->GetBndCondExpansions()[entry.region];
        // restore phys and coeff values
        std::copy(entry.phys.begin(), entry.phys.end(),
                  exp->UpdatePhys().data());
        std::copy(entry.coeffs.begin(), entry.coeffs.end(),
                  exp->UpdateCoeffs().data());
    }
}

// ===========================================================================
// Mode-data helpers
// ===========================================================================
// These helpers store DO modes and define the mass inner product. Modes live
// outside m_fields, so their coeffs and phys values are stored in ModeData.
// Mode operations (MGS, projections, normalisation) use this inner product.

/**
 * Stores one DO mode component-by-component.
 * coeffs[c] and phys[c] are the representations of component c.
 */
struct ModeData
{
    std::vector<std::vector<NekDouble>> coeffs;
    std::vector<std::vector<NekDouble>> phys;
};

/**
 * Computes the vector mass inner product
 *  <u,v>_M=\sum_c u[c]_coeffs^T M v[c]_coeffs, summed over components.
 */

// ===========================================================================
// Constant subspace projection helper
// ===========================================================================
// If no Dirichlet BCs, a constant velocity component leaks into some modes
// and they drift to uniform vector fields. To prevent this, the constant
// velocity subspace v_c(x) = (1/sqrt(|Omega|)) e_c   (c = 0,...,nVel-1)
// is treated as a fixed sub-basis and the DO modes (and their explicit RHS)
// are projected orthogonal to it under the velocity mass inner product.
// The required geometric data (domain area, onesCoeffs, admissibility) is
// held in DOVelocityCorrectionScheme::m_constSubspace, filled once per
// session by BuildConstantSubspaceCache.

/**
 * Projects a mode orthogonal to the constant subspace component-wise, in both
 * phys and coeff representations. We own the mode itself:
 *      - mode.phys[c] -> mode.phys[c] - mean;
 *      - mode.coeffs[c] -> mode.coeffs[c] - mean * onesCoeffs[c];
 * with mean = spatial mean of the mode
 *
 * Used in ReOrthonormalise after each Stokes projection to remove the
 * constant component the projector preserves.
 */
void ProjectOutConstantsFromMode(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    const Array<OneD, int>                            &velocity,
    const DOVelocityCorrectionScheme::ConstantSubspaceCache &cache,
    ModeData                                          &mode)
{
    if (cache.domainArea <= 0.0) return;

    const int nVel = velocity.size();
    for (int c = 0; c < nVel; ++c)              // loop over components
    {
        if (!cache.admissible[c]) continue;     // Dirichlet component: skip

        auto      f     = fields[velocity[c]];
        const int nPhys = f->GetTotPoints();
        const int nCo   = f->GetNcoeffs();

        // \int mode.phys[c] d\Omega via a non-owning view (eArrayWrapper;
        // the default constructor mode would copy)
        Array<OneD, NekDouble> physView(nPhys, mode.phys[c].data(),
                                        eArrayWrapper);
        const NekDouble integ = f->Integral(physView);
        const NekDouble mean  = integ / cache.domainArea;

        // phys: subtract the spatial mean
        for (auto &v : mode.phys[c])
        {
            v -= mean;
        }

        // coeffs: subtract mean * onesCoeffs[c]
        const NekDouble *oc = cache.onesCoeffs[c].data();
        NekDouble       *cc = mode.coeffs[c].data();
        Vmath::Svtvp(nCo, -mean, oc, 1, cc, 1, cc, 1);  // cc -= mean * oc
    }
}

/**
 * Symmetric eigendecomposition A = V diag(w) V^T via LAPACK Dspev.
 * A is the n*n row-major symmetric matrix. Returns:
 *      - eigvals is size n, sorted ascending (LAPACK convention);
 *      - V is size n*n, row-major, with columns as the eigenvectors.
 */
void SymmetricEig(int n,
                  const std::vector<NekDouble> &A,
                  std::vector<NekDouble> &V,
                  std::vector<NekDouble> &eigvals)
{
    // pack upper triangle of A in LAPACK packed format:
    //   AP[i + j*(j+1)/2] = A_{ij}  for i <= j.
    std::vector<NekDouble> AP(n*(n+1)/2);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= j; ++i)
            AP[i + j*(j+1)/2] = A[i*n + j];

    eigvals.assign(n, 0.0);
    std::vector<NekDouble> z(n*n), work(3*n);
    int info = 0;
    Lapack::Dspev('V', 'U', n, AP.data(), eigvals.data(),
                  z.data(), n, work.data(), info);
    ASSERTL0(info == 0, "Dspev failed in SymmetricEig");

    // transpose column-major z into row-major V (V[row*n + col] = z(row, col))
    V.assign(n*n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            V[i*n + k] = z[k*n + i];
}

/**
 * Decodes a hex string (two characters per byte, as written by
 * FilterDOArchive) into nDoubles doubles at dst. Returns false without
 * writing anything when the string length does not match.
 */
bool DecodeHexDoubles(const std::string &hex, NekDouble *dst, int nDoubles)
{
    const int nBytes = nDoubles * (int)sizeof(NekDouble);
    if ((int)hex.size() != 2 * nBytes)
    {
        return false;
    }
    unsigned char *bytes = reinterpret_cast<unsigned char *>(dst);
    for (int b = 0; b < nBytes; ++b)
    {
        unsigned val = 0;
        std::sscanf(hex.c_str() + 2*b, "%02x", &val);
        bytes[b] = static_cast<unsigned char>(val);
    }
    return true;
}

} // namespace

// ===========================================================================
// Class registration
// ===========================================================================

// register DOVelocityCorrectionScheme with the factory
std::string DOVelocityCorrectionScheme::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "DOVelocityCorrectionScheme", DOVelocityCorrectionScheme::create);

// register DOVelocityCorrectionScheme enum value for SolverType
std::string DOVelocityCorrectionScheme::solverTypeLookupId =
    LibUtilities::SessionReader::RegisterEnumValue(
        "SolverType", "DOVelocityCorrectionScheme",
        eDOVelocityCorrectionScheme);

// constructor
DOVelocityCorrectionScheme::DOVelocityCorrectionScheme(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph),
      VelocityCorrectionScheme(pSession, pGraph)
{
}

/**
 * Initialises the solver:
 * - runs base VCS initialisation;
 * - reads DO & forcing parameters;
 * - allocates memory for modes (vel. and pressure) & Yi;
 * - initialises the time integrator m_doScheme.
 */
void DOVelocityCorrectionScheme::v_InitObject(bool DeclareField)
{
    VelocityCorrectionScheme::v_InitObject(DeclareField);   // VCS init
    ASSERTL0(!m_ALESolver,
        "ALE not supported with DOVelocityCorrectionScheme.");
    ASSERTL0(!m_meshDistorted,
        "Distorted mesh not supported with DOVelocityCorrectionScheme.");
    for (int c = 0; c < (int)m_velocity.size(); ++c)
    {
        ASSERTL0(std::dynamic_pointer_cast<MultiRegions::ContField>(
            m_fields[m_velocity[c]]),
            "DOVelocityCorrectionScheme: CG only.");
    }

    // sizes
    const int nPhys   = m_fields[0]->GetTotPoints();
    const int nCoeffs = m_fields[0]->GetNcoeffs();
    const int nPC     = m_pressure->GetNcoeffs();
    const int nDim    = m_velocity.size();
    m_session->LoadParameter("DOModes", m_nDOModes);
    m_session->LoadParameter("DOParticles", m_nDOParticles);

    // initial mode basis: "Laplacian" (eigenmodes) or "POD" from samples
    m_session->LoadSolverInfo("DOInitModeBasis", m_doInitBasis, "Laplacian");
    ASSERTL0(m_doInitBasis == "Laplacian" || m_doInitBasis == "POD",
             "DOInitModeBasis must be 'Laplacian' or 'POD'.");

    m_session->MatchSolverInfo("DOAllowConstantModes", "True",
                               m_doAllowConstantModes, false);
    m_session->LoadParameter("DOYiSeed", m_doYiSeed, m_doYiSeed);
    m_session->LoadParameter("DOYiSigma", m_doYiSigma, m_doYiSigma);

    // Tikhonov regularisation strength
    m_session->LoadParameter("DOInvCovRegEps", m_invCovRegEps, m_invCovRegEps);

    // additive stochastic forcing
    m_session->LoadParameter("DOForcingNumChannels", m_nForcingChannels,
                             m_nForcingChannels);
    m_session->LoadParameter("DOForcingSigma", m_forcingSigma, m_forcingSigma);
    m_session->LoadParameter("DOForcingTau", m_forcingTau, m_forcingTau);
    m_session->LoadParameter("DOForcingSeed", m_forcingSeed, m_forcingSeed);
    if (m_nForcingChannels > 0)
    {
        const int K = m_nForcingChannels;
        m_forcingBasisPhys   = Array<OneD, NekDouble>(K*nDim*nPhys,   0.0);
        m_forcingBasisCoeffs = Array<OneD, NekDouble>(K*nDim*nCoeffs, 0.0);
        m_forcingEta         = Array<OneD, NekDouble>(m_nDOParticles*K, 0.0);
        m_forcingG.assign(m_nDOModes*K, 0.0);
        m_forcingA.assign(m_nDOModes*K, 0.0);
        m_forcingRng.seed(
            static_cast<std::mt19937::result_type>(m_forcingSeed));
    }

    // Particle shard: contiguous block of the global population. Draws are
    // replicated (every rank runs the same RNG stream), so the population
    // is identical for any rank count; only storage and per-particle work
    // are distributed. Eta (small) stays fully replicated.
    {
        LibUtilities::CommSharedPtr rc = m_fields[0]->GetComm()->GetRowComm();
        const int nR  = rc->GetSize();
        const int rk  = rc->GetRank();
        ASSERTL0(m_nDOParticles >= nR,
                 "DOVelocityCorrectionScheme: DOParticles must be >= the "
                 "number of MPI ranks (every rank needs a non-empty "
                 "particle shard).");
        const int rem = m_nDOParticles % nR;
        m_npLocal  = m_nDOParticles / nR + (rk < rem ? 1 : 0);
        m_npOffset = rk * (m_nDOParticles / nR) + std::min(rk, rem);
    }

    // allocate arrays (history owned by m_doScheme, the time integrator)
    m_DOModePhys = Array<OneD, NekDouble>(nDim*nPhys*m_nDOModes, 0.0);
    m_DOModeCoeffs = Array<OneD, NekDouble>(nDim*nCoeffs*m_nDOModes, 0.0);
    m_DOModePCoeffs = Array<OneD, NekDouble>(nPC * m_nDOModes, 0.0);
    m_Yi = Array<OneD, NekDouble>(
        std::max(1, m_npLocal * m_nDOModes), 0.0);
    m_Cij.assign(m_nDOModes*m_nDOModes, 0.0);
    m_Mkli.assign(m_nDOModes*m_nDOModes*m_nDOModes, 0.0);
    m_Zbuf      = Array<OneD, NekDouble>(
        std::max(1, m_npLocal * m_nDOModes * m_nDOModes), 0.0);
    m_NAllBuf   = Array<OneD, NekDouble>(m_nDOModes * nDim * nPhys, 0.0);
    m_modeLap   = Array<OneD, NekDouble>(m_nDOModes * nDim * nPhys, 0.0);
    // One ComputeNModeBody scratch slab per worker: the mode loop indexes
    // slabs by thread when built with OpenMP, and uses a single slab in
    // serial builds (per-mode slabs would waste S*(4*nVel+2)*nPhys doubles).
    int nSlabs = 1;
#ifdef _OPENMP
    nSlabs = std::min(m_nDOModes, omp_get_max_threads());
#endif
    m_NBodyBuf  = Array<OneD, NekDouble>(
        nSlabs * (4*nDim + 2) * nPhys, 0.0);
    // PrecomputeGradients is serial (PhysDeriv allocates via the MemPool,
    // which is not thread-safe), so a single tmp + d2u pair suffices.
    m_gradScratch = Array<OneD, NekDouble>(2 * nPhys, 0.0);

    // physWeights[k] = w_k * J_k at each quadrature point k.
    // Supports tensor-product elements: 2D quads, 3D hexahedra.
    // Tet/prism/pyramid (non-tensor-product) are not supported.
    // Flat index convention: direction 0 is innermost (stride 1),
    // direction 1 next, direction 2 outermost -- matching Nektar's
    // physical array layout and GeomFactors::GetJac() storage order.
    ASSERTL0(nDim == 2 || nDim == 3,
        "DOVelocityCorrectionScheme: physWeights supports "
        "2D and 3D only.");
    m_physWeights = Array<OneD, NekDouble>(nPhys, 0.0);
    for (int e = 0; e < m_fields[0]->GetExpSize(); ++e)
    {
        auto exp_e = m_fields[0]->GetExp(e);
        ASSERTL0(exp_e->DetShapeType() == LibUtilities::eQuadrilateral ||
                     exp_e->DetShapeType() == LibUtilities::eHexahedron,
                 "DOVelocityCorrectionScheme: physWeights supports "
                 "tensor-product elements (quads, hexes) only.");
        const int nq0  = exp_e->GetBasis(0)->GetNumPoints();
        const int nq1  = exp_e->GetBasis(1)->GetNumPoints();
        const auto &w0 = exp_e->GetBasis(0)->GetW();
        const auto &w1 = exp_e->GetBasis(1)->GetW();
        auto *gf = exp_e->GetGeomFactors();
        const auto &jac = gf->GetJac();
        const bool deformed =
            (gf->GetGtype() == SpatialDomains::eDeformed);
        const int off = m_fields[0]->GetPhys_Offset(e);
        if (nDim == 2)
        {
            // q increments in the same order as the phys array:
            // dir0 innermost (stride 1), dir1 outer (stride nq0).
            int q = 0;
            for (int k1 = 0; k1 < nq1; ++k1)
                for (int k0 = 0; k0 < nq0; ++k0, ++q)
                {
                    const NekDouble J = deformed ? jac[q] : jac[0];
                    m_physWeights[off + q] = w0[k0] * w1[k1] * J;
                }
        }
        else  // nDim == 3
        {
            const int nq2  = exp_e->GetBasis(2)->GetNumPoints();
            const auto &w2 = exp_e->GetBasis(2)->GetW();
            int q = 0;
            for (int k2 = 0; k2 < nq2; ++k2)
                for (int k1 = 0; k1 < nq1; ++k1)
                    for (int k0 = 0; k0 < nq0; ++k0, ++q)
                    {
                        const NekDouble J = deformed
                            ? jac[q] : jac[0];
                        m_physWeights[off + q] =
                            w0[k0] * w1[k1] * w2[k2] * J;
                    }
        }
    }

    // Time integrator state-vector layout:
    // - variables 0, ..., m_nDOModes*nVel-1 : mode phys components
    //   (one variable per (mode i, comp c), each of size nPhys);
    // - variable S*nVel : local Y shard (size m_npLocal*m_nDOModes).
    m_doNumModeVars = m_nDOModes * nDim;
    m_doYIdx        = m_doNumModeVars;
    auto timeInt = m_session->GetTimeIntScheme();
    m_doScheme =
        LibUtilities::GetTimeIntegrationSchemeFactory().CreateInstance(
            timeInt.method, timeInt.variant, timeInt.order, timeInt.freeParams);
    m_doOps.DefineOdeRhs(&DOVelocityCorrectionScheme::DOExplicitRhs, this);
    m_doOps.DefineImplicitSolve(
        &DOVelocityCorrectionScheme::DOImplicitSolve, this);

    // constant-subspace data for the strip-constants gauge (collective)
    BuildConstantSubspaceCache();
}

/**
 * Fills m_constSubspace: domainArea = |Omega|, onesCoeffs[c] = FwdTrans of
 * the constant function 1, admissible[c] = true iff component c has no
 * Dirichlet boundary DOFs anywhere (AllReduce: interior ranks see none and
 * must agree globally). Collective; called once from v_InitObject on every
 * rank so the AllReduce and FwdTrans stay aligned.
 */
void DOVelocityCorrectionScheme::BuildConstantSubspaceCache()
{
    const int nVel = m_velocity.size();
    m_constSubspace.onesCoeffs.resize(nVel);
    m_constSubspace.admissible.assign(nVel, false);

    // domain area: |Omega| = \int_\Omega d\Omega (Integral AllReduces)
    auto field0 = m_fields[m_velocity[0]];
    Array<OneD, NekDouble> ones(field0->GetTotPoints(), 1.0);
    m_constSubspace.domainArea = field0->Integral(ones);

    for (int c = 0; c < nVel; ++c)              // loop over components
    {
        auto cf = std::dynamic_pointer_cast<MultiRegions::ContField>(
            m_fields[m_velocity[c]]);
        int numDirBnd =
            (int)cf->GetLocalToGlobalMap()->GetNumGlobalDirBndCoeffs();
        field0->GetComm()->GetRowComm()->AllReduce(numDirBnd,
                                                   LibUtilities::ReduceMax);
        m_constSubspace.admissible[c] = (numDirBnd == 0);
        if (!m_constSubspace.admissible[c])
        {
            continue;
        }

        // coeff representation of 1 (used in ProjectOutConstantsFromMode)
        const int nc = cf->GetNcoeffs();
        m_constSubspace.onesCoeffs[c] = Array<OneD, NekDouble>(nc, 0.0);
        Array<OneD, NekDouble> physOnes(cf->GetTotPoints(), 1.0);
        cf->FwdTrans(physOnes, m_constSubspace.onesCoeffs[c]);
    }
}

/**
 * Gathers the sharded Yi into out (global m_nDOParticles * m_nDOModes, on
 * every rank). Block shards are contiguous in the global particle index,
 * so an AllGatherv with per-rank counts reproduces the global ordering.
 */
void DOVelocityCorrectionScheme::GatherYi(Array<OneD, NekDouble> &out) const
{
    const int S = m_nDOModes;
    LibUtilities::CommSharedPtr rc = m_fields[0]->GetComm()->GetRowComm();
    const int nR = rc->GetSize();
    if (nR == 1)
    {
        Vmath::Vcopy(m_npLocal * S, m_Yi.data(), 1, out.data(), 1);
        return;
    }
    Array<OneD, int> counts(nR), offsets(nR);
    const int rem = m_nDOParticles % nR;
    for (int r = 0; r < nR; ++r)
    {
        counts[r]  = (m_nDOParticles / nR + (r < rem ? 1 : 0)) * S;
        offsets[r] = (r * (m_nDOParticles / nR) + std::min(r, rem)) * S;
    }
    Array<OneD, NekDouble> send(std::max(1, m_npLocal * S));
    Vmath::Vcopy(m_npLocal * S, m_Yi.data(), 1, send.data(), 1);
    rc->AllGatherv(send, out, counts, offsets);
}

/**
 * Runs the VCS IC setup (mean velocity + pressure), and on the first call:
 * - initialises modes;
 * - initialises Yi as i.i.d. Gaussian samples;
 * - rotates the joint (modes, Yi) state so C(0) is diagonal;
 * - orthonormalises modes, projects them to finite-element basis;
 * - recomputes Yi by projection (if POD-initialised), to make (modes, Yi)
 *   self-consistent;
 * - initialises forcing channels;
 * - packs the (modes, Y) state into m_doState and calls
 *   m_doScheme->InitializeScheme so the integrator knows the t=0 state and
 *   computes the initial RHS for its history.
 */
void DOVelocityCorrectionScheme::v_DoInitialise(bool dumpInitialConditions)
{
    VelocityCorrectionScheme::v_DoInitialise(dumpInitialConditions);

    if (!m_modesInitialised)    // only on first call
    {
        if (m_session->DefinesSolverInfo("DORestartFile"))
        {
            // Archive already holds an orthonormal basis; skip eigenbasis/Yi
            // init entirely and restore directly.
            RestoreFromDOArchive(m_session->GetSolverInfo("DORestartFile"));
            DiagonaliseCov();
        }
        else
        {
            if (m_doInitBasis == "POD")
            {
                InitialiseModesFromPOD();
            }
            else
            {
                InitialiseModesFromEllipticEigenbasis();
            }
            InitialiseYi();         // Y: i.i.d. Gaussian, sample mean removed
            DiagonaliseCov();
            ReOrthonormalise();
            // re-project snapshots onto the new basis
            if (m_doInitBasis == "POD" && m_podInitialiser)
            {
                m_podInitialiser->RecomputeYiByProjection(
                    m_DOModePhys, m_DOModeCoeffs, m_Yi, m_nDOParticles,
                    m_npLocal, m_npOffset);
                m_podInitialiser.reset();  // POD state freed

                // de-mean Yi across the GLOBAL particle population: local
                // shard sums, one AllReduce, subtract from the local rows
                const NekDouble invNp =
                    1.0 / static_cast<NekDouble>(m_nDOParticles);
                std::vector<NekDouble> mu(m_nDOModes, 0.0);
                for (int p = 0; p < m_npLocal; ++p)
                    for (int i = 0; i < m_nDOModes; ++i)
                        mu[i] += m_Yi[p * m_nDOModes + i];
                m_fields[m_velocity[0]]->GetComm()->GetRowComm()->AllReduce(
                    mu, LibUtilities::ReduceSum);
                for (int p = 0; p < m_npLocal; ++p)
                    for (int i = 0; i < m_nDOModes; ++i)
                        m_Yi[p * m_nDOModes + i] -= invNp * mu[i];
                DiagonaliseCov();
            }
        }

        InitialiseForcingBasis();   // forcing channels

        // Pack the (modes, Y) initial state into m_doState and initialise the
        // DO subsystem integrator. From here on the scheme owns the multi-step
        // history; DOVelocityCorrectionScheme only sees the post-step
        const int nVel  = m_velocity.size();
        const int nPhys = m_fields[0]->GetTotPoints();
        const int nVar  = m_doNumModeVars + 1;
        m_doState = Array<OneD, Array<OneD, NekDouble>>(nVar);
        // copy modes into state vector
        for (int i = 0; i < m_nDOModes; ++i)    // loop over modes
            for (int c = 0; c < nVel; ++c)      // loop over components
            {
                const int v = i*nVel + c;
                m_doState[v] = Array<OneD, NekDouble>(nPhys);
                Vmath::Vcopy(nPhys, m_DOModePhys.data() + v*nPhys, 1,
                             m_doState[v].data(), 1);
            }
        // copy Yi into state vector
        const int nY = m_npLocal * m_nDOModes;
        m_doState[m_doYIdx] = Array<OneD, NekDouble>(nY);
        Vmath::Vcopy(nY, m_Yi.data(), 1, m_doState[m_doYIdx].data(), 1);

        m_doScheme->InitializeScheme(m_timestep, m_doState, m_time, m_doOps);
        m_doSchemeInited = true;
        m_modesInitialised = true;
    }
}

/**
 * Initialises m_Yi[p*S + i] = Y_{i,p} (mode i for particle p) by drawing
 * i.i.d. Gaussian samples Y_{i,p} ~ N(0, m_doYiSigma^2)
 *
 * Decorrelation: the resulting sample covariance C[i,j] can have non-zero
 * diagonals (of order m_doYiSigma^2/sqrt(Np)). Since ComputeNModeBody uses
 * the simplification mu_i = C[i,i], the caller (v_DoInitialise) must invoke
 * DiagonaliseCov once after this routine to diagonalise the initial C
 * and rotate the modes/Y consistently before the first integration step.
 */
void DOVelocityCorrectionScheme::InitialiseYi()
{
    Vmath::Zero(m_Yi.size(), m_Yi.data(), 1);

    if (m_nDOModes == 0 || m_nDOParticles == 0) return;

    const bool podPath = !m_podSigmas.empty();
    if (podPath)
    {
        ASSERTL0((int)m_podSigmas.size()  == m_nDOModes &&
                 (int)m_podEigVecs.size() == m_nDOModes,
                 "DOVelocityCorrectionScheme::InitialiseYi: "
                 "POD spectrum present but inconsistent "
                 "with m_nDOModes (internal logic error).");
    }
    // Replicated RNG: every rank runs the identical stream and builds the
    // FULL population in yAll, then keeps its shard. This makes the global
    // particle set independent of the rank count.
    std::mt19937 rng(static_cast<std::mt19937::result_type>(m_doYiSeed));
    std::vector<NekDouble> yAll((size_t)m_nDOParticles * m_nDOModes, 0.0);

    // initial Y values
    if (!podPath)
    {
        std::normal_distribution<NekDouble> dist(0.0, m_doYiSigma);
        for (int p = 0; p < m_nDOParticles; ++p)        // loop over particles
            for (int i = 0; i < m_nDOModes; ++i)        // loop over modes
                yAll[p * m_nDOModes + i] = dist(rng);   // sample Y_{i,p}
    }
    else
    {
        // POD path. If K saved snapshots, Np particles, sigma_i the i-th
        // singular value of the snapshot matrix, and v_{p,i} the corresponding
        // right singular vectors:
        // - p < K: Y_{p,i} = sigma_i * v_{p,i}
        // - K <= p < Np: Y_{p,i} ~ N(0, sigma_i^2 / \sqrt{K}) i.i.d.
        const int Kproj = std::min(m_podNumSnapshots, m_nDOParticles);
        const NekDouble invSqrtK =
            (m_podNumSnapshots > 0)
                ? 1.0 / std::sqrt(static_cast<NekDouble>(m_podNumSnapshots))
                : 1.0;
        for (int i = 0; i < m_nDOModes; ++i)            // loop over modes
        {
            const NekDouble sigma_i     = m_podSigmas[i];
            const NekDouble sigma_i_iid = sigma_i * invSqrtK;
            std::normal_distribution<NekDouble> dist(0.0, sigma_i_iid);
            for (int p = 0; p < Kproj; ++p)             // first K particles
            {
                yAll[p * m_nDOModes + i] = sigma_i * m_podEigVecs[i][p];
            }
            for (int p = Kproj; p < m_nDOParticles; ++p)// remaining particles
            {
                yAll[p * m_nDOModes + i] = dist(rng);
            }
        }
    }

    // de-mean each column over the full population (no comm needed: yAll is
    // replicated), then keep this rank's shard
    const NekDouble invNp = 1.0 / static_cast<NekDouble>(m_nDOParticles);
    for (int i = 0; i < m_nDOModes; ++i)
    {
        NekDouble mu = 0.0;
        for (int p = 0; p < m_nDOParticles; ++p)
            mu += yAll[p * m_nDOModes + i];
        mu *= invNp;
        for (int p = 0; p < m_nDOParticles; ++p)
            yAll[p * m_nDOModes + i] -= mu;
    }
    Vmath::Vcopy(m_npLocal * m_nDOModes,
                 yAll.data() + (size_t)m_npOffset * m_nDOModes, 1,
                 m_Yi.data(), 1);
}


/**
 * Reads the XML "ForcingChannels" function block, evaluates each channel's
 * spatial template at quadrature points, FwdTrans -> BwdTrans (FE projection),
 * and mass-normalises each channel so ||g_k||_M = 1. Variable naming convention
 * in the XML: "g{k}_{component}" with k in 1..K and component in {u, v, w}.
 *
 * Two channels with the same shape stay as two independent OU streams
 * driving identical templates.
 */
void DOVelocityCorrectionScheme::InitialiseForcingBasis()
{
    if (m_nForcingChannels == 0) return;
    ASSERTL0(m_session->DefinesFunction("ForcingChannels"),
             "DOVelocityCorrectionScheme: DOForcingNumChannels > 0 but "
              "no XML <FUNCTION NAME=\"ForcingChannels\"> block.");

    const int K       = m_nForcingChannels;
    const int nVel    = m_velocity.size();
    const int nPhys   = m_fields[0]->GetTotPoints();
    const int nCoeffs = m_fields[0]->GetNcoeffs();
    const auto vars   = m_session->GetVariables();  // {"u", "v", [...], "p"}

    // quadrature coordinates (same grid for every velocity component)
    Array<OneD, NekDouble> xq(nPhys), yq(nPhys), zq(nPhys, 0.0);
    m_fields[m_velocity[0]]->GetCoords(xq, yq, zq);
    Array<OneD, NekDouble> phys(nPhys), coeffs(nCoeffs), ip(nCoeffs);

    // read & evaluate the template for every channel and component
    for (int k = 0; k < K; ++k)         // loop over channels
    {
        for (int c = 0; c < nVel; ++c)  // loop over components
        {
            const std::string vname = "g" + std::to_string(k+1) + "_"
                                      + vars[m_velocity[c]];
            ASSERTL0(m_session->GetFunctionType("ForcingChannels", vname) ==
                     LibUtilities::eFunctionTypeExpression,
                     "DOVelocityCorrectionScheme: ForcingChannels VAR \""
                     + vname + "\" missing or not an expression.");
            auto eq = m_session->GetFunction("ForcingChannels", vname);
            eq->Evaluate(xq, yq, zq, phys); // analytical -> phys
            m_fields[m_velocity[c]]->FwdTrans(phys, coeffs); // phys -> coeffs
            m_fields[m_velocity[c]]->BwdTrans(coeffs, phys); // for consistency
            const int pOff = (k*nVel + c)*nPhys;   // physical offset
            const int cOff = (k*nVel + c)*nCoeffs; // coeff offset
            Vmath::Vcopy(nPhys, phys.data(), 1,    // store
                         m_forcingBasisPhys.data() + pOff, 1);
            Vmath::Vcopy(nCoeffs, coeffs.data(), 1,
                         m_forcingBasisCoeffs.data() + cOff, 1);
        }

        // mass-normalise this channel's spatial shape
        NekDouble nrm2 = 0.0;
        for (int c = 0; c < nVel; ++c)  // loop over components
        {
            NekDouble *g_kc_phys = m_forcingBasisPhys.data()
                                   + (k*nVel + c)*nPhys;
            const NekDouble *g_kc_coeffs = m_forcingBasisCoeffs.data()
                                           + (k*nVel + c)*nCoeffs;
            // non-owning view (eArrayWrapper; the default would copy)
            Array<OneD, NekDouble> physView(nPhys, g_kc_phys, eArrayWrapper);
            // ip = M g_kc_coeffs (in coeff space)
            // nrm2 = g_kc_coeffs^T M g_kc_coeffs, summed over components
            m_fields[m_velocity[c]]->IProductWRTBase(physView, ip);
            nrm2 += Vmath::Dot(nCoeffs, g_kc_coeffs, 1, ip.data(), 1);
        }
        m_fields[m_velocity[0]]->GetComm()->GetRowComm()->AllReduce(
            nrm2, LibUtilities::ReduceSum); // MPI: global ||g_k||^2_M
        ASSERTL0(nrm2 > 1e-30, "DOVelocityCorrectionScheme: forcing channel "
                               + std::to_string(k+1) + " has zero mass-norm.");
        const NekDouble inv = 1.0 / std::sqrt(nrm2);    // mass-normalise
        Vmath::Smul(nVel*nPhys, inv, m_forcingBasisPhys.data() + k*nVel*nPhys,
                    1, m_forcingBasisPhys.data() + k*nVel*nPhys, 1);
        Vmath::Smul(nVel*nCoeffs, inv,
                    m_forcingBasisCoeffs.data() + k*nVel*nCoeffs,
                    1, m_forcingBasisCoeffs.data() + k*nVel*nCoeffs, 1);
    }
}

/**
 * One step of the additive forcing's stochastic state, run once at the start
 * of every v_PostIntegrate (before AdvanceModes/AdvanceYi):
 *   - exact OU update per (particle, channel):
 *        tau > 0: eta_{n+1} = alpha eta_n + sigma*sqrt(1-alpha^2) xi,
 *        tau = 0: eta_{n+1} <- sigma sqrt{dt} xi,
 *     where alpha = exp(-dt/tau).
 *   - enforce 0 sample mean across particles.
 *   - computes the mode RHS contribution from the forcing:
 *      <E[f_{stoch}Y_i], u_p> = \sum_k <g_k, u_p> E[eta_k Y_i]
 *                             = \sum_k m_forcingG[i*K + k]
 *                               * m_forcingA[i*K + k].
 */
void DOVelocityCorrectionScheme::AdvanceForcingState()
{
    if (m_nForcingChannels == 0) return;

    const int K       = m_nForcingChannels;
    const int Np      = m_nDOParticles;
    const int S       = m_nDOModes;
    const int nVel    = m_velocity.size();
    const int nPhys   = m_fields[0]->GetTotPoints();
    const int nCoeffs = m_fields[0]->GetNcoeffs();
    const NekDouble sigma = m_forcingSigma;
    const NekDouble tau   = m_forcingTau;
    const NekDouble dt    = m_timestep;

    // OU exact step
    std::normal_distribution<NekDouble> dist(0.0, 1.0);
    if (tau > 0.0)              // coloured in time
    {
        const NekDouble alpha = std::exp(-dt / tau);
        const NekDouble beta  =
            sigma * std::sqrt(std::max(0.0, 1.0 - alpha*alpha));
        for (int q = 0; q < Np*K; ++q)
        {
            m_forcingEta[q] =
                alpha * m_forcingEta[q] + beta * dist(m_forcingRng);
        }
    }
    else                        // white-in-time
    {
        const NekDouble beta = sigma * std::sqrt(dt);
        for (int q = 0; q < Np*K; ++q)
        {
            m_forcingEta[q] = beta * dist(m_forcingRng);
        }
    }

    // per-channel centering across particles
    const NekDouble invNp = 1.0 / static_cast<NekDouble>(Np);
    for (int k = 0; k < K; ++k) // loop over channels
    {
        NekDouble mean = 0.0;
        for (int p = 0; p < Np; ++p)
        {
            mean += m_forcingEta[p*K + k];
        }
        mean *= invNp;
        for (int p = 0; p < Np; ++p)
        {
            m_forcingEta[p*K + k] -= mean;
        }
    }

    // <g_k,u_p>_M =
    //     \sum_c g_{k,coeffs}[c] * IProductWRTBase(u_{p,phys})_{coeffs}[c]
    Array<OneD, NekDouble> ip(nCoeffs);
    std::fill(m_forcingG.begin(), m_forcingG.end(), 0.0);
    for (int k = 0; k < K; ++k)         // loop over channels
        for (int c = 0; c < nVel; ++c)  // loop over components
        {
            NekDouble *g_kc_phys =
                m_forcingBasisPhys.data() + (k*nVel + c)*nPhys;
            // non-owning view (eArrayWrapper; the default would copy)
            Array<OneD, NekDouble> physView(nPhys, g_kc_phys, eArrayWrapper);
            // ip = M g_kc_coeffs
            m_fields[m_velocity[c]]->IProductWRTBase(physView, ip);
            for (int i = 0; i < S; ++i) // loop over modes
            {
                const NekDouble *u_ic_coeffs = m_DOModeCoeffs.data()
                                               + (i*nVel + c)*nCoeffs;
                m_forcingG[i*K + k] += Vmath::Dot(nCoeffs, u_ic_coeffs, 1,
                                                  ip.data(), 1);
            }
        }

    // m_forcingA[i, k] = (1/Np) sum_p Y_{p,i} eta_{p,k}: local particle
    // shard against the replicated eta rows at the global particle index
    std::fill(m_forcingA.begin(), m_forcingA.end(), 0.0);
    for (int p = 0; p < m_npLocal; ++p)
    {
        const NekDouble *Yp = m_Yi.data() + p*S;
        const NekDouble *Ep = m_forcingEta.data() + (m_npOffset + p)*K;
        for (int i = 0; i < S; ++i)
            for (int k = 0; k < K; ++k)
                m_forcingA[i*K + k] += Yp[i] * Ep[k];
    }

    // MPI: G holds quadrature partial sums, A holds particle-shard partial
    // sums; one fused AllReduce covers both
    std::vector<NekDouble> red(2*S*K);
    std::copy(m_forcingG.begin(), m_forcingG.end(), red.begin());
    std::copy(m_forcingA.begin(), m_forcingA.end(), red.begin() + S*K);
    m_fields[m_velocity[0]]->GetComm()->GetRowComm()->AllReduce(
        red, LibUtilities::ReduceSum);
    std::copy(red.begin(), red.begin() + S*K, m_forcingG.begin());
    std::copy(red.begin() + S*K, red.end(), m_forcingA.begin());
    Vmath::Smul((int)m_forcingA.size(), invNp, m_forcingA.data(), 1,
                m_forcingA.data(), 1);
}

/**
 * Recomputes the sample moments via BLAS:
 * - C[i,j] = E[Y_i Y_j] in m_Cij[i*m_nDOModes + j],
 * - M[i,j,k] = E[Y_i Y_j Y_k] in m_Mkli[(i*m_nDOModes+j)*m_nDOModes+k].
 * The particle outer products Z[p*S*S + i*S + j] = Y_{p,i} Y_{p,j} are
 * stored in m_Zbuf and reused by AssembleYRhs (same Yi state within one
 * RHS evaluation).
 *
 * BLAS layout: m_Yi[p*S+i] read column-major with lda=S is the S x Np
 * matrix A(i,p) = Y_{p,i}. C = invN * A A^T is symmetric, so the
 * column-major result equals the row-major m_Cij layout. m_Zbuf read
 * column-major with ld=S*S is Zf((ij),p); M = invN * A Zf^T lands
 * Mf(k,(ij)) at m_Mkli[(i*S+j)*S+k], matching the row-major layout.
 */
void DOVelocityCorrectionScheme::ComputeYMoments()
{
    if (m_nDOModes == 0 || m_nDOParticles == 0) return;
    const int S           = m_nDOModes;
    const NekDouble invN  = 1.0 / static_cast<NekDouble>(m_nDOParticles);

    for (int p = 0; p < m_npLocal; ++p)
    {
        const NekDouble *y = m_Yi.data() + p*S;
        NekDouble       *z = m_Zbuf.data() + p*S*S;
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j)
                z[i*S + j] = y[i] * y[j];
    }
    // shard partials; the global 1/Np weight is applied before the reduce
    Blas::Dgemm('N', 'T', S, S, m_npLocal, invN, m_Yi.data(), S,
                m_Yi.data(), S, 0.0, m_Cij.data(), S);
    Blas::Dgemm('N', 'T', S, S*S, m_npLocal, invN, m_Yi.data(), S,
                m_Zbuf.data(), S*S, 0.0, m_Mkli.data(), S);
    // one fused AllReduce for C and M
    std::vector<NekDouble> red(S*S + S*S*S);
    std::copy(m_Cij.begin(), m_Cij.end(), red.begin());
    std::copy(m_Mkli.begin(), m_Mkli.end(), red.begin() + S*S);
    m_fields[m_velocity[0]]->GetComm()->GetRowComm()->AllReduce(
        red, LibUtilities::ReduceSum);
    std::copy(red.begin(), red.begin() + S*S, m_Cij.begin());
    std::copy(red.begin() + S*S, red.end(), m_Mkli.begin());

    // diagonality and symmetry diagnostics for C (root rank prints; the
    // moments are replicated so every rank would print the same numbers)
    if (m_verbose && m_nDOModes > 1 &&
        m_session->GetComm()->GetSpaceComm()->GetRank() == 0)
    {
        NekDouble maxOff = 0.0, maxDiag = 0.0;
        NekDouble fOff2 = 0.0, fAll2 = 0.0;
        NekDouble symMax = 0.0;
        NekDouble traceC = 0.0,
            minDiag = std::numeric_limits<NekDouble>::max();
        for (int i = 0; i < m_nDOModes; ++i)
        {
            traceC  += m_Cij[i*m_nDOModes + i];
            minDiag  = std::min(minDiag, m_Cij[i*m_nDOModes + i]);
            maxDiag  = std::max(maxDiag, std::abs(m_Cij[i*m_nDOModes + i]));
            for (int j = 0; j < m_nDOModes; ++j)
            {
                const NekDouble v = m_Cij[i*m_nDOModes + j];
                fAll2 += v*v;
                if (i != j)
                {
                    maxOff = std::max(maxOff, std::abs(v));
                    fOff2 += v*v;
                }
                symMax = std::max(symMax,
                    std::abs(m_Cij[i*m_nDOModes + j] -
                             m_Cij[j*m_nDOModes + i]));
            }
        }
        NekDouble fAll = std::sqrt(fAll2);
        NekDouble fOff = std::sqrt(fOff2);
        NekDouble ratio    = (maxDiag > 0) ? maxOff / maxDiag : 0.0;
        NekDouble fRatio   = (fAll  > 0)   ? fOff   / fAll    : 0.0;
        std::cout << "[DOVelocityCorrectionScheme] C-diag: |off|max=" << maxOff
                  << " |diag|max=" << maxDiag
                  << " r=" << ratio
                  << " ||C-diag(C)||/||C||=" << fRatio
                  << " sym_err=" << symMax
                  << " trace(C)=" << traceC
                  << " min(diag(C))=" << minDiag << "\n";
    }
}

/**
 * Precomputes physical-space gradients of every DO mode and the mean field
 * from the current m_DOModePhys and m_fields.  Called twice per outer step:
 * once from v_EvaluateAdvection_SetPressureBCs at modes^n (for mean coupling),
 * once from DOExplicitRhs at modes^{n+1} (for the explicit DO RHS).
 *
 * Fills (nP = GetTotPoints()):
 *   m_modeGrad1[(i*nVel+c)*nVel+d : *nP] = \partial_d u_i[c]
 *   m_meanGrad1[(c*nVel+d)*nP]           = \partial_d u_mean[c]
 * m_modeGrad2 (\partial_d^2 u_i[c]) is no longer cached; the Laplacian
 * in ComputeNMode is recomputed on the fly from u_i to avoid the memory
 * cost (~nModes * nVel^2 * nPhys doubles, ~720 MB for 3D at scale).
 */
void DOVelocityCorrectionScheme::PrecomputeGradients()
{
    const int nVel  = m_velocity.size();
    const int nPhys = m_fields[0]->GetTotPoints();
    const int n1    = m_nDOModes * nVel * nVel * nPhys;
    const int nm    = nVel * nVel * nPhys;
    const int nLin  = m_nDOModes * nVel * nPhys;

    if ((int)m_modeGrad1.size() != n1)
        m_modeGrad1 = Array<OneD, NekDouble>(n1);
    if ((int)m_meanGrad1.size() != nm)
        m_meanGrad1 = Array<OneD, NekDouble>(nm);
    if ((int)m_modeLinRhs.size() != nLin)
        m_modeLinRhs = Array<OneD, NekDouble>(nLin);

    // Mode gradients and Laplacians: sequential -- PhysDeriv internally
    // constructs Array<OneD,NekDouble> via MemPool, which is not thread-safe.
    // m_gradScratch[0..2*nPhys) provides reusable tmp and d2u buffers.
    // First derivatives use the all-directions PhysDeriv: the per-direction
    // overload computes every parametric derivative anyway and discards all
    // but one, so one batched call replaces nVel redundant tensor sweeps.
    NekDouble *tmp_ptr = m_gradScratch.data();
    NekDouble *d2u_ptr = tmp_ptr + nPhys;
    for (int i = 0; i < m_nDOModes; ++i)
        for (int c = 0; c < nVel; ++c)
        {
            const NekDouble *u_ic = m_DOModePhys.data() + (i*nVel+c)*nPhys;
            Vmath::Vcopy(nPhys, u_ic, 1, tmp_ptr, 1);
            Array<OneD, NekDouble> tmp_a(nPhys, tmp_ptr, eArrayWrapper);
            Array<OneD, NekDouble> g0 =
                m_modeGrad1 + ((i*nVel+c)*nVel + 0)*nPhys;
            Array<OneD, NekDouble> g1 =
                m_modeGrad1 + ((i*nVel+c)*nVel + 1)*nPhys;
            if (nVel == 2)
            {
                m_fields[m_velocity[c]]->PhysDeriv(tmp_a, g0, g1);
            }
            else
            {
                Array<OneD, NekDouble> g2 =
                    m_modeGrad1 + ((i*nVel+c)*nVel + 2)*nPhys;
                m_fields[m_velocity[c]]->PhysDeriv(tmp_a, g0, g1, g2);
            }
            NekDouble *lap_ic = m_modeLap.data() + (i*nVel+c)*nPhys;
            Vmath::Zero(nPhys, lap_ic, 1);
            Array<OneD, NekDouble> d2u_a(nPhys, d2u_ptr, eArrayWrapper);
            for (int d = 0; d < nVel; ++d)
            {
                Array<OneD, NekDouble> du(nPhys,
                    m_modeGrad1.data() + ((i*nVel+c)*nVel+d)*nPhys,
                    eArrayWrapper);
                m_fields[m_velocity[c]]->PhysDeriv(d, du, d2u_a);
                Vmath::Vadd(nPhys, d2u_ptr, 1, lap_ic, 1, lap_ic, 1);
            }
        }

    // Mean field gradients: one all-directions PhysDeriv per component.
    {
        Array<OneD, NekDouble> tmp(nPhys);
        for (int c = 0; c < nVel; ++c)
        {
            Vmath::Vcopy(nPhys,
                         m_fields[m_velocity[c]]->GetPhys().data(), 1,
                         tmp.data(), 1);
            Array<OneD, NekDouble> mg0 = m_meanGrad1 + (c*nVel + 0)*nPhys;
            Array<OneD, NekDouble> mg1 = m_meanGrad1 + (c*nVel + 1)*nPhys;
            if (nVel == 2)
            {
                m_fields[m_velocity[c]]->PhysDeriv(tmp, mg0, mg1);
            }
            else
            {
                Array<OneD, NekDouble> mg2 =
                    m_meanGrad1 + (c*nVel + 2)*nPhys;
                m_fields[m_velocity[c]]->PhysDeriv(tmp, mg0, mg1, mg2);
            }
        }
    }
}

/**
 * Adds the DO contribution to the mean velocity's explicit term,
 *   doCorr[c][k] = -\sum_{i,j} C[i,j] (u_i(x_k) . grad) u_j(x_k)
 *                = -\sum_{i,j} C[i,j]
 *                  \sum_d u_i[d](x_k) \partial_d u_j[c](x_k).
 * doCorr[c][k] is the c-th spatial component at quadrature point k.
 */
void DOVelocityCorrectionScheme::ComputeDOMeanCoupling(
    Array<OneD, Array<OneD, NekDouble>> &doCorr)
{
    const int nVel  = m_velocity.size();
    const int nPhys = m_fields[0]->GetTotPoints();

    for (int c = 0; c < nVel; ++c)
    {
        Vmath::Zero(nPhys, doCorr[c].data(), 1);
    }
    if (m_nDOModes == 0) return;

    // doCorr[c] = -sum_{i,j} C_{ij} * (u_i . grad) u_j[c].
    // m_modeGrad1[(j*nVel+c)*nVel+d : *nPhys] = \partial_d u_j[c].
    Array<OneD, NekDouble> prod(nPhys);
    for (int i = 0; i < m_nDOModes; ++i)
        for (int j = 0; j < m_nDOModes; ++j)
        {
            const NekDouble Cij = m_Cij[i*m_nDOModes + j];
            if (std::abs(Cij) < 1e-12) continue;
            for (int c = 0; c < nVel; ++c)
                for (int d = 0; d < nVel; ++d)
                {
                    const NekDouble *u_id = m_DOModePhys.data()
                                          + (i*nVel + d)*nPhys;
                    const NekDouble *du = m_modeGrad1.data()
                                          + ((j*nVel + c)*nVel + d)*nPhys;
                    Vmath::Vmul(nPhys, u_id, 1, du, 1, prod.data(), 1);
                    Vmath::Svtvp(nPhys, -Cij, prod.data(), 1,
                                 doCorr[c].data(), 1, doCorr[c].data(), 1);
                }
        }
}

/**
 * Computes the nonlinear term of mode i's PDE (pre-projection):
 *      - computes the regularisation parameter invMuReg for C inverse;
 *      - computes the triple moment contribution triple;
 *      - computes the stochastic forcing contribution addStochN;
 *      - assembles:
 *          N = cross + invMuReg * (triple + addStochN)
 *          innerArg = N + nu*lap
 *      - betasOut[p]    = local part of <innerArg, u_p> (DO constraint);
 *      - constIntOut[c] = local part of \int innerArg[c] d\Omega (strip-
 *        constants gauge; zero when the gauge is inactive).
 * The caller AllReduces betasOut / constIntOut across ranks, then subtracts
 * the global innerArg mean and the beta projections from N.
 * Cross and Laplacian are read from m_modeGrad1 and m_modeLap respectively,
 * cached by PrecomputeGradients; no PhysDeriv calls here.
 *
 * Thread-safe: uses only pre-allocated scratch (bodyBuf, one slab per mode)
 * and per-mode output slots, and reads shared read-only arrays. No Nektar
 * MemPool allocations happen here, so DOExplicitRhs may call it inside an
 * OpenMP parallel region. Scratch layout per mode (SLOTS = 4*nVel+2 slabs
 * of nPhys):
 *   [0..nVel)        cross[c],    [nVel..2*nVel)   triple[c],
 *   [2*nVel..3*nVel) innerArg[c], [3*nVel..4*nVel) addStochN[c],
 *   [4*nVel]         prod,        [4*nVel+1]       wArg.
 */
void DOVelocityCorrectionScheme::ComputeNModeBody(
    int i, Array<OneD, Array<OneD, NekDouble>> &N, NekDouble *bodyBuf,
    NekDouble *betasOut, NekDouble *constIntOut)
{
    const int nVel  = m_velocity.size();
    const int nPhys = m_fields[0]->GetTotPoints();
    const NekDouble nu  = m_kinvis;
    const NekDouble mui = m_Cij[i*m_nDOModes + i];
    const NekDouble eps = 1e-12;

    NekDouble muMax = 0.0;
    for (int q = 0; q < m_nDOModes; ++q)
        muMax = std::max(muMax, std::abs(m_Cij[q*m_nDOModes + q]));
    const NekDouble lambdaReg = m_invCovRegEps * muMax;
    const NekDouble invMuReg  = mui / (mui*mui + lambdaReg*lambdaReg);

    // Carve named slabs out of the pre-allocated bodyBuf (no MemPool).
    NekDouble *cross_raw     = bodyBuf + 0*nVel*nPhys;
    NekDouble *triple_raw    = bodyBuf + 1*nVel*nPhys;
    NekDouble *innerArg_raw  = bodyBuf + 2*nVel*nPhys;
    NekDouble *addStochN_raw = bodyBuf + 3*nVel*nPhys;
    NekDouble *prod_raw      = bodyBuf + 4*nVel*nPhys;
    NekDouble *wArg_raw      = bodyBuf + (4*nVel+1)*nPhys;

    for (int c = 0; c < nVel; ++c)
    {
        Vmath::Zero(nPhys, cross_raw     + c*nPhys, 1);
        Vmath::Zero(nPhys, triple_raw    + c*nPhys, 1);
        Vmath::Zero(nPhys, innerArg_raw  + c*nPhys, 1);
        Vmath::Zero(nPhys, addStochN_raw + c*nPhys, 1);
        Vmath::Zero(nPhys, N[c].data(),             1);
    }

    // cross[c] = -[(u_mean . \nabla) u_i + (u_i . \nabla) u_mean][c]
    for (int c = 0; c < nVel; ++c)
    {
        NekDouble *cross_c = cross_raw + c*nPhys;
        for (int d = 0; d < nVel; ++d)
        {
            const NekDouble *du_icd =
                m_modeGrad1.data() + ((i*nVel+c)*nVel+d)*nPhys;
            const NekDouble *uBar_d =
                m_fields[m_velocity[d]]->GetPhys().data();
            Vmath::Vmul(nPhys, uBar_d, 1, du_icd, 1, prod_raw, 1);
            Vmath::Svtvp(nPhys, -1.0, prod_raw, 1, cross_c, 1, cross_c, 1);
            const NekDouble *u_id    = m_DOModePhys.data() + (i*nVel+d)*nPhys;
            const NekDouble *dBar_cd = m_meanGrad1.data() + (c*nVel+d)*nPhys;
            Vmath::Vmul(nPhys, u_id, 1, dBar_cd, 1, prod_raw, 1);
            Vmath::Svtvp(nPhys, -1.0, prod_raw, 1, cross_c, 1, cross_c, 1);
        }
    }

    // triple[c] = -\sum_{m,l} M_{mli} (u_m . \nabla) u_l[c]
    if (invMuReg > eps)
    {
        for (int mm = 0; mm < m_nDOModes; ++mm)
            for (int ll = 0; ll < m_nDOModes; ++ll)
            {
                const NekDouble Mml =
                    m_Mkli[(mm*m_nDOModes + ll)*m_nDOModes + i];
                if (std::abs(Mml) < eps) continue;
                for (int c = 0; c < nVel; ++c)
                {
                    NekDouble *triple_c = triple_raw + c*nPhys;
                    for (int d = 0; d < nVel; ++d)
                    {
                        const NekDouble *u_md = m_DOModePhys.data()
                                               + (mm*nVel + d)*nPhys;
                        const NekDouble *du   = m_modeGrad1.data()
                            + ((ll*nVel + c)*nVel + d)*nPhys;
                        Vmath::Vmul(nPhys, u_md, 1, du, 1, prod_raw, 1);
                        Vmath::Svtvp(nPhys, -Mml, prod_raw, 1,
                                     triple_c, 1, triple_c, 1);
                    }
                }
            }
    }

    // stochastic contribution into addStochN_raw
    if (m_nForcingChannels > 0 && invMuReg > eps)
    {
        for (int k = 0; k < m_nForcingChannels; ++k)
        {
            const NekDouble Aik = m_forcingA[i*m_nForcingChannels + k];
            if (std::abs(Aik) < eps) continue;
            for (int c = 0; c < nVel; ++c)
            {
                const NekDouble *gk =
                    m_forcingBasisPhys.data() + (k*nVel + c)*nPhys;
                NekDouble *stoch_c = addStochN_raw + c*nPhys;
                Vmath::Svtvp(nPhys, Aik, gk, 1, stoch_c, 1, stoch_c, 1);
            }
        }
    }

    // cache F_i = cross + nu*lap for BuildYRhsTensors
    for (int c = 0; c < nVel; ++c)
    {
        const NekDouble *lap_ic = m_modeLap.data() + (i*nVel + c)*nPhys;
        NekDouble       *fic    = m_modeLinRhs.data() + (i*nVel + c)*nPhys;
        const NekDouble *cc     = cross_raw + c*nPhys;
        for (int k = 0; k < nPhys; ++k)
            fic[k] = cc[k] + nu * lap_ic[k];
    }

    // assemble N = cross + invMuReg*(triple+stoch); innerArg = N + nu*lap
    for (int c = 0; c < nVel; ++c)
    {
        const NekDouble *lap_ic    = m_modeLap.data() + (i*nVel + c)*nPhys;
        const NekDouble *cross_c   = cross_raw     + c*nPhys;
        const NekDouble *triple_c  = triple_raw    + c*nPhys;
        const NekDouble *stoch_c   = addStochN_raw + c*nPhys;
        NekDouble       *inner_c   = innerArg_raw  + c*nPhys;
        for (int k = 0; k < nPhys; ++k)
        {
            const NekDouble triple_scaled = invMuReg * triple_c[k];
            const NekDouble stoch_scaled  = invMuReg * stoch_c[k];
            const NekDouble rough = cross_c[k] + triple_scaled + stoch_scaled;
            N[c][k]    = rough;
            inner_c[k] = rough + nu * lap_ic[k];
        }
    }

    // local (pre-AllReduce) integrals of innerArg for the strip-constants
    // gauge; the caller reduces them and subtracts the global mean from N
    for (int c = 0; c < nVel; ++c)
    {
        constIntOut[c] = 0.0;
    }
    if (!m_doAllowConstantModes && m_constSubspace.domainArea > 0.0)
    {
        for (int c = 0; c < nVel; ++c)
        {
            if (!m_constSubspace.admissible[c]) continue;
            const NekDouble *inner_c = innerArg_raw + c*nPhys;
            constIntOut[c] = Vmath::Dot(nPhys, inner_c, 1,
                                        m_physWeights.data(), 1);
        }
    }

    // local (pre-AllReduce) inner products <innerArg, u_p>: one Dgemv per
    // component against the mode matrix, whose column p for component c
    // starts at m_DOModePhys + (p*nVel + c)*nPhys (leading dim nVel*nPhys)
    for (int c = 0; c < nVel; ++c)
    {
        const NekDouble *inner_c = innerArg_raw + c*nPhys;
        Vmath::Vmul(nPhys, m_physWeights.data(), 1, inner_c, 1, wArg_raw, 1);
        Blas::Dgemv('T', nPhys, m_nDOModes, 1.0,
                    m_DOModePhys.data() + c*nPhys, nVel*nPhys,
                    wArg_raw, 1, (c == 0) ? 0.0 : 1.0, betasOut, 1);
    }
}

/**
 * Explicit RHS for the per-particle coefficients Y_{p,i}:
 *      RHS_{p,i} = \sum_k Y_{p,k} <F_k - grad(p_k), u_i>
 *                + \sum_{k,l} (Y_{p,k}Y_{p,l} - C_{kl}) <F_{kl}, u_i>
 *                + \sum_k \eta_{p,k} m_forcingG_{k,i},
 * split into two phases around the caller's single AllReduce:
 * BuildYRhsTensors fills the local (pre-reduce) inner-product tensors
 *      - ipKi[k*S+i]        = <F_k - grad(p_k), u_i>;
 *      - ipKli[(k*S+l)*S+i] = <F_{kl}, u_i>;
 * and AssembleYRhs contracts the reduced tensors against the particles.
 *
 * grad(p_k) uses one all-directions PhysDeriv (the per-direction overload
 * recomputes every parametric derivative anyway). The inner products
 * against the modes are Dgemv calls: the component-c mode matrix has
 * column i at (i*nVel + c)*nPhys with leading dimension nVel*nPhys.
 */
void DOVelocityCorrectionScheme::BuildYRhsTensors(NekDouble *ipKi,
                                                  NekDouble *ipKli)
{
    const int S       = m_nDOModes;
    const int nVel    = m_velocity.size();
    const int nPhys   = m_fields[0]->GetTotPoints();
    const int nPC     = m_pressure->GetNcoeffs();
    const int nPP     = m_pressure->GetTotPoints();

    Array<OneD, NekDouble> wFk(nPhys);
    Array<OneD, Array<OneD, NekDouble>> Fk(nVel), dpk(nVel);
    Array<OneD, NekDouble> pkPhys(nPP);
    for (int c = 0; c < nVel; ++c)
    {
        Fk[c]  = Array<OneD, NekDouble>(nPhys);
        dpk[c] = Array<OneD, NekDouble>(nPhys);
    }
    for (int k = 0; k < S; ++k)
    {
        for (int c = 0; c < nVel; ++c)
            Vmath::Vcopy(nPhys, m_modeLinRhs.data() + (k*nVel+c)*nPhys, 1,
                         Fk[c].data(), 1);
        Array<OneD, NekDouble> pkCoeffs = m_DOModePCoeffs + k*nPC;
        m_pressure->BwdTrans(pkCoeffs, pkPhys);
        if (nVel == 2)
        {
            m_pressure->PhysDeriv(pkPhys, dpk[0], dpk[1]);
        }
        else
        {
            m_pressure->PhysDeriv(pkPhys, dpk[0], dpk[1], dpk[2]);
        }
        for (int c = 0; c < nVel; ++c)
        {
            Vmath::Vsub(nPhys, Fk[c].data(), 1, dpk[c].data(), 1,
                        Fk[c].data(), 1);
            Vmath::Vmul(nPhys, m_physWeights.data(), 1, Fk[c].data(), 1,
                        wFk.data(), 1);
            Blas::Dgemv('T', nPhys, S, 1.0,
                        m_DOModePhys.data() + c*nPhys, nVel*nPhys,
                        wFk.data(), 1, (c == 0) ? 0.0 : 1.0, ipKi + k*S, 1);
        }
    }

    Array<OneD, NekDouble> prod(nPhys);
    Array<OneD, Array<OneD, NekDouble>> Fkl(nVel);
    for (int c = 0; c < nVel; ++c)
        Fkl[c] = Array<OneD, NekDouble>(nPhys);
    for (int k = 0; k < S; ++k)
    {
        for (int l = 0; l < S; ++l)
        {
            for (int c = 0; c < nVel; ++c)
                Vmath::Zero(nPhys, Fkl[c].data(), 1);
            for (int c = 0; c < nVel; ++c)
                for (int d = 0; d < nVel; ++d)
                {
                    const NekDouble *u_kd =
                        m_DOModePhys.data() + (k*nVel + d)*nPhys;
                    const NekDouble *du =
                        m_modeGrad1.data() + ((l*nVel + c)*nVel + d)*nPhys;
                    Vmath::Vmul(nPhys, u_kd, 1, du, 1, prod.data(), 1);
                    Vmath::Svtvp(nPhys, -1.0, prod.data(), 1,
                                 Fkl[c].data(), 1, Fkl[c].data(), 1);
                }
            for (int c = 0; c < nVel; ++c)
            {
                Vmath::Vmul(nPhys, m_physWeights.data(), 1, Fkl[c].data(),
                            1, wFk.data(), 1);
                Blas::Dgemv('T', nPhys, S, 1.0,
                            m_DOModePhys.data() + c*nPhys, nVel*nPhys,
                            wFk.data(), 1, (c == 0) ? 0.0 : 1.0,
                            ipKli + (k*S + l)*S, 1);
            }
        }
    }
}

/**
 * Contracts the globally reduced Y-RHS tensors against the particles:
 *      rhs[p*S+i] = \sum_k Y_{p,k} ipKi[k*S+i]
 *                 + \sum_{k,l} (Y_{p,k}Y_{p,l} - C_{kl}) ipKli[(k*S+l)*S+i]
 *                 + \sum_k \eta_{p,k} m_forcingG[i*K+k].
 * BLAS layout: with row-major Yi (Np x S) read column-major (lda = S),
 * Dgemm('N','N', S, Np, S) forms rhs(p,i) = \sum_k ipKi(k,i) Y_{p,k}
 * directly in the row-major rhs. The quadratic term reuses m_Zbuf (built
 * by ComputeYMoments from the same Yi) and accumulates with beta = 1; its
 * C part, t0[i] = \sum_{kl} C_{kl} ipKli[(kl)*S+i], is particle-
 * independent and subtracted once per particle row.
 */
void DOVelocityCorrectionScheme::AssembleYRhs(const NekDouble *ipKi,
                                              const NekDouble *ipKli,
                                              Array<OneD, NekDouble> &rhs)
{
    const int S  = m_nDOModes;
    const int Kf = m_nForcingChannels;

    // linear term over the local particle shard: rhs = Y ipKi
    Blas::Dgemm('N', 'N', S, m_npLocal, S, 1.0, ipKi, S,
                m_Yi.data(), S, 0.0, rhs.data(), S);
    // quadratic term: rhs += Z ipKli ...
    Blas::Dgemm('N', 'N', S, m_npLocal, S*S, 1.0, ipKli, S,
                m_Zbuf.data(), S*S, 1.0, rhs.data(), S);
    // ... minus its C contraction, identical for every particle
    std::vector<NekDouble> t0(S, 0.0);
    for (int kl = 0; kl < S*S; ++kl)
    {
        const NekDouble Ckl = m_Cij[kl];
        for (int i = 0; i < S; ++i)
            t0[i] += Ckl * ipKli[kl*S + i];
    }
    for (int p = 0; p < m_npLocal; ++p)
    {
        NekDouble *Rp = rhs.data() + p*S;
        for (int i = 0; i < S; ++i)
            Rp[i] -= t0[i];
        if (Kf > 0)
        {
            // replicated eta row of this local particle's global index
            const NekDouble *Ep =
                m_forcingEta.data() + (m_npOffset + p)*Kf;
            for (int i = 0; i < S; ++i)
            {
                NekDouble frc = 0.0;
                for (int k = 0; k < Kf; ++k)
                    frc += Ep[k] * m_forcingG[i*Kf + k];
                Rp[i] += frc;
            }
        }
    }
}

/**
 * VCS override:
 *      - adds correction doCorr to mean's explicit RHS outarray on top of
 *        advection;
 *      - also captures the current mean field into m_meanAtTn for use in
 *        DOExplicitRhs (needed by DO subsystem).
 */
void DOVelocityCorrectionScheme::v_EvaluateAdvection_SetPressureBCs(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time)
{
    // base VCS: outarray = -(u_mean . grad)u_mean + pressure BCs
    VelocityCorrectionScheme::v_EvaluateAdvection_SetPressureBCs(
        inarray, outarray, time);

    if (m_nDOModes == 0) return;

    const int nVel  = m_velocity.size();
    const int nPhys = m_fields[0]->GetTotPoints();

    if (m_meanAtTn.size() == 0)
    {
        m_meanAtTn = Array<OneD, Array<OneD, NekDouble>>(nVel);
        for (int c = 0; c < nVel; ++c)
            m_meanAtTn[c] = Array<OneD, NekDouble>(nPhys, 0.0);
    }
    for (int c = 0; c < nVel; ++c)  // capture current inarray in m_meanAtTn
        Vmath::Vcopy(nPhys, inarray[c].data(), 1, m_meanAtTn[c].data(), 1);
    m_meanSnapshotValid = true;

    ComputeYMoments();

    // Precompute mode and mean gradients at u^n for ComputeDOMeanCoupling.
    // DOExplicitRhs will re-run PrecomputeGradients at modes^{n+1} (the
    // post-implicit state it receives as in from the GLM stage loop).
    PrecomputeGradients();

    // initialise doCorr & its components, fill it
    Array<OneD, Array<OneD, NekDouble>> doCorr(nVel);
    for (int c = 0; c < nVel; ++c)
        doCorr[c] = Array<OneD, NekDouble>(nPhys, 0.0);
    ComputeDOMeanCoupling(doCorr);

    for (int c = 0; c < nVel; ++c)  // outarray[c] += doCorr[c]
        Vmath::Vadd(nPhys, doCorr[c].data(), 1, outarray[c].data(), 1,
                    outarray[c].data(), 1);
}

/**
 * Solves the pressure Poisson equation for one DO mode (core, no save/restore).
 * Caller is responsible for zeroing pbnd and restoring m_pressure state.
 */
void DOVelocityCorrectionScheme::ModePressureSolve(
    const Array<OneD, Array<OneD, NekDouble>> &uhatPhys,
    NekDouble aii_Dt, const Array<OneD, const NekDouble> &pGuess,
    Array<OneD, NekDouble> &pCoeffsOut)
{
    const int nVel  = m_velocity.size();
    const int nPhys = m_fields[0]->GetTotPoints();
    const int npC   = m_pressure->GetNcoeffs();

    // poisson_RHS = div(uhat) / aii_Dt
    Array<OneD, NekDouble> poisson_RHS(nPhys, 0.0), tmp(nPhys, 0.0);
    for (int c = 0; c < nVel; ++c)
    {
        m_fields[m_velocity[c]]->PhysDeriv(c, uhatPhys[c], tmp);
        Vmath::Vadd(nPhys, tmp.data(), 1, poisson_RHS.data(), 1,
                    poisson_RHS.data(), 1);
    }
    Vmath::Smul(nPhys, 1.0/aii_Dt, poisson_RHS.data(), 1,
                poisson_RHS.data(), 1);

    // Save and zero pressure BCs (modes need homogeneous Neumann).
    auto pbnd = m_pressure->GetBndCondExpansions();
    std::vector<std::vector<NekDouble>> savedPBnd(pbnd.size());
    for (int n = 0; n < (int)pbnd.size(); ++n)
    {
        const int nc = pbnd[n]->GetNcoeffs();
        savedPBnd[n].assign(pbnd[n]->GetCoeffs().data(),
                            pbnd[n]->GetCoeffs().data() + nc);
        Vmath::Zero(nc, pbnd[n]->UpdateCoeffs().data(), 1);
    }

    StdRegions::ConstFactorMap factors;
    factors[StdRegions::eFactorLambda] = 0.0;
    // Warm start from this mode's previous pressure solution: iterative
    // solvers converge in fewer iterations, direct solvers ignore it, and
    // the answer is guess-independent (a pinned Dirichlet vertex removes
    // the pure-Neumann null space at the assembly-map level).
    Vmath::Vcopy(npC, pGuess.data(), 1, m_pressure->UpdateCoeffs().data(), 1);
    m_pressure->HelmSolve(poisson_RHS, m_pressure->UpdateCoeffs(), factors);
    Vmath::Vcopy(npC, m_pressure->GetCoeffs().data(), 1, pCoeffsOut.data(), 1);

    // Restore pressure BCs.
    for (int n = 0; n < (int)pbnd.size(); ++n)
        std::copy(savedPBnd[n].begin(), savedPBnd[n].end(),
                  pbnd[n]->UpdateCoeffs().data());
}

/**
 * Solves the viscous Helmholtz step for one DO mode, component-wise.
 * Builds Helmholtz RHS = (- uhat / aii_Dt + grad(p)) / \nu, then calls
 * HelmSolve for each velocity component (zeroing its coeffs first).
 * Leaves m_fields and m_pressure in the mode state; DOImplicitSolve
 * restores the mean-field state after the full mode loop.
 *
 * The equation solved is valid for any IMEX order:
 *  (lap - 1/(\nu aii_Dt))u_k^{n+1} = -uhat_k/(\nu aii_Dt)
 *                                    + \partial_k p^{n+1} / \nu.
 */
void DOVelocityCorrectionScheme::ModeViscousSolve(
    const Array<OneD, Array<OneD, NekDouble>> &uhatPhys,
    const Array<OneD, NekDouble> &pCoeffsIn, NekDouble aii_Dt,
    const Array<OneD, const NekDouble> &uGuessCoeffs,
    Array<OneD, Array<OneD, NekDouble>> &uNewPhys,
    Array<OneD, Array<OneD, NekDouble>> &uNewCoeffs)
{
    const int nVel  = m_velocity.size();
    const int nPhys = m_fields[0]->GetTotPoints();
    const int npC   = m_pressure->GetNcoeffs();

    // Helmholtz RHS [k] = (- uhat_k / aii_Dt + \partial_k p) / nu
    Array<OneD, Array<OneD, NekDouble>> helmholtz_RHS(m_nConvectiveFields);
    // copy pCoeffsIn into m_pressure, compute gradient
    Vmath::Vcopy(npC, pCoeffsIn.data(), 1,
                 m_pressure->UpdateCoeffs().data(), 1);
    m_pressure->BwdTrans(m_pressure->GetCoeffs(), m_pressure->UpdatePhys());
    for (int k = 0; k < m_nConvectiveFields; ++k)       // allocate space
        helmholtz_RHS[k] = Array<OneD, NekDouble>(nPhys, 0.0);
    if (nVel == 2)  // differentiates p wrt all convective directions
        m_pressure->PhysDeriv(m_pressure->GetPhys(), helmholtz_RHS[0],
                              helmholtz_RHS[1]);
    else
        m_pressure->PhysDeriv(m_pressure->GetPhys(), helmholtz_RHS[0],
                              helmholtz_RHS[1], helmholtz_RHS[2]);
    // helmholtz_RHS[k] -= uhatPhys[k]/(\nu aii_Dt)
    for (int k = 0; k < m_nConvectiveFields; ++k)
    {
        if (k < nVel)
            Vmath::Svtvp(nPhys, -1.0/aii_Dt, uhatPhys[k].data(), 1,
                         helmholtz_RHS[k].data(), 1,
                         helmholtz_RHS[k].data(), 1);
        Vmath::Smul(nPhys, 1.0/m_diffCoeff[k],
                    helmholtz_RHS[k].data(), 1, helmholtz_RHS[k].data(), 1);
    }

    // solve (lap - lambda) u_k = helmholtz_RHS[k]
    StdRegions::ConstFactorMap factors;
    for (int k = 0; k < m_nConvectiveFields; ++k)
    {
        // eFactorLambda = 1/(aii_Dt*nu), where aii_Dt is the implicit weight
        factors[StdRegions::eFactorLambda] = 1.0/aii_Dt/m_diffCoeff[k];
        // warm start from this mode's previous velocity solution (used by
        // iterative solvers as the initial guess, ignored by direct ones)
        if (k < nVel)
        {
            Vmath::Vcopy(m_fields[m_velocity[k]]->GetNcoeffs(),
                         uGuessCoeffs.data() +
                             k*m_fields[m_velocity[k]]->GetNcoeffs(), 1,
                         m_fields[m_velocity[k]]->UpdateCoeffs().data(), 1);
        }
        else
        {
            Vmath::Zero(m_fields[m_velocity[k]]->GetNcoeffs(),
                        m_fields[m_velocity[k]]->UpdateCoeffs().data(), 1);
        }
        m_fields[m_velocity[k]]->HelmSolve(helmholtz_RHS[k],
            m_fields[m_velocity[k]]->UpdateCoeffs(), factors);
        m_fields[m_velocity[k]]->BwdTrans(
            m_fields[m_velocity[k]]->GetCoeffs(), uNewPhys[k]);
        Vmath::Vcopy(m_fields[m_velocity[k]]->GetNcoeffs(),
                     m_fields[m_velocity[k]]->GetCoeffs().data(), 1,
                     uNewCoeffs[k].data(), 1);
    }
}

/**
 * RHS for explicit (Poisson) step, called by the time integrator m_doScheme.
 *      - in[0,...,m_nDOModes*nVel-1]: mode phys at time t (one per (i,c));
 *      - in[m_doYIdx]: local Y shard at time t (m_npLocal*m_nDOModes).
 *      - out: unscaled explicit RHS (dt weight applied by the integrator).
 *
 * m_fields phys swapped to m_meanAtTn for the duration of the callback.
 * m_DOModePhys/m_DOModeCoeffs/m_Yi are synced from in so ComputeYMoments,
 * ComputeNMode operates on the correct state.
 */
void DOVelocityCorrectionScheme::DOExplicitRhs(
    const Array<OneD, const Array<OneD, NekDouble>> &in,
    Array<OneD, Array<OneD, NekDouble>>             &out,
    [[maybe_unused]] const NekDouble                 time)
{
    if (m_nDOModes == 0) return;

    const int nVel    = m_velocity.size();
    const int nPhys   = m_fields[0]->GetTotPoints();

    // copy modes from in to m_DOModePhys
    for (int i = 0; i < m_nDOModes; ++i)
        for (int c = 0; c < nVel; ++c)
            Vmath::Vcopy(nPhys, in[i*nVel + c].data(), 1,
                         m_DOModePhys.data()
                         + (i*nVel + c)*nPhys, 1);
    // copy the local Y shard from in to m_Yi
    Vmath::Vcopy(m_npLocal*m_nDOModes,
                 in[m_doYIdx].data(), 1, m_Yi.data(), 1);

    // swap m_fields phys to m_meanAtTn, save mean^{n+1} to meanSaved
    Array<OneD, Array<OneD, NekDouble>> meanSaved(nVel);
    if (m_meanSnapshotValid)
    {
        for (int c = 0; c < nVel; ++c)  // loop over components
        {
            meanSaved[c] = Array<OneD, NekDouble>(nPhys);
            Vmath::Vcopy(nPhys, m_fields[m_velocity[c]]->GetPhys().data(), 1,
                         meanSaved[c].data(), 1);
            Vmath::Vcopy(nPhys, m_meanAtTn[c].data(), 1,
                         m_fields[m_velocity[c]]->UpdatePhys().data(), 1);
        }
    }

    ComputeYMoments();

    // For IMEX BDF2, A_implicit[0][0] = 2/3 > 0: the GLM stage loop calls
    // DoImplicitSolve before DoOdeRhs, so in = modes^{n+1} (post-implicit).
    // v_EvaluateAdvection_SetPressureBCs cached gradients at modes^n; those
    // are stale here.  Recompute from the freshly synced m_DOModePhys.
    PrecomputeGradients();

    // Explicit RHS for all modes.  NiAll[i][c] is a non-owning view into
    // m_NAllBuf (eArrayWrapper), so ComputeNModeBody writes directly there.
    // (Default eArrayCopy would allocate a separate buffer, leaving m_NAllBuf
    // unfilled; beta projection and copy-to-out both become wrong.)
    const int SLOTS = 4*nVel + 2;
    const int S     = m_nDOModes;
    std::vector<Array<OneD, Array<OneD, NekDouble>>> NiAll(S);
    for (int i = 0; i < S; ++i)
    {
        NiAll[i] = Array<OneD, Array<OneD, NekDouble>>(nVel);
        for (int c = 0; c < nVel; ++c)
            NiAll[i][c] = Array<OneD, NekDouble>(
                nPhys, m_NAllBuf.data() + (i*nVel+c)*nPhys, eArrayWrapper);
    }

    // Each mode writes only its per-worker slab and per-mode reduction
    // slots, so the loop parallelises when built with OpenMP. One AllReduce
    // covers the S*S beta projections, the S*nVel strip-constants integrals
    // and the S*S + S*S*S Y-RHS tensors.
    const bool dbgDiag =
        m_verbose && m_session->GetComm()->GetSpaceComm()->GetRank() == 0;
    std::vector<NekDouble> dbgMax;
    if (dbgDiag)
    {
        dbgMax.assign(4*S, 0.0);
    }
    std::vector<NekDouble> allRed(S*S + S*nVel + S*S + S*S*S, 0.0);
    NekDouble *betasAll = allRed.data();
    NekDouble *intsAll  = betasAll + S*S;
    NekDouble *ipKiAll  = intsAll + S*nVel;
    NekDouble *ipKliAll = ipKiAll + S*S;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < S; ++i)
    {
        int slab = 0;
#ifdef _OPENMP
        slab = omp_get_thread_num();
#endif
        NekDouble *bodyBuf = m_NBodyBuf.data() + slab*SLOTS*nPhys;
        ComputeNModeBody(i, NiAll[i], bodyBuf, betasAll + i*S,
                         intsAll + i*nVel);
        if (dbgDiag)
        {
            // per-mode maxima, taken before the slab is reused
            NekDouble mc = 0.0, mt = 0.0, ms = 0.0, mn = 0.0;
            for (int c = 0; c < nVel; ++c)
                for (int k = 0; k < nPhys; ++k)
                {
                    mc = std::max(mc, std::abs(bodyBuf[c*nPhys + k]));
                    mt = std::max(mt,
                                  std::abs(bodyBuf[(nVel + c)*nPhys + k]));
                    ms = std::max(
                        ms, std::abs(bodyBuf[(3*nVel + c)*nPhys + k]));
                    mn = std::max(
                        mn, std::abs(m_NAllBuf[(i*nVel + c)*nPhys + k]));
                }
            dbgMax[4*i + 0] = mc;
            dbgMax[4*i + 1] = mt;
            dbgMax[4*i + 2] = ms;
            dbgMax[4*i + 3] = mn;
        }
    }
    // Y-RHS tensors (serial: BwdTrans/PhysDeriv allocate via the MemPool)
    BuildYRhsTensors(ipKiAll, ipKliAll);

    if (dbgDiag)
    {
        NekDouble muMax = 0.0;
        for (int q = 0; q < S; ++q)
            muMax = std::max(muMax, std::abs(m_Cij[q*S + q]));
        const NekDouble lambdaReg = m_invCovRegEps * muMax;
        for (int i = 0; i < S; ++i)
        {
            const NekDouble mui      = m_Cij[i*S + i];
            const NekDouble invMuReg = mui / (mui*mui + lambdaReg*lambdaReg);
            std::cout << "[DBG ComputeNMode step=" << m_doStepCounter
                      << " mode=" << i
                      << " mui=" << mui
                      << " lambdaReg=" << lambdaReg
                      << " invMuReg=" << invMuReg
                      << " |cross|max=" << dbgMax[4*i + 0]
                      << " |triple_raw|max=" << dbgMax[4*i + 1]
                      << " |triple_scaled|max=" << invMuReg * dbgMax[4*i + 1]
                      << " |stoch_raw|max=" << dbgMax[4*i + 2]
                      << " |stoch_scaled|max=" << invMuReg * dbgMax[4*i + 2]
                      << " |N_preproj|max=" << dbgMax[4*i + 3]
                      << "]\n";
        }
    }

    m_fields[m_velocity[0]]->GetComm()->GetRowComm()->AllReduce(
        allRed, LibUtilities::ReduceSum);

    // strip-constants gauge: subtract the global innerArg mean from N,
    // before the beta projection to match the serial operation order
    if (!m_doAllowConstantModes && m_constSubspace.domainArea > 0.0)
    {
        for (int i = 0; i < S; ++i)
            for (int c = 0; c < nVel; ++c)
            {
                if (!m_constSubspace.admissible[c]) continue;
                const NekDouble mean =
                    intsAll[i*nVel + c] / m_constSubspace.domainArea;
                NekDouble *Nic = m_NAllBuf.data() + (i*nVel + c)*nPhys;
                Vmath::Sadd(nPhys, -mean, Nic, 1, Nic, 1);
            }
    }

    for (int i = 0; i < S; ++i)
        for (int p = 0; p < S; ++p)
        {
            const NekDouble beta_ip = betasAll[i*S + p];
            if (beta_ip == 0.0) continue;
            for (int c = 0; c < nVel; ++c)
            {
                NekDouble *Nic = m_NAllBuf.data() + (i*nVel + c)*nPhys;
                const NekDouble *u_pc =
                    m_DOModePhys.data() + (p*nVel + c)*nPhys;
                Vmath::Svtvp(nPhys, -beta_ip, u_pc, 1, Nic, 1, Nic, 1);
            }
        }
    for (int i = 0; i < S; ++i)
        for (int c = 0; c < nVel; ++c)
            Vmath::Vcopy(nPhys, m_NAllBuf.data() + (i*nVel + c)*nPhys, 1,
                         out[i*nVel + c].data(), 1);

    // explicit RHS for Y from the reduced tensors
    AssembleYRhs(ipKiAll, ipKliAll, out[m_doYIdx]);

    // restore m_fields phys (mean^{n+1} for the rest of the step)
    if (m_meanSnapshotValid)
    {
        for (int c = 0; c < nVel; ++c)
            Vmath::Vcopy(nPhys, meanSaved[c].data(), 1,
                         m_fields[m_velocity[c]]->UpdatePhys().data(), 1);
    }
}

/**
 * Implicit solve for one IMEX step, called by time integrator m_doScheme.
 *      - in[0,...,m_nDOModes*nVel-1] : mode predictor for each (i,c);
 *      - in[m_doYIdx] : local Y shard predictor (m_npLocal*m_nDOModes);
 *      - out: post-implicit state: modes u_i^{n+1}, Y identity-copied;
 *      - lambda: IMEX implicit weight ((2/3)dt in BDF2, dt in BDF1 startup).
 * For each mode, runs ModePressureSolve, then ModeViscousSolve. Both helpers
 * take lambda as their implicit-weight argument (for any IMEX order).
 * Also, m_DOModePCoeffs[i] is overwritten with p_i^{n+1} so the next
 * DOExplicitRhs sees consistent grad(p_k).
 */

void DOVelocityCorrectionScheme::DOImplicitSolve(
    const Array<OneD, const Array<OneD, NekDouble>> &in,
    Array<OneD, Array<OneD, NekDouble>>             &out,
    [[maybe_unused]] const NekDouble                 time,
    const NekDouble                                  lambda)
{
    if (m_nDOModes == 0) return;

    const int nVel     = m_velocity.size();
    const int nPhys    = m_fields[0]->GetTotPoints();
    const int nCoeffs  = m_fields[0]->GetNcoeffs();
    const int nPC      = m_pressure->GetNcoeffs();
    const int nPP      = m_pressure->GetTotPoints();

    // save mean-field state to restore (helpers modify m_fields/m_pressure)
    std::vector<Array<OneD, NekDouble>> svc(nVel), svp(nVel);
    for (int c = 0; c < nVel; ++c)
    {
        const int nc = m_fields[m_velocity[c]]->GetNcoeffs();
        const int np = m_fields[m_velocity[c]]->GetTotPoints();
        svc[c] = Array<OneD, NekDouble>(nc);
        svp[c] = Array<OneD, NekDouble>(np);
        Vmath::Vcopy(nc, m_fields[m_velocity[c]]->GetCoeffs().data(),
                     1, svc[c].data(), 1);
        Vmath::Vcopy(np, m_fields[m_velocity[c]]->GetPhys().data(),
                     1, svp[c].data(), 1);
    }
    Array<OneD, NekDouble> sPC(nPC), sPP(nPP);
    Vmath::Vcopy(nPC, m_pressure->GetCoeffs().data(), 1, sPC.data(), 1);
    Vmath::Vcopy(nPP, m_pressure->GetPhys().data(),   1, sPP.data(), 1);

    // homogenise velocity BCs for the per-mode solves; restored at the end
    auto bcState = CaptureVelocityBCState(m_fields, m_velocity);
    HomogenizeVelocityBCsForModes(m_fields, m_velocity);

    // arrays reused across modes
    Array<OneD, Array<OneD, NekDouble>> uhat(nVel), uNewPhys(nVel),
                                        uNewCoeffs(nVel);
    Array<OneD, NekDouble> pMode(nPC, 0.0);
    for (int c = 0; c < nVel; ++c)
    {
        uhat[c]       = Array<OneD, NekDouble>(nPhys, 0.0);
        uNewPhys[c]   = Array<OneD, NekDouble>(nPhys, 0.0);
        uNewCoeffs[c] = Array<OneD, NekDouble>(nCoeffs, 0.0);
    }

    // Mode solves grouped by operator: all pressure Poissons first, then
    // all viscous Helmholtz solves. Consecutive solves of the same matrix
    // keep any SuccessiveRHS projection basis coherent and reuse the warm
    // preconditioner; each solve's inputs are unchanged, so direct-solver
    // results are identical to the interleaved ordering. The previous
    // solutions (m_DOModePCoeffs / m_DOModeCoeffs, still holding the last
    // stage's values) warm-start the iterative solvers.
    for (int i = 0; i < m_nDOModes; ++i)
    {
        for (int c = 0; c < nVel; ++c)
            Vmath::Vcopy(nPhys, in[i*nVel + c].data(), 1,
                        uhat[c].data(), 1);
        ModePressureSolve(uhat, lambda, m_DOModePCoeffs + i*nPC, pMode);
        Vmath::Vcopy(nPC, pMode.data(), 1,      // cache p_i for next Y-RHS
                     m_DOModePCoeffs.data() + i*nPC, 1);
    }
    for (int i = 0; i < m_nDOModes; ++i)
    {
        for (int c = 0; c < nVel; ++c)
            Vmath::Vcopy(nPhys, in[i*nVel + c].data(), 1,
                        uhat[c].data(), 1);
        ModeViscousSolve(uhat, m_DOModePCoeffs + i*nPC, lambda,
                         m_DOModeCoeffs + i*nVel*nCoeffs,
                         uNewPhys, uNewCoeffs);
        for (int c = 0; c < nVel; ++c)
        {
            Vmath::Vcopy(nPhys, uNewPhys[c].data(), 1,
                         out[i*nVel + c].data(), 1);
            Vmath::Vcopy(nCoeffs, uNewCoeffs[c].data(), 1,
                         m_DOModeCoeffs.data()
                         + (i*nVel + c)*nCoeffs, 1);
        }
    }

    // Y: identity implicit operator (the Y system has no implicit term)
    Vmath::Vcopy(m_npLocal*m_nDOModes, in[m_doYIdx].data(), 1,
                 out[m_doYIdx].data(), 1);

    // Restore mean-field state.
    RestoreVelocityBCState(m_fields, bcState);
    for (int c = 0; c < nVel; ++c)
    {
        Vmath::Vcopy(svc[c].size(), svc[c].data(), 1,
                     m_fields[m_velocity[c]]->UpdateCoeffs().data(), 1);
        Vmath::Vcopy(svp[c].size(), svp[c].data(), 1,
                     m_fields[m_velocity[c]]->UpdatePhys().data(), 1);
    }
    Vmath::Vcopy(nPC, sPC.data(), 1, m_pressure->UpdateCoeffs().data(), 1);
    Vmath::Vcopy(nPP, sPP.data(), 1, m_pressure->UpdatePhys().data(),   1);
}

/**
 * Rotates the basis to diagonalise C = E[Y Y^T].
 *
 * Eigendecomposes C using LAPACK and applies the orthogonal V to:
 *   - per-mode arrays: m_DOModePhys, m_DOModeCoeffs, m_DOModePCoeffs;
 *   - per-particle Y:   m_Yi (Y_new = V^T Y_old);
 *   - history values in the integrator.
 */
void DOVelocityCorrectionScheme::DiagonaliseCov(
    std::vector<NekDouble> *deferredV)
{
    if (deferredV)
    {
        deferredV->clear();
    }
    if (m_nDOModes <= 1) return;
    ComputeYMoments();

    // symmetric eigendecomposition via LAPACK Dspev:
    // w returns ascending eigenvalues; columns of V are eigenvectors
    std::vector<NekDouble> V, w;
    SymmetricEig(m_nDOModes, m_Cij, V, w);

    // sort eigenvalues descending
    std::vector<int> ord(m_nDOModes);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(),
      [&](int i, int j){ return w[i] > w[j]; });
    std::vector<NekDouble> Vs(m_nDOModes*m_nDOModes, 0.0);
    for (int i = 0; i < m_nDOModes; ++i)
        for (int k = 0; k < m_nDOModes; ++k)
            Vs[k*m_nDOModes + i] = V[k*m_nDOModes + ord[i]];
    V = std::move(Vs);

    // eigendecomposition determines the new basis up to
    //    - a per-column sign for non-degenerate eigenvalues, and
    //    - an arbitrary orthogonal change of basis within any
    //       (near-)degenerate eigenvalue cluster.
    // Thus, we implement Procrustes alignment to avoid mode flipping/rotation.
    //
    // For each cluster B = {a, a+1, ..., a+m-1}:
    //   - m == 1 (singleton):  Q[a,a] = sign(V[a,a]).
    //   - m > 1 : Q_B = polar(V_B^T) = V_B^T (V_B V_B^T)^{-1/2}.
    {
        std::vector<NekDouble> dvec(m_nDOModes);    // descending eigenvalues
        for (int i = 0; i < m_nDOModes; ++i)
        {
            dvec[i] = w[ord[i]];
        }
        NekDouble dMax = 0.0;
        for (NekDouble v : dvec)
        {
            dMax = std::max(dMax, std::abs(v));
        }
        const NekDouble degTol = 1e-3 * std::max(dMax, NekDouble{1e-300});
        // identify clusters of (near-)equal sorted eigenvalues.
        std::vector<std::pair<int,int>> blocks;
        {
            int b0 = 0;
            for (int i = 1; i <= m_nDOModes; ++i)
                if (i == m_nDOModes || std::abs(dvec[i-1] - dvec[i]) > degTol)
                {
                    blocks.emplace_back(b0, i);
                    b0 = i;
                }
        }
        // build block-diagonal Q
        std::vector<NekDouble> Q(m_nDOModes*m_nDOModes, 0.0);
        for (auto [a, b] : blocks)
        {
            const int m = b - a;
            // non-degenerate eigenvalues (only sign fix)
            if (m == 1)
            {
                Q[a*m_nDOModes + a] = (V[a*m_nDOModes + a] >= 0.0) ? 1.0 : -1.0;
                continue;
            }
            // Q_B for m > 1
            std::vector<NekDouble> Vt(m*m), VVt(m*m, 0.0);
            // populate Vt = V^T
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < m; ++j)
                    Vt[i*m + j] = V[(a+j)*m_nDOModes + (a+i)];
            // compute VVt (= Vt^T Vt) = V V^T
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < m; ++j)
                {
                    NekDouble s = 0.0;
                    for (int k = 0; k < m; ++k)
                        s += Vt[k*m + i] * Vt[k*m + j];
                    VVt[i*m + j] = s;
                }
            // symmetric eigendecomposition of VVt via Dspev
            std::vector<NekDouble> E, dE;
            SymmetricEig(m, VVt, E, dE);
            // floor 0 or negative eigenvalues (only from roundoff)
            for (int i = 0; i < m; ++i)
                dE[i] = std::max(dE[i], NekDouble{1e-300});
            // (V V^T)^{-1/2} = E diag(1/sqrt(d)) E^T
            std::vector<NekDouble> InvSqrt(m*m, 0.0);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < m; ++j)
                {
                    NekDouble s = 0.0;
                    for (int k = 0; k < m; ++k)
                        s += E[i*m + k] * (1.0/std::sqrt(dE[k])) * E[j*m + k];
                    InvSqrt[i*m + j] = s;
                }
            // Q_block = V^T (V V^T)^{-1/2}
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < m; ++j)
                {
                    NekDouble s = 0.0;
                    for (int k = 0; k < m; ++k)
                        s += Vt[i*m + k] * InvSqrt[k*m + j];
                    Q[(a+i)*m_nDOModes + (a+j)] = s;
                }
        }
        // V <- VQ
        std::vector<NekDouble> Veff(m_nDOModes*m_nDOModes, 0.0);
        for (int i = 0; i < m_nDOModes; ++i)
            for (int j = 0; j < m_nDOModes; ++j)
            {
                NekDouble s = 0.0;
                for (int k = 0; k < m_nDOModes; ++k)
                    s += V[i*m_nDOModes + k] * Q[k*m_nDOModes + j];
                Veff[i*m_nDOModes + j] = s;
            }
        V = std::move(Veff);
    }

    // apply V^T to all current timestep arrays
    auto rotateModeBlocks = [&](Array<OneD, NekDouble> &arr, int blockSize) {
        if (arr.size() == 0) return;
        std::vector<NekDouble> tmp(m_nDOModes * blockSize, 0.0);
        for (int i = 0; i < m_nDOModes; ++i)
            for (int j = 0; j < m_nDOModes; ++j)
            {
                const NekDouble Vji = V[j*m_nDOModes + i];
                if (Vji == 0.0) continue;
                const NekDouble *src = arr.data() + j*blockSize;
                NekDouble *dst       = tmp.data() + i*blockSize;
                for (int n = 0; n < blockSize; ++n)
                    dst[n] += Vji * src[n];
            }
        Vmath::Vcopy(m_nDOModes*blockSize, tmp.data(), 1, arr.data(), 1);
    };
    auto rotateYi = [&](Array<OneD, NekDouble> &Yarr) {
        if (Yarr.size() == 0) return;
        std::vector<NekDouble> tmp(m_nDOModes, 0.0);
        for (int p = 0; p < m_npLocal; ++p)          // local particle shard
        {
            for (int i = 0; i < m_nDOModes; ++i)     // new mode i
            {
                NekDouble s = 0.0;
                for (int j = 0; j < m_nDOModes; ++j) // old mode j
                {
                    s += V[j*m_nDOModes + i] * Yarr[p*m_nDOModes + j];
                }
                tmp[i] = s;
            }
            Vmath::Vcopy(m_nDOModes, tmp.data(), 1,
                         Yarr.data() + p*m_nDOModes, 1);
        }
    };

    const int nVel    = m_velocity.size();
    const int nPhys   = m_fields[0]->GetTotPoints();
    const int nCoeffs = m_fields[0]->GetNcoeffs();
    const int nPC     = m_pressure->GetNcoeffs();

    rotateModeBlocks(m_DOModePhys, nVel * nPhys);
    rotateModeBlocks(m_DOModeCoeffs, nVel * nCoeffs);
    rotateModeBlocks(m_DOModePCoeffs, nPC);
    rotateYi(m_Yi);

    // apply V^T to all the time integrator's history, too -- unless the
    // caller asked to defer it (v_PostIntegrate: ReOrthonormalise composes
    // V with its own basis change into a single history pass)
    if (m_doSchemeInited)
    {
        if (deferredV)
        {
            *deferredV = V;
        }
        else
        {
            auto &solVec = m_doScheme->UpdateSolutionVector();
            const int nSteps = static_cast<int>(solVec.size());
            Array<OneD, NekDouble> buf(m_nDOModes * nPhys);
            for (int step = 0; step < nSteps; ++step)
            {
                for (int c = 0; c < nVel; ++c)
                {
                    for (int i = 0; i < m_nDOModes; ++i)
                    {
                        Vmath::Vcopy(nPhys,
                                     solVec[step][i*nVel + c].data(), 1,
                                     buf.data() + i*nPhys, 1);
                    }
                    rotateModeBlocks(buf, nPhys);
                    for (int i = 0; i < m_nDOModes; ++i)
                    {
                        Vmath::Vcopy(nPhys, buf.data() + i*nPhys, 1,
                                     solVec[step][i*nVel + c].data(), 1);
                    }
                }
                rotateYi(solVec[step][m_doYIdx]);
            }
        }
    }
    ComputeYMoments();
}

/**
 * After the base VCS step has advanced the mean field, this method:
 *      - advances (modes, Y) atomically via m_doScheme -- all RHS terms
 *        evaluated at the same t^n state thanks to the integrator passing
 *        a single in snapshot to DOExplicitRhs / DOImplicitSolve. Mean^n is
 *        read from m_meanAtTn (snapshotted by the EXT operator at t^n).
 *      - unpacks the post-step (modes, Y) back into m_DOModePhys / m_Yi
 *        (the ground-truth members consumed by DiagonaliseCov,
 *        ReOrthonormalise, the archive writer, and the next
 *        step's DOExplicitRhs).
 *      - diagonalises the covariance and orthonormalises the basis.
 */
bool DOVelocityCorrectionScheme::v_PreIntegrate(int step)
{
    m_stepTimer.Start();
    return VelocityCorrectionScheme::v_PreIntegrate(step);
}

bool DOVelocityCorrectionScheme::v_PostIntegrate(int step)
{
    // base VCS (m_fields <- mean^{n+1}); a true return value requests
    // termination of the time loop (e.g. steady-state) and must propagate
    bool terminate = VelocityCorrectionScheme::v_PostIntegrate(step);
    ++m_doStepCounter;  // per-step verbose-only counter

    if (m_nDOModes > 0 && m_doSchemeInited)
    {
        AdvanceForcingState();

        const auto &advanced = m_doScheme->TimeIntegrate(step, m_timestep);

        // update m_DOModePhys / m_DOModeCoeffs / m_Yi.
        const int nVel    = m_velocity.size();
        const int nPhys   = m_fields[0]->GetTotPoints();
        const int nCoeffs = m_fields[0]->GetNcoeffs();
        auto piBcState = CaptureVelocityBCState(m_fields, m_velocity);
        HomogenizeVelocityBCsForModes(m_fields, m_velocity);
        Array<OneD, NekDouble> tmpPhys(nPhys, 0.0), tmpCoef(nCoeffs, 0.0);
        for (int i = 0; i < m_nDOModes; ++i)    // loop over modes
            for (int c = 0; c < nVel; ++c)      // loop over components
            {
                Vmath::Vcopy(nPhys, advanced[i*nVel + c].data(), 1,
                             m_DOModePhys.data() + (i*nVel + c)*nPhys, 1);
                Vmath::Vcopy(nPhys, advanced[i*nVel + c].data(), 1,
                             tmpPhys.data(), 1);
                // warm-start the mass solve from last step's coefficients
                Vmath::Vcopy(nCoeffs,
                             m_DOModeCoeffs.data() + (i*nVel + c)*nCoeffs, 1,
                             tmpCoef.data(), 1);
                m_fields[m_velocity[c]]->FwdTrans(tmpPhys, tmpCoef);
                Vmath::Vcopy(nCoeffs, tmpCoef.data(), 1,
                             m_DOModeCoeffs.data() + (i*nVel + c)*nCoeffs, 1);
            }
        RestoreVelocityBCState(m_fields, piBcState);
        Vmath::Vcopy(m_npLocal*m_nDOModes, advanced[m_doYIdx].data(), 1,
                     m_Yi.data(), 1);

        // DiagonaliseCov defers its history rotation (deferredV);
        // ReOrthonormalise composes it with its own basis change and
        // applies a single transform pass to the integrator history.
        std::vector<NekDouble> deferredV;
        DiagonaliseCov(&deferredV);
        ReOrthonormalise(&deferredV);

        // calling ReOrthonormalise modifies m_DOModePhys and m_Yi after
        // DiagonaliseCov, which rotates solVec[*]. Thus, we need to
        // update solVec[] so that DOExplicitRhs uses the orthonormalised basis
        {
            auto &solVec = m_doScheme->UpdateSolutionVector();
            for (int i = 0; i < m_nDOModes; ++i)
                for (int c = 0; c < nVel; ++c)
                {
                    Vmath::Vcopy(nPhys,
                                 m_DOModePhys.data() + (i*nVel + c)*nPhys, 1,
                                 solVec[0][i*nVel + c].data(), 1);
                }
            Vmath::Vcopy(m_npLocal*m_nDOModes, m_Yi.data(), 1,
                         solVec[0][m_doYIdx].data(), 1);
        }

    }

    m_stepTimer.Stop();
    m_stepAccumTime += m_stepTimer.TimePerTest(1);

    if (m_infosteps && !((step + 1) % m_infosteps) &&
        m_session->GetComm()->GetSpaceComm()->GetRank() == 0)
    {
        std::stringstream ss;
        ss << m_stepAccumTime << "s";
        if (m_comm->IsParallelInTime())
            std::cout << "RANK "
                      << m_session->GetComm()->GetTimeComm()->GetRank()
                      << " ";
        std::cout << "Steps: "    << std::setw(8)  << std::left << step + 1
                  << " Time: "    << std::setw(12) << std::left << m_time;
        if (m_cflSafetyFactor)
            std::cout << " Time-step: "
                      << std::setw(12) << std::left << m_timestep;
        std::cout << " CPU Time: " << std::setw(8) << std::left
                  << ss.str() << std::endl;
        m_stepAccumTime = 0.0;
    }
    return terminate;
}

void DOVelocityCorrectionScheme::v_PrintStatusInformation(
    const int /*step*/, const NekDouble /*cpuTime*/)
{
    // printing deferred to v_PostIntegrate where the full step time is known
}

/**
 * Orthonormalise the DO mode basis after each integration step.
 *
 * Per mode:
 *   - Helmholtz-Hodge projection onto the discrete div-free subspace:
 *           lap(phi) = div(u),    (HelmSolve with lambda = 0)
 *           u <- u - grad(phi);
 *   - optional constants strip (if DOAllowConstantModes = false);
 *   - classical Gram-Schmidt with reorthogonalisation against the accepted
 *     basis (two passes; four during initialisation);
 *   - divergence-L2 sanity check; re-project once if dirty;
 *   - normalise; abort if collapsed.
 *
 * After all modes are processed, perform the Y / mode-pressure / history-
 * mode co-transform so the realisation u_p = ubar + sum_i Y_{p,i} u_i is
 * invariant under the MGS basis change (kept verbatim from the prior
 * implementation --
 * load-bearing for time-integration state consistency).
 *
 * Velocity- and pressure-BC brackets at the top zero homogeneous BC DOFs;
 * both are restored on exit.
 */
void DOVelocityCorrectionScheme::ReOrthonormalise(
    const std::vector<NekDouble> *pendingV)
{
    if (m_nDOModes == 0) return;

    // homogenise BCs
    auto vBcState = CaptureVelocityBCState(m_fields, m_velocity);
    HomogenizeVelocityBCsForModes(m_fields, m_velocity);
    auto pBnd = m_pressure->GetBndCondExpansions();
    std::vector<std::vector<NekDouble>> savedPBndC(pBnd.size()),
                                        savedPBndP(pBnd.size());
    for (int n = 0; n < (int)pBnd.size(); ++n)
    {
        const int nc = pBnd[n]->GetNcoeffs();
        const int np = pBnd[n]->GetTotPoints();
        savedPBndC[n].assign(pBnd[n]->GetCoeffs().data(),
                             pBnd[n]->GetCoeffs().data() + nc);
        savedPBndP[n].assign(pBnd[n]->GetPhys().data(),
                             pBnd[n]->GetPhys().data() + np);
        Vmath::Zero(nc, pBnd[n]->UpdateCoeffs().data(), 1);
        Vmath::Zero(np, pBnd[n]->UpdatePhys().data(),   1);
    }
    // m_pressure values are mutated by div-free projection; save & restore
    Array<OneD, NekDouble> savedPC(m_pressure->GetNcoeffs()),
                           savedPP(m_pressure->GetTotPoints());
    Vmath::Vcopy(m_pressure->GetNcoeffs(),   m_pressure->GetCoeffs().data(),
                 1, savedPC.data(), 1);
    Vmath::Vcopy(m_pressure->GetTotPoints(), m_pressure->GetPhys().data(),
                 1, savedPP.data(), 1);

    const int nVel    = (int)m_velocity.size();
    const int nCoeffs = m_fields[m_velocity[0]]->GetNcoeffs();
    const int nPhys   = m_fields[m_velocity[0]]->GetTotPoints();

    std::vector<ModeData> basis;

    // Helmholtz-Hodge decomposition
    auto projectHHD = [&](Array<OneD, Array<OneD, NekDouble>> &u) {
        Array<OneD, NekDouble> divU(nPhys, 0.0), tmp(nPhys);
        // compute div(u)
        for (int c = 0; c < nVel; ++c)
        {
            m_fields[m_velocity[c]]->PhysDeriv(c, u[c], tmp);
            Vmath::Vadd(nPhys, tmp.data(), 1, divU.data(), 1, divU.data(), 1);
        }
        // HelmSolve with lambda = 0 to find phi: lap(phi) = div(u)
        StdRegions::ConstFactorMap fac;
        fac[StdRegions::eFactorLambda] = 0.0;
        Vmath::Zero(m_pressure->GetNcoeffs(),
                    m_pressure->UpdateCoeffs().data(), 1);
        m_pressure->HelmSolve(divU, m_pressure->UpdateCoeffs(), fac);
        m_pressure->BwdTrans(m_pressure->GetCoeffs(), m_pressure->UpdatePhys());
        // compute grad(phi)
        Array<OneD, Array<OneD, NekDouble>> grad(nVel);
        for (int c = 0; c < nVel; ++c)
            grad[c] = Array<OneD, NekDouble>(nPhys, 0.0);
        if (nVel == 2)
            m_pressure->PhysDeriv(m_pressure->GetPhys(), grad[0], grad[1]);
        else
            m_pressure->PhysDeriv(m_pressure->GetPhys(),
                                  grad[0], grad[1], grad[2]);
        // u -= grad(phi) is now divergence-free
        for (int c = 0; c < nVel; ++c)
            Vmath::Vsub(nPhys, u[c].data(), 1, grad[c].data(), 1,
                        u[c].data(), 1);
    };

    // 4-pass modified Gram-Schmidt against accepted basis.
    // All k = basis.size() inner products <basis[j], cand> for j=0..k-1 are
    // computed locally and reduced in one AllReduce per pass instead of k.
    auto runMgs = [&](ModeData &cand) {
        const int k = (int)basis.size();
        if (k == 0) return;
        auto comm = m_fields[m_velocity[0]]->GetComm()->GetRowComm();
        std::vector<NekDouble> alphas(k);
        Array<OneD, NekDouble> wCand(nPhys);
        // Two classical-GS passes suffice ("twice is enough") during time
        // stepping, where candidates enter near-orthonormal (O(dt) drift).
        // Initialisation keeps four passes: POD candidates can be strongly
        // rank-deficient there.
        const int nPasses = (m_doStepCounter <= 1) ? 4 : 2;
        for (int pass = 0; pass < nPasses; ++pass)
        {
            std::fill(alphas.begin(), alphas.end(), 0.0);
            for (int c = 0; c < nVel; ++c)
            {
                // wCand[q] = cand.phys[c][q] * physWeights[q]
                Vmath::Vmul(nPhys, m_physWeights.data(), 1,
                            cand.phys[c].data(), 1, wCand.data(), 1);
                for (int j = 0; j < k; ++j)
                    alphas[j] += Vmath::Dot(nPhys,
                                            basis[j].phys[c].data(), 1,
                                            wCand.data(), 1);
            }
            // one AllReduce for all k inner products
            comm->AllReduce(alphas, LibUtilities::ReduceSum);
            for (int j = 0; j < k; ++j)
            {
                const NekDouble a = alphas[j];
                for (int c = 0; c < nVel; ++c)
                {
                    Vmath::Svtvp(nCoeffs, -a, basis[j].coeffs[c].data(), 1,
                                 cand.coeffs[c].data(), 1,
                                 cand.coeffs[c].data(), 1);
                    Vmath::Svtvp(nPhys,  -a, basis[j].phys[c].data(), 1,
                                 cand.phys[c].data(), 1,
                                 cand.phys[c].data(), 1);
                }
            }
        }
    };

    // scratch buffers
    Array<OneD, Array<OneD, NekDouble>> uTmp(nVel);
    for (int c = 0; c < nVel; ++c)
    {
        uTmp[c] = Array<OneD, NekDouble>(nPhys);
    }
    Array<OneD, NekDouble> coefTmp(nCoeffs), physTmp(nPhys);

    // physWeights squared norm: 1 AllReduce.
    auto physWeightsNorm2 = [&](ModeData &md) -> NekDouble {
        NekDouble s = 0.0;
        Array<OneD, NekDouble> wBuf(nPhys);
        for (int c = 0; c < nVel; ++c)
        {
            Vmath::Vmul(nPhys, m_physWeights.data(), 1,
                        md.phys[c].data(), 1, wBuf.data(), 1);
            s += Vmath::Dot(nPhys, md.phys[c].data(), 1, wBuf.data(), 1);
        }
        m_fields[m_velocity[0]]->GetComm()->GetRowComm()->AllReduce(
            s, LibUtilities::ReduceSum);
        return s;
    };

    // Helmholtz-Hodge project, enforce BCs
    auto projectAndSync = [&](ModeData &cand) {
        for (int c = 0; c < nVel; ++c)
            std::copy(cand.phys[c].begin(), cand.phys[c].end(), uTmp[c].data());
        projectHHD(uTmp);
        for (int c = 0; c < nVel; ++c)
        {
            auto cf = std::dynamic_pointer_cast<MultiRegions::ContField>(
                m_fields[m_velocity[c]]);
            Vmath::Zero(nCoeffs, coefTmp.data(), 1);
            cf->FwdTrans(uTmp[c], coefTmp);
            cf->ImposeDirichletConditions(coefTmp);
            cf->BwdTrans(coefTmp, physTmp);
            cand.coeffs[c].assign(coefTmp.data(), coefTmp.data() + nCoeffs);
            cand.phys[c].assign(physTmp.data(),   physTmp.data()  + nPhys);
        }
    };

    basis.reserve(m_nDOModes);

    // project, MGS, normalise, accept for each mode
    for (int i = 0; i < m_nDOModes; ++i)
    {
        // create candidate cand, populate with mode values
        ModeData cand;
        cand.coeffs.resize(nVel);
        cand.phys.resize(nVel);
        for (int c = 0; c < nVel; ++c)
        {
            const int cOff = (i*nVel + c)*nCoeffs;
            const int pOff = (i*nVel + c)*nPhys;
            cand.coeffs[c].assign(m_DOModeCoeffs.data() + cOff,
                                  m_DOModeCoeffs.data() + cOff + nCoeffs);
            cand.phys[c].assign(m_DOModePhys.data() + pOff,
                                m_DOModePhys.data() + pOff + nPhys);
        }
        // norm pre-MGS: diagnostic only and costs an AllReduce, so skip it
        // unless verbose (m_verbose is uniform across ranks, keeping the
        // collective aligned)
        const NekDouble preMgsNrm =
            m_verbose ? std::sqrt(std::max(physWeightsNorm2(cand), 0.0))
                      : 0.0;

        // HHD: DOImplicitSolve has already run a Poisson pressure-correction
        // step for every mode before this function is called, so weak
        // div-freedom is guaranteed from step 2 onward.  Run HHD only on the
        // very first step as a conservative safety net for initialisation;
        // all subsequent steps skip the Poisson solve entirely, saving 6
        // HelmSolves + 6 FwdTrans per step and all their AllReduces.
        if (m_doStepCounter <= 1)
        {
            projectAndSync(cand);
        }
        if (!m_doAllowConstantModes)
            ProjectOutConstantsFromMode(m_fields, m_velocity,
                                        m_constSubspace, cand);
        runMgs(cand);

        // norm post-MGS
        const NekDouble nrm = std::sqrt(std::max(
            physWeightsNorm2(cand), 0.0));
        if (!(nrm > 1e-12) && m_verbose)
        {
            std::cout << "[DOVelocityCorrectionScheme][reorth-FATAL] step="
                      << m_doStepCounter << " mode i=" << i
                      << " preMGS=" << preMgsNrm << " nrm=" << nrm << "\n";
        }
        ASSERTL0(nrm > 1e-12,
                 "DOVelocityCorrectionScheme: orthonormalisation collapsed.");
        const NekDouble invNrm = 1.0 / nrm;
        for (int c = 0; c < nVel; ++c)
        {
            for (auto &v : cand.coeffs[c])
            {
                v *= invNrm;
            }
            for (auto &v : cand.phys[c])
            {
                v *= invNrm;
            }
        }
        basis.push_back(std::move(cand));
    }

    // Y / mode-pressure / history need to be transformed
    //  - G[j,i] = <basis[j], u_old[i]>_M;
    //  - Y_new = G*Y_old;
    //  - mode-pressure rotates by G^{-T};
    //  - history (Y, modes) in m_doScheme by (G, G^{-1}).
    if (m_doSchemeInited)
    {
        std::vector<NekDouble> G(m_nDOModes*m_nDOModes, 0.0);
        // G[j,i] = <basis[j], u_old[i]>_w; physWeights dot replaces
        // IProductWRTBase + Dot, same result when phys+coeffs consistent.
        Array<OneD, NekDouble> wOld(nPhys);
        for (int i = 0; i < m_nDOModes; ++i)
            for (int c = 0; c < nVel; ++c)
            {
                Vmath::Vmul(nPhys,
                            m_DOModePhys.data() + (i*nVel + c)*nPhys, 1,
                            m_physWeights.data(), 1, wOld.data(), 1);
                for (int j = 0; j < m_nDOModes; ++j)
                    G[j*m_nDOModes + i] +=
                        Vmath::Dot(nPhys, basis[j].phys[c].data(), 1,
                                   wOld.data(), 1);
            }
        m_fields[m_velocity[0]]->GetComm()->GetRowComm()->AllReduce(
            G, LibUtilities::ReduceSum);

        // Y rotation (local particle shard; the current arrays were already
        // rotated by DiagonaliseCov, so plain G applies here)
        std::vector<NekDouble> Ytmp(m_nDOModes*m_npLocal);
        Blas::Dgemm('T', 'N', m_nDOModes, m_npLocal, m_nDOModes, 1.0,
                    G.data(), m_nDOModes, m_Yi.data(), m_nDOModes,
                    0.0, Ytmp.data(), m_nDOModes);
        Vmath::Vcopy(m_nDOModes*m_npLocal, Ytmp.data(), 1,
                     m_Yi.data(), 1);

        // rotate pressure modes
        if (m_DOModePCoeffs.size() > 0)
        {
            const int nPC = m_pressure->GetNcoeffs();
            std::vector<NekDouble> P(m_nDOModes*nPC);
            for (int k = 0; k < m_nDOModes; ++k)
            {
                for (int q = 0; q < nPC; ++q)
                {
                    P[k + q*m_nDOModes] = m_DOModePCoeffs[k*nPC + q];
                }
            }
            int info = 0;
            // G stored C-style as G[j*S+i]; BLAS sees G_blas = G_math^T.
            // G_math is upper triangular (MGS: basis[j] orthogonal to u_old[i]
            // for j>i), so G_blas = G_math^T is lower triangular.
            Lapack::Dtrtrs('L', 'N', 'N', m_nDOModes, nPC,
                           G.data(), m_nDOModes, P.data(), m_nDOModes, info);
            ASSERTL0(info == 0, "Dtrtrs failed on mode-pressure rotation");
            for (int k = 0; k < m_nDOModes; ++k)
            {
                for (int q = 0; q < nPC; ++q)
                {
                    m_DOModePCoeffs[k*nPC + q] = P[k + q*m_nDOModes];
                }
            }
        }
        // rotate history
        // G_math is upper triangular: backward substitution
        // to compute G_math^{-1}.
        std::vector<NekDouble> Ginv(m_nDOModes*m_nDOModes, 0.0);
        for (int k = 0; k < m_nDOModes; ++k)
            for (int i = m_nDOModes-1; i >= 0; --i)
            {
                NekDouble s = (i == k) ? 1.0 : 0.0;
                for (int j = i+1; j < m_nDOModes; ++j)
                    s -= G[i*m_nDOModes + j] * Ginv[j*m_nDOModes + k];
                Ginv[i*m_nDOModes + k] = s / G[i*m_nDOModes + i];
            }

        // Compose with DiagonaliseCov's deferred rotation V when present,
        // so the history is transformed in a single pass:
        //   mode fields transform by right-multiplication, sequentially
        //   old*V then *Ginv  =>  Rm = V*Ginv, Rm[j*S+i] = sum_k V[j*S+k]
        //   Ginv[k*S+i];
        //   Y transforms by left-multiplication, G*(V^T y)  =>  Ry = G*V^T,
        //   Ry[i*S+k] = sum_m G[i*S+m] V[k*S+m].
        const int  S     = m_nDOModes;
        const bool haveV = (pendingV && !pendingV->empty());
        std::vector<NekDouble> Rm(Ginv), Ry(G);
        if (haveV)
        {
            const std::vector<NekDouble> &V = *pendingV;
            for (int j = 0; j < S; ++j)
                for (int i = 0; i < S; ++i)
                {
                    NekDouble s = 0.0;
                    for (int k = 0; k < S; ++k)
                        s += V[j*S + k] * Ginv[k*S + i];
                    Rm[j*S + i] = s;
                }
            for (int i = 0; i < S; ++i)
                for (int k = 0; k < S; ++k)
                {
                    NekDouble s = 0.0;
                    for (int m = 0; m < S; ++m)
                        s += G[i*S + m] * V[k*S + m];
                    Ry[i*S + k] = s;
                }
        }

        auto &solVec = m_doScheme->UpdateSolutionVector();
        std::vector<NekDouble> tmpModes(m_nDOModes * nPhys, 0.0);
        std::vector<NekDouble> gatherBuf(m_nDOModes*nPhys);
        for (auto &slot : solVec)
        {
            if (slot[m_doYIdx].size() != 0)
            {
                Blas::Dgemm('T', 'N', m_nDOModes, m_npLocal, m_nDOModes,
                            1.0, Ry.data(), m_nDOModes, slot[m_doYIdx].data(),
                            m_nDOModes, 0.0, Ytmp.data(), m_nDOModes);
                Vmath::Vcopy(m_nDOModes*m_npLocal, Ytmp.data(), 1,
                             slot[m_doYIdx].data(), 1);
            }
            for (int c = 0; c < nVel; ++c)
            {
                for (int i = 0; i < m_nDOModes; ++i)
                {
                    Vmath::Vcopy(nPhys, slot[i*nVel + c].data(), 1,
                                 gatherBuf.data() + i*nPhys, 1);
                }
                Blas::Dgemm('N', 'T', nPhys, m_nDOModes, m_nDOModes, 1.0,
                            gatherBuf.data(), nPhys, Rm.data(), m_nDOModes,
                            0.0, tmpModes.data(), nPhys);
                for (int j = 0; j < m_nDOModes; ++j)
                {
                    Vmath::Vcopy(nPhys, tmpModes.data() + j*nPhys, 1,
                                 slot[j*nVel + c].data(), 1);
                }
            }
        }
    }

    // write basis back to m_DOModePhys, m_DOModeCoeffs
    for (int i = 0; i < m_nDOModes; ++i)
        for (int c = 0; c < nVel; ++c)
        {
            const int cOff = (i * nVel + c) * nCoeffs;
            const int pOff = (i * nVel + c) * nPhys;
            Vmath::Vcopy(nCoeffs, basis[i].coeffs[c].data(), 1,
                         m_DOModeCoeffs.data() + cOff, 1);
            Vmath::Vcopy(nPhys,   basis[i].phys[c].data(),   1,
                         m_DOModePhys.data() + pOff, 1);
        }
    // restore m_pressure state and BCs
    Vmath::Vcopy(m_pressure->GetNcoeffs(),   savedPC.data(), 1,
                 m_pressure->UpdateCoeffs().data(), 1);
    Vmath::Vcopy(m_pressure->GetTotPoints(), savedPP.data(), 1,
                 m_pressure->UpdatePhys().data(),   1);
    for (int n = 0; n < (int)pBnd.size(); ++n)
    {
        std::copy(savedPBndC[n].begin(), savedPBndC[n].end(),
                  pBnd[n]->UpdateCoeffs().data());
        std::copy(savedPBndP[n].begin(), savedPBndP[n].end(),
                  pBnd[n]->UpdatePhys().data());
    }
    RestoreVelocityBCState(m_fields, vBcState);
}

/**
 * Initialises the modes from the m_nDOModes smallest-eigenvalue eigenvectors
 * of the homogeneous CG Laplacian (built by DOReducedCGEigenBasis). Each
 * eigenpair generates up to nVel modes by placing the eigenvector into each
 * spatial component slot in turn (and setting the others to 0).
 * Then, calls ReOrthonormalise.
 */
void DOVelocityCorrectionScheme::InitialiseModesFromEllipticEigenbasis()
{
    if (m_nDOModes == 0) return;

    // DOReducedCGEigenBasis runs ARPACK on the per-rank local CG-reduced space;
    // its iteration arithmetic is not MPI-aware, so under multi-rank runs
    // each rank produces a DIFFERENT eigenbasis and the subsequent collective
    // HelmSolve / MGS calls deadlock on inconsistent state. Detect and abort
    // with a clear pointer to the workaround.
    ASSERTL0(m_session->GetComm()->GetSize() == 1,
             "DOInitModeBasis=Laplacian is single-rank only (the underlying "
             "ARPACK eigensolver is not MPI-aware). For multi-rank runs use "
             "DOInitModeBasis=POD (set <I PROPERTY=\"DOInitModeBasis\" "
             "VALUE=\"POD\"/> in your casefile and"
             " provide a snapshot pattern).");

    auto field0 = std::dynamic_pointer_cast<MultiRegions::ContField>(
        m_fields[m_velocity[0]]);
    DOReducedCGEigenBasis eigenBasis(field0);
    ASSERTL0(eigenBasis.GetNumHomCoeffs() > m_nDOModes,
             "Number of DO modes must be smaller than the"
             " reduced homogeneous CG size.");

    const int nVel    = m_velocity.size();
    const int nCoeffs = field0->GetNcoeffs();
    const int nPhys   = field0->GetTotPoints();
    // ceil(S / nVel) eigenpairs to fill all modes; +1 for headroom in case
    // the smallest eigenpair is the constant (lambda ~= 0) -- skipped below
    const int nSeeds =
        (int)std::ceil((NekDouble)m_nDOModes / (NekDouble)nVel) + 1;
    auto eigenpairs = eigenBasis.ComputeSmallest(nSeeds);

    // skip threshold: constant null-space mode (lambda ~= 0 for periodic /
    // pure-Neumann domains) must not become a DO mode.
    const NekDouble eps_const = 1e-10;

    // for each eigenpair, place it into a spatial component of a zeroed mode
    Array<OneD, NekDouble> localCoeffs(field0->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> phys(field0->GetTotPoints(), 0.0);
    int modeCount = 0;
    for (int j = 0; j < (int)eigenpairs.size() && modeCount < m_nDOModes; ++j)
    {
        if (eigenpairs[j].lambda < eps_const) continue;
        eigenBasis.ExportToLocalAndPhys(
            eigenpairs[j].reduced, localCoeffs, phys);
        for (int c = 0; c < nVel && modeCount < m_nDOModes; ++c, ++modeCount)
        {
            Vmath::Zero(nVel*nCoeffs,
                        m_DOModeCoeffs.data() + modeCount*nVel*nCoeffs, 1);
            Vmath::Zero(nVel*nPhys,
                        m_DOModePhys.data()   + modeCount*nVel*nPhys,   1);
            const int cOff = (modeCount*nVel + c)*nCoeffs;
            const int pOff = (modeCount*nVel + c)*nPhys;
            Vmath::Vcopy(nCoeffs, localCoeffs.data(), 1,
                         m_DOModeCoeffs.data() + cOff, 1);
            Vmath::Vcopy(nPhys,   phys.data(),        1,
                         m_DOModePhys.data() + pOff, 1);
        }
    }

    ReOrthonormalise();
}

/**
 * Initialises the modes via POD from a set of velocity snapshots stored in
 * .chk/.fld files. Reads session keys (all under <SOLVERINFO>):
 *   <I PROPERTY="PODSnapshotPattern" VALUE="snap_*.chk"/>      (required)
 *   <I PROPERTY="PODMeanType"
 *              VALUE="TimeMean|FirstSnapshot|ProvidedMeanField"/>
 *   <I PROPERTY="PODMeanFile"  VALUE="..."/>   (only if ProvidedMeanField)
 *   <P> PODtmin = 40.0 </P>      (optional; warm-up cutoff, requires PODdt)
 *   <P> PODdt   = 0.5  </P>      (sampling output interval = ChkSteps*DT)
 * Snapshot list is the glob expansion of PODSnapshotPattern (sorted by trailing
 * _<index>.<ext> if present, else lexicographically). Method-of-snapshots POD
 * is computed in the velocity mass inner product. The leading m_nDOModes modes
 * are written into m_DOMode*; ReOrthonormalise() then projects them to the
 * discrete divergence-free space, applies homogeneous-BC bracketing, and mass-
 * orthonormalises. The POD spectrum (sigma_k, eigenvectors v_{p,k}) is also
 * stashed into m_podSigmas / m_podEigVecs / m_podNumSnapshots so that the
 * subsequent InitialiseYi() can build Y from snapshot projection (see there).
 */
void DOVelocityCorrectionScheme::InitialiseModesFromPOD()
{
    if (m_nDOModes == 0) return;

    DOPODInitialiser::Config cfg;

    ASSERTL0(m_session->DefinesSolverInfo("PODSnapshotPattern"),
             "DOInitModeBasis=POD requires "
             "<I PROPERTY=\"PODSnapshotPattern\" VALUE=\"...\"/>.");
    const std::string pattern = m_session->GetSolverInfo("PODSnapshotPattern");
    cfg.snapshotFiles = DOPODInitialiser::ExpandGlob(pattern);

    NekDouble podTmin = 0.0, podDt = 0.0;
    m_session->LoadParameter("PODtmin", podTmin, podTmin);
    m_session->LoadParameter("PODdt",   podDt,   podDt);
    ASSERTL0(!(podTmin > 0.0 && podDt <= 0.0),
             "PODtmin requires PODdt (sampling interval) to be > 0.");
    if (podTmin > 0.0)
    {
        size_t k0 = std::min(cfg.snapshotFiles.size(),
                             (size_t)std::ceil(podTmin / podDt));
        cfg.snapshotFiles.erase(cfg.snapshotFiles.begin(),
                                cfg.snapshotFiles.begin() + k0);
    }

    ASSERTL0(!cfg.snapshotFiles.empty(),
             "DOInitModeBasis=POD: PODSnapshotPattern matched zero files: "
             + pattern);
    ASSERTL0((int)cfg.snapshotFiles.size() >= m_nDOModes,
             "DOInitModeBasis=POD: number of snapshots (" +
                 std::to_string(cfg.snapshotFiles.size()) +
                 ") must be >= DOModes (" + std::to_string(m_nDOModes) +
                 "). Run a longer sampling case or reduce DOModes.");

    cfg.numModes = m_nDOModes;     // exactly the DO rank, no extra modes

    std::string meanType;
    m_session->LoadSolverInfo("PODMeanType", meanType, "TimeMean");
    if (meanType == "TimeMean")
    {
        cfg.meanType = DOPODInitialiser::MeanType::TimeMean;
    }
    else if (meanType == "FirstSnapshot")
    {
        cfg.meanType = DOPODInitialiser::MeanType::FirstSnapshot;
    }
    else if (meanType == "ProvidedMeanField")
    {
        cfg.meanType = DOPODInitialiser::MeanType::ProvidedMeanField;
        ASSERTL0(m_session->DefinesSolverInfo("PODMeanFile"),
                 "PODMeanType=ProvidedMeanField requires PODMeanFile.");
        cfg.meanFile = m_session->GetSolverInfo("PODMeanFile");
    }
    else
    {
        ASSERTL0(false,
                 "PODMeanType must be 'TimeMean', 'FirstSnapshot' or "
                 "'ProvidedMeanField'.");
    }

    cfg.verbose = m_verbose;

    m_podInitialiser =
        std::make_unique<DOPODInitialiser>(
            m_session, m_fields, m_velocity, cfg);
    DOPODInitialiser &pod = *m_podInitialiser;
    pod.Compute();
    pod.ExportToDOMode(m_DOModePhys, m_DOModeCoeffs);

    // Install POD mean as the simulation mean field, overriding whatever
    // VelocityCorrectionScheme::v_DoInitialise set from <InitialConditions>.
    // Without this, the mean (= <InitialConditions>) and modes (= POD modes
    // around the POD time mean) are inconsistent at t=0.
    {
        const int nVel    = (int)m_velocity.size();
        const int nCoeffs = m_fields[m_velocity[0]]->GetNcoeffs();
        Array<OneD, NekDouble> meanCoeffs(nVel * nCoeffs);
        pod.ExportMean(meanCoeffs);
        for (int c = 0; c < nVel; ++c)
        {
            auto &f = m_fields[m_velocity[c]];
            Vmath::Vcopy(nCoeffs, meanCoeffs.data() + c * nCoeffs, 1,
                         f->UpdateCoeffs().data(), 1);
            f->BwdTrans(f->GetCoeffs(), f->UpdatePhys());
        }
    }

    // Stash POD spectrum + eigenvectors for InitialiseYi (Y from projection).
    m_podSigmas       = pod.SingularValues();
    m_podEigVecs      = pod.EigenVectors();
    m_podNumSnapshots = pod.NumSnapshots();

    if (m_verbose)
    {
        std::cout << "[DOVelocityCorrectionScheme][POD] init done: "
                  << cfg.snapshotFiles.size()
                  << " snapshots, " << m_nDOModes << " modes, energy="
                  << pod.EnergyFraction() << "\n";
    }
}

void DOVelocityCorrectionScheme::RestoreFromDOArchive(
    const std::string &fldPath)
{
    const int nVel    = m_velocity.size();
    const int nPhys   = m_fields[0]->GetTotPoints();
    const int nCoeffs = m_fields[m_velocity[0]]->GetNcoeffs();

    std::vector<LibUtilities::FieldDefinitionsSharedPtr> FieldDef;
    std::vector<std::vector<NekDouble>>                  FieldData;
    LibUtilities::FieldMetaDataMap                       meta;
    LibUtilities::FieldIOSharedPtr fio =
        LibUtilities::FieldIO::CreateForFile(m_session, fldPath);
    fio->Import(fldPath, FieldDef, FieldData, meta);

    for (int m = 0; m < m_nDOModes; ++m)
    {
        for (int c = 0; c < nVel; ++c)
        {
            const std::string fieldName =
                "mode_" + std::to_string(m) + "_" +
                m_session->GetVariable(m_velocity[c]);
            Array<OneD, NekDouble> coeffArr(nCoeffs, 0.0);
            std::string mutableName = fieldName;
            bool found = false;
            for (size_t i = 0; i < FieldDef.size(); ++i)
            {
                const auto &flds = FieldDef[i]->m_fields;
                if (std::find(flds.begin(), flds.end(), fieldName) ==
                    flds.end()) continue;
                m_fields[m_velocity[c]]->ExtractDataToCoeffs(
                    FieldDef[i], FieldData[i], mutableName, coeffArr);
                found = true;
            }
            ASSERTL0(found, "DORestartFile: field " + fieldName +
                            " not found in archive " + fldPath +
                            " (was the archive written with fewer modes?).");
            Vmath::Vcopy(nCoeffs, coeffArr.data(), 1,
                         m_DOModeCoeffs.data() + (m*nVel + c)*nCoeffs, 1);
            Array<OneD, NekDouble> physArr(
                nPhys, m_DOModePhys.data() + (m*nVel + c)*nPhys, eArrayWrapper);
            m_fields[m_velocity[c]]->BwdTrans(coeffArr, physArr);
        }
    }

    const bool root = (m_session->GetComm()->GetRank() == 0);

    auto it = meta.find("DOVelocityCorrectionScheme_Yi_hex");
    ASSERTL0(it != meta.end(), "DORestartFile: Yi_hex not found in metadata");
    {
        // archives store the GLOBAL population; decode fully, keep shard
        std::vector<NekDouble> yAll((size_t)m_nDOParticles * m_nDOModes);
        ASSERTL0(DecodeHexDoubles(it->second, yAll.data(),
                                  m_nDOParticles * m_nDOModes),
                 "DORestartFile: Yi_hex size mismatch");
        Vmath::Vcopy(m_npLocal * m_nDOModes,
                     yAll.data() + (size_t)m_npOffset * m_nDOModes, 1,
                     m_Yi.data(), 1);
    }

    // Pressure modes: prefer the partition-aware mode_<i>_p fields; fall
    // back to the legacy rank-local PCoeffs_hex (valid single-rank only).
    // Missing pressure modes are benign either way: DOImplicitSolve rewrites
    // them before the first Y-RHS reads grad(p_k).
    const int nPC = m_pressure->GetNcoeffs();
    bool pFromFields = (m_nDOModes > 0);
    for (int m = 0; m < m_nDOModes && pFromFields; ++m)
    {
        const std::string fieldName = "mode_" + std::to_string(m) + "_p";
        Array<OneD, NekDouble> coeffArr(nPC, 0.0);
        std::string mutableName = fieldName;
        bool found = false;
        for (size_t i = 0; i < FieldDef.size(); ++i)
        {
            const auto &flds = FieldDef[i]->m_fields;
            if (std::find(flds.begin(), flds.end(), fieldName) ==
                flds.end()) continue;
            m_pressure->ExtractDataToCoeffs(
                FieldDef[i], FieldData[i], mutableName, coeffArr);
            found = true;
        }
        if (!found)
        {
            pFromFields = false;
            break;
        }
        Vmath::Vcopy(nPC, coeffArr.data(), 1,
                     m_DOModePCoeffs.data() + m*nPC, 1);
    }
    if (pFromFields)
    {
        if (root)
        {
            std::cout << "[DORestartFile] restored pressure modes"
                         " from fields\n";
        }
    }
    else
    {
        auto itP = meta.find("DOVelocityCorrectionScheme_PCoeffs_hex");
        if (itP != meta.end())
        {
            if (DecodeHexDoubles(itP->second, m_DOModePCoeffs.data(),
                                 (int)m_DOModePCoeffs.size()))
            {
                if (root)
                {
                    std::cout << "[DORestartFile] restored pressure modes"
                                 " from legacy PCoeffs_hex\n";
                }
            }
            else if (root)
            {
                std::cout << "[DORestartFile] WARNING: PCoeffs_hex size "
                             "mismatch (rank-local legacy encoding) -- "
                             "pressure modes not restored.\n";
            }
        }
    }

    // Stochastic forcing state (soft-skip if absent: old archives still work).
    if (m_nForcingChannels > 0)
    {
        auto itEta = meta.find("DOVelocityCorrectionScheme_ForcingEta_hex");
        auto itRng = meta.find("DOVelocityCorrectionScheme_ForcingRng");
        if (itEta != meta.end() && itRng != meta.end())
        {
            DecodeHexDoubles(itEta->second, m_forcingEta.data(),
                             m_nDOParticles * m_nForcingChannels);
            std::istringstream rngSs(itRng->second);
            rngSs >> m_forcingRng;
            if (root)
            {
                std::cout
                    << "[DORestartFile] restored stochastic forcing state\n";
            }
        }
        else if (root)
        {
            std::cout << "[DORestartFile] WARNING: forcing state not found in "
                         "archive -- OU process restarts from seed.\n";
        }
    }

    PrecomputeGradients();
    if (root)
    {
        std::cout << "[DORestartFile] restored from " << fldPath << "\n";
    }
}

} // namespace Nektar
