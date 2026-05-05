#include <IncNavierStokesSolver/EquationSystems/DOVelocityCorrectionScheme.h>
#include <IncNavierStokesSolver/EquationSystems/DOPODInitialiser.h>
#include <IncNavierStokesSolver/EquationSystems/DOReducedCGEigenBasis.h>

#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <MultiRegions/ContField.h>

#include <algorithm>
#include <cmath>
#include <cstdio>       // snprintf for hex hash strings (verbose-only diagnostics)
#include <cstring>      // memcpy for type-punning doubles in non-contamination hash
#include <iostream>
#include <limits>
#include <numeric>
#include <random>       // mt19937 + normal_distribution for Yi sampling
#include <vector>

namespace Nektar
{
namespace
{

// ============================================================================
// BC capture / homogenize / restore helpers
// ============================================================================

/**
 * Stores one BC state for one velocity component and one boundary region.
 * Used to capture and restore BCs around mode projection.
 * - `fieldId`: which component;
 * - `region`: which boundary region;
 * - `phys` and `coeffs`: the BC arrays for that region, which get overwritten during
 *      mode projection and need to be restored afterward.
 */
struct BcArrayState
{
    int fieldId = -1;
    int region  = -1;
    std::vector<NekDouble> phys;
    std::vector<NekDouble> coeffs;
};

/**
 * A collection of BcArrayState s, representing the full BC state for
 * all velocity components in all regions
 */
struct VelocityBCState
{
    std::vector<BcArrayState> entries;
};

/**
 * Saves the current boundary data for all velocity components for all boundary regions.
 */
VelocityBCState CaptureVelocityBCState(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    const Array<OneD, int> &velocity)
{
    VelocityBCState state;
    for (int c = 0; c < velocity.size(); ++c)               // loop over components
    {
        const int fieldId = velocity[c];
        auto bcs = fields[fieldId]->GetBndConditions();
        auto bnd = fields[fieldId]->GetBndCondExpansions();
        for (int region = 0; region < bcs.size(); ++region) // loop over regions
        {
            // keep only Dirichlet & Neumann
            const auto type = bcs[region]->GetBoundaryConditionType();
            if (type != SpatialDomains::eDirichlet &&
                type != SpatialDomains::eNeumann)
            {
                continue;
            }

            // copy phys and coeff values
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
    return state;   // vector of BcArrayState entries for all velocity components and all regions
}

/**
 * Zeroes velocity BC data for modes.
 */
void HomogenizeVelocityBCsForModes(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    const Array<OneD, int> &velocity)
{
    for (int c = 0; c < velocity.size(); ++c)               // loop over components
    {
        const int fieldId = velocity[c];
        auto bcs = fields[fieldId]->GetBndConditions();
        auto bnd = fields[fieldId]->GetBndCondExpansions();
        for (int region = 0; region < bcs.size(); ++region) // loop over regions
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
        std::copy(entry.phys.begin(), entry.phys.end(), exp->UpdatePhys().data());
        std::copy(entry.coeffs.begin(), entry.coeffs.end(),
                  exp->UpdateCoeffs().data());
    }
}

// ============================================================================
// Mode-data helper
// ============================================================================

/**
 * Stores one DO mode component-by-component.
 * `coeffs[c]` and `phys[c]` are the representations of component `c`.
 */
struct ModeData
{
    std::vector<std::vector<NekDouble>> coeffs;
    std::vector<std::vector<NekDouble>> phys;
};

/**
 * Computes the vector mass inner product <u,v>_M=\sum_c u_c_coeffs^T M v_c_coeffs, summed over components.
 * M[i,j] = \int \phi_i \phi_j d\Omega
 */
NekDouble VectorMassInner(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    const Array<OneD, int> &velocity, const ModeData &u, const ModeData &v)
{
    NekDouble sum = 0.0;                                            // local partial sum
    for (int c = 0; c < velocity.size(); ++c)                       // loop over components
    {
        auto field = fields[velocity[c]];
        Array<OneD, NekDouble> ip(field->GetNcoeffs(), 0.0);        // initialise
        Array<OneD, NekDouble> phys(field->GetTotPoints(), 0.0);    // initialise phys
        std::copy(v.phys[c].begin(), v.phys[c].end(), phys.data()); // phys: v_c_phys
        field->IProductWRTBase(phys, ip);                           // ip: M v_c_coeffs
        sum += Vmath::Dot(field->GetNcoeffs(),u.coeffs[c].data(), 1, ip.data(), 1);
    }
    // Global AllReduce so callers (MGS, normalisation, G-matrix in ReOrth)
    // see the full-domain mass IP under MPI partitioning.
    fields[velocity[0]]->GetComm()->GetRowComm()->AllReduce(
        sum, LibUtilities::ReduceSum);
    return sum;
}

// ============================================================================
// Constant subspace cache + projection helpers
// ============================================================================
//
// The discrete FEM Stokes projector preserves spatially-constant velocity
// fields (a constant is divergence-free), so without explicit treatment a
// non-zero constant component leaks into the modes during time stepping and
// the modes drift to uniform velocity vectors. To prevent this, we treat
// the constant velocity subspace
//      c_α(x) = (1/√|Ω|) e_α       (α = 0,...,nVel-1)
// as a fixed, non-evolving sub-basis and continuously project the DO modes
// (and the rate N) orthogonal to it under the velocity mass inner product.
// In purely-Dirichlet problems this subspace is empty (constants violate the
// BCs), so the projection becomes a no-op there.
//
// Mathematical reduction. With c_α normalised by √|Ω|, the inner product
//      <v, c_α>_M = (1/√|Ω|) ∫_Ω v_α dΩ
// so the per-component update v_α ← v_α - <v,c_α>_M c_α reduces to
//      v_α(x) ← v_α(x) - (1/|Ω|) ∫_Ω v_α dΩ              (phys representation)
// i.e. subtract the spatial mean of component α from itself. The coefficient
// counterpart is `coeffs[α] -= mean * onesCoeffs`, with `onesCoeffs` the
// (cached) coefficient representation of the constant 1.

/**
 * Cached state for the constant-subspace projection.
 * - `domainArea`: |Ω| = ∫_Ω 1 dΩ.
 * - `onesCoeffs`: coefficient vector of the constant function 1 (per component;
 *      identical across components since all velocity fields share the same
 *      expansion, but cached as an array for symmetry with the per-component API).
 * - `admissible[c]`: whether constants are admissible for component c (true iff
 *      that component has no Dirichlet BC; false for hard-walled components).
 *      When admissible[c] is false the projection is a no-op for that component.
 * - `inited`: lazy-init flag.
 */
struct ConstantSubspaceCache
{
    NekDouble                                 domainArea = 0.0;
    std::vector<Array<OneD, NekDouble>>       onesCoeffs;       // size nVel
    std::vector<bool>                         admissible;       // size nVel
    bool                                      inited     = false;
};

/**
 * Lazily computes (on first call) and returns the constant-subspace cache.
 * `domainArea` is computed via `field0->Integral(ones)`; `onesCoeffs[c]` via
 * `field[c]->FwdTrans(ones, ...)`; `admissible[c]` is true iff the
 * AssemblyMapCG of component `c` has no Dirichlet bnd-coeffs.
 */
ConstantSubspaceCache &GetConstantSubspaceCache(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    const Array<OneD, int> &velocity)
{
    static ConstantSubspaceCache s_cache;
    if (s_cache.inited) return s_cache;

    const int nVel = velocity.size();
    s_cache.onesCoeffs.resize(nVel);
    s_cache.admissible.assign(nVel, false);

    // domain area: |Ω| = \int_\Omega 1 dΩ via Integral on a constant-1 phys array
    auto field0     = fields[velocity[0]];
    const int nPhys = field0->GetTotPoints();
    Array<OneD, NekDouble> ones(nPhys, 1.0);
    s_cache.domainArea = field0->Integral(ones);

    // per-component: admissibility flag + onesCoeffs (FwdTrans of constant 1)
    for (int c = 0; c < nVel; ++c)
    {
        auto cf = std::dynamic_pointer_cast<MultiRegions::ContField>(
            fields[velocity[c]]);
        ASSERTL0(cf, "DOVelocityCorrectionScheme: CG only.");
        // constants admissible iff this component has no Dirichlet bnd-coeffs
        s_cache.admissible[c] =
            (cf->GetLocalToGlobalMap()->GetNumGlobalDirBndCoeffs() == 0);
        if (!s_cache.admissible[c]) continue;       // skip FwdTrans when subspace is empty

        const int nc = cf->GetNcoeffs();
        s_cache.onesCoeffs[c] = Array<OneD, NekDouble>(nc, 0.0);
        Array<OneD, NekDouble> physOnes(cf->GetTotPoints(), 1.0);
        cf->FwdTrans(physOnes, s_cache.onesCoeffs[c]);
    }

    s_cache.inited = true;
    return s_cache;
}

/**
 * Projects `N` orthogonal to the constant subspace under the velocity mass
 * inner product (component-wise), using `innerArg` as the projection argument
 * (so that the projection coefficient matches the one used downstream in the
 * DO projection of N onto the existing modes).
 *
 * For each spatial component c with admissible constants:
 *     β_c = <innerArg[c], c_c>_M = (1/√|Ω|) ∫_Ω innerArg[c] dΩ
 *     N[c](x) ← N[c](x) - β_c · (1/√|Ω|)
 *             = N[c](x) - (1/|Ω|) ∫_Ω innerArg[c] dΩ
 * i.e. subtract the spatial mean of innerArg[c] from N[c].
 */
void ProjectOutConstantsFromN(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    const Array<OneD, int>                            &velocity,
    const Array<OneD, Array<OneD, NekDouble>>         &innerArg,
    Array<OneD, Array<OneD, NekDouble>>               &N)
{
    const auto &cache = GetConstantSubspaceCache(fields, velocity);
    if (cache.domainArea <= 0.0) return;        // degenerate; nothing safe to do

    const int nVel  = velocity.size();
    const int nPhys = fields[0]->GetTotPoints();
    Array<OneD, NekDouble> ones(nPhys, 1.0);
    for (int c = 0; c < nVel; ++c)              // loop over components
    {
        if (!cache.admissible[c]) continue;     // Dirichlet component: constants not admissible
        const NekDouble integ = fields[velocity[c]]->Integral(innerArg[c]);     // ∫ innerArg[c] dΩ
        const NekDouble mean  = integ / cache.domainArea;                       // spatial mean
        Vmath::Sadd(nPhys, -mean, N[c].data(), 1, N[c].data(), 1);              // N[c] -= mean
    }
}

/**
 * Projects a candidate mode `cand` orthogonal to the constant subspace under
 * the velocity mass inner product, in BOTH phys and coeff representations.
 *
 * For each spatial component c with admissible constants:
 *     β_c = <cand, c_c>_M = (1/√|Ω|) ∫_Ω cand.phys[c] dΩ
 *     cand.phys[c]   ← cand.phys[c]   - mean
 *     cand.coeffs[c] ← cand.coeffs[c] - mean · onesCoeffs[c]
 * with mean = (1/|Ω|) ∫_Ω cand.phys[c] dΩ.
 *
 * Used after each Stokes projection inside ReOrthonormalise to remove the
 * constant component the projector preserves (constants are divergence-free).
 */
void ProjectOutConstantsFromCand(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    const Array<OneD, int>                            &velocity,
    ModeData                                          &cand)
{
    const auto &cache = GetConstantSubspaceCache(fields, velocity);
    if (cache.domainArea <= 0.0) return;

    const int nVel = velocity.size();
    for (int c = 0; c < nVel; ++c)              // loop over components
    {
        if (!cache.admissible[c]) continue;     // Dirichlet component: constants not admissible

        auto      f     = fields[velocity[c]];
        const int nPhys = f->GetTotPoints();
        const int nCo   = f->GetNcoeffs();

        // ∫ cand.phys[c] dΩ via a borrowed (non-owning) view over the std::vector storage
        Array<OneD, NekDouble> physView(nPhys, cand.phys[c].data());
        const NekDouble integ = f->Integral(physView);
        const NekDouble mean  = integ / cache.domainArea;

        // phys: subtract the spatial mean
        for (auto &v : cand.phys[c]) v -= mean;

        // coeffs: subtract mean · onesCoeffs[c] (coefficient image of constant 1)
        const NekDouble *oc = cache.onesCoeffs[c].data();
        NekDouble       *cc = cand.coeffs[c].data();
        Vmath::Svtvp(nCo, -mean, oc, 1, cc, 1, cc, 1);                          // cc -= mean · oc
    }
}

} // namespace

// ============================================================================
// Class registration
// ============================================================================

// register DOVelocityCorrectionScheme with the factory
std::string DOVelocityCorrectionScheme::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "DOVelocityCorrectionScheme", DOVelocityCorrectionScheme::create);

// register DOVelocityCorrectionScheme enum value for SolverType
std::string DOVelocityCorrectionScheme::solverTypeLookupId =
    LibUtilities::SessionReader::RegisterEnumValue(
        "SolverType", "DOVelocityCorrectionScheme", eDOVelocityCorrectionScheme);

// constructor
DOVelocityCorrectionScheme::DOVelocityCorrectionScheme(const LibUtilities::SessionReaderSharedPtr &pSession,
               const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph), VelocityCorrectionScheme(pSession, pGraph)
{
}

/**
 * Initialises the solver:
 * - runs base VCS initialisation;
 * - reads DO parameters;
 * - allocates memory for modes and Yi;
 *      - modes + mode history,
 *      - mode pressures,
 *      - Yi + Yi history,
 */
void DOVelocityCorrectionScheme::v_InitObject(bool DeclareField)
{
    VelocityCorrectionScheme::v_InitObject(DeclareField);   // VCS init
    ASSERTL0(!m_ALESolver, "ALE not supported with DOVelocityCorrectionScheme.");
    ASSERTL0(!m_meshDistorted, "Distorted mesh not supported with DOVelocityCorrectionScheme.");

    // sizes
    const int nPhys   = m_fields[0]->GetTotPoints();
    const int nCoeffs = m_fields[0]->GetNcoeffs();
    const int nPC     = m_pressure->GetNcoeffs();
    const int nDim    = m_velocity.size();

    m_nDOModes     = m_session->GetParameter("DOModes");
    m_nDOParticles = m_session->GetParameter("DOParticles");

    // allocate arrays. NB the m_doScheme owns the multi-step history for both
    // the modes and Y (m_F[1]/m_F[0]/u^{n-1}/etc); DOVelocityCorrectionScheme no longer carries
    // m_DOModePhysPrev/m_DOModeNExpl/m_DOModeNExplPrev/m_YiPrev/m_YiRhsPrev.
    m_DOModePhys      = Array<OneD, NekDouble>(nDim*nPhys*m_nDOModes, 0.0);     // modes' physical values
    m_DOModeCoeffs    = Array<OneD, NekDouble>(nDim*nCoeffs*m_nDOModes, 0.0);   // modes' coeffs
    m_DOModePCoeffs   = Array<OneD, NekDouble>(nPC * m_nDOModes, 0.0);          // mode pressure coeffs (read by Y RHS)
    m_Yi              = Array<OneD, NekDouble>(m_nDOParticles*m_nDOModes, 0.0); // stoch coeffs
    m_Cij.assign(m_nDOModes*m_nDOModes, 0.0);                                   // second moment
    m_Mkli.assign(m_nDOModes*m_nDOModes*m_nDOModes, 0.0);                       // third moment


    // Initial mode basis: "Laplacian" (default) or "POD". When POD is
    // selected, additional keys consulted in InitialiseModesFromPOD():
    //     <S> PODSnapshotPattern </S>     (glob with one '*', e.g. "snap_*.chk")
    //     <I PROPERTY="PODMeanType" VALUE="TimeMean|FirstSnapshot|ProvidedMeanField"/>
    //     <S> PODMeanFile </S>            (only if PODMeanType=ProvidedMeanField)
    //     <P> PODNumModes = DOModes </P>  (defaults to DOModes)
    m_session->LoadSolverInfo("DOInitModeBasis", m_doInitBasis, "Laplacian");
    ASSERTL0(m_doInitBasis == "Laplacian" || m_doInitBasis == "POD",
             "DOInitModeBasis must be 'Laplacian' or 'POD'.");

    // Gaussian parameters for Yi init
    if (m_session->DefinesParameter("DOYiSeed"))
        m_doYiSeed = (int)m_session->GetParameter("DOYiSeed");
    if (m_session->DefinesParameter("DOYiSigma"))
        m_doYiSigma = m_session->GetParameter("DOYiSigma");

    // additive stochastic forcing (channels): K templates from "ForcingChannels"
    // XML block, σ = OU equilibrium std (per-channel), τ = OU correlation time
    // (0 ⇒ white in time), seed = mt19937 seed
    if (m_session->DefinesParameter("DOForcingNumChannels"))
        m_nForcingChannels = (int)m_session->GetParameter("DOForcingNumChannels");
    if (m_session->DefinesParameter("DOForcingSigma"))
        m_forcingSigma = m_session->GetParameter("DOForcingSigma");
    if (m_session->DefinesParameter("DOForcingTau"))
        m_forcingTau = m_session->GetParameter("DOForcingTau");
    if (m_session->DefinesParameter("DOForcingSeed"))
        m_forcingSeed = (int)m_session->GetParameter("DOForcingSeed");
    if (m_nForcingChannels > 0)
    {
        const int K = m_nForcingChannels;
        m_forcingBasisPhys   = Array<OneD, NekDouble>(K*nDim*nPhys,   0.0);     // K channels, each a vector field on phys grid
        m_forcingBasisCoeffs = Array<OneD, NekDouble>(K*nDim*nCoeffs, 0.0);     // FE coefficients of each channel
        m_forcingEta         = Array<OneD, NekDouble>(m_nDOParticles*K, 0.0);   // per-particle OU amplitudes (start quiescent)
        m_forcingG.assign(m_nDOModes*K, 0.0);                                   // <g_k, u_i>_M (recomputed each step)
        m_forcingA.assign(m_nDOModes*K, 0.0);                                   // (1/Np) Σ_p Y_{p,i} η_{p,k}
        m_forcingRng.seed(static_cast<std::mt19937::result_type>(m_forcingSeed));
    }

    // === DO subsystem time integrator ===========================================
    // State-vector layout: variables 0..S*nVel-1 are the mode phys components
    // (one variable per (mode i, comp c), each of size nPhys); variable S*nVel is
    // Y_flat (size Np*S). Heterogeneous-size GLM API supplies the rest.
    m_doNumModeVars = m_nDOModes * nDim;
    m_doYIdx        = m_doNumModeVars;

    // Use the SAME TIMEINTEGRATIONSCHEME as VCS reads from session (typically
    // IMEX/2 → BDF2/EXT2). DOImplicitSolve relies on the scheme passing
    // lambda = (2/3)·dt to the implicit callback (BDF2 weight); other schemes
    // would need a generalised Dt = lambda·(scheme-specific factor) inside
    // ModePressureSolve/ModeViscousSolve.
    auto timeInt = m_session->GetTimeIntScheme();
    m_doScheme   = LibUtilities::GetTimeIntegrationSchemeFactory().CreateInstance(
        timeInt.method, timeInt.variant, timeInt.order, timeInt.freeParams);
    m_doOps.DefineOdeRhs(&DOVelocityCorrectionScheme::DOOdeRhs, this);                              // explicit RHS for (modes, Y)
    m_doOps.DefineImplicitSolve(&DOVelocityCorrectionScheme::DOImplicitSolve, this);                // per-mode press+visc Helm; identity for Y
}

/**
 * Runs the VCS IC setup (mean velocity + pressure), and on the first call:
 * - initialises modes from elliptic eigenbasis;
 * - initialises Yi as i.i.d. Gaussian samples;
 * - rotates the joint (modes, Yi) state so C(0) is diagonal;
 * - initialises forcing channels;
 * - packs the (modes, Y) state into m_doState and calls
 *   m_doScheme->InitializeScheme so the integrator knows the t=0 state and
 *   computes the initial RHS for its history.
 */
void DOVelocityCorrectionScheme::v_DoInitialise(bool dumpInitialConditions)
{
    VelocityCorrectionScheme::v_DoInitialise(dumpInitialConditions);    // VCS ICs

    if (!m_modesInitialised)    // only on first call
    {
        if (m_doInitBasis == "POD")
        {
            InitialiseModesFromPOD();               // modes: POD on velocity snapshots (mass IP)
        }
        else
        {
            InitialiseModesFromEllipticEigenbasis();// modes: Laplacian eigenbasis (already orthonormalised)
        }
        InitialiseYi();                             // Y: i.i.d. Gaussian, sample mean removed
        RotateToEigenbasisOfC();                    // diagonalise C(0); rotates modes and Y consistently (no-op if R<=1)
        ReOrthonormalise();                         // Stokes-project each mode to discrete div-free space

        // POD path: ReOrthonormalise's Stokes projection is non-unitary
        // (changes the subspace, not just rotates within it) and is
        // non-deterministic under MPI. Re-project the snapshots onto the
        // post-reorth basis to make (modes, Yi) self-consistent:
        //   u_p = ubar + Σ_k Yi[p,k] * mode_k_new ≈ snap_p.
        if (m_doInitBasis == "POD" && m_podInitialiser)
        {
            // The owned-scratch refactor in RecomputeYiByProjection (Tier B1)
            // reduces but does not eliminate the post-projection NaN flake
            // (measured: n=2 failure 10/10 → 8/10). The remaining trigger
            // appears to be inside IProductWRTBase or a related Nektar
            // codepath. Keep the storage guard as belt-and-suspenders.
            CallRecomputeYiWithStorageGuard();
            m_podInitialiser.reset();   // snapshots/POD state no longer needed

            // Re-zero-mean Yi across particles per mode (DO requires the
            // sample mean of Y to be zero; the projection block is mean 0
            // for TimeMean but not for FirstSnapshot/ProvidedMeanField).
            const NekDouble invNp =
                1.0 / static_cast<NekDouble>(m_nDOParticles);
            for (int i = 0; i < m_nDOModes; ++i)
            {
                NekDouble mu = 0.0;
                for (int p = 0; p < m_nDOParticles; ++p)
                    mu += m_Yi[p * m_nDOModes + i];
                mu *= invNp;
                for (int p = 0; p < m_nDOParticles; ++p)
                    m_Yi[p * m_nDOModes + i] -= mu;
            }
        }
        InitialiseForcingBasis();                   // forcing channels (XML-driven; no-op if K==0)

        // Pack the (modes, Y) initial state into m_doState and initialise the
        // DO subsystem integrator. From here on the scheme owns the multi-step
        // history; DOVelocityCorrectionScheme only sees the post-step (modes, Y) via Unpack.
        const int nVel  = m_velocity.size();
        const int nPhys = m_fields[0]->GetTotPoints();
        const int nVar  = m_doNumModeVars + 1;
        m_doState = Array<OneD, Array<OneD, NekDouble>>(nVar);
        for (int i = 0; i < m_nDOModes; ++i)                                    // mode phys per (i, c)
            for (int c = 0; c < nVel; ++c)
            {
                const int v = i*nVel + c;
                m_doState[v] = Array<OneD, NekDouble>(nPhys);
                Vmath::Vcopy(nPhys, m_DOModePhys.data() + v*nPhys, 1,
                             m_doState[v].data(), 1);
            }
        const int nY = m_nDOParticles * m_nDOModes;
        m_doState[m_doYIdx] = Array<OneD, NekDouble>(nY);
        Vmath::Vcopy(nY, m_Yi.data(), 1, m_doState[m_doYIdx].data(), 1);

        m_doScheme->InitializeScheme(m_timestep, m_doState, m_time, m_doOps);
        m_doSchemeInited = true;
        m_modesInitialised = true;
    }
}

/**
 * Initialises `m_Yi[p*R + i]` = Y_{i,p} (mode i for particle p) by drawing
 * i.i.d. Gaussian samples Y_{i,p} ~ N(0, σ²) with σ = m_doYiSigma, using a
 * Mersenne-Twister seeded by m_doYiSeed (any int is a valid seed).
 *
 * Centering: each column is shifted by its sample mean so E[Y_i] = 0 exactly
 * across the m_nDOParticles realisations (sample mean is O(σ/√Np) without
 * this; subtracting it removes the leading bias and keeps Σ_p Y_{i,p} = 0).
 *
 * Decorrelation: the resulting sample covariance C_{ij} = (1/Np)Σ_p Y_{i,p}Y_{j,p}
 * has off-diagonals of order σ²/√Np. Since `ComputeNMode` uses the simplification
 * μ_i = C_{ii} (valid only when C is diagonal, which `RotateToEigenbasisOfC`
 * normally enforces at the end of every step), the caller (`v_DoInitialise`)
 * must invoke `RotateToEigenbasisOfC` once after this routine to diagonalise
 * the initial C and rotate the modes/Y consistently before the first integration
 * step.
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
                 "DOVelocityCorrectionScheme::InitialiseYi: POD spectrum present but inconsistent "
                 "with m_nDOModes (internal logic error).");
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(m_doYiSeed));

    // 1) Initial Y values.
    if (!podPath)
    {
        // Laplacian path: keep the historical iid sampling order — outer
        // particle, inner mode — with a single std::normal_distribution
        // instance so the RNG sequence is bit-identical to pre-POD-fix runs.
        std::normal_distribution<NekDouble> dist(0.0, m_doYiSigma);
        for (int p = 0; p < m_nDOParticles; ++p)
            for (int i = 0; i < m_nDOModes; ++i)
                m_Yi[p * m_nDOModes + i] = dist(rng);
    }
    else
    {
        // POD path. Conventions:
        //   sigma_i = sqrt(lambda_i),   lambda_i = i-th eigenvalue of the
        //                               (K x K) snapshot correlation matrix
        //                               C[i,j] = <snap_i, snap_j>_M.
        //   v_{p,i} = K-vector eigenvector of C, 2-normalised: sum_p v_{p,i}^2 = 1.
        //
        // Snapshot projection: a_{p,i} = <snap_p - mean, phi_i>_M = sigma_i * v_{p,i}.
        // Setting Y_{p,i} = a_{p,i} for p in [0, K) makes u_p = ubar + sum_i Y_{p,i}
        // phi_i agree with snap_p (truncated to S modes).
        //
        // Variance scaling: DOVelocityCorrectionScheme's sample covariance is
        //   C_ii^{DO} = (1/Np) sum_p Y_{p,i}^2.
        // For pure projection (Np = K), C_ii^{DO} = sigma_i^2 / K  (= lambda_i / K).
        // To keep this scale CONSISTENT for any Np >= K, the iid filler for
        // particles p in [K, Np) must use sigma_i^{iid} = sigma_i / sqrt(K)
        // (NOT the raw sigma_i): then E[(Np-K)/Np * sigma_i^{iid 2}] +
        // (1/Np) * sigma_i^2  =  sigma_i^2 / K. Without this scaling the iid
        // block would dominate at Np >> K and C_ii would jump by a factor of K
        // between the projection and statistical regimes.
        const int Kproj = std::min(m_podNumSnapshots, m_nDOParticles);
        const NekDouble invSqrtK =
            (m_podNumSnapshots > 0)
                ? 1.0 / std::sqrt(static_cast<NekDouble>(m_podNumSnapshots))
                : 1.0;
        for (int i = 0; i < m_nDOModes; ++i)
        {
            const NekDouble sigma_i     = m_podSigmas[i];
            const NekDouble sigma_i_iid = sigma_i * invSqrtK;
            std::normal_distribution<NekDouble> dist(0.0, sigma_i_iid);
            for (int p = 0; p < Kproj; ++p)
            {
                m_Yi[p * m_nDOModes + i] = sigma_i * m_podEigVecs[i][p];
            }
            for (int p = Kproj; p < m_nDOParticles; ++p)
            {
                m_Yi[p * m_nDOModes + i] = dist(rng);
            }
        }
    }

    // 2) Zero-mean each column (mode-i sample mean -> 0 exactly).
    //    For POD with TimeMean and Np == K this is already a no-op (the POD
    //    eigenvectors are orthogonal to the all-ones vector after time-mean
    //    subtraction); for FirstSnapshot/ProvidedMeanField, or when Np != K,
    //    this enforces the zero-mean property the rest of DO assumes.
    const NekDouble invNp = 1.0 / static_cast<NekDouble>(m_nDOParticles);
    for (int i = 0; i < m_nDOModes; ++i)
    {
        NekDouble mu = 0.0;
        for (int p = 0; p < m_nDOParticles; ++p)
            mu += m_Yi[p * m_nDOModes + i];
        mu *= invNp;
        for (int p = 0; p < m_nDOParticles; ++p)
            m_Yi[p * m_nDOModes + i] -= mu;
    }
}

/**
 * Wrapper around DOPODInitialiser::RecomputeYiByProjection() that captures
 * each velocity ExpList's m_coeffs/m_phys BEFORE the call, runs the call,
 * then writes the captured values back through UpdateCoeffs()/UpdatePhys()
 * AFTER the call.
 *
 * Symptom prevented:
 *   On n=2/4 MPI ranks (cylinder POD-init case), without this guard the
 *   first time step's pressure or viscous CG diverged to NaN immediately
 *   after init:
 *     "CG iterations made = 5001 ... error = nan" repeated, then
 *     "DOVelocityCorrectionScheme: orthonormalisation collapsed" or
 *     "DOVelocityCorrectionScheme: forcing channel ... has zero mass-norm".
 *
 * Why this guard works (and why it has to look like this):
 *   RecomputeYiByProjection constructs non-owning eArrayWrapper views into
 *   m_DOModePhys and passes them through Nektar's IProductWRTBase /
 *   ExtractDataToCoeffs codepaths. Those calls leave the velocity ExpLists'
 *   Array<OneD,T> shared-storage / refcount state in a configuration that
 *   confuses the next iterative solve. A bisect showed:
 *     - rewriting m_coeffs/m_phys (with any data) AT THE CALL SITE after
 *       the function returns and its local Array views destruct fixes
 *       n=2 and n=3; merely writing through UpdateCoeffs()/UpdatePhys()
 *       once is enough on n=3 but n=2 needs the full rewrite;
 *     - identical rewrite inside the projection function (before return)
 *       does not fix n=4;
 *     - the saved values must come from BEFORE the call (saving the
 *       post-call values and writing them back fails on n=2);
 *     - byte-by-byte the buffer VALUES are identical before vs after the
 *       projection, so this is numerically a no-op — the side effect is
 *       purely Array internal book-keeping (refcount / detach-on-write).
 *
 * Idempotent. Safe on n=1. See Tier B refactor: the cleaner long-term fix
 * is to remove the eArrayWrapper views inside RecomputeYiByProjection.
 */
void DOVelocityCorrectionScheme::CallRecomputeYiWithStorageGuard()
{
    const int nVel = m_velocity.size();
    std::vector<Array<OneD, NekDouble>> sC(nVel), sP(nVel);
    for (int c = 0; c < nVel; ++c)
    {
        auto &f = m_fields[m_velocity[c]];
        sC[c] = Array<OneD, NekDouble>(f->GetNcoeffs());
        sP[c] = Array<OneD, NekDouble>(f->GetTotPoints());
        Vmath::Vcopy(f->GetNcoeffs(),   f->GetCoeffs().data(), 1, sC[c].data(), 1);
        Vmath::Vcopy(f->GetTotPoints(), f->GetPhys().data(),   1, sP[c].data(), 1);
    }
    m_podInitialiser->RecomputeYiByProjection(
        m_DOModePhys, m_DOModeCoeffs, m_Yi, m_nDOParticles);
    for (int c = 0; c < nVel; ++c)
    {
        auto &f = m_fields[m_velocity[c]];
        Vmath::Vcopy(f->GetNcoeffs(),   sC[c].data(), 1, f->UpdateCoeffs().data(), 1);
        Vmath::Vcopy(f->GetTotPoints(), sP[c].data(), 1, f->UpdatePhys().data(),   1);
    }
}

/**
 * Reads the XML "ForcingChannels" function block, evaluates each channel's
 * spatial template at quadrature points, FwdTrans -> BwdTrans (FE projection),
 * and mass-normalises each channel so ‖g_k‖_M = 1. Variable naming convention
 * in the XML: "g{k}_{component}" with k in 1..K and component in {u, v, w}.
 *
 * Two channels with the same shape stay as two independent OU streams
 * driving identical templates.
 */
void DOVelocityCorrectionScheme::InitialiseForcingBasis()
{
    if (m_nForcingChannels == 0) return;
    ASSERTL0(m_session->DefinesFunction("ForcingChannels"),
             "DOVelocityCorrectionScheme: DOForcingNumChannels > 0 but no XML <FUNCTION NAME=\"ForcingChannels\"> block.");

    const int K       = m_nForcingChannels;
    const int nVel    = m_velocity.size();
    const int nPhys   = m_fields[0]->GetTotPoints();
    const int nCoeffs = m_fields[0]->GetNcoeffs();
    const auto vars   = m_session->GetVariables();      // {"u", "v", [...], "p"}

    // quadrature coordinates (same grid for every velocity component)
    Array<OneD, NekDouble> xq(nPhys), yq(nPhys), zq(nPhys, 0.0);
    m_fields[m_velocity[0]]->GetCoords(xq, yq, zq);

    Array<OneD, NekDouble> phys(nPhys), coeffs(nCoeffs), ip(nCoeffs);
    for (int k = 0; k < K; ++k)                         // loop over channels
    {
        for (int c = 0; c < nVel; ++c)                  // loop over components
        {
            const std::string vname = "g" + std::to_string(k+1) + "_" + vars[m_velocity[c]];
            ASSERTL0(m_session->GetFunctionType("ForcingChannels", vname) ==
                     LibUtilities::eFunctionTypeExpression,
                     "DOVelocityCorrectionScheme: ForcingChannels VAR \"" + vname + "\" missing or not an expression.");
            auto eq = m_session->GetFunction("ForcingChannels", vname);
            eq->Evaluate(xq, yq, zq, phys);                                     // analytical -> phys
            m_fields[m_velocity[c]]->FwdTrans(phys, coeffs);                    // -> FE coefficients
            m_fields[m_velocity[c]]->BwdTrans(coeffs, phys);                    // FE-projected phys (consistent with coeffs)
            const int pOff = (k*nVel + c)*nPhys;
            const int cOff = (k*nVel + c)*nCoeffs;
            Vmath::Vcopy(nPhys,   phys.data(),   1, m_forcingBasisPhys.data()   + pOff, 1); // store the physical values
            Vmath::Vcopy(nCoeffs, coeffs.data(), 1, m_forcingBasisCoeffs.data() + cOff, 1); // store the coeffs
        }

        // ‖g_k‖²_M = Σ_c <g_{k,c}, g_{k,c}>_M = Σ_c coeffs · IProductWRTBase(phys)
        NekDouble nrm2 = 0.0;
        for (int c = 0; c < nVel; ++c)                  // loop over components
        {
            const NekDouble *g_kc_phys   = m_forcingBasisPhys.data()   + (k*nVel + c)*nPhys;    // pointer to this component's physical values of noise
            const NekDouble *g_kc_coeffs = m_forcingBasisCoeffs.data() + (k*nVel + c)*nCoeffs;
            Array<OneD, NekDouble> physView(nPhys, const_cast<NekDouble*>(g_kc_phys));
            m_fields[m_velocity[c]]->IProductWRTBase(physView, ip);             // ip = M g_kc_coeffs (in coeff space)
            nrm2 += Vmath::Dot(nCoeffs, g_kc_coeffs, 1, ip.data(), 1);
        }
        m_fields[m_velocity[0]]->GetComm()->GetRowComm()->AllReduce(
            nrm2, LibUtilities::ReduceSum);                                     // MPI: global ‖g_k‖²_M
        ASSERTL0(nrm2 > 1e-30, "DOVelocityCorrectionScheme: forcing channel " + std::to_string(k+1)
                                + " has zero mass-norm.");
        const NekDouble inv = 1.0 / std::sqrt(nrm2);                            // mass-normalise to ‖g_k‖_M = 1
        Vmath::Smul(nVel*nPhys,   inv, m_forcingBasisPhys.data()   + k*nVel*nPhys,   1,
                    m_forcingBasisPhys.data()   + k*nVel*nPhys,   1);
        Vmath::Smul(nVel*nCoeffs, inv, m_forcingBasisCoeffs.data() + k*nVel*nCoeffs, 1,
                    m_forcingBasisCoeffs.data() + k*nVel*nCoeffs, 1);
    }
}

/**
 * One step of the additive forcing's stochastic state, run once at the start
 * of every v_PostIntegrate (before AdvanceModes/AdvanceYi see η):
 *   1) Exact OU update per (particle, channel):
 *        τ > 0: η ← α η + σ √(1-α²) ξ,  α = exp(-Δt/τ)         (eq-std = σ)
 *        τ = 0: η ← σ √Δt ξ                                     (white limit)
 *   2) Subtract per-channel sample mean across particles so <f>_p = 0 exactly.
 *   3) Recompute the inner-product matrix G[i,k] = <g_k, u_i>_M (S × K) and
 *      the cross-correlation matrix A[i,k] = (1/Np) Σ_p Y_{p,i} η_{p,k}, both
 *      consumed downstream by AdvanceModes (mode RHS) and AdvanceYi (Y RHS).
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

    // 1) OU exact step (white limit when tau=0)
    std::normal_distribution<NekDouble> dist(0.0, 1.0);
    if (tau > 0.0)    // coloured in time
    {
        const NekDouble alpha = std::exp(-dt / tau);
        const NekDouble beta  = sigma * std::sqrt(std::max(0.0, 1.0 - alpha*alpha));
        for (int q = 0; q < Np*K; ++q)
            m_forcingEta[q] = alpha * m_forcingEta[q] + beta * dist(m_forcingRng);
    }
    else            // white-in-time
    {
        const NekDouble beta = sigma * std::sqrt(dt);
        for (int q = 0; q < Np*K; ++q)
            m_forcingEta[q] = beta * dist(m_forcingRng);
    }

    // 2) per-channel centering across particles (zero ensemble-mean forcing)
    const NekDouble invNp = 1.0 / static_cast<NekDouble>(Np);
    for (int k = 0; k < K; ++k)
    {
        NekDouble mean = 0.0;
        for (int p = 0; p < Np; ++p) mean += m_forcingEta[p*K + k];
        mean *= invNp;
        for (int p = 0; p < Np; ++p) m_forcingEta[p*K + k] -= mean;
    }

    // 3a) G[i,k] = Σ_c <g_{k,c}, u_{i,c}>_M = Σ_c coeffs(u_{i,c}) · IProductWRTBase(phys(g_{k,c}))
    Array<OneD, NekDouble> ip(nCoeffs);
    std::fill(m_forcingG.begin(), m_forcingG.end(), 0.0);
    for (int k = 0; k < K; ++k)
        for (int c = 0; c < nVel; ++c)
        {
            const NekDouble *g_kc_phys = m_forcingBasisPhys.data() + (k*nVel + c)*nPhys;
            Array<OneD, NekDouble> physView(nPhys, const_cast<NekDouble*>(g_kc_phys));
            m_fields[m_velocity[c]]->IProductWRTBase(physView, ip);             // ip = M g_kc_coeffs
            for (int i = 0; i < S; ++i)                                         // loop over modes
            {
                const NekDouble *u_ic_coeffs = m_DOModeCoeffs.data()
                                               + (i*nVel + c)*nCoeffs;
                m_forcingG[i*K + k] += Vmath::Dot(nCoeffs, u_ic_coeffs, 1, ip.data(), 1);
            }
        }
    // MPI: m_forcingG entries are partial sums until reduced.
    if (!m_forcingG.empty())
    {
        m_fields[m_velocity[0]]->GetComm()->GetRowComm()->AllReduce(
            m_forcingG, LibUtilities::ReduceSum);
    }

    // 3b) A[i,k] = (1/Np) Σ_p Y_{p,i} η_{p,k}
    std::fill(m_forcingA.begin(), m_forcingA.end(), 0.0);
    for (int p = 0; p < Np; ++p)
    {
        const NekDouble *Yp = m_Yi.data()         + p*S;
        const NekDouble *Ep = m_forcingEta.data() + p*K;
        for (int i = 0; i < S; ++i)
            for (int k = 0; k < K; ++k)
                m_forcingA[i*K + k] += Yp[i] * Ep[k];
    }
    Vmath::Smul((int)m_forcingA.size(), invNp, m_forcingA.data(), 1, m_forcingA.data(), 1);
}

/**
 * Recomputes the sample moments:
 * - C_{ij} = E[Y_i Y_j] in m_Cij[i*m_nDOModes + j]
 * - M_{kli} = E[Y_k Y_l Y_i] in m_Mkli[(k*m_nDOModes + l)*m_nDOModes + i]
 * from the current m_Yi.
 * If verbose, 
 */
void DOVelocityCorrectionScheme::ComputeYMoments()
{
    if (m_nDOModes == 0 || m_nDOParticles == 0) return;
    const NekDouble invN = 1.0 / static_cast<NekDouble>(m_nDOParticles);

    Vmath::Zero(m_nDOModes*m_nDOModes,              m_Cij.data(),  1);
    Vmath::Zero(m_nDOModes*m_nDOModes*m_nDOModes,   m_Mkli.data(), 1);

    // accumulate sums over particles: C_{ij} = sum_p Y_{i,p} Y_{j,p}; M_{kli} = sum_p Y_{k,p} Y_{l,p} Y_{i,p}
    for (int p = 0; p < m_nDOParticles; ++p)
    {
        const NekDouble *y = m_Yi.data() + p*m_nDOModes;    // pointer to Y_{0,p}, Y_{1,p}, ..., Y_{R-1,p}

        for (int i = 0; i < m_nDOModes; ++i)
            for (int j = 0; j < m_nDOModes; ++j)
                {
                    m_Cij[i*m_nDOModes + j] += y[i] * y[j]; // fill C_{ij}
                    for (int k = 0; k < m_nDOModes; ++k)    // fill M_{kli}
                        m_Mkli[(i*m_nDOModes + j)*m_nDOModes + k] += y[i] * y[j] * y[k];

                }
    }
    // normalise for expected values
    Vmath::Smul(m_nDOModes*m_nDOModes,            invN, m_Cij.data(),  1, m_Cij.data(),  1);
    Vmath::Smul(m_nDOModes*m_nDOModes*m_nDOModes, invN, m_Mkli.data(), 1, m_Mkli.data(), 1);

    if (m_verbose && m_nDOModes > 1)
    {
        // Diagonality / symmetry diagnostics for C
        NekDouble maxOff = 0.0, maxDiag = 0.0;
        NekDouble fOff2 = 0.0, fAll2 = 0.0;
        NekDouble symMax = 0.0;
        NekDouble traceC = 0.0, minDiag = std::numeric_limits<NekDouble>::max();
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
                    std::abs(m_Cij[i*m_nDOModes + j] - m_Cij[j*m_nDOModes + i]));
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
 * Adds the DO contribution to the mean velocity's explicit term (vector field),
 *      doCorr[c][k] = -\sum_{i,j} C_{ij} (u_i(x_k) . grad) u_j(x_k)
 *                   = -\sum_{i,j} C_{ij} \sum_d u_{i,d}(x_k) ∂_d u_{j,c}(x_k)
 *      where u_{i,c} is mode i's velocity field's component c and writes it into `doCorr`.
 * doCorr[c][k] is the c-th spatial component at quadrature point k.
 */
void DOVelocityCorrectionScheme::ComputeDOMeanCoupling(
    Array<OneD, Array<OneD, NekDouble>> &doCorr)
{
    const int nVel  = m_velocity.size();
    const int nPhys = m_fields[0]->GetTotPoints();

    for (int c = 0; c < nVel; ++c)
        Vmath::Zero(nPhys, doCorr[c].data(), 1);    // zero output
    if (m_nDOModes == 0) return;

    // precompute ∂_d u_{j,c} for all (j, c, d)
    Array<OneD, NekDouble> duJcd(m_nDOModes * nVel * nVel * nPhys); // will hold the derivative terms
    Array<OneD, NekDouble> tmp(nPhys);
    for (int j = 0; j < m_nDOModes; ++j)                            // loop over modes
    {
        for (int c = 0; c < nVel; ++c)                              // loop over components of u_j to differentiate
        {
            const NekDouble *u_jc = m_DOModePhys.data()             // pointer to mode j, component c
                                   + (j*nVel + c)*nPhys;
            Vmath::Vcopy(nPhys, u_jc, 1, tmp.data(), 1);            // tmp = u_jc
            for (int d = 0; d < nVel; ++d)                          // contraction inside (u_i . grad) u_j
            {
                Array<OneD, NekDouble> duOut = duJcd +              // points to duJcd with the appropriate offset (mode j, component c, )
                                              ((j*nVel + c)*nVel+ d)*nPhys;
                m_fields[m_velocity[c]]->PhysDeriv(d, tmp, duOut);  // differentiate tmp wrt d, write to duOut
            }
        }
    }

    // accumulate doCorr[c]
    Array<OneD, NekDouble> prod(nPhys);
    for (int i = 0; i < m_nDOModes; ++i)
    {
        for (int j = 0; j < m_nDOModes; ++j)
        {
            const NekDouble Cij = m_Cij[i*m_nDOModes + j];
            if (std::abs(Cij) < 1e-12) continue;

            for (int c = 0; c < nVel; ++c)                          // output spatial component
            {
                for (int d = 0; d < nVel; ++d)                      // contraction inside (u_i . grad) u_j
                {
                    const NekDouble *u_id = m_DOModePhys.data()     // pointer to first value of u_{i,d} (mode i, component d)
                                          + (i*nVel +d)*nPhys;
                    const NekDouble *du = duJcd.data()              // pointer to first value of ∂_d u_{j,c}
                                          + ((j*nVel + c)*nVel + d)*nPhys;

                    Vmath::Vmul(nPhys, u_id, 1,                     // prod = u_id * du
                                du, 1, prod.data(), 1);
                    Vmath::Svtvp(nPhys, -Cij, prod.data(), 1,       // scaled-vector-times-vector-plus
                                doCorr[c].data(), 1,                // doCorr[c][k] = -Cij * prod[k] + doCorr[c][k] for all ks
                                doCorr[c].data(), 1);
                }
            }
        }
    }

    if (m_verbose && m_doStepCounter == 1)
    {
        NekDouble m = 0.0;
        for (int c = 0; c < nVel; ++c)
            for (int k = 0; k < nPhys; ++k)
                m = std::max(m, std::abs(doCorr[c][k]));
        std::cout << "[DOVelocityCorrectionScheme][diag] meanCoup max|C_{ij} F_{ij}| = " << m << "\n";
    }
}

/**
 * Cross terms for one mode:
 *      cross[c] = -[(ū . grad) u_i + (u_i . grad) ū]_c
 *               = -[\sum_d (ū_d ∂_d u_{i,c}) + (u_{i,d} ∂_d ū_c)].
 * Output overwritten in `cross`.
 */
void DOVelocityCorrectionScheme::ComputeModeCross(int i,
                              Array<OneD, Array<OneD, NekDouble>> &cross)
{
    const int nVel  = m_velocity.size();
    const int nPhys = m_fields[0]->GetTotPoints();

    Array<OneD, NekDouble> tmp_uic(nPhys), tmp_uBarc(nPhys);                    // PhysDeriv inputs
    Array<OneD, NekDouble> du(nPhys), prod(nPhys);

    for (int c = 0; c < nVel; ++c)                                              // output spatial component
    {
        Vmath::Zero(nPhys, cross[c].data(), 1);                                 // zero output

        const NekDouble *u_ic = m_DOModePhys.data() + (i*nVel + c)*nPhys;       // pointer to mode i, component c
        Vmath::Vcopy(nPhys, u_ic, 1, tmp_uic.data(), 1);                        // tmp_uic = u_{i,c}
        Vmath::Vcopy(nPhys, m_fields[m_velocity[c]]->GetPhys().data(), 1,       // tmp_uBarc = ū_c 
                     tmp_uBarc.data(), 1);

        for (int d = 0; d < nVel; ++d)                                          // contraction 
        {
            // term 1: cross[c] -= ū_d * ∂_d u_{i,c}
            m_fields[m_velocity[c]]->PhysDeriv(d, tmp_uic, du);                 // du = ∂_d u_{i,c}
            const auto &uBar_d = m_fields[m_velocity[d]]->GetPhys();            // mean's d-component
            Vmath::Vmul(nPhys, uBar_d.data(), 1, du.data(), 1, prod.data(), 1); // prod = uBar_d * du
            Vmath::Svtvp(nPhys, -1.0, prod.data(), 1,                           // cross[c][k] -= prod[k] for all ks
                         cross[c].data(), 1, cross[c].data(), 1);

            // term 2: cross[c] -= u_{i,d} * ∂_d ū_c
            m_fields[m_velocity[c]]->PhysDeriv(d, tmp_uBarc, du);               // du = ∂_d ū_c
            const NekDouble *u_id = m_DOModePhys.data() + (i*nVel + d)*nPhys;   // pointer to mode i, component d
            Vmath::Vmul(nPhys, u_id, 1, du.data(), 1, prod.data(), 1);          // prod = u_id * du
            Vmath::Svtvp(nPhys, -1.0, prod.data(), 1,                           // cross[c][k] -= prod[k] for all ks
                         cross[c].data(), 1, cross[c].data(), 1);
        }
    }
}

/**
 * Computes the laplacian of mode i in physical space. Writes result in `lap`.
 */
void DOVelocityCorrectionScheme::ComputeModeLaplacian(int i,
                                  Array<OneD, Array<OneD, NekDouble>> &lap)
{
    const int nVel  = m_velocity.size();
    const int nPhys = m_fields[0]->GetTotPoints();

    Array<OneD, NekDouble> phys(nPhys), du(nPhys), d2u(nPhys);

    for (int c = 0; c < nVel; ++c)                  // output spatial component
    {
        Vmath::Zero(nPhys, lap[c].data(), 1);       // zero it out
        const NekDouble *u_ic = m_DOModePhys.data() // pointer to mode i, component c
                              + (i*nVel + c)*nPhys;
        Vmath::Vcopy(nPhys, u_ic, 1,                // phys = u_{i,c}
                     phys.data(), 1);
        for (int d = 0; d < nVel; ++d)              // sum over derivative directions
        {
            m_fields[m_velocity[c]]->PhysDeriv(d, phys, du);    // du = ∂_d u_{i,c}
            m_fields[m_velocity[c]]->PhysDeriv(d, du, d2u);     // d2u = ∂²_d u_{i,c}
            Vmath::Vadd(nPhys, d2u.data(), 1, lap[c].data(),    // lap[c] += d2u
                        1, lap[c].data(), 1);
        }
    }
}

void DOVelocityCorrectionScheme::ComputeNMode(int i,
                          Array<OneD, Array<OneD, NekDouble>> &N)
{
    const int nVel  = m_velocity.size();
    const int nPhys = m_fields[0]->GetTotPoints();
    const int nCoeffs   = m_fields[0]->GetNcoeffs();
    const NekDouble nu  = m_kinvis;
    const NekDouble mui = (i < m_nDOModes) ? m_Cij[i*m_nDOModes + i] : 0.0;
    const NekDouble eps = 1e-12;

    Array<OneD, Array<OneD, NekDouble>> cross(nVel), triple(nVel), lap(nVel), innerArg(nVel);

    for (int c = 0; c < nVel; ++c)  // for each component: array of 0s for each quadrature point
    {
        cross[c]    = Array<OneD, NekDouble>(nPhys, 0.0);
        triple[c]   = Array<OneD, NekDouble>(nPhys, 0.0);
        lap[c]      = Array<OneD, NekDouble>(nPhys, 0.0);
        innerArg[c] = Array<OneD, NekDouble>(nPhys, 0.0);
        Vmath::Zero(nPhys, N[c].data(), 1); // zero output
    }

    // 1) cross = -[(ū . grad) u_i + (u_i . grad) ū]
    ComputeModeCross(i, cross);

    // 2) triple-moment F_{ml}M_{mli}/mu_i (pseudoinverse: zero when mu_i below eps)
    //     triple[c](x) = \sum_{m,l} (M_{mli}/mu_i) F_{ml,c}(x)
    if (mui > eps)
    {
        // precompute duLcd = ∂_d u_{l,c}
        Array<OneD, NekDouble> duLcd(m_nDOModes * nVel * nVel * nPhys);
        Array<OneD, NekDouble> tmp(nPhys);
        for (int l = 0; l < m_nDOModes; ++l)
        {
            for (int c = 0; c < nVel; ++c)                      // component to differentiate
            {
                const NekDouble *u_lc = m_DOModePhys.data()     // pointer to mode l, component c
                                       + (l*nVel + c)*nPhys;
                Vmath::Vcopy(nPhys, u_lc, 1, tmp.data(), 1);    // tmp = u_{l,c}
                for (int d = 0; d < nVel; ++d)                  // derivative direction
                {
                    Array<OneD, NekDouble> duOut = duLcd        // pointer to duLcd
                                                  + ((l*nVel + c)*nVel + d)*nPhys;
                    m_fields[m_velocity[c]]->PhysDeriv(d, tmp,  // duOut = ∂_d u_{l,c}
                                                       duOut);
                }
            }
        }

        Array<OneD, NekDouble> prod(nPhys);
        for (int m = 0; m < m_nDOModes; ++m)                        // first triple-moment index
        {
            for (int l = 0; l < m_nDOModes; ++l)                    // second triple-moment index
            {
                const NekDouble Mml = m_Mkli[(m*m_nDOModes          // Mml = M_mli / mui
                                              + l)*m_nDOModes + i] / mui;
                if (std::abs(Mml) < eps) continue;

                for (int c = 0; c < nVel; ++c)                      // output spatial component
                {
                    for (int d = 0; d < nVel; ++d)                  // contraction inside (u_m . grad) u_l
                    {
                        const NekDouble *u_md = m_DOModePhys.data() // pointer to mode m, component d
                                               + (m*nVel + d)*nPhys;
                        const NekDouble *du   = duLcd.data()        // pointer to precomputed duLcd
                                               + ((l*nVel + c)*nVel + d)*nPhys;
                        Vmath::Vmul(nPhys, u_md, 1, du, 1,          // prod = u_md * du
                                    prod.data(), 1);
                        Vmath::Svtvp(nPhys, -Mml, prod.data(), 1,   // triple[c][k] -= Mml * prod[k] for all ks
                                    triple[c].data(), 1, triple[c].data(), 1);
                    }
                }
            }
        }
    }

    // 3) additive stochastic forcing contribution to mode i:
    //      addStochN_{i,c}(x) = (1/μ_i) Σ_k A_{i,k} g_{k,c}(x)
    //      Included in `rough` below so it flows into both N and innerArg, and
    //      so the existing DO projection (step 5) automatically applies the
    //      ⊥-to-modes orthogonalisation that the canonical DO mode equation
    //      requires for the forcing term.
    Array<OneD, Array<OneD, NekDouble>> addStochN(nVel);
    for (int c = 0; c < nVel; ++c) addStochN[c] = Array<OneD, NekDouble>(nPhys, 0.0);
    if (m_nForcingChannels > 0 && mui > eps)
    {
        const int K = m_nForcingChannels;
        for (int k = 0; k < K; ++k)                         // loop over channels
        {
            const NekDouble Aik = m_forcingA[i*K + k] / mui;
            if (std::abs(Aik) < eps) continue;
            for (int c = 0; c < nVel; ++c)                  // loop over components
            {
                const NekDouble *gk = m_forcingBasisPhys.data() + (k*nVel + c)*nPhys;
                Vmath::Svtvp(nPhys, Aik, gk, 1, addStochN[c].data(), 1, addStochN[c].data(), 1);
            }
        }
    }

    // 4) lap(u_i)
    ComputeModeLaplacian(i, lap);

    // 5) "rough" = cross + triple + addStochN ; innerArg = rough + nu * Lap u_i
    NekDouble maxTriple = 0.0, maxCross = 0.0, maxStoch = 0.0;
    for (int c = 0; c < nVel; ++c)      // loop over components
        for (int k = 0; k < nPhys; ++k) // loop over points
        {
            const NekDouble rough = cross[c][k] + triple[c][k]    // mean-mode + triple-moment
                                  + addStochN[c][k];               // + stochastic forcing
            N[c][k]         = rough;
            innerArg[c][k]  = rough + nu * lap[c][k];
            maxTriple = std::max(maxTriple, std::abs(triple[c][k]));
            maxCross  = std::max(maxCross , std::abs(cross [c][k]));
            maxStoch  = std::max(maxStoch , std::abs(addStochN[c][k]));
        }
    if (m_verbose && m_doStepCounter == 1)
        std::cout << "[DOVelocityCorrectionScheme][diag] mode i=" << i
                  << " max|cross|="    << maxCross
                  << " max|triple|="   << maxTriple
                  << " max|addStoch|=" << maxStoch
                  << " mu_i="          << mui << "\n";

    // 5.5) project N orthogonal to the constant subspace (uses innerArg as the
    //      projection argument so the coefficient matches the downstream DO
    //      projection; no-op for components with Dirichlet BCs)
    ProjectOutConstantsFromN(m_fields, m_velocity, innerArg, N);

    // 6) DO projection: N -= Σ_p β_p u_p   with β_p = <innerArg, u_p>_M
    NekDouble maxBeta = 0.0;
    Array<OneD, NekDouble> ip(nCoeffs);
    // Pre-AllReduce: assemble all S β_p into one buffer, single global reduce.
    std::vector<NekDouble> betas(m_nDOModes, 0.0);
    for (int p = 0; p < m_nDOModes; ++p)
    {
        for (int c = 0; c < nVel; ++c)
        {
            m_fields[m_velocity[c]]->IProductWRTBase(innerArg[c], ip);
            const NekDouble *u_pc_coeffs = m_DOModeCoeffs.data()
                                  + (p*nVel + c)*nCoeffs;
            betas[p] += Vmath::Dot(nCoeffs, ip.data(), 1, u_pc_coeffs, 1);
        }
    }
    m_fields[m_velocity[0]]->GetComm()->GetRowComm()->AllReduce(
        betas, LibUtilities::ReduceSum);
    for (int p = 0; p < m_nDOModes; ++p)    // loop over modes
    {
        const NekDouble beta_p = betas[p];
        maxBeta = std::max(maxBeta, std::abs(beta_p));
        for (int c = 0; c < nVel; ++c)                                   // subtract β_p * u_p from N
        {
            const NekDouble *u_pc = m_DOModePhys.data()                  // pointer to mode p, component c (phys)
                                   + (p*nVel + c)*nPhys;
            Vmath::Svtvp(nPhys, -beta_p, u_pc, 1, N[c].data(), 1,        // N[c][k] -= β_p * u_pc[k] for all ks
                         N[c].data(), 1);
        }
    }
    if (m_verbose && m_doStepCounter == 1)
        std::cout << "[DOVelocityCorrectionScheme][diag] mode i=" << i
                  << " DO-projection |beta_p|max=" << maxBeta << "\n";
}

/**
 * VCS override. Adds the `doCorr` correction to the mean velocity's explicit RHS (`outarray`)
 * on top of the VCS advection. Also captures the t^n mean field into
 * m_meanAtTn: the integrator passes `inarray` = mean phys at t^n, which is
 * EXACTLY what the DO subsystem needs later (in DOOdeRhs) to evaluate the
 * mode RHS cross/triple/mean-coupling terms at the same t^n as modes/Y.
 */
void DOVelocityCorrectionScheme::v_EvaluateAdvection_SetPressureBCs(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time)
{
    VelocityCorrectionScheme::v_EvaluateAdvection_SetPressureBCs(   // base VCS: outarray = -(ū . grad)ū + pressure BCs
        inarray, outarray, time);

    if (m_nDOModes == 0) return;

    const int nVel  = m_velocity.size();
    const int nPhys = m_fields[0]->GetTotPoints();

    // Snapshot mean^n. `inarray` carries the velocity components at the time
    // level the integrator is currently evaluating (= t^n for IMEX BDF2/EXT2).
    if (m_meanAtTn.size() == 0)                                         // lazy-allocate
    {
        m_meanAtTn = Array<OneD, Array<OneD, NekDouble>>(nVel);
        for (int c = 0; c < nVel; ++c)
            m_meanAtTn[c] = Array<OneD, NekDouble>(nPhys, 0.0);
    }
    for (int c = 0; c < nVel; ++c)                                      // copy mean^n
        Vmath::Vcopy(nPhys, inarray[c].data(), 1, m_meanAtTn[c].data(), 1);
    m_meanSnapshotValid = true;

    ComputeYMoments();

    Array<OneD, Array<OneD, NekDouble>> doCorr(nVel);                   // initialise correction
    for (int c = 0; c < nVel; ++c)
        doCorr[c] = Array<OneD, NekDouble>(nPhys, 0.0);                 // initialise each component's arrays
    ComputeDOMeanCoupling(doCorr);                                      // call `ComputeDOMeanCoupling` for the full vector

    for (int c = 0; c < nVel; ++c)
        Vmath::Vadd(nPhys, doCorr[c].data(), 1, outarray[c].data(), 1,  // outarray[c] += doCorr[c]
                    outarray[c].data(), 1);
}

/**
 * Solves the pressure Poisson equation for one DO mode (spec eq. 92-95):
 *
 *     Δ p_i^{n+1} = ∇·uhat / Δt        (HelmSolve with lambda = 0)
 *
 * with homogeneous Neumann BCs on the pressure mode. `Δt` here is the
 * full timestep `m_timestep` — DOImplicitSolve always passes this exact
 * value as `Dt`; the BDF order is encoded only in the `uhat` predictor
 * (see DOImplicitSolve's "scale" logic), not in `Dt`.
 *
 * The mean-field pressure (m_pressure) is saved before the solve and
 * restored after — otherwise this call would mutate the VCS solver
 * state mid-step.
 */
void DOVelocityCorrectionScheme::ModePressureSolve(
    const Array<OneD, Array<OneD, NekDouble>> &uhatPhys,
    NekDouble Dt, Array<OneD, NekDouble> &pCoeffsOut)
{
    const int nVel  = m_velocity.size();
    const int nPhys = m_fields[0]->GetTotPoints();
    const int npC   = m_pressure->GetNcoeffs();
    const int npP   = m_pressure->GetTotPoints();

    // saved pressure coeffs and physical values to restore later
    Array<OneD, NekDouble> savedPC(npC), savedPP(npP);  
    Vmath::Vcopy(npC, m_pressure->GetCoeffs().data(), 1, savedPC.data(), 1);
    Vmath::Vcopy(npP, m_pressure->GetPhys().data(),   1, savedPP.data(), 1);

    // save and zero pressure mode BC coeffs
    auto pbnd = m_pressure->GetBndCondExpansions();
    std::vector<std::vector<NekDouble>> savedPBnd(pbnd.size());
    for (int n = 0; n < (int)pbnd.size(); ++n)
    {
        const int nc = pbnd[n]->GetNcoeffs();
        savedPBnd[n].assign(pbnd[n]->GetCoeffs().data(),                // savedPBnd[n] = pbnd[n] (coeffs)
                            pbnd[n]->GetCoeffs().data() + nc);
        Vmath::Zero(nc, pbnd[n]->UpdateCoeffs().data(), 1);             // zero pbnd[n] (coeffs)
    }

    // Poisson RHS: divUhat = ∇·uhat / Δt
    Array<OneD, NekDouble> divUhat(nPhys, 0.0), tmp(nPhys, 0.0);
    m_fields[m_velocity[0]]->PhysDeriv(0, uhatPhys[0], divUhat);        // divUhat =  ∂_0 uhat[0] (derivative wrt first spatial direction (x))
    for (int c = 1; c < nVel; ++c)                                      // accumulate  ∂_c uhat[c]
    {
        m_fields[m_velocity[c]]->PhysDeriv(c, uhatPhys[c], tmp);
        Vmath::Vadd(nPhys, tmp.data(), 1, divUhat.data(), 1, divUhat.data(), 1);
    }
    Vmath::Smul(nPhys, 1.0/Dt, divUhat.data(), 1, divUhat.data(), 1);   // divUhat /= Δt

    // solve Δp = divUhat with homogeneous Neumann
    StdRegions::ConstFactorMap factors;
    factors[StdRegions::eFactorLambda] = 0.0;
    m_pressure->HelmSolve(divUhat, m_pressure->UpdateCoeffs(), factors);

    Vmath::Vcopy(npC, m_pressure->GetCoeffs().data(), 1,                // pCoeffsOut = m_pressure coeffs
                 pCoeffsOut.data(), 1);

    // restore pressure BCs and m_pressure state
    for (int n = 0; n < (int)pbnd.size(); ++n)
        std::copy(savedPBnd[n].begin(), savedPBnd[n].end(),
                  pbnd[n]->UpdateCoeffs().data());
    Vmath::Vcopy(npC, savedPC.data(), 1, m_pressure->UpdateCoeffs().data(), 1);
    Vmath::Vcopy(npP, savedPP.data(), 1, m_pressure->UpdatePhys().data(),   1);
}

/**
 * Solves the viscous Helmholtz step for one DO mode, component-wise
 * (spec eq. 97-99):
 *
 *     (Δ - 3/(2νΔt)) u_k^{n+1} = -uhat_k/(νΔt) + ∂_k p^{n+1} / ν
 *
 * The implicit weight aii_Dt = (2/3)·Dt is hard-coded for BDF2; on the
 * BDF1 startup step (DOImplicitSolve passes the same `Dt = m_timestep`)
 * this introduces a small order-1 startup error on the implicit term —
 * the standard VCS BDF1→BDF2 transition behaviour.
 *
 * Both m_pressure and m_fields coefficient arrays are saved on entry
 * and restored on exit. Results written to `uNewPhys` and `uNewCoeffs`.
 */
void DOVelocityCorrectionScheme::ModeViscousSolve(
    const Array<OneD, Array<OneD, NekDouble>> &uhatPhys,
    const Array<OneD, NekDouble> &pCoeffsIn,
    NekDouble Dt,
    Array<OneD, Array<OneD, NekDouble>> &uNewPhys,
    Array<OneD, Array<OneD, NekDouble>> &uNewCoeffs)
{
    const int nVel   = m_velocity.size();
    const int nPhys  = m_fields[0]->GetTotPoints();
    const int npC    = m_pressure->GetNcoeffs();
    const int npP    = m_pressure->GetTotPoints();
    const NekDouble aii_Dt = (2.0/3.0)*Dt;

    // save `m_pressure` state
    Array<OneD, NekDouble> savedPC(npC), savedPP(npP);
    Vmath::Vcopy(npC, m_pressure->GetCoeffs().data(), 1, savedPC.data(), 1);
    Vmath::Vcopy(npP, m_pressure->GetPhys().data(),   1, savedPP.data(), 1);

    // save `m_fields` velocity coeffs (otherwise HelmSolve overwrites them)
    std::vector<Array<OneD, NekDouble>> savedVC(nVel);
    for (int c = 0; c < nVel; ++c)
    {
        const int nc = m_fields[m_velocity[c]]->GetNcoeffs();
        savedVC[c]   = Array<OneD, NekDouble>(nc);
        Vmath::Vcopy(nc, m_fields[m_velocity[c]]->GetCoeffs().data(), 1,
                     savedVC[c].data(), 1);
    }

    // load `pCoeffsIn` into `m_pressure` to use PhysDeriv on it
    Vmath::Vcopy(npC, pCoeffsIn.data(), 1,
                 m_pressure->UpdateCoeffs().data(), 1);
    m_pressure->BwdTrans(m_pressure->GetCoeffs(), m_pressure->UpdatePhys());

    // Forcing[k] := ∂_k p  for k < nVel ; zero for any extra convective fields
    Array<OneD, Array<OneD, NekDouble>> Forcing(m_nConvectiveFields);
    for (int k = 0; k < m_nConvectiveFields; ++k)       // allocate space
        Forcing[k] = Array<OneD, NekDouble>(nPhys, 0.0);

    if (nVel == 2)  // 2d fill derivatives
        m_pressure->PhysDeriv(m_pressure->GetPhys(), Forcing[0], Forcing[1]);
    else            // 3d fill derivatives
        m_pressure->PhysDeriv(m_pressure->GetPhys(),
                              Forcing[0], Forcing[1], Forcing[2]);

    // Forcing[k] = (∂_k p - uhat_k / Dt) / ν                          
    for (int k = 0; k < m_nConvectiveFields; ++k)
    {
        if (k < nVel)
            Vmath::Svtvp(nPhys, -1.0/Dt, uhatPhys[k].data(), 1,         // Forcing[k] -=u hatPhys[k]/Dt for all ks
                         Forcing[k].data(), 1, Forcing[k].data(), 1);
        Vmath::Smul(nPhys, 1.0/m_diffCoeff[k],                          // Forcing[k] /= ν
                    Forcing[k].data(), 1, Forcing[k].data(), 1);
    }

    // solve (Δ - λ) u_k = Forcing[k]  with λ = 1/(aii_Dt · ν) = 3/(2νΔt)
    StdRegions::ConstFactorMap factors;
    for (int k = 0; k < m_nConvectiveFields; ++k)
    {
        factors[StdRegions::eFactorLambda] = 1.0/aii_Dt/m_diffCoeff[k];
        m_fields[m_velocity[k]]->HelmSolve(Forcing[k],              // coeffs stored in m_fields[k]
            m_fields[m_velocity[k]]->UpdateCoeffs(), factors);
        m_fields[m_velocity[k]]->BwdTrans(                          // uNewPhys
            m_fields[m_velocity[k]]->GetCoeffs(), uNewPhys[k]);
        Vmath::Vcopy(m_fields[m_velocity[k]]->GetNcoeffs(),
                     m_fields[m_velocity[k]]->GetCoeffs().data(), 1,
                     uNewCoeffs[k].data(), 1);
    }

    // restore `m_fields` velocity coeffs & `m_pressure` state
    for (int c = 0; c < nVel; ++c)
        Vmath::Vcopy(m_fields[m_velocity[c]]->GetNcoeffs(),
                     savedVC[c].data(), 1,
                     m_fields[m_velocity[c]]->UpdateCoeffs().data(), 1);
    Vmath::Vcopy(npC, savedPC.data(), 1, m_pressure->UpdateCoeffs().data(), 1);
    Vmath::Vcopy(npP, savedPP.data(), 1, m_pressure->UpdatePhys().data(),   1);
}

/**
 * Explicit-RHS callback registered with m_doScheme. Replaces the explicit
 * portions of the deleted AdvanceModes() and AdvanceYi(). The integrator
 * passes:
 *      `in[0..S*nVel-1]` = mode phys at time t (one variable per (i,c));
 *      `in[m_doYIdx]`    = Y_flat at time t (size Np*S).
 * It expects in `out` the *un-multiplied* explicit RHS (the BDF2/EXT2 weight
 * Δt is applied internally by the integrator).
 *
 * For consistency with mean^t, we swap m_fields phys to m_meanAtTn (captured
 * by the EXT operator at the same time level the integrator is asking for)
 * for the duration of the callback. m_DOModePhys/m_DOModeCoeffs/m_Yi are
 * synced from `in` so the existing helpers (ComputeYMoments, ComputeNMode,
 * ComputeModeCross/Laplacian) operate on the correct state.
 */
void DOVelocityCorrectionScheme::DOOdeRhs(
    const Array<OneD, const Array<OneD, NekDouble>> &in,
    Array<OneD, Array<OneD, NekDouble>>             &out,
    const NekDouble                                  time)
{
    boost::ignore_unused(time);                                                 // explicit terms here are autonomous in t
    if (m_nDOModes == 0) return;

    const int S       = m_nDOModes;
    const int nVel    = m_velocity.size();
    const int nPhys   = m_fields[0]->GetTotPoints();
    const int nCoeffs = m_fields[0]->GetNcoeffs();
    const int nPC     = m_pressure->GetNcoeffs();
    const int nPP     = m_pressure->GetTotPoints();

    // 1) sync m_DOModePhys / m_DOModeCoeffs / m_Yi from `in`
    Array<OneD, NekDouble> tmpPhys(nPhys), tmpCoef(nCoeffs);
    for (int i = 0; i < S; ++i)                                                 // loop over modes
        for (int c = 0; c < nVel; ++c)                                          // loop over components
        {
            const int v    = i*nVel + c;
            const int pOff = v*nPhys;
            const int cOff = v*nCoeffs;
            Vmath::Vcopy(nPhys, in[v].data(), 1,
                         m_DOModePhys.data() + pOff, 1);
            Vmath::Vcopy(nPhys, in[v].data(), 1, tmpPhys.data(), 1);            // FwdTrans wants its own buffer
            m_fields[m_velocity[c]]->FwdTrans(tmpPhys, tmpCoef);                // refresh coeffs
            Vmath::Vcopy(nCoeffs, tmpCoef.data(), 1,
                         m_DOModeCoeffs.data() + cOff, 1);
        }
    Vmath::Vcopy(m_nDOParticles*S, in[m_doYIdx].data(), 1, m_Yi.data(), 1);

    // 2) swap m_fields phys to mean^n (so ComputeModeCross/DOMeanCoupling read
    //    the right time level), saving current mean^{n+1} aside for restore
    Array<OneD, Array<OneD, NekDouble>> meanSaved(nVel);
    if (m_meanSnapshotValid)
    {
        for (int c = 0; c < nVel; ++c)
        {
            meanSaved[c] = Array<OneD, NekDouble>(nPhys);
            Vmath::Vcopy(nPhys, m_fields[m_velocity[c]]->GetPhys().data(), 1,
                         meanSaved[c].data(), 1);
            Vmath::Vcopy(nPhys, m_meanAtTn[c].data(), 1,
                         m_fields[m_velocity[c]]->UpdatePhys().data(), 1);
        }
    }

    ComputeYMoments();                                                          // refresh m_Cij, m_Mkli for current Y

    // 3) explicit RHS for each mode: out[i*nVel+c] = N_i,c (cross + triple +
    //    addStochN; the constant projection + ⊥ to span{u_p} are inside ComputeNMode).
    Array<OneD, Array<OneD, NekDouble>> N(nVel);
    for (int c = 0; c < nVel; ++c) N[c] = Array<OneD, NekDouble>(nPhys, 0.0);
    for (int i = 0; i < S; ++i)
    {
        ComputeNMode(i, N);                                                     // existing helper, untouched
        for (int c = 0; c < nVel; ++c)
        {
            const int v = i*nVel + c;
            Vmath::Vcopy(nPhys, N[c].data(), 1, out[v].data(), 1);
        }
    }

    // 4) explicit RHS for Y: out[m_doYIdx][p*S+i] = R^t[p,i]
    //    R^t[p,i] = Σ_k Y_p,k <F_k − ∇p_k, u_i> + Σ_{k,l}(Y_p,k Y_p,l − C_kl)<F_kl, u_i> + Σ_k η_p,k G[i,k].
    //    First compute the inner-product tensors ipKi (linear) and ipKli (quadratic).
    std::vector<NekDouble> ipKi (S*S,   0.0);
    std::vector<NekDouble> ipKli(S*S*S, 0.0);
    Array<OneD, NekDouble> ip(nCoeffs);
    Array<OneD, Array<OneD, NekDouble>> Fk(nVel), Lk(nVel);
    Array<OneD, NekDouble> pkPhys(nPP), dpk(nPhys);
    for (int c = 0; c < nVel; ++c)
    {
        Fk[c] = Array<OneD, NekDouble>(nPhys, 0.0);
        Lk[c] = Array<OneD, NekDouble>(nPhys, 0.0);
    }

    // 4a) ipKi[k,i] = <F_k − ∇p_k, u_i>_M
    for (int k = 0; k < S; ++k)                                                 // loop over modes k
    {
        ComputeModeCross(k, Fk);                                                // cross_k = -[(ū·∇)u_k + (u_k·∇)ū]
        ComputeModeLaplacian(k, Lk);                                            // lap_k = Δu_k
        for (int c = 0; c < nVel; ++c)
            Vmath::Svtvp(nPhys, m_kinvis, Lk[c].data(), 1,                      // Fk[c] += ν · lap[c]
                         Fk[c].data(), 1, Fk[c].data(), 1);

        // Fk[c] -= ∂_c p_k from cached mode pressure m_DOModePCoeffs[k] (this
        // is pressure^t — set by the LAST DOImplicitSolve that wrote mode k)
        Array<OneD, NekDouble> pkCoeffs = m_DOModePCoeffs + k*nPC;              // non-owning view of p_k coeffs
        m_pressure->BwdTrans(pkCoeffs, pkPhys);                                 // pkPhys = p_k(x)
        for (int c = 0; c < nVel; ++c)
        {
            m_pressure->PhysDeriv(c, pkPhys, dpk);                              // dpk = ∂_c p_k
            Vmath::Vsub(nPhys, Fk[c].data(), 1, dpk.data(), 1,
                        Fk[c].data(), 1);                                       // Fk[c] -= dpk
        }

        for (int i = 0; i < S; ++i)                                             // <F_k − ∇p_k, u_i>
        {
            NekDouble s = 0.0;
            for (int c = 0; c < nVel; ++c)
            {
                m_fields[m_velocity[c]]->IProductWRTBase(Fk[c], ip);            // ip = M Fk[c] coeffs
                const NekDouble *u_ic = m_DOModeCoeffs.data() + (i*nVel + c)*nCoeffs;
                s += Vmath::Dot(nCoeffs, ip.data(), 1, u_ic, 1);
            }
            ipKi[k*S + i] = s;
        }
    }

    // 4b) ipKli[k,l,i] = <F_kl, u_i>_M with F_kl,c = -(u_k·∇)u_{l,c}
    Array<OneD, NekDouble> duLcd(S * nVel * nVel * nPhys);                      // cached ∂_d u_{l,c}
    Array<OneD, NekDouble> tmp(nPhys), prod(nPhys);
    Array<OneD, Array<OneD, NekDouble>> Fkl(nVel);
    for (int c = 0; c < nVel; ++c) Fkl[c] = Array<OneD, NekDouble>(nPhys);
    for (int l = 0; l < S; ++l)                                                 // precompute derivatives
        for (int c = 0; c < nVel; ++c)
        {
            const NekDouble *u_lc = m_DOModePhys.data() + (l*nVel + c)*nPhys;
            Vmath::Vcopy(nPhys, u_lc, 1, tmp.data(), 1);
            for (int d = 0; d < nVel; ++d)
            {
                Array<OneD, NekDouble> duOut = duLcd
                                               + ((l*nVel + c)*nVel + d)*nPhys;
                m_fields[m_velocity[c]]->PhysDeriv(d, tmp, duOut);
            }
        }
    for (int k = 0; k < S; ++k)                                                 // assemble F_kl + project
        for (int l = 0; l < S; ++l)
        {
            for (int c = 0; c < nVel; ++c) Vmath::Zero(nPhys, Fkl[c].data(), 1);
            for (int c = 0; c < nVel; ++c)                                      // output spatial component
                for (int d = 0; d < nVel; ++d)                                  // contraction direction
                {
                    const NekDouble *u_kd = m_DOModePhys.data() + (k*nVel + d)*nPhys;
                    const NekDouble *du   = duLcd.data() + ((l*nVel + c)*nVel + d)*nPhys;
                    Vmath::Vmul(nPhys, u_kd, 1, du, 1, prod.data(), 1);
                    Vmath::Svtvp(nPhys, -1.0, prod.data(), 1,
                                 Fkl[c].data(), 1, Fkl[c].data(), 1);           // Fkl[c] -= u_kd · ∂_d u_lc
                }
            for (int i = 0; i < S; ++i)
            {
                NekDouble s = 0.0;
                for (int c = 0; c < nVel; ++c)
                {
                    m_fields[m_velocity[c]]->IProductWRTBase(Fkl[c], ip);
                    const NekDouble *u_ic = m_DOModeCoeffs.data() + (i*nVel + c)*nCoeffs;
                    s += Vmath::Dot(nCoeffs, ip.data(), 1, u_ic, 1);
                }
                ipKli[(k*S + l)*S + i] = s;
            }
        }

    // MPI: ipKi and ipKli are partial sums; AllReduce so every rank's per-
    // particle Y-RHS uses globally-consistent inner products.
    auto comm = m_fields[m_velocity[0]]->GetComm()->GetRowComm();
    if (!ipKi.empty())  comm->AllReduce(ipKi,  LibUtilities::ReduceSum);
    if (!ipKli.empty()) comm->AllReduce(ipKli, LibUtilities::ReduceSum);

    // 4c) per-particle Y RHS into out[m_doYIdx]
    const int Kf = m_nForcingChannels;
    NekDouble *Rout = out[m_doYIdx].data();
    NekDouble maxLin = 0.0, maxTri = 0.0, maxFor = 0.0;
    for (int p = 0; p < m_nDOParticles; ++p)                                    // loop over particles
    {
        const NekDouble *Yp = m_Yi.data()         + p*S;
        const NekDouble *Ep = (Kf > 0) ? (m_forcingEta.data() + p*Kf) : nullptr;
        NekDouble       *Rp = Rout + p*S;
        for (int i = 0; i < S; ++i)
        {
            NekDouble lin = 0.0, tri = 0.0, frc = 0.0;
            for (int k = 0; k < S; ++k)
                lin += Yp[k] * ipKi[k*S + i];                                   // Σ_k Y_p,k <F_k − ∇p_k, u_i>
            for (int k = 0; k < S; ++k)
                for (int l = 0; l < S; ++l)
                    tri += (Yp[k]*Yp[l] - m_Cij[k*S + l])
                         * ipKli[(k*S + l)*S + i];                              // demeaned-quadratic
            for (int k = 0; k < Kf; ++k)
                frc += Ep[k] * m_forcingG[i*Kf + k];                            // <f̃_p, u_i>
            Rp[i]  = lin + tri + frc;
            if (m_verbose && m_doStepCounter == 1)
            {
                maxLin = std::max(maxLin, std::abs(lin));
                maxTri = std::max(maxTri, std::abs(tri));
                maxFor = std::max(maxFor, std::abs(frc));
            }
        }
    }
    if (m_verbose && m_doStepCounter == 1)
        std::cout << "[DOVelocityCorrectionScheme][diag] Yi RHS:"
                  << " max|Y.(F-grad p)| = " << maxLin
                  << " max|triple|="          << maxTri
                  << " max|forcing|="         << maxFor << "\n";

    if (m_verbose && m_doStepCounter == 1)
        std::cout << "[DOVelocityCorrectionScheme][int] step=" << m_doStepCounter
                  << " callIdx=" << m_doOdeRhsCallIdx
                  << " (DOOdeRhs)\n";
    ++m_doOdeRhsCallIdx;

    // 5) restore m_fields phys (mean^{n+1} for the rest of the step)
    if (m_meanSnapshotValid)
    {
        for (int c = 0; c < nVel; ++c)
            Vmath::Vcopy(nPhys, meanSaved[c].data(), 1,
                         m_fields[m_velocity[c]]->UpdatePhys().data(), 1);
    }
}

/**
 * Implicit-solve callback registered with m_doScheme. The integrator passes:
 *      `in[0..S*nVel-1]` = u^* for each (mode, component) — i.e. the BDF2/EXT2
 *                          predictor (4/3)u^n − (1/3)u^{n-1} + (2/3)Δt(2N^n − N^{n-1});
 *      `in[m_doYIdx]`    = Y^* (no implicit term acts on Y).
 * `lambda` carries the implicit-step weight a_iixDt = (2/3)Δt for IMEX/BDF2.
 *
 * For each mode: re-run the existing pressure-Poisson + viscous-Helmholtz
 * pipeline (ModePressureSolve + ModeViscousSolve hard-code aii_Dt = (2/3)·Dt
 * internally, so we pass Dt = (3/2)·lambda). For Y: identity.
 *
 * Side effect: m_DOModePCoeffs[i] is overwritten with the new mode pressure
 * p_i^{n+1} so the next DOOdeRhs call sees consistent ∇p_k.
 */
void DOVelocityCorrectionScheme::DOImplicitSolve(
    const Array<OneD, const Array<OneD, NekDouble>> &in,
    Array<OneD, Array<OneD, NekDouble>>             &out,
    const NekDouble                                  time,
    const NekDouble                                  lambda)
{
    boost::ignore_unused(time);
    if (m_nDOModes == 0) return;

    const int S        = m_nDOModes;
    const int nVel     = m_velocity.size();
    const int nPhys    = m_fields[0]->GetTotPoints();
    const int nCoeffs  = m_fields[0]->GetNcoeffs();
    const int nPC      = m_pressure->GetNcoeffs();
    const int nPP      = m_pressure->GetTotPoints();
    // Scale from m_tmp to uhat_manual is dt/lambda (=1 for BDF1, =1.5 for BDF2/EXT2).
    // Helpers expect Dt = full timestep; their internal aii_Dt = (2/3)·Dt assumes BDF2,
    // which is correct only when the scheme is in BDF2 phase (lambda = (2/3)·dt). The
    // BDF1 phase (step 0) uses lambda = dt; in that case we still pass Dt = dt and
    // tolerate the (2/3)·dt aii_Dt mismatch (BDF1 startup error is O(dt²) on the
    // implicit term — bounded once BDF2 takes over).
    const NekDouble Dt    = m_timestep;
    const NekDouble scale = m_timestep / lambda;                                // 1.0 (BDF1) or 1.5 (BDF2)
    if (m_verbose && (m_doStepCounter == 1 || m_doStepCounter == 2))
    {
        // gamma0 = inverse of "aii" coefficient on u^{n+1} — for BDF1: 1, BDF2: 3/2.
        // The integrator's lambda equals dt/gamma0, so gamma0 = m_timestep/lambda.
        const NekDouble gamma0 = m_timestep / lambda;
        const NekDouble aii    = (2.0/3.0) * Dt;        // hard-coded BDF2 weight in helpers
        std::cout << "[DOVelocityCorrectionScheme][int] step=" << m_doStepCounter
                  << " gamma0=" << gamma0
                  << " aii(helper)=" << aii
                  << " lambda=" << lambda
                  << " dt=" << m_timestep
                  << " scale=" << scale << "\n";
    }

    // save mean-field state (velocity & pressure) — ModePressureSolve and
    // ModeViscousSolve transiently modify m_fields/m_pressure
    std::vector<Array<OneD, NekDouble>> svc(nVel), svp(nVel);
    for (int c = 0; c < nVel; ++c)
    {
        const int nc = m_fields[m_velocity[c]]->GetNcoeffs();
        const int np = m_fields[m_velocity[c]]->GetTotPoints();
        svc[c] = Array<OneD, NekDouble>(nc);
        svp[c] = Array<OneD, NekDouble>(np);
        Vmath::Vcopy(nc, m_fields[m_velocity[c]]->GetCoeffs().data(), 1, svc[c].data(), 1);
        Vmath::Vcopy(np, m_fields[m_velocity[c]]->GetPhys().data(),   1, svp[c].data(), 1);
    }
    Array<OneD, NekDouble> sPC(nPC), sPP(nPP);
    Vmath::Vcopy(nPC, m_pressure->GetCoeffs().data(), 1, sPC.data(), 1);
    Vmath::Vcopy(nPP, m_pressure->GetPhys().data(),   1, sPP.data(), 1);

    // homogenise velocity BCs for the per-mode solves; restore at the end
    auto bcState = CaptureVelocityBCState(m_fields, m_velocity);
    HomogenizeVelocityBCsForModes(m_fields, m_velocity);

    // scratch arrays reused across modes
    Array<OneD, Array<OneD, NekDouble>> uhat(nVel), uNewPhys(nVel), uNewCoeffs(nVel);
    Array<OneD, NekDouble> pMode(nPC, 0.0);
    for (int c = 0; c < nVel; ++c)
    {
        uhat[c]       = Array<OneD, NekDouble>(nPhys, 0.0);
        uNewPhys[c]   = Array<OneD, NekDouble>(nPhys, 0.0);
        uNewCoeffs[c] = Array<OneD, NekDouble>(nCoeffs, 0.0);
    }

    // Per-mode pressure Poisson + viscous Helmholtz.
    //
    // Predictor scaling: the GLM integrator passes its `m_tmp` linear
    // combination as `in`. For IMEXOrder1 (BDF1, step 0):
    //     m_tmp = u^n + dt·R^n,                           lambda = dt
    // For IMEXOrder2 (BDF2/EXT2, step 1+):
    //     m_tmp = (4/3)u^n - (1/3)u^{n-1}
    //             + (4/3)dt·R^n - (2/3)dt·R^{n-1},        lambda = (2/3)dt
    // The helpers ModePressureSolve / ModeViscousSolve below were written for
    // the manual form
    //     uhat_manual = γ_0 u^n − γ_1 u^{n-1}
    //                 + dt(γ_N0 R^n − γ_N1 R^{n-1}),
    // with γ_0 set by the BDF order (1 for BDF1, 2 for BDF2). The relation is
    //     m_tmp · (dt/lambda) = uhat_manual,
    // so `scale = dt/lambda` recovers the helpers' input contract for BOTH
    // phases (1 for BDF1, 1.5 for BDF2). The helpers themselves take Dt =
    // m_timestep — they hard-code aii_Dt = (2/3)·Dt internally, which
    // is the BDF2 implicit weight. For BDF1 startup that is technically
    // off by 3/2 on the implicit term; the resulting startup error is the
    // standard VCS BDF1→BDF2 transition artefact (also present in the
    // original AB2 code, which uses the same helpers).
    for (int i = 0; i < S; ++i)
    {
        for (int c = 0; c < nVel; ++c)
            Vmath::Smul(nPhys, scale, in[i*nVel + c].data(), 1,
                        uhat[c].data(), 1);                                     // uhat = scale · m_tmp

        ModePressureSolve(uhat, Dt, pMode);                                     // p_i^{n+1}: Δp = ∇·uhat / Dt
        Vmath::Vcopy(nPC, pMode.data(), 1,                                      // cache for next DOOdeRhs's
                     m_DOModePCoeffs.data() + i*nPC, 1);                        //   Yi-RHS pressure-gradient term

        ModeViscousSolve(uhat, pMode, Dt, uNewPhys, uNewCoeffs);                // u_i^{n+1}: (Δ - 3/(2νΔt))u = ...

        for (int c = 0; c < nVel; ++c)                                          // pack into out
            Vmath::Vcopy(nPhys, uNewPhys[c].data(), 1,
                         out[i*nVel + c].data(), 1);
    }

    // Y: identity implicit operator. The Y system has no implicit term in
    // the spec (eq. 28-33: dY_i/dt = ⟨..., u_i⟩, all explicit). With this
    // identity, the GLM IMEXOrder2 phase reduces — for the Y variable — to
    // the explicit half EXT2:
    //     Y^{n+1} = (4/3) Y^n - (1/3) Y^{n-1} + (4/3) dt R^n - (2/3) dt R^{n-1}
    // (and the BDF1 startup phase reduces to forward Euler:
    //     Y^{n+1} = Y^n + dt R^n).
    // This is *not* the original code's Adams-Bashforth-2 for Y; it is a
    // different 2nd-order explicit scheme implied by the IMEX/Order=2 the
    // session asks for. Both give comparable Y std at engineering accuracy.
    Vmath::Vcopy(m_nDOParticles*S, in[m_doYIdx].data(), 1,
                 out[m_doYIdx].data(), 1);

    // restore BCs and mean-field state
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
 * Eigendecomposes C = V Λ V^T (symmetric Jacobi, ascending-sorted then
 * resorted descending) and applies the orthogonal V to:
 *   - per-mode arrays:  m_DOModePhys, m_DOModeCoeffs, m_DOModePCoeffs
 *                       (transformation: u_new[i] = Σ_j V[j,i] u_old[j])
 *   - per-particle Y:   m_Yi (transformation: Y_new = V^T Y_old)
 *   - integrator GLM solVec[step][var] for every mode and Y variable, every
 *     multistep entry (values AND explicit derivatives). Without this, the
 *     next BDF2 predictor would mix u^{n+1} (rotated) with u^{n-1} and
 *     R^{n-1} (still in the OLD basis), corrupting the predictor.
 *
 * Why C must be diagonal: ComputeNMode uses μ_i = C_{ii} as a pseudoinverse
 * for the mode-equation 1/μ_i factor (spec eq. 38-42, 50). That formula is
 * only valid when C is diagonal; it has to be re-diagonalised every step
 * because the IMEX2 step generally drives Y-correlations off-diagonal.
 *
 * Eigenvalues are sorted descending so mode index 0 carries the largest
 * variance — preserves a stable, physically-meaningful mode ordering.
 */
void DOVelocityCorrectionScheme::RotateToEigenbasisOfC()
{
    const int S = m_nDOModes;
    if (S <= 1) return;

    // refresh m_Cij from CURRENT m_Yi so V diagonalises the matrix that is
    // about to be rotated. Otherwise we would build V from the pre-step
    // covariance and it would not zero the off-diagonals of the post-step C.
    ComputeYMoments();

    if (m_verbose && m_doStepCounter == 1)
    {
        NekDouble offMax = 0.0, diagMax = 0.0;
        for (int ii = 0; ii < S; ++ii)
        {
            diagMax = std::max(diagMax, std::abs(m_Cij[ii*S + ii]));
            for (int jj = 0; jj < S; ++jj)
                if (ii != jj)
                    offMax = std::max(offMax, std::abs(m_Cij[ii*S + jj]));
        }
        std::cout << "[DOVelocityCorrectionScheme][rot] before: |off|max/|diag|max = "
                  << ((diagMax > 0) ? offMax/diagMax : 0.0)
                  << " (offMax=" << offMax << " diagMax=" << diagMax << ")\n";
    }

    // 1) Symmetric Jacobi: A := C, V := I, sweep until off^2 < tol
    const NekDouble tol = 1e-14;
    std::vector<NekDouble> A = m_Cij;               // SxS symmetric copy
    std::vector<NekDouble> V(S*S, 0.0);
    for (int i = 0; i < S; ++i) V[i*S + i] = 1.0;   // V = I
    const int maxSweeps = 80;
    for (int sweep = 0; sweep < maxSweeps; ++sweep) // loop over Jacobi sweeps
    {
        NekDouble off2 = 0.0;
        for (int i = 0; i < S; ++i)
            for (int j = i+1; j < S; ++j)
                off2 += A[i*S + j]*A[i*S + j];      // off2 = \sum_{i<j} A_{ij}^2
        if (off2 < tol*tol) break;                  // converged
        for (int i = 0; i < S; ++i)                 // pair index i
            for (int j = i+1; j < S; ++j)           // pair index j (> i)
            {
                const NekDouble aij = A[i*S + j];
                if (std::abs(aij) < 1e-300) continue;                   // already 0
                const NekDouble aii = A[i*S + i], ajj = A[j*S + j];
                NekDouble theta = 0.5 * std::atan2(2.0*aij, aii - ajj); // angle zeroing A_ij
                const NekDouble c = std::cos(theta), s = std::sin(theta);

                // off-diagonal rows/cols other than (i,j)
                for (int k = 0; k < S; ++k)
                {
                    if (k == i || k == j) continue;
                    const NekDouble Aki = A[k*S + i], Akj = A[k*S + j];
                    A[k*S + i] = c*Aki + s*Akj;
                    A[k*S + j] = -s*Aki + c*Akj;

                    // keep symmetric
                    A[i*S + k] = A[k*S + i];
                    A[j*S + k] = A[k*S + j];
                }
                // (i,j) 2x2 block
                A[i*S + i] = c*c*aii + 2.0*s*c*aij + s*s*ajj;   // A_ii rotated
                A[j*S + j] = s*s*aii - 2.0*s*c*aij + c*c*ajj;   // A_jj rotated
                A[i*S + j] = 0.0; A[j*S + i] = 0.0;             // by choice of 0

                // accumulate V := V * G(i,j,0)
                for (int k = 0; k < S; ++k)
                {
                    const NekDouble Vki = V[k*S + i], Vkj = V[k*S + j];
                    V[k*S + i] = c*Vki + s*Vkj;
                    V[k*S + j] = -s*Vki + c*Vkj;
                }
            }
    }

    // 2) sort eigenvalues descending
    std::vector<int> ord(S);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(),
        [&](int i, int j){ return A[i*S + i] > A[j*S + j]; });
    std::vector<NekDouble> Vs(S*S, 0.0);
    for (int i = 0; i < S; ++i)
        for (int k = 0; k < S; ++k)
            Vs[k*S + i] = V[k*S + ord[i]];
    V.swap(Vs);

    // 2.5) Procrustes mode-alignment.
    //
    // The Jacobi eigendecomposition determines the new basis only up to
    //   (a) a per-column sign for non-degenerate eigenvalues, and
    //   (b) an arbitrary orthogonal change of basis within any
    //       (near-)degenerate eigenvalue cluster.
    // Without alignment, modes flip and rotate frame-to-frame in a way that
    // is mathematically valid but visually meaningless.
    //
    // We post-multiply V by a block-diagonal Q such that V·Q is the closest
    // valid eigenbasis to the previous step's modes. Q is block-diagonal
    // w.r.t. the eigenvalue clusters, so applying it preserves the
    // diagonality of C up to the within-cluster eigenvalue spread.
    //
    // Key shortcut: since the previous-step modes u_prev are M-orthonormal
    // at entry, the Gram matrix G[k,j] := <u_jacobi[k], u_prev[j]>_M reduces
    // analytically to G[k,j] = V[j,k]. So Q is determined entirely by V —
    // we don't need to save or compare mode arrays.
    //
    // For each cluster B = {a, a+1, ..., a+m-1}:
    //   - m == 1 (singleton):  Q[a,a] = sign(V[a,a]).
    //   - m  > 1:  Q_block = polar(G_block) = G_block (G_block^T G_block)^{-1/2},
    //              the orthogonal m×m matrix maximizing trace(G_block^T Q).
    {
        std::vector<NekDouble> dvec(S);
        for (int i = 0; i < S; ++i) dvec[i] = A[ord[i]*S + ord[i]];
        NekDouble dMax = 0.0;
        for (NekDouble v : dvec) dMax = std::max(dMax, std::abs(v));
        const NekDouble degTol = 1e-3 * std::max(dMax, NekDouble{1e-300});

        // Identify [a, b) clusters of (near-)equal sorted eigenvalues.
        std::vector<std::pair<int,int>> blocks;
        {
            int b0 = 0;
            for (int i = 1; i <= S; ++i)
                if (i == S || std::abs(dvec[i-1] - dvec[i]) > degTol)
                {
                    blocks.emplace_back(b0, i);
                    b0 = i;
                }
        }

        // Build block-diagonal Q.
        std::vector<NekDouble> Q(S*S, 0.0);
        for (auto [a, b] : blocks)
        {
            const int m = b - a;
            if (m == 1)
            {
                Q[a*S + a] = (V[a*S + a] >= 0.0) ? 1.0 : -1.0;
                continue;
            }
            // m>1: Q_block = polar(G_block) where G_block[i,j] = V[a+j, a+i].
            std::vector<NekDouble> Gb(m*m), GtG(m*m, 0.0);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < m; ++j)
                    Gb[i*m + j] = V[(a+j)*S + (a+i)];
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < m; ++j)
                {
                    NekDouble s = 0.0;
                    for (int k = 0; k < m; ++k)
                        s += Gb[k*m + i] * Gb[k*m + j];
                    GtG[i*m + j] = s;
                }
            // Symmetric Jacobi on GtG (m×m).
            std::vector<NekDouble> dE(m), E(m*m, 0.0);
            for (int i = 0; i < m; ++i) E[i*m + i] = 1.0;
            for (int sw = 0; sw < 60; ++sw)
            {
                NekDouble off2 = 0.0;
                for (int i = 0; i < m; ++i)
                    for (int j = i+1; j < m; ++j)
                        off2 += GtG[i*m + j] * GtG[i*m + j];
                if (off2 < 1e-28) break;
                for (int i = 0; i < m; ++i)
                    for (int j = i+1; j < m; ++j)
                    {
                        NekDouble aij = GtG[i*m + j];
                        if (std::abs(aij) < 1e-300) continue;
                        NekDouble aii = GtG[i*m + i], ajj = GtG[j*m + j];
                        NekDouble theta = 0.5 * std::atan2(2*aij, aii - ajj);
                        NekDouble c = std::cos(theta), sn = std::sin(theta);
                        for (int k = 0; k < m; ++k)
                        {
                            if (k == i || k == j) continue;
                            NekDouble Aki = GtG[k*m + i], Akj = GtG[k*m + j];
                            GtG[k*m + i] = c*Aki + sn*Akj;
                            GtG[k*m + j] = -sn*Aki + c*Akj;
                            GtG[i*m + k] = GtG[k*m + i];
                            GtG[j*m + k] = GtG[k*m + j];
                        }
                        GtG[i*m + i] = c*c*aii + 2*sn*c*aij + sn*sn*ajj;
                        GtG[j*m + j] = sn*sn*aii - 2*sn*c*aij + c*c*ajj;
                        GtG[i*m + j] = 0.0; GtG[j*m + i] = 0.0;
                        for (int k = 0; k < m; ++k)
                        {
                            NekDouble Eki = E[k*m + i], Ekj = E[k*m + j];
                            E[k*m + i] = c*Eki + sn*Ekj;
                            E[k*m + j] = -sn*Eki + c*Ekj;
                        }
                    }
            }
            for (int i = 0; i < m; ++i)
                dE[i] = std::max(GtG[i*m + i], NekDouble{1e-300});

            // (G^T G)^{-1/2} = E · diag(1/sqrt(d)) · E^T
            std::vector<NekDouble> InvSqrt(m*m, 0.0);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < m; ++j)
                {
                    NekDouble s = 0.0;
                    for (int k = 0; k < m; ++k)
                        s += E[i*m + k] * (1.0/std::sqrt(dE[k])) * E[j*m + k];
                    InvSqrt[i*m + j] = s;
                }
            // Q_block = G_block · (G^T G)^{-1/2}
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < m; ++j)
                {
                    NekDouble s = 0.0;
                    for (int k = 0; k < m; ++k)
                        s += Gb[i*m + k] * InvSqrt[k*m + j];
                    Q[(a+i)*S + (a+j)] = s;
                }
        }

        // V_eff = V · Q  (replaces V for the rest of the routine)
        std::vector<NekDouble> Veff(S*S, 0.0);
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j)
            {
                NekDouble s = 0.0;
                for (int k = 0; k < S; ++k)
                    s += V[i*S + k] * Q[k*S + j];
                Veff[i*S + j] = s;
            }
        V.swap(Veff);
    }

    // 3) rotate any [S*size]-laid-out array by U_new[i] = \sum_j V[j,i] U_old[j]
    auto rotateModeBlocks = [&](Array<OneD, NekDouble> &arr, int blockSize) {
        if (arr.size() == 0) return;
        std::vector<NekDouble> tmp(S * blockSize, 0.0);
        for (int i = 0; i < S; ++i)                                 // new mode i
            for (int j = 0; j < S; ++j)                             // old mode j
            {
                const NekDouble Vji = V[j*S + i];
                if (Vji == 0.0) continue;
                const NekDouble *src = arr.data() + j*blockSize;    // old slab
                NekDouble *dst       = tmp.data() + i*blockSize;    // new slab
                for (int n = 0; n < blockSize; ++n)
                    dst[n] += Vji * src[n];                         // tmp[i,n] += V_{ji} . arr[j,n]
            }
        Vmath::Vcopy(S*blockSize, tmp.data(), 1, arr.data(), 1);
    };

    const int nVel    = m_velocity.size();
    const int nPhys   = m_fields[0]->GetTotPoints();
    const int nCoeffs = m_fields[0]->GetNcoeffs();
    const int nPC     = m_pressure->GetNcoeffs();
    const int physBlk = nVel * nPhys;
    const int coefBlk = nVel * nCoeffs;

    rotateModeBlocks(m_DOModePhys,        physBlk);
    rotateModeBlocks(m_DOModeCoeffs,      coefBlk);
    rotateModeBlocks(m_DOModePCoeffs,     nPC);

    // ------------------------------------------------------------------
    // Rotate the integrator's GLM multi-step history.
    //
    // m_doScheme->UpdateSolutionVector() is a TripleArray indexed
    //     solVec[step][var][p]
    // where `step` runs over multi-step values and explicit derivatives
    // (4 entries for IMEXOrder2: 2 values u^n, u^{n-1}; 2 derivs dt·R^n,
    // dt·R^{n-1}); `var` runs over the heterogeneous-size variable list
    // (S*nVel mode-component scalars then 1 flat-Y vector); `p` runs
    // over per-variable inner size (nPhys for modes, Np*S for Y).
    //
    // For mode entries we rotate across the mode index `i` for each
    // component `c` (gather S variables into an [S × nPhys] block, apply
    // the standard rotateModeBlocks rule, scatter back). For Y we just
    // call rotateYi on solVec[step][m_doYIdx], whose layout
    // [particle*S + mode] matches what rotateYi expects.
    //
    // This is the central fix that makes Option B match the original
    // multi-step DO integrator: without it, the next BDF2 step would
    // build m_tmp from u^n in the new basis and u^{n-1}/R^{n-1} in the
    // old basis, which produces a garbage predictor and silently breaks
    // the mode pressure that subsequently drives the Yi RHS.
    // ------------------------------------------------------------------
    if (m_doSchemeInited)
    {
        auto &solVec = m_doScheme->UpdateSolutionVector();
        const int nSteps = static_cast<int>(solVec.size());
        for (int step = 0; step < nSteps; ++step)
        {
            // Gather the S mode variables for each component c into a contiguous
            // [S*nPhys] buffer, rotate, then scatter back.
            std::vector<NekDouble> buf(S * nPhys);
            for (int c = 0; c < nVel; ++c)
            {
                // gather: buf[i,p] = solVec[step][i*nVel + c][p]
                for (int i = 0; i < S; ++i)
                {
                    const NekDouble *src = solVec[step][i*nVel + c].data();
                    NekDouble *dst       = buf.data() + i*nPhys;
                    for (int p = 0; p < nPhys; ++p) dst[p] = src[p];
                }
                // rotate in place (using the same rule as rotateModeBlocks):
                // tmp[i,p] = Σ_j V[j,i] · buf[j,p]
                std::vector<NekDouble> tmp(S * nPhys, 0.0);
                for (int i = 0; i < S; ++i)
                    for (int j = 0; j < S; ++j)
                    {
                        const NekDouble Vji = V[j*S + i];
                        if (Vji == 0.0) continue;
                        const NekDouble *src = buf.data() + j*nPhys;
                        NekDouble *dst       = tmp.data() + i*nPhys;
                        for (int p = 0; p < nPhys; ++p) dst[p] += Vji * src[p];
                    }
                // scatter back: solVec[step][i*nVel + c][p] = tmp[i,p]
                for (int i = 0; i < S; ++i)
                {
                    NekDouble *dst       = solVec[step][i*nVel + c].data();
                    const NekDouble *src = tmp.data() + i*nPhys;
                    for (int p = 0; p < nPhys; ++p) dst[p] = src[p];
                }
            }
        }
    }

    // 4) rotate Yi (per particle): y_new = V^T y_old, i.e. y'_i = Σ_j V_{j,i} y_j
    auto rotateYi = [&](Array<OneD, NekDouble> &Yarr) {
        if (Yarr.size() == 0) return;
        std::vector<NekDouble> tmp(S, 0.0);
        for (int p = 0; p < m_nDOParticles; ++p)    // loop over particles
        {
            for (int i = 0; i < S; ++i)             // new mode i
            {
                NekDouble s = 0.0;
                for (int j = 0; j < S; ++j)         // old mode j
                    s += V[j*S + i] * Yarr[p*S + j];
                tmp[i] = s;
            }
            for (int i = 0; i < S; ++i)             // write back
                Yarr[p*S + i] = tmp[i];
        }
    };
    rotateYi(m_Yi);

    // Rotate the integrator's Y multi-step history (variable index m_doYIdx).
    if (m_doSchemeInited)
    {
        auto &solVec = m_doScheme->UpdateSolutionVector();
        const int nSteps = static_cast<int>(solVec.size());
        for (int step = 0; step < nSteps; ++step)
        {
            // The Y data is stored flat as [particle*S + mode]; rotateYi
            // expects exactly that layout, so we can call it directly.
            rotateYi(solVec[step][m_doYIdx]);
        }
    }

    // 5) recompute C, M3
    ComputeYMoments();

    // clamp tiny-negative diagonal noise to 0 (preserves nonneg eigenvalues).
    for (int i = 0; i < S; ++i)
        if (m_Cij[i*S+i] < 0.0 && std::abs(m_Cij[i*S + i]) < 1e-14)
            m_Cij[i*S + i] = 0.0;

    if (m_verbose && m_doStepCounter == 1)
    {
        NekDouble maxOff = 0.0, maxDiag = 0.0;
        for (int i = 0; i < S; ++i)
        {
            maxDiag = std::max(maxDiag, std::abs(m_Cij[i*S + i]));
            for (int j = 0; j < S; ++j)
                if (i != j) maxOff = std::max(maxOff,
                                              std::abs(m_Cij[i*S + j]));
        }
        std::cout << "[DOVelocityCorrectionScheme][rot] after:  |off|max/|diag|max = "
                  << ((maxDiag > 0) ? maxOff/maxDiag : 0.0)
                  << " (offMax=" << maxOff << " diagMax=" << maxDiag << ")\n";
    }
}

/**
 * After the base VCS step has advanced the mean field, this method:
 *      - advances (modes, Y) atomically via m_doScheme — all RHS terms
 *        evaluated at the same t^n state thanks to the integrator passing
 *        a single `in` snapshot to DOOdeRhs / DOImplicitSolve. Mean^n is
 *        read from m_meanAtTn (snapshotted by the EXT operator at t^n).
 *      - unpacks the post-step (modes, Y) back into m_DOModePhys / m_Yi
 *        (the ground-truth members consumed by RotateToEigenbasisOfC,
 *        ReOrthonormalise, the archive writer, and the next step's DOOdeRhs).
 *      - diagonalises the covariance and orthonormalises the basis.
 */
bool DOVelocityCorrectionScheme::v_PostIntegrate(int step)
{
    VelocityCorrectionScheme::v_PostIntegrate(step);    // base VCS (m_fields → mean^{n+1})

    // Per-step verbose-only counters (used to gate diagnostic prints to
    // step 1 / step 2 in DOOdeRhs / DOImplicitSolve / RotateToEigenbasisOfC,
    // and to assert DO-subsystem non-contamination of the mean field).
    ++m_doStepCounter;
    m_doOdeRhsCallIdx = 0;

    // ---- non-contamination sentinel: hash mean coeffs BEFORE DO subsystem ----
    auto md5_of_coeffs = [&]() -> std::string {
        // tiny FNV-1a 64-bit hash over the mean coefficient array (sufficient
        // to detect bit-level mutation; not cryptographic)
        uint64_t h = 1469598103934665603ull;
        for (int v : m_velocity)
        {
            const auto &c = m_fields[v]->GetCoeffs();
            for (size_t k = 0; k < c.size(); ++k)
            {
                uint64_t bits;
                std::memcpy(&bits, &c[k], sizeof(bits));
                h ^= bits;
                h *= 1099511628211ull;
            }
        }
        char buf[32]; std::snprintf(buf, sizeof(buf), "%016llx",
                                    (unsigned long long)h);
        return buf;
    };
    std::string hashPre;
    if (m_verbose && m_doStepCounter == 1) hashPre = md5_of_coeffs();

    if (m_nDOModes > 0 && m_doSchemeInited)
    {
        AdvanceForcingState();      // OU step + center η + recompute G^n[i,k], A^n[i,k]

        // m_doScheme advances (modes, Y) one IMEX/BDF2 step. Inside, DOOdeRhs
        // is called once at t^n; DOImplicitSolve once with lambda = (2/3)·dt.
        const auto &advanced = m_doScheme->TimeIntegrate(step, m_timestep);

        // Unpack `advanced` back into m_DOModePhys / m_DOModeCoeffs / m_Yi.
        const int nVel    = m_velocity.size();
        const int nPhys   = m_fields[0]->GetTotPoints();
        const int nCoeffs = m_fields[0]->GetNcoeffs();
        Array<OneD, NekDouble> tmpPhys(nPhys), tmpCoef(nCoeffs);
        for (int i = 0; i < m_nDOModes; ++i)
            for (int c = 0; c < nVel; ++c)
            {
                const int v = i*nVel + c;
                Vmath::Vcopy(nPhys, advanced[v].data(), 1,
                             m_DOModePhys.data() + v*nPhys, 1);
                Vmath::Vcopy(nPhys, advanced[v].data(), 1, tmpPhys.data(), 1);
                m_fields[m_velocity[c]]->FwdTrans(tmpPhys, tmpCoef);            // refresh coeffs
                Vmath::Vcopy(nCoeffs, tmpCoef.data(), 1,
                             m_DOModeCoeffs.data() + v*nCoeffs, 1);
            }
        Vmath::Vcopy(m_nDOParticles*m_nDOModes, advanced[m_doYIdx].data(), 1,
                     m_Yi.data(), 1);

        RotateToEigenbasisOfC();
        ReOrthonormalise();

        // Push the post-rotation+reorth m_DOModePhys / m_Yi back into the
        // integrator's solVec[0] so the next step's DOOdeRhs (which syncs
        // m_DOModePhys from `in[v]`) sees the orthonormalised u^{n+1} that
        // downstream consumers (archive, ComputeNMode in DOOdeRhs, etc.)
        // expect. RotateToEigenbasisOfC already rotated solVec[*] in
        // lockstep with m_DOModePhys; ReOrthonormalise then made an
        // additional in-place change to m_DOModePhys that we mirror here.
        // (This induces a mild basis asymmetry between u^{n+1} (reorth'd)
        // and u^n (rotated only) in the next step's BDF2 predictor — the
        // original AB2 code has the same asymmetry, see uNSnap vs
        // m_DOModePhysPrev in AdvanceModes.)
        {
            auto &solVec = m_doScheme->UpdateSolutionVector();
            for (int i = 0; i < m_nDOModes; ++i)
                for (int c = 0; c < nVel; ++c)
                {
                    const int v = i*nVel + c;
                    Vmath::Vcopy(nPhys,
                                 m_DOModePhys.data() + v*nPhys, 1,
                                 solVec[0][v].data(), 1);
                }
            Vmath::Vcopy(m_nDOParticles*m_nDOModes, m_Yi.data(), 1,
                         solVec[0][m_doYIdx].data(), 1);
        }
    }

    // ---- non-contamination sentinel: hash mean coeffs AFTER DO subsystem ----
    if (m_verbose && m_doStepCounter == 1)
    {
        const std::string hashPost = md5_of_coeffs();
        std::cout << "[DOVelocityCorrectionScheme][hash] mean coeffs md5 (FNV1a-64):"
                  << " pre="  << hashPre
                  << " post=" << hashPost
                  << ((hashPre == hashPost) ? " UNCHANGED" : " CHANGED")
                  << "\n";
    }

    return false;
}

/**
 * Orthonormalises modes via Stokes projection + modified Gram-Schmidt.
 * 
 * For each candidate mode:
 * - projects into divergence-free space via SolveUnsteadyStokesSystem
 *   (Stokes projection, lambda = 1, source = 0);
 * - 4-pass modified Gram-Schmidt against the already-accepted basis, in
 *   per-component coefficient/phys space (the homogeneous-BC bracket keeps
 *   Dirichlet entries at zero in both cand and basis, so the subtraction
 *   preserves them; BwdTrans is linear, so subtracting in coeffs and phys
 *   simultaneously preserves their consistency);
 * - check L2-norm of divergence (re-do previous steps if above tolerance);
 * - normalise.
 */
void DOVelocityCorrectionScheme::ReOrthonormalise()
{
    if (m_nDOModes == 0) return;
    auto field0 = std::dynamic_pointer_cast<MultiRegions::ContField>(
        m_fields[m_velocity[0]]);
    ASSERTL0(field0, "DOVelocityCorrectionScheme: CG only."); // CG only

    // homogeneous-BC bracket: zero velocity Dirichlet+Neumann coeffs for the
    // duration of the per-mode Stokes projections; restore at the end
    auto bcState = CaptureVelocityBCState(m_fields, m_velocity);
    HomogenizeVelocityBCsForModes(m_fields, m_velocity);

    // Pressure-BC bracket: SolveUnsteadyStokesSystem inside projectMode reads
    // the pressure BC expansions populated by the most recent VCS step
    // (Karniadakis HOBC + outflow Dirichlet). Those values are mode-independent
    // and dominate the per-mode Stokes projection on bluff-body geometries,
    // injecting a fixed BC-driven shape with norm O(10) regardless of input.
    // Mirror the pattern ModePressureSolve already uses internally: save the
    // pressure BC coeffs+phys, zero them, restore at the end.
    auto pBnd = m_pressure->GetBndCondExpansions();
    std::vector<std::vector<NekDouble>> savedPBndCoeffs(pBnd.size());
    std::vector<std::vector<NekDouble>> savedPBndPhys(pBnd.size());
    for (int n = 0; n < (int)pBnd.size(); ++n)
    {
        const int nc = pBnd[n]->GetNcoeffs();
        const int np = pBnd[n]->GetTotPoints();
        savedPBndCoeffs[n].assign(pBnd[n]->GetCoeffs().data(),
                                  pBnd[n]->GetCoeffs().data() + nc);
        savedPBndPhys[n].assign(pBnd[n]->GetPhys().data(),
                                pBnd[n]->GetPhys().data() + np);
        Vmath::Zero(nc, pBnd[n]->UpdateCoeffs().data(), 1);
        Vmath::Zero(np, pBnd[n]->UpdatePhys().data(),   1);
    }

    const int nVel    = m_velocity.size();
    const int nCoeffs = field0->GetNcoeffs();
    const int nPhys   = field0->GetTotPoints();

    std::vector<ModeData> basis;        // accepted modes (local coeffs + phys)
    basis.reserve(m_nDOModes);

    // scratch for projectMode
    std::vector<Array<OneD, NekDouble>> pmSavedC(nVel), pmSavedP(nVel);     // saved m_fields velocity coeffs/phys
    std::vector<Array<OneD, NekDouble>> pmLocC(nVel), pmLocP(nVel);         // extracted mode coeffs/phys
    Array<OneD, Array<OneD, NekDouble>> pmIn(nVel), pmOut(nVel);            // in/out for SolveUnsteadyStokesSystem
    for (int c = 0; c < nVel; ++c)
    {
        const int nc = m_fields[m_velocity[c]]->GetNcoeffs();
        const int np = m_fields[m_velocity[c]]->GetTotPoints();
        pmSavedC[c] = Array<OneD, NekDouble>(nc);
        pmSavedP[c] = Array<OneD, NekDouble>(np);
        pmLocC[c]   = Array<OneD, NekDouble>(nc);
        pmLocP[c]   = Array<OneD, NekDouble>(np);
        pmIn[c]     = Array<OneD, NekDouble>(np);
        pmOut[c]    = Array<OneD, NekDouble>(np);
    }
    Array<OneD, NekDouble> pmSPC(m_pressure->GetNcoeffs());                 // saved m_pressure coeffs
    Array<OneD, NekDouble> pmSPP(m_pressure->GetTotPoints());               // saved m_pressure phys

    // projectMode: projects input onto the discrete divergence-free space
    // via a Stokes solve.
    auto projectMode = [&](ModeData &mode) {
        // save m_fields velocity (coeffs+phys) and m_pressure (coeffs+phys)
        for (int c = 0; c < nVel; ++c)
        {
            auto &f = m_fields[m_velocity[c]];
            Vmath::Vcopy(f->GetNcoeffs(),   f->GetCoeffs().data(), 1, pmSavedC[c].data(), 1);
            Vmath::Vcopy(f->GetTotPoints(), f->GetPhys().data(),   1, pmSavedP[c].data(), 1);
        }
        Vmath::Vcopy(m_pressure->GetNcoeffs(),   m_pressure->GetCoeffs().data(), 1, pmSPC.data(), 1);
        Vmath::Vcopy(m_pressure->GetTotPoints(), m_pressure->GetPhys().data(),   1, pmSPP.data(), 1);

        // load mode phys into the Stokes-solve input and call the projector
        for (int c = 0; c < nVel; ++c)
            std::copy(mode.phys[c].begin(), mode.phys[c].end(), pmIn[c].data());
        SolveUnsteadyStokesSystem(pmIn, pmOut, 0.0, 1.0);   // projection into div-free space

        // extract solved coeffs from m_fields, reimpose Dirichlet, BwdTrans -> mode
        for (int c = 0; c < nVel; ++c)
        {
            auto cf      = std::dynamic_pointer_cast<MultiRegions::ContField>(
                m_fields[m_velocity[c]]);
            const int nc = cf->GetNcoeffs();
            const int np = cf->GetTotPoints();
            Vmath::Vcopy(nc, m_fields[m_velocity[c]]->GetCoeffs().data(), 1,
                         pmLocC[c].data(), 1);
            cf->ImposeDirichletConditions(pmLocC[c]);
            cf->BwdTrans(pmLocC[c], pmLocP[c]);
            mode.coeffs[c].assign(pmLocC[c].data(), pmLocC[c].data() + nc);
            mode.phys[c].assign(pmLocP[c].data(),   pmLocP[c].data()  + np);
        }

        // restore m_fields velocity and m_pressure state
        for (int c = 0; c < nVel; ++c)
        {
            auto &f = m_fields[m_velocity[c]];
            Vmath::Vcopy(f->GetNcoeffs(),   pmSavedC[c].data(), 1, f->UpdateCoeffs().data(), 1);
            Vmath::Vcopy(f->GetTotPoints(), pmSavedP[c].data(), 1, f->UpdatePhys().data(),   1);
        }
        Vmath::Vcopy(m_pressure->GetNcoeffs(),   pmSPC.data(), 1, m_pressure->UpdateCoeffs().data(), 1);
        Vmath::Vcopy(m_pressure->GetTotPoints(), pmSPP.data(), 1, m_pressure->UpdatePhys().data(),   1);
    };

    // runMgs: 4-pass modified Gram-Schmidt directly on the per-component
    // coefficient and phys arrays. Each pass: alpha = <basis[j], cand>_M
    // and cand -= alpha * basis[j] in both coeff and phys space.
    auto runMgs = [&](ModeData &cand) {
        if (basis.empty()) return;
        for (int pass = 0; pass < 4; ++pass)                // 4 passes
            for (size_t j = 0; j < basis.size(); ++j)       // loop over accepted basis modes
            {
                const NekDouble alpha =
                    VectorMassInner(m_fields, m_velocity, basis[j], cand);  // alpha = <basis[j], cand>
                for (int c = 0; c < nVel; ++c)
                {
                    const int nC = (int)cand.coeffs[c].size();
                    const int nP = (int)cand.phys[c].size();
                    Vmath::Svtvp(nC, -alpha, basis[j].coeffs[c].data(), 1,
                                 cand.coeffs[c].data(), 1, cand.coeffs[c].data(), 1);
                    Vmath::Svtvp(nP, -alpha, basis[j].phys[c].data(), 1,
                                 cand.phys[c].data(), 1, cand.phys[c].data(), 1);
                }
            }
    };

    for (int i = 0; i < m_nDOModes; ++i)
    {
        // pull mode i from m_DOModePhys/Coeffs into candidate ModeData
        ModeData cand;
        cand.coeffs.resize(nVel);
        cand.phys.resize(nVel);
        for (int c = 0; c < nVel; ++c)     // loop over components
        {
            const int cOff = (i * nVel + c)*nCoeffs;
            const int pOff = (i * nVel + c)*nPhys;
            cand.coeffs[c].assign(m_DOModeCoeffs.data() + cOff,
                                     m_DOModeCoeffs.data() + cOff + nCoeffs);
            cand.phys[c].assign(m_DOModePhys.data() + pOff,
                                   m_DOModePhys.data() + pOff + nPhys);
        }
        // pre-MGS candidate norm: cheap one inner product, used only by the
        // collapse-diagnostic print below to give context if ASSERTL0 fires.
        const NekDouble preMgsNrm = std::sqrt(std::max(
            VectorMassInner(m_fields, m_velocity, cand, cand), 0.0));

        // 1) Stokes projection -> 2) modified GS
        projectMode(cand);
        ProjectOutConstantsFromCand(m_fields, m_velocity, cand);    // strip the constant component the Stokes projector preserves
        runMgs(cand);

        // 3) ‖∇·u_cand‖_L2 — trigger a second projection if not clean
        Array<OneD, NekDouble> divCand(nPhys, 0.0), derivBuf(nPhys);
        for (int c = 0; c < nVel; ++c)
        {
            Array<OneD, NekDouble> phys(nPhys);
            std::copy(cand.phys[c].begin(), cand.phys[c].end(), phys.data());
            m_fields[m_velocity[c]]->PhysDeriv(c, phys, derivBuf);  // ∂_c u_cand,c
            Vmath::Vadd(nPhys, derivBuf.data(), 1,
                        divCand.data(), 1, divCand.data(), 1);
        }
        if (m_pressure->L2(divCand) > 1e-3)     // if dirty -> second pass
        {
            projectMode(cand);
            ProjectOutConstantsFromCand(m_fields, m_velocity, cand);    // strip constants from second-pass output too
            runMgs(cand);
        }

        // 4) normalise
        const NekDouble nrm = std::sqrt(std::max(
            VectorMassInner(m_fields, m_velocity, cand, cand), 0.0));
        if (!(nrm > 1e-12) && m_verbose)
        {
            std::cout << "[DOVelocityCorrectionScheme][reorth-FATAL] step=" << m_doStepCounter
                      << " mode i=" << i
                      << " preMGS=" << preMgsNrm
                      << " nrm="    << nrm << "\n";
        }
        ASSERTL0(nrm > 1e-12, "DOVelocityCorrectionScheme: orthonormalisation collapsed.");
        const NekDouble inv = 1.0/nrm;
        for (int c = 0; c < nVel; ++c)
        {
            for (auto &v : cand.coeffs[c]) v *= inv;    // scale coeffs
            for (auto &v : cand.phys[c])   v *= inv;    // scale phys
        }

        basis.push_back(std::move(cand));               // accept
    }

    // Y co-transform: realisation u = ū + Σ Y_i u_i is invariant under the
    // MGS basis change. G[j,i] = <basis[j], u_old[i]>_M, where u_old is the
    // modes before reorth — m_DOMode* still holds them at this point (the
    // writeback below hasn't run yet). Y_new = G · Y_old preserves the
    // realisation. Skipped at init because reorth's Stokes projection
    // homogenises BCs and is not a unitary rotation — at init the
    // realisation-consistency fix is RecomputeYiFromSnapshots() (POD path),
    // called by the caller after ReOrthonormalise().
    if (m_doSchemeInited)
    {
        const int S = m_nDOModes;
        std::vector<NekDouble> G(S*S, 0.0);
        Array<OneD, NekDouble> ip(nCoeffs);
        for (int j = 0; j < S; ++j)
            for (int i = 0; i < S; ++i)
                for (int c = 0; c < nVel; ++c)
                {
                    Array<OneD, NekDouble> physView(nPhys,
                        m_DOModePhys.data() + (i*nVel + c)*nPhys);
                    m_fields[m_velocity[c]]->IProductWRTBase(physView, ip);
                    G[j*S + i] += Vmath::Dot(nCoeffs,
                        basis[j].coeffs[c].data(), 1, ip.data(), 1);
                }
        // MPI: G entries are partial sums; bulk-AllReduce S² values once.
        m_fields[m_velocity[0]]->GetComm()->GetRowComm()->AllReduce(
            G, LibUtilities::ReduceSum);
        std::vector<NekDouble> y(S);
        for (int p = 0; p < m_nDOParticles; ++p)
        {
            for (int j = 0; j < S; ++j)
            {
                NekDouble s = 0.0;
                for (int i = 0; i < S; ++i) s += G[j*S + i] * m_Yi[p*S + i];
                y[j] = s;
            }
            for (int j = 0; j < S; ++j) m_Yi[p*S + j] = y[j];
        }

        // Mode pressure rotation: p_new = G^{-T} · p_old. G is upper-triangular
        // (QR R), so G^T is lower-triangular and we forward-substitute per
        // pressure coeff. No-op at init (m_DOModePCoeffs is zero). Needed at
        // runtime so the next DOOdeRhs uses ∇p_k consistent with the new basis.
        if (m_DOModePCoeffs.size() > 0)
        {
            const int nPC = m_pressure->GetNcoeffs();
            std::vector<NekDouble> pn(S);
            for (int q = 0; q < nPC; ++q)
            {
                for (int k = 0; k < S; ++k)
                {
                    NekDouble s = m_DOModePCoeffs[k*nPC + q];
                    for (int j = 0; j < k; ++j) s -= G[j*S + k] * pn[j];
                    pn[k] = s / G[k*S + k];
                }
                for (int k = 0; k < S; ++k) m_DOModePCoeffs[k*nPC + q] = pn[k];
            }
        }

        // Past Y AND past modes in integrator history: rotate so the BDF2
        // predictor combines time levels in a single basis.
        //   Y_post = G · Y_pre  (Y values transform by G)
        //   u_post[j] = Σ_i (G^{-1})[i,j] u_pre[i]  (mode values by G^{-1})
        // Compute G^{-1} (upper triangular) via column-wise back-sub.
        // Skipped at init: m_doScheme has no history yet.
        if (m_doSchemeInited)
        {
            std::vector<NekDouble> Ginv(S*S, 0.0);
            for (int k = 0; k < S; ++k)                 // k-th column of G^{-1}
                for (int i = S-1; i >= 0; --i)          // back-substitution
                {
                    NekDouble s = (i == k) ? 1.0 : 0.0;
                    for (int j = i+1; j < S; ++j)
                        s -= G[i*S + j] * Ginv[j*S + k];
                    Ginv[i*S + k] = s / G[i*S + i];
                }

            auto &solVec = m_doScheme->UpdateSolutionVector();
            std::vector<NekDouble> tmpY(S);
            std::vector<NekDouble> tmpModes(S * nPhys, 0.0);
            for (auto &slot : solVec)
            {
                // Y values: Y_new[p,j] = Σ_i G[j,i] Y_old[p,i]
                if (slot[m_doYIdx].size() != 0)
                {
                    for (int p = 0; p < m_nDOParticles; ++p)
                    {
                        for (int j = 0; j < S; ++j)
                        {
                            NekDouble s = 0.0;
                            for (int i = 0; i < S; ++i)
                                s += G[j*S + i] * slot[m_doYIdx][p*S + i];
                            tmpY[j] = s;
                        }
                        for (int j = 0; j < S; ++j)
                            slot[m_doYIdx][p*S + j] = tmpY[j];
                    }
                }
                // Mode values: u_new[j*nVel+c] = Σ_i (G^{-1})[i,j] u_old[i*nVel+c]
                for (int c = 0; c < nVel; ++c)
                {
                    std::fill(tmpModes.begin(), tmpModes.end(), 0.0);
                    for (int j = 0; j < S; ++j)
                        for (int i = 0; i < S; ++i)
                        {
                            const NekDouble Gij = Ginv[i*S + j];
                            if (Gij == 0.0) continue;
                            const NekDouble *src = slot[i*nVel + c].data();
                            NekDouble *dst       = tmpModes.data() + j*nPhys;
                            for (int q = 0; q < nPhys; ++q) dst[q] += Gij * src[q];
                        }
                    for (int j = 0; j < S; ++j)
                        std::copy(tmpModes.data() + j*nPhys,
                                  tmpModes.data() + (j+1)*nPhys,
                                  slot[j*nVel + c].data());
                }
            }
        }
    }

    // write the orthonormalised basis back into m_DOModePhys/Coeffs
    for (int i = 0; i < m_nDOModes; ++i)
        for (int c = 0; c < nVel; ++c)
        {
            const int cOff = (i*nVel + c) * nCoeffs;
            const int pOff = (i*nVel + c) * nPhys;
            Vmath::Vcopy(nCoeffs, basis[i].coeffs[c].data(), 1,
                         m_DOModeCoeffs.data() + cOff, 1);
            Vmath::Vcopy(nPhys,   basis[i].phys[c].data(), 1,
                         m_DOModePhys.data() + pOff, 1);
        }

    // restore pressure BC state captured at the top of ReOrthonormalise
    for (int n = 0; n < (int)pBnd.size(); ++n)
    {
        std::copy(savedPBndCoeffs[n].begin(), savedPBndCoeffs[n].end(),
                  pBnd[n]->UpdateCoeffs().data());
        std::copy(savedPBndPhys[n].begin(), savedPBndPhys[n].end(),
                  pBnd[n]->UpdatePhys().data());
    }

    RestoreVelocityBCState(m_fields, bcState);
}

/**
 * Initialises the modes from the `m_nDOModes` smallest-eigenvalue eigenvectors
 * of the homogeneous CG Laplacian (built by DOReducedCGEigenBasis). Each
 * eigenpair generates up to nVel modes by placing the eigenvector into each
 * spatial component slot in turn (and setting the others to 0).
 * Then, calls ReOrthonormalise.
 */
void DOVelocityCorrectionScheme::InitialiseModesFromEllipticEigenbasis()
{
    if (m_nDOModes == 0) return;
    auto field0 = std::dynamic_pointer_cast<MultiRegions::ContField>(
        m_fields[m_velocity[0]]);
    ASSERTL0(field0, "DOVelocityCorrectionScheme: CG only.");       // CG only

    // DOReducedCGEigenBasis runs ARPACK on the per-rank local CG-reduced space;
    // its iteration arithmetic is not MPI-aware, so under multi-rank runs
    // each rank produces a DIFFERENT eigenbasis and the subsequent collective
    // HelmSolve / MGS calls deadlock on inconsistent state. Detect and abort
    // with a clear pointer to the workaround.
    ASSERTL0(m_session->GetComm()->GetSize() == 1,
             "DOInitModeBasis=Laplacian is single-rank only (the underlying "
             "ARPACK eigensolver is not MPI-aware). For multi-rank runs use "
             "DOInitModeBasis=POD (set <I PROPERTY=\"DOInitModeBasis\" "
             "VALUE=\"POD\"/> in your casefile and provide a snapshot pattern).");

    DOReducedCGEigenBasis eigenBasis(field0);   // reduced homogeneous Laplacian eigenbasis            
    ASSERTL0(eigenBasis.GetNumHomCoeffs() > m_nDOModes,
             "Number of DO modes must be smaller than the reduced homogeneous CG size.");

    const int nVel    = m_velocity.size();
    const int nCoeffs = field0->GetNcoeffs();
    const int nPhys   = field0->GetTotPoints();
    // ceil(S / nVel) eigenpairs to fill all modes, +1 for headroom in case the
    // smallest eigenpair is the constant function (lambda ≈ 0) — skipped below
    const int nSeeds = std::max(m_nDOModes,
        (int)std::ceil((NekDouble)m_nDOModes / (NekDouble)nVel)) + 1;
    auto eigenpairs = eigenBasis.ComputeSmallest(nSeeds);   // smallest-eigenvalue pairs

    // skip threshold for the constant-function eigenpair (periodic / pure-Neumann
    // domains have lambda = 0 for the constant; we don't want it as a DO mode)
    const NekDouble eps_const = 1e-10;

    // for each eigenpair, place it into a spacial component of
    // a zeroed mode
    Array<OneD, NekDouble> localCoeffs(field0->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> phys(field0->GetTotPoints(), 0.0);
    int modeCount = 0;
    for (int j = 0; j < (int)eigenpairs.size() && modeCount < m_nDOModes; ++j)
      {
          if (eigenpairs[j].lambda < eps_const) continue;                 // skip constant null-space eigenpair
          eigenBasis.ExportToLocalAndPhys(eigenpairs[j].reduced,          // localCoeffs / phys = scalar eigenpair j
                                          localCoeffs, phys);                                                                            
          for (int c = 0; c < nVel && modeCount < m_nDOModes; ++c, ++modeCount)
          {                                                                                                                              
              // zero all nVel spatial components of mode `modeCount`
              Vmath::Zero(nVel*nCoeffs, m_DOModeCoeffs.data() + modeCount*nVel*nCoeffs, 1);                                              
              Vmath::Zero(nVel*nPhys,   m_DOModePhys.data()   + modeCount*nVel*nPhys,   1);                                              
                                                                                                                                         
              // place the scalar eigenpair into spatial component c of mode `modeCount`                                                 
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
 *   <I PROPERTY="PODMeanType"        VALUE="TimeMean|FirstSnapshot|ProvidedMeanField"/>
 *   <I PROPERTY="PODMeanFile"        VALUE="..."/>             (only if ProvidedMeanField)
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
        std::make_unique<DOPODInitialiser>(m_session, m_fields, m_velocity, cfg);
    DOPODInitialiser &pod = *m_podInitialiser;
    pod.Compute();
    pod.ExportToDOMode(m_DOModePhys, m_DOModeCoeffs);

    // Stash POD spectrum + eigenvectors for InitialiseYi (Y from projection).
    m_podSigmas       = pod.SingularValues();
    m_podEigVecs      = pod.EigenVectors();
    m_podNumSnapshots = pod.NumSnapshots();

    if (m_verbose)
    {
        std::cout << "[DOVelocityCorrectionScheme][POD] init done: " << cfg.snapshotFiles.size()
                  << " snapshots, " << m_nDOModes << " modes, energy="
                  << pod.EnergyFraction() << "\n";
    }

    ReOrthonormalise();
}

} // namespace Nektar
