///////////////////////////////////////////////////////////////////////////////
//
// File: DOVelocityCorrectionScheme.h
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
// correction incompressible solver. Auxiliary non-field time-integration
// state is supported for non-ALE runs only; moving-mesh / ALE
// configurations assume a field-only state layout and are not supported.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERS_DOVELOCITYCORRECTIONSCHEME_H
#define NEKTAR_SOLVERS_DOVELOCITYCORRECTIONSCHEME_H

#include <IncNavierStokesSolver/EquationSystems/DOPODInitialiser.h>
#include <IncNavierStokesSolver/EquationSystems/VelocityCorrectionScheme.h>

#include <LibUtilities/BasicUtils/Timer.h>
#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>

#include <memory>
#include <random>   // mt19937 for the OU forcing RNG state
#include <vector>

namespace Nektar
{
class DOVelocityCorrectionScheme : public VelocityCorrectionScheme
{
public:
    friend class MemoryManager<DOVelocityCorrectionScheme>;

    /// Creates an instance of this class
    static SolverUtils::EquationSystemSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph)
    {
        SolverUtils::EquationSystemSharedPtr p =
            MemoryManager<DOVelocityCorrectionScheme>::AllocateSharedPtr(
                pSession, pGraph);
        p->InitObject();
        return p;
    }

    /// Name of class
    static std::string className;

    /// Accessors for FilterDOArchive
    int GetNumDOModes() const { return m_nDOModes; }
    int GetNumDOParticles() const { return m_nDOParticles; }
    const Array<OneD, NekDouble>
        &GetDOModePhys() const { return m_DOModePhys; }
    const Array<OneD, NekDouble>
        &GetDOModeCoeffs() const { return m_DOModeCoeffs; }
    const Array<OneD, NekDouble> &GetYi() const { return m_Yi; }
    /// gathers the sharded Yi into out (global Np * S, every rank); used by
    /// FilterDOArchive so the archive format stays rank-count independent
    void GatherYi(Array<OneD, NekDouble> &out) const;
    const Array<OneD, NekDouble>
        &GetDOModePCoeffs() const { return m_DOModePCoeffs; }
    const Array<OneD, int> &GetVelocityIdx()  const { return m_velocity; }
    int GetNumForcingChannels() const { return m_nForcingChannels; }
    const Array<OneD, NekDouble>
        &GetForcingEta() const { return m_forcingEta; }
    void SerializeForcingRng(std::ostream &os) const { os << m_forcingRng; }

    /// Constant-velocity-subspace data for the strip-constants gauge:
    /// domainArea is |Omega|; onesCoeffs[c] holds the FE coefficients of
    /// the constant function 1; admissible[c] is true iff component c has
    /// no Dirichlet boundary anywhere (so constants are representable).
    struct ConstantSubspaceCache
    {
        NekDouble                           domainArea = 0.0;
        std::vector<Array<OneD, NekDouble>> onesCoeffs;
        std::vector<bool>                   admissible;
    };

protected:
    // ========== DO parameters ==========
    /// number of modes
    int m_nDOModes;
    /// GLOBAL number of Monte-Carlo particles for Yi evolution
    int m_nDOParticles;
    /// Particle shard: this rank owns the contiguous global-index block
    /// [m_npOffset, m_npOffset + m_npLocal). Random draws stay replicated
    /// (identical streams on every rank), so the global population is
    /// independent of the rank count; only storage and per-particle work
    /// are distributed.
    int m_npLocal  = 0;
    int m_npOffset = 0;
    /// velocity modes layout: (mode * nVel + comp) * nPhys/nCoeffs
    Array<OneD, NekDouble> m_DOModePhys;
    Array<OneD, NekDouble> m_DOModeCoeffs;
    /// pressure modes coefficients, updated by DOImplicitSolve and needed by
    /// Yi RHS for grad(p_k).
    Array<OneD, NekDouble> m_DOModePCoeffs;
    /// LOCAL Yi shard, particle-major: Y_{i,p_local} =
    /// m_Yi[p_local*m_nDOModes + i], global index p = m_npOffset + p_local
    Array<OneD, NekDouble> m_Yi;

    // ========== Mode/Yi init ==========
    bool m_modesInitialised = false;
    /// Yi seed (any int) for the (Laplacian) Gaussian initialisation.
    int m_doYiSeed = 0;
    /// std of the Gaussian Yi initialisation
    NekDouble m_doYiSigma = 0.5;
    /// relative Tikhonov regularisation strength for the inverse-covariance
    NekDouble m_invCovRegEps = 1e-2;
    /// selected initial-mode basis: "Laplacian" or "POD".
    std::string m_doInitBasis = "Laplacian";
    /// allow modes to have non-zero spatial mean (constant component).
    /// Required for fully-periodic ICs whose dominant variability is a uniform
    /// flow direction (e.g. Mowlavi & Sapsis 2018 Fig. 10). When false, the
    /// strip-constants gauge is enforced both at init and during evolution.
    bool m_doAllowConstantModes = false;
    /// POD-init outputs (empty unless POD init ran successfully)
    std::vector<NekDouble>              m_podSigmas;    ///< POD singular values
    std::vector<std::vector<NekDouble>> m_podEigVecs;   ///< POD eigenvectors
    int                                 m_podNumSnapshots = 0;
    /// POD initialiser: kept after InitialiseModesFromPOD() for the Y
    /// re-projection, reset after (null if Laplacian init)
    std::unique_ptr<DOPODInitialiser>   m_podInitialiser;

    // ========== IMEX integration ==========
    /// IMEX scheme advancing the (modes, Y) coupled subsystem at
    /// the end of each VCS step. The state vector is heterogeneous-size
    LibUtilities::TimeIntegrationSchemeSharedPtr m_doScheme;
    LibUtilities::TimeIntegrationSchemeOperators m_doOps;
    /// current state for m_doScheme
    Array<OneD, Array<OneD, NekDouble>> m_doState;
    bool                                m_doSchemeInited = false;
    /// number of mode variables in m_doState (i.e. m_nDOModes * nVel)
    int m_doNumModeVars = 0;
    /// index of the first Yi variable in m_doState
    int m_doYIdx = 0;
    /// snapshot of the mean velocity (phys) at t^n. Needed to advance modes
    /// with consistent mean-mode coupling
    Array<OneD, Array<OneD, NekDouble>> m_meanAtTn;
    bool                                m_meanSnapshotValid = false;

    // ========== Stochastic forcing ==========
    /// number of spatial shapes to force
    int m_nForcingChannels = 0;
    /// OU equilibrium std
    NekDouble m_forcingSigma = 0.0;
    /// OU correlation time
    NekDouble m_forcingTau = 0.0;
    /// OU seed
    int m_forcingSeed = 0;
    /// fixed channel templates
    Array<OneD, NekDouble> m_forcingBasisPhys;
    /// FE coefficients of channels
    Array<OneD, NekDouble> m_forcingBasisCoeffs;
    /// per-particle, per-channel OU amplitudes at current times
    Array<OneD, NekDouble> m_forcingEta;
    /// RNG state
    std::mt19937 m_forcingRng;
    /// m_forcingG[k,i]: projection of forcing shape k onto mode i
    std::vector<NekDouble> m_forcingG;
    /// m_forcingA[k,i]: Monte-Carlo estimate of the expectation E[eta_k Y_i]
    std::vector<NekDouble> m_forcingA;
    /// second moment
    std::vector<NekDouble> m_Cij;
    /// third moment
    std::vector<NekDouble> m_Mkli;
    /// particle outer products Z[p*S*S + i*S + j] = Y_{p,i} Y_{p,j}; built
    /// by ComputeYMoments (BLAS third moment) and reused by AssembleYRhs
    Array<OneD, NekDouble> m_Zbuf;

    // ========== Gradient caches (filled by PrecomputeGradients) ==========
    // m_modeGrad1[(i*nVel+c)*nVel+d : *nPhys] = \partial_d u_i[c]
    // m_meanGrad1[(c*nVel+d)*nPhys]           = \partial_d u_mean[c]
    // m_modeLinRhs[(i*nVel+c)*nPhys]          = cross_i[c] + nu*lap_i[c]
    // m_modeGrad2 (\partial_d^2 u_i[c]) is not cached; recomputed on
    // the fly in ComputeNMode to avoid ~nModes*nVel^2*nPhys overhead.
    Array<OneD, NekDouble> m_modeGrad1, m_meanGrad1, m_modeLinRhs;
    /// Quadrature weights times Jacobian at each physical point:
    /// m_physWeights[k] = w_k * J_k.  Pre-computed in v_InitObject.
    Array<OneD, NekDouble> m_physWeights;
    /// Pre-allocated scratch for parallel ComputeNModeBody in DOExplicitRhs.
    /// Layout: (i*nVel+c)*nPhys for mode i, component c.
    Array<OneD, NekDouble> m_NAllBuf;
    /// Laplacian cache: m_modeLap[(i*nVel+c)*nPhys] = sum_d d^2/dx_d^2 u_i[c].
    /// Filled by PrecomputeGradients so ComputeNModeBody needs no PhysDeriv.
    Array<OneD, NekDouble> m_modeLap;
    /// Per-mode scratch for ComputeNModeBody: (4*nVel+2)*nPhys doubles.
    /// Eliminates all MemPool allocations inside the OMP parallel region.
    Array<OneD, NekDouble> m_NBodyBuf;
    /// Scratch for PrecomputeGradients (serial; PhysDeriv allocates via
    /// the MemPool): 2*nPhys doubles (tmp copy + d2u).
    Array<OneD, NekDouble> m_gradScratch;
    /// Constant-subspace cache; filled once by BuildConstantSubspaceCache.
    ConstantSubspaceCache m_constSubspace;

    /// verbose-only
    int m_doStepCounter = 0;
    LibUtilities::Timer m_stepTimer;
    NekDouble m_stepAccumTime = 0.0;

    static std::string solverTypeLookupId;  // Nektar++ class registration key

    DOVelocityCorrectionScheme(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph);

    ~DOVelocityCorrectionScheme() override = default;

    void v_InitObject(bool DeclareField = true) override;
    void v_DoInitialise(bool dumpInitialConditions = true) override;
    bool v_PreIntegrate(int step) override;
    bool v_PostIntegrate(int step) override;
    void v_PrintStatusInformation(int step, NekDouble cpuTime) override;
    void v_EvaluateAdvection_SetPressureBCs(
        const Array<OneD, const Array<OneD, NekDouble>> &inarray,
        Array<OneD, Array<OneD, NekDouble>>             &outarray,
        const NekDouble                                 time) override;

    /// precompute mode and mean gradients into m_modeGrad1, m_meanGrad1
    void PrecomputeGradients();
    /// fill m_constSubspace (collective: AllReduce plus FwdTrans per
    /// admissible component); called once from v_InitObject on every rank
    void BuildConstantSubspaceCache();
    /// compute moments C_ij, M_kli from m_Yi
    void ComputeYMoments();

    /// compute the DO contribution to the mean explicit term (vector)
    void ComputeDOMeanCoupling(Array<OneD, Array<OneD, NekDouble>> &doCorr);

    /// Explicit nonlinear RHS body for mode i: fills N (pre-projection),
    /// betasOut[0..nDOModes) with the local (pre-AllReduce) inner products
    /// <N + nu*lap, u_p>, and constIntOut[0..nVel) with the local integrals
    /// of N + nu*lap used by the strip-constants gauge (zero when the gauge
    /// is inactive). Thread-safe: writes only the per-mode bodyBuf slab and
    /// the per-mode output slots, with no MemPool allocations, so
    /// DOExplicitRhs may run it inside an OpenMP parallel region.
    void ComputeNModeBody(int i, Array<OneD, Array<OneD, NekDouble>> &N,
                          NekDouble *bodyBuf, NekDouble *betasOut,
                          NekDouble *constIntOut);

    /// local (pre-AllReduce) Y-RHS tensors ipKi[k*S+i] = <F_k - grad(p_k),
    /// u_i> and ipKli[(k*S+l)*S+i] = <F_kl, u_i>; DOExplicitRhs reduces them
    /// in the same AllReduce as the mode-projection terms
    void BuildYRhsTensors(NekDouble *ipKi, NekDouble *ipKli);
    /// per-particle Y-RHS assembly from the reduced tensors (BLAS)
    void AssembleYRhs(const NekDouble *ipKi, const NekDouble *ipKli,
                      Array<OneD, NekDouble> &rhs);

    /// Poisson explicit solve for one mode; pGuess warm-starts iterative
    /// solvers (ignored by direct ones)
    void ModePressureSolve(
        const Array<OneD, Array<OneD, NekDouble>> &uhatPhys,
        NekDouble aii_Dt,
        const Array<OneD, const NekDouble> &pGuess,
        Array<OneD, NekDouble> &pCoeffsOut);
    /// Helmholtz implicit solve for one mode; uGuessCoeffs (nVel*nCoeffs
    /// slab) warm-starts iterative solvers (ignored by direct ones)
    void ModeViscousSolve(
        const Array<OneD, Array<OneD, NekDouble>> &uhatPhys,
        const Array<OneD, NekDouble> &pCoeffsIn,
        NekDouble aii_Dt,
        const Array<OneD, const NekDouble> &uGuessCoeffs,
        Array<OneD, Array<OneD, NekDouble>> &uNewPhys,
        Array<OneD, Array<OneD, NekDouble>> &uNewCoeffs);

    /// DO subsystem explicit-RHS registered with m_doScheme.
    /// in holds the packed state at time t (modes^t + Y^t); out receives
    /// the explicit RHS for each variable. Reads m_meanAtTn.
    /// After spatial discretisation, the mode PDEs and the coeffs ODEs
    /// become a system of ODEs, one per FE coefficient.
    void DOExplicitRhs(
        const Array<OneD, const Array<OneD, NekDouble>> &in,
        Array<OneD, Array<OneD, NekDouble>>             &out,
        const NekDouble                                  time);

    /// DO subsystem implicit-solve callback registered with m_doScheme. For
    /// each mode variable, runs the Poisson + Helmholtz pipeline
    /// (ModePressureSolve + ModeViscousSolve). The Y variable has no implicit
    /// term (identity copy).
    void DOImplicitSolve(
        const Array<OneD, const Array<OneD, NekDouble>> &in,
        Array<OneD, Array<OneD, NekDouble>>             &out,
        const NekDouble                                  time,
        const NekDouble                                  lambda);

    /// Rotates the basis to diagonalise C. When deferredV is non-null and
    /// the integrator history exists, the history rotation is NOT applied;
    /// the rotation matrix is stored there instead so ReOrthonormalise can
    /// compose it with its own basis change into a single history pass.
    void DiagonaliseCov(std::vector<NekDouble> *deferredV = nullptr);
    /// pendingV: deferred history rotation from DiagonaliseCov, composed
    /// with this basis change and applied to the history in one pass
    void ReOrthonormalise(const std::vector<NekDouble> *pendingV = nullptr);

private:
    void InitialiseModesFromEllipticEigenbasis();
    void InitialiseModesFromPOD();
    void InitialiseYi();
    void InitialiseForcingBasis();
    void AdvanceForcingState();
    void RestoreFromDOArchive(const std::string &fldPath);
};

typedef std::shared_ptr<DOVelocityCorrectionScheme>
    DOVelocityCorrectionSchemeSharedPtr;

} // namespace Nektar

#endif // NEKTAR_SOLVERS_DOVELOCITYCORRECTIONSCHEME_H
