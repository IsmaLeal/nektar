/**
 * DO extension of the velocity-correction incompressible solver.
 *
 * Auxiliary non-field time-integration state is currently supported only
 * for non-ALE runs. Moving-mesh / ALE configurations are not supported yet,
 * because the ALE helper path assumes a field-only state layout.
 */
#ifndef NEKTAR_SOLVERS_DOMINE_H
#define NEKTAR_SOLVERS_DOMINE_H

#include <IncNavierStokesSolver/EquationSystems/DOPODInitialiser.h>
#include <IncNavierStokesSolver/EquationSystems/VelocityCorrectionScheme.h>

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

    static SolverUtils::EquationSystemSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph)
    {
        SolverUtils::EquationSystemSharedPtr p =
            MemoryManager<DOVelocityCorrectionScheme>::AllocateSharedPtr(pSession, pGraph);
        p->InitObject();
        return p;
    }

    static std::string className;

    /// Accessors for FilterDOArchive (read-only views of the DO state).
    int GetNumDOModes()      const { return m_nDOModes; }
    int GetNumDOParticles()  const { return m_nDOParticles; }
    const Array<OneD, NekDouble> &GetDOModePhys()   const { return m_DOModePhys; }
    const Array<OneD, NekDouble> &GetDOModeCoeffs() const { return m_DOModeCoeffs; }
    const Array<OneD, NekDouble> &GetYi()           const { return m_Yi; }
    const Array<OneD, int>       &GetVelocityIdx()  const { return m_velocity; }

protected:
    /// number of modes
    int m_nDOModes;
    /// number of particles for Yi evolution
    int m_nDOParticles;
    /// modes layout: (mode * nVel + comp) * nPhys/nCoeffs
    Array<OneD, NekDouble> m_DOModePhys;
    Array<OneD, NekDouble> m_DOModeCoeffs;
    /// Yi coefficients (size m_nDOParticles*m_nDOModes), particle-major:
    /// Y_{i,alpha} = m_Yi[alpha*m_nDOModes + i]
    Array<OneD, NekDouble> m_Yi;
    bool m_modesInitialised = false;

    /// Mode pressure coefficients (size R * nPC), updated by DOImplicitSolve
    /// for each mode and consumed by DOOdeRhs in the Y RHS to form ∇p_k.
    Array<OneD, NekDouble> m_DOModePCoeffs;

    /// Snapshot of the mean velocity (phys) at t^n. VCS's integrator calls
    /// the EXT operator with m_fields holding mean^n; we capture it there
    /// so the DO subsystem's RHS callbacks (run later, when m_fields holds
    /// mean^{n+1}) can swap mean^n into m_fields for ComputeModeCross /
    /// ComputeDOMeanCoupling reads. See v_EvaluateAdvection_SetPressureBCs.
    Array<OneD, Array<OneD, NekDouble>> m_meanAtTn;
    bool                                m_meanSnapshotValid = false;

    /// Separate IMEX scheme advancing the (modes, Y) coupled subsystem at
    /// the end of each VCS step. The state vector is heterogeneous-size
    /// (variables 0..S*nVel-1 are mode phys of size nPhys; variable S*nVel
    /// is Y_flat of size Np*S). The integrator stores its own multi-step
    /// history internally, so DOVelocityCorrectionScheme no longer needs the *Prev arrays.
    LibUtilities::TimeIntegrationSchemeSharedPtr m_doScheme;
    LibUtilities::TimeIntegrationSchemeOperators m_doOps;
    Array<OneD, Array<OneD, NekDouble>>          m_doState; ///< current state for m_doScheme
    bool                                         m_doSchemeInited = false;
    int m_doNumModeVars = 0;        ///< S * nVel (cached)
    int m_doYIdx        = 0;        ///< index of Y_flat variable in m_doState (= S*nVel)
    /// Yi RNG seed (any int) for the i.i.d. Gaussian initialisation.
    int m_doYiSeed = 0;
    /// Std-dev σ of the i.i.d. Gaussian Yi initialisation: Y_{p,i} ~ N(0, σ²).
    NekDouble m_doYiSigma = 0.5;

    /// Additive stochastic forcing (channel-based, fixed-template, OU-in-time).
    /// Disabled when m_nForcingChannels == 0.
    int       m_nForcingChannels = 0;
    NekDouble m_forcingSigma     = 0.0;     ///< OU equilibrium std-dev (per channel)
    NekDouble m_forcingTau       = 0.0;     ///< OU correlation time (0 => white in time)
    int       m_forcingSeed      = 0;       ///< mt19937 seed
    /// Relative Tikhonov regularisation strength for the inverse-covariance
    /// operator Σ^{-1} acting on the mode RHS in eigenbasis. Defines
    ///   invMuReg_i = μ_i / (μ_i² + λ²),  λ = m_forcingRegEps · μ_max,
    /// where μ_max is the largest current diagonal of C. Bounded for μ_i → 0;
    /// reduces to 1/μ_i when μ_i ≫ λ. Loaded from session parameter
    /// "DOForcingRegEps" (default 1e-2).
    NekDouble m_forcingRegEps    = 1e-2;
    Array<OneD, NekDouble> m_forcingBasisPhys;   ///< K * nVel * nPhys, fixed channel templates
    Array<OneD, NekDouble> m_forcingBasisCoeffs; ///< K * nVel * nCoeffs, FE coefficients of channels
    Array<OneD, NekDouble> m_forcingEta;         ///< Np * K, current per-particle OU amplitudes
    std::mt19937           m_forcingRng;         ///< persistent RNG state
    std::vector<NekDouble> m_forcingG;           ///< S * K, G[i,k] = <g_k, u_i>_M (recomputed each step)
    std::vector<NekDouble> m_forcingA;           ///< S * K, A[i,k] = (1/Np) Σ_p Y_{p,i} η_{p,k} (recomputed each step)

    /// Sample moments cached each step
    std::vector<NekDouble> m_Cij;   ///< C_{ij} = E[Y_i Y_j], size R*R
    std::vector<NekDouble> m_Mkli;  ///< M_{kli} = E[Y_k Y_l Y_i], size R*R*R

    /// Verbose-only: integrator-step counter (incremented in v_PostIntegrate).
    /// Used to gate per-step diagnostic prints to step 1 (and step 2 for
    /// the gamma0/scale snapshot in Part 3).
    int m_doStepCounter   = 0;
    /// Verbose-only: DOOdeRhs invocation counter, reset at the top of every
    /// integrator step (in v_PostIntegrate before TimeIntegrate). Used to
    /// confirm the integrator's call pattern (1 explicit call per BDF1 step,
    /// 1 explicit call per BDF2 step plus 1 fresh-deriv call after solve).
    int m_doOdeRhsCallIdx = 0;

    static std::string solverTypeLookupId;

    DOVelocityCorrectionScheme(const LibUtilities::SessionReaderSharedPtr &pSession,
           const SpatialDomains::MeshGraphSharedPtr &pGraph);

    ~DOVelocityCorrectionScheme() override = default;

    void v_InitObject(bool DeclareField = true) override;
    void v_DoInitialise(bool dumpInitialConditions = true) override;

    bool v_PostIntegrate(int step) override;

    /// Override to inject DO mean-field coupling: outarray += sum_ij C_ij F_ij
    void v_EvaluateAdvection_SetPressureBCs(
        const Array<OneD, const Array<OneD, NekDouble>> &inarray,
        Array<OneD, Array<OneD, NekDouble>> &outarray,
        const NekDouble time) override;

    /// Compute moments C_ij, M_kli, mu_i from m_Yi
    void ComputeYMoments();

    /// Compute the DO Reynolds-stress contribution to the mean explicit term
    /// (added on top of standard advection by VCS).
    /// On entry, doCorr[alpha] should be zeroed; on exit it contains
    /// -(u_i ∂_x u_j + v_i ∂_y u_j + w_i ∂_z u_j) C_ij  (per spatial component).
    void ComputeDOMeanCoupling(Array<OneD, Array<OneD, NekDouble>> &doCorr);

    /// Compute the cross terms -(u_bar . ∇ u_i + u_i . ∇ u_bar) for one mode
    /// (vector field, all components).
    void ComputeModeCross(int i,
                          Array<OneD, Array<OneD, NekDouble>> &cross);

    /// Compute the strong Laplacian of one mode's vector field, in phys space.
    void ComputeModeLaplacian(int i,
                              Array<OneD, Array<OneD, NekDouble>> &lap);

    /// Compute the explicit N for one mode (cross + triple-moment + DO projection)
    /// Result: N[c] = N_mode_i,c (phys-space, vector).
    void ComputeNMode(int i,
                      Array<OneD, Array<OneD, NekDouble>> &N);

    /// IMEX2 viscous Helmholtz solve for one mode (from the formula above).
    void ModePressureSolve(
        const Array<OneD, Array<OneD, NekDouble>> &uhatPhys,
        NekDouble Dt,
        Array<OneD, NekDouble> &pCoeffsOut);

    void ModeViscousSolve(
        const Array<OneD, Array<OneD, NekDouble>> &uhatPhys,
        const Array<OneD, NekDouble> &pCoeffsIn,
        NekDouble Dt,
        Array<OneD, Array<OneD, NekDouble>> &uNewPhys,
        Array<OneD, Array<OneD, NekDouble>> &uNewCoeffs);

    /// DO subsystem explicit-RHS callback registered with m_doScheme.
    /// `in` holds the packed state at time t (modes^t + Y^t); `out` receives
    /// the explicit RHS for each variable. Reads m_meanAtTn (snapshotted by
    /// the VCS EXT operator) so the mean field used in the cross / triple-
    /// moment / mean-coupling terms is consistent with the mode/Y state in
    /// `in` — every term is evaluated at the SAME t.
    void DOOdeRhs(
        const Array<OneD, const Array<OneD, NekDouble>> &in,
        Array<OneD, Array<OneD, NekDouble>>             &out,
        const NekDouble                                  time);

    /// DO subsystem implicit-solve callback registered with m_doScheme.
    /// For each mode variable, runs the existing pressure Poisson + viscous
    /// Helmholtz pipeline (ModePressureSolve + ModeViscousSolve). The Y
    /// variable has no implicit term (identity copy in→out).
    /// `lambda` (= a_iixDt from the integrator) carries the IMEX/BDF2 weight
    /// (2/3)·dt; ModePressureSolve / ModeViscousSolve are called with
    /// Dt = (3/2)·lambda since they hard-code aii_Dt = (2/3)·Dt internally.
    void DOImplicitSolve(
        const Array<OneD, const Array<OneD, NekDouble>> &in,
        Array<OneD, Array<OneD, NekDouble>>             &out,
        const NekDouble                                  time,
        const NekDouble                                  lambda);

    /// Rotate (modes + Yi + histories + mode pressures) so that
    /// C = E[Y Y^T] becomes diagonal in the new basis. C is recomputed
    /// at the end so m_Cij/m_Mkli are consistent.
    void RotateToEigenbasisOfC();

    void ReOrthonormalise();

    /// Selected initial-mode basis: "Laplacian" (default) or "POD".
    std::string m_doInitBasis = "Laplacian";

    /// POD-init outputs (consumed by InitialiseYi when m_doInitBasis=="POD"):
    ///   m_podSigmas[k]        = sigma_k = sqrt(lambda_k)
    ///   m_podEigVecs[k][p]    = v_{p,k}, K eigenvector entries per mode k
    ///   m_podNumSnapshots     = K (number of snapshots used for POD)
    /// Empty unless POD init ran successfully.
    std::vector<NekDouble>              m_podSigmas;
    std::vector<std::vector<NekDouble>> m_podEigVecs;
    int                                 m_podNumSnapshots = 0;

    /// POD initialiser instance, kept alive between InitialiseModesFromPOD()
    /// and the post-ReOrthonormalise Y re-projection. Reset to free snapshot
    /// metadata after the projection finishes. Empty for the Laplacian path.
    std::unique_ptr<DOPODInitialiser>   m_podInitialiser;

private:
    void InitialiseModesFromEllipticEigenbasis();
    void InitialiseModesFromPOD();
    void InitialiseYi();
    /// Read ForcingChannels XML, evaluate at quadrature points, FwdTrans, mass-normalise.
    void InitialiseForcingBasis();
    /// One OU step of m_forcingEta + per-channel centering + recompute G[i,k] and A[i,k].
    void AdvanceForcingState();
};

typedef std::shared_ptr<DOVelocityCorrectionScheme> DOVelocityCorrectionSchemeSharedPtr;

} // namespace Nektar

#endif // DO_MINE_H