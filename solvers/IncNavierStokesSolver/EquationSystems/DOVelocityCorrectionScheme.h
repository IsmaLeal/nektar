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
            MemoryManager<DOVelocityCorrectionScheme>::AllocateSharedPtr(pSession, pGraph);
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
    const Array<OneD, int> &GetVelocityIdx()  const { return m_velocity; }

protected:
    // ========== DO parameters ==========
    /// number of modes
    int m_nDOModes;
    /// number of Monte-Carlo particles for Yi evolution
    int m_nDOParticles;
    /// velocity modes layout: (mode * nVel + comp) * nPhys/nCoeffs
    Array<OneD, NekDouble> m_DOModePhys;
    Array<OneD, NekDouble> m_DOModeCoeffs;
    /// pressure modes coefficients, updated by DOImplicitSolve and needed by
    /// Yi RHS for grad(p_k).
    Array<OneD, NekDouble> m_DOModePCoeffs;
    /// Yi coefficients, particle-major: Y_{i,p} = m_Yi[p*m_nDOModes + i]
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
    /// POD initialiser: kept after `InitialiseModesFromPOD()` for the Y
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
    /// number of mode variables in `m_doState` (i.e. m_nDOModes * nVel)
    int m_doNumModeVars = 0;
    /// index of the first Yi variable in `m_doState`
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
    ///  FE coefficients of channels
    Array<OneD, NekDouble> m_forcingBasisCoeffs;
    /// per-particle, per-channel OU amplitudes at current times
    Array<OneD, NekDouble> m_forcingEta;
    /// RNG state
    std::mt19937           m_forcingRng;
    /// m_forcingG[k,i]: projection of forcing shape k onto mode i
    std::vector<NekDouble> m_forcingG;
    /// m_forcingA[k,i]: Monte-Carlo estimate of the expectation E[eta_k Y_i]
    std::vector<NekDouble> m_forcingA;
    /// second moment
    std::vector<NekDouble> m_Cij;
    /// third moment
    std::vector<NekDouble> m_Mkli;

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

    /// verbose-only
    int                   m_doStepCounter  = 0;
    LibUtilities::Timer   m_stepTimer;
    NekDouble             m_stepAccumTime  = 0.0;

    // Set to true by v_EvaluateAdvection_SetPressureBCs after calling
    // PrecomputeGradients with u^n.  DOExplicitRhs clears it and skips its
    // own PrecomputeGradients call when the flag is set, because both see
    // the same m_DOModePhys = u^n and the same mean field via m_meanAtTn.
    bool m_gradientsStaged = false;

    static std::string solverTypeLookupId;

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

    /// precompute mode and mean physical gradients into m_modeGrad1, m_meanGrad1
    void PrecomputeGradients();
    /// compute moments C_ij, M_kli from m_Yi
    void ComputeYMoments();

    /// compute the DO contribution to the mean explicit term (vector)
    void ComputeDOMeanCoupling(Array<OneD, Array<OneD, NekDouble>> &doCorr);

    /// compute the cross terms -((u_bar . \nabla) u_i + (u_i . \nabla) u_bar)
    /// for one mode (vector)
    void ComputeModeCross(int i,
                          Array<OneD, Array<OneD, NekDouble>> &cross);

    /// compute the strong Laplacian of one mode (phys-space, vector)
    void ComputeModeLaplacian(int i, Array<OneD, Array<OneD, NekDouble>> &lap);

    /// compute the explicit N (cross + triple-moment + DO projection)
    /// for one mode (phys-space, vector).
    void ComputeNMode(int i,
                      Array<OneD, Array<OneD, NekDouble>> &N);
    
    /// compute the explicit RHS for the Y coefficients per particle (vector)
    void ComputeYRhs(Array<OneD, NekDouble> &rhs);

    /// Poisson explicit solve for one mode
    void ModePressureSolve(
        const Array<OneD, Array<OneD, NekDouble>> &uhatPhys,
        NekDouble aii_Dt,
        Array<OneD, NekDouble> &pCoeffsOut);
    /// Helmholtz implicit solve for one mode
    void ModeViscousSolve(
        const Array<OneD, Array<OneD, NekDouble>> &uhatPhys,
        const Array<OneD, NekDouble> &pCoeffsIn,
        NekDouble aii_Dt,
        Array<OneD, Array<OneD, NekDouble>> &uNewPhys,
        Array<OneD, Array<OneD, NekDouble>> &uNewCoeffs);

    /// DO subsystem explicit-RHS registered with m_doScheme.
    /// `in` holds the packed state at time t (modes^t + Y^t); `out` receives
    /// the explicit RHS for each variable. Reads m_meanAtTn.
    /// After spatial discretisation, the mode PDEs and the coeffs ODEs become a
    /// system of ODEs, one per FE coefficient. 
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

    void DiagonaliseCov();
    void ReOrthonormalise();

private:
    void InitialiseModesFromEllipticEigenbasis();
    void InitialiseModesFromPOD();
    void InitialiseYi();
    void InitialiseForcingBasis();
    void AdvanceForcingState();
};

typedef std::shared_ptr<DOVelocityCorrectionScheme> DOVelocityCorrectionSchemeSharedPtr;

} // namespace Nektar

#endif // DO_MINE_H