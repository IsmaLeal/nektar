/////////////////////////////////////////////////////////////////////////////
//
// File: UnsteadyAdvection.cpp
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
// Description: Unsteady linear advection solve routines
//
///////////////////////////////////////////////////////////////////////////////

#include <ADRSolver/EquationSystems/UnsteadyAdvection.h>
#include <LibUtilities/BasicUtils/Timer.h>
#include <MultiRegions/ContField.h>
#include <SolverUtils/Advection/AdvectionWeakDG.h>
#include <iostream>

namespace Nektar
{

std::string UnsteadyAdvection::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "UnsteadyAdvection", UnsteadyAdvection::create,
        "Unsteady Advection equation.");

UnsteadyAdvection::UnsteadyAdvection(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph), AdvectionSystem(pSession, pGraph)
{
}

/**
 * @brief Initialisation object for the unsteady linear advection equation.
 */
void UnsteadyAdvection::v_InitObject(bool DeclareFields)
{
    // Call to the initialisation object of UnsteadySystem
    AdvectionSystem::v_InitObject(DeclareFields);

    // Read the advection velocities from session file
    m_session->LoadParameter("wavefreq", m_waveFreq, 0.0);

    // check to see if it is explicity turned off
    m_session->MatchSolverInfo("GJPStabilisation", "False",
                               m_useGJPStabilisation, true);

    // if GJPStabilisation set to False bool will be true and
    // if not false so negate/revese bool
    m_useGJPStabilisation = !m_useGJPStabilisation;

    m_session->LoadParameter("GJPJumpScale", m_GJPJumpScale, 1.0);

    // Define Velocity fields
    std::vector<std::string> vel;
    vel.push_back("Vx");
    vel.push_back("Vy");
    vel.push_back("Vz");

    // Resize the advection velocities vector to dimension of the problem
    vel.resize(m_spacedim);

    // Store in the global variable m_velocity the advection velocities
    m_velocity = Array<OneD, Array<OneD, NekDouble>>(m_spacedim);
    GetFunction("AdvectionVelocity")->Evaluate(vel, m_velocity);

    // Set advection velocities
    StdRegions::VarCoeffType varcoefftypes[] = {StdRegions::eVarCoeffVelX,
                                                StdRegions::eVarCoeffVelY,
                                                StdRegions::eVarCoeffVelZ};
    for (int i = 0; i < m_spacedim; i++)
    {
        m_varcoeffs[varcoefftypes[i]] = m_velocity[i];
    }

    // Type of advection class to be used
    switch (m_projectionType)
    {
        // Discontinuous field
        case MultiRegions::eDiscontinuous:
        {
            // Do not forwards transform initial condition
            m_homoInitialFwd = false;

            // Define the normal velocity fields
            if (m_fields[0]->GetTrace())
            {
                m_traceVn = Array<OneD, NekDouble>(GetTraceNpoints());
            }

            std::string advName;
            std::string riemName;
            m_session->LoadSolverInfo("AdvectionType", advName, "WeakDG");
            m_advObject = SolverUtils::GetAdvectionFactory().CreateInstance(
                advName, advName);
            if (m_specHP_dealiasing)
            {
                m_advObject->SetFluxVector(
                    &UnsteadyAdvection::GetFluxVectorDeAlias, this);
            }
            else
            {
                m_advObject->SetFluxVector(&UnsteadyAdvection::GetFluxVector,
                                           this);
            }
            m_session->LoadSolverInfo("UpwindType", riemName, "Upwind");
            m_riemannSolver =
                SolverUtils::GetRiemannSolverFactory().CreateInstance(
                    riemName, m_session);
            m_riemannSolver->SetScalar(
                "Vn", &UnsteadyAdvection::GetNormalVelocity, this);
            m_advObject->SetRiemannSolver(m_riemannSolver);
            m_advObject->InitObject(m_session, m_fields);
            break;
        }
        // Continuous field
        case MultiRegions::eGalerkin:
        case MultiRegions::eMixed_CG_Discontinuous:
        {
            std::string advName;
            m_session->LoadSolverInfo("AdvectionType", advName,
                                      "NonConservative");
            m_advObject = SolverUtils::GetAdvectionFactory().CreateInstance(
                advName, advName);
            if (m_specHP_dealiasing)
            {
                m_advObject->SetFluxVector(
                    &UnsteadyAdvection::GetFluxVectorDeAlias, this);
            }
            else
            {
                m_advObject->SetFluxVector(&UnsteadyAdvection::GetFluxVector,
                                           this);
            }
            break;
        }
        default:
        {
            ASSERTL0(false, "Unsupported projection type.");
            break;
        }
    }

    // Forcing terms
    m_forcing = SolverUtils::Forcing::Load(m_session, shared_from_this(),
                                           m_fields, m_fields.size());

    // If explicit it computes RHS and PROJECTION for the time integration
    if (m_explicitAdvection)
    {
        m_ode.DefineOdeRhs(&UnsteadyAdvection::DoOdeRhs, this);
        m_ode.DefineProjection(&UnsteadyAdvection::DoOdeProjection, this);
    }
    // Otherwise it gives an error (no implicit integration)
    else
    {
        m_ode.DefineImplicitSolve(&UnsteadyAdvection::DoImplicitSolve, this);
        ASSERTL0(m_projectionType == MultiRegions::eGalerkin,
                 "Implicit UnsteadyAdvection is not implemented for a "
                 "Discontinuous Galerkin discretisation.")
    }
}

/**
 * @brief Compute the right-hand side for the linear advection equation.
 *
 * @param inarray    Given fields.
 * @param outarray   Calculated solution.
 * @param time       Time.
 */
void UnsteadyAdvection::DoOdeRhs(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time)
{
    // Number of fields (variables of the problem)
    int nVariables = inarray.size();

    LibUtilities::Timer timer;
    if (m_meshDistorted)
    {
        timer.Start();
        Array<OneD, Array<OneD, NekDouble>> tmpIn(nVariables);
        // If ALE we must take Mu coefficient space to u physical space
        ALEHelper::ALEDoElmtInvMassBwdTrans(inarray, tmpIn);
        auto advWeakDGObject =
            std::dynamic_pointer_cast<SolverUtils::AdvectionWeakDG>(
                m_advObject);
        advWeakDGObject->AdvectCoeffs(nVariables, m_fields, m_velocity, tmpIn,
                                      outarray, time);
        timer.Stop();
    }
    else
    {
        timer.Start();
        m_advObject->Advect(nVariables, m_fields, m_velocity, inarray, outarray,
                            time);
        timer.Stop();
    }

    // Elapsed time
    timer.AccumulateRegion("Advect");

    // Negate the RHS
    for (int i = 0; i < nVariables; ++i)
    {
        Vmath::Neg(outarray[i].size(), outarray[i], 1);
    }

    // Add forcing terms
    for (auto &x : m_forcing)
    {
        // set up non-linear terms
        x->Apply(m_fields, inarray, outarray, time);
    }
}

/**
 * @brief Compute the projection for the linear advection equation.
 *
 * @param inarray    Given fields.
 * @param outarray   Calculated solution.
 * @param time       Time.
 */
void UnsteadyAdvection::DoOdeProjection(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time)
{
    // Number of fields (variables of the problem)
    int nVariables = inarray.size();

    // Perform ALE movement
    if (m_ALESolver)
    {
        MoveMesh(time, m_traceNormals);
    }

    // Set the boundary conditions
    SetBoundaryConditions(time);

    // Switch on the projection type (Discontinuous or Continuous)
    switch (m_projectionType)
    {
        // Discontinuous projection
        case MultiRegions::eDiscontinuous:
        {
            // Just copy over array
            if (inarray != outarray)
            {
                int npoints = GetNpoints();

                for (int i = 0; i < nVariables; ++i)
                {
                    Vmath::Vcopy(npoints, inarray[i], 1, outarray[i], 1);
                }
            }
            break;
        }
        // Continuous projection
        case MultiRegions::eGalerkin:
        case MultiRegions::eMixed_CG_Discontinuous:
        {
            int ncoeffs = m_fields[0]->GetNcoeffs();
            Array<OneD, NekDouble> coeffs(ncoeffs, 0.0);
            if (m_useGJPStabilisation)
            {
                StdRegions::ConstFactorMap factors;
                StdRegions::MatrixType mtype = StdRegions::eMass;

                Array<OneD, NekDouble> wsp(ncoeffs);

                for (int i = 0; i < nVariables; ++i)
                {
                    MultiRegions::ContFieldSharedPtr cfield =
                        std::dynamic_pointer_cast<MultiRegions::ContField>(
                            m_fields[i]);

                    m_fields[i]->IProductWRTBase(inarray[i], wsp);

                    const MultiRegions::GJPStabilisationSharedPtr GJPData =
                        cfield->GetGJPForcing();

                    factors[StdRegions::eFactorGJP] =
                        m_GJPJumpScale * m_timestep;

                    if (GJPData->IsSemiImplicit())
                    {
                        mtype = StdRegions::eMassGJP;
                    }

                    // to set up forcing need initial guess in
                    // physical space
                    NekDouble scale = -factors[StdRegions::eFactorGJP];

                    GJPData->Apply(inarray[i], wsp, NullNekDouble1DArray,
                                   scale);

                    // Solve the system
                    MultiRegions::GlobalLinSysKey key(
                        mtype, cfield->GetLocalToGlobalMap(), factors);

                    cfield->GlobalSolve(key, wsp, coeffs, NullNekDouble1DArray);

                    m_fields[i]->BwdTrans(coeffs, outarray[i]);
                }
            }
            else
            {
                for (int i = 0; i < nVariables; ++i)
                {
                    m_fields[i]->FwdTrans(inarray[i], coeffs);
                    m_fields[i]->BwdTrans(coeffs, outarray[i]);
                }
            }
            break;
        }
        default:
            ASSERTL0(false, "Unknown projection scheme");
            break;
    }
}

/**
 * @brief Implicit solution of the unsteady advection problem.
 */
void UnsteadyAdvection::DoImplicitSolve(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray,
    [[maybe_unused]] const NekDouble time, const NekDouble lambda)
{
    StdRegions::ConstFactorMap factors;

    int nvariables                     = inarray.size();
    int npoints                        = m_fields[0]->GetNpoints();
    factors[StdRegions::eFactorLambda] = -1.0 / lambda;

    for (int i = 0; i < nvariables; ++i)
    {
        // Multiply 1.0/timestep/lambda
        Vmath::Smul(npoints, factors[StdRegions::eFactorLambda], inarray[i], 1,
                    outarray[i], 1);

        // Solve a system of equations
        m_fields[i]->LinearAdvectionReactionSolve(
            outarray[i], m_fields[i]->UpdateCoeffs(), factors, m_varcoeffs);

        m_fields[i]->BwdTrans(m_fields[i]->GetCoeffs(), outarray[i]);

        m_fields[i]->SetPhysState(false);
    }
}

/**
 * @brief Get the normal velocity for the linear advection equation.
 */
Array<OneD, NekDouble> &UnsteadyAdvection::GetNormalVelocity()
{
    GetNormalVel(m_velocity);
    return m_traceVn;
}

Array<OneD, NekDouble> &UnsteadyAdvection::GetNormalVel(
    const Array<OneD, const Array<OneD, NekDouble>> &velfield)
{
    // Number of trace (interface) points
    int nTracePts = GetTraceNpoints();
    int nPts      = m_velocity[0].size();

    // Auxiliary variable to compute the normal velocity
    Array<OneD, NekDouble> tmp(nPts), tmp2(nTracePts);

    // Reset the normal velocity
    Vmath::Zero(nTracePts, m_traceVn, 1);

    for (int i = 0; i < velfield.size(); ++i)
    {
        // velocity - grid velocity for ALE before getting trace velocity
        Vmath::Vsub(nPts, velfield[i], 1, m_gridVelocity[i], 1, tmp, 1);

        m_fields[0]->ExtractTracePhys(tmp, tmp2);

        Vmath::Vvtvp(nTracePts, m_traceNormals[i], 1, tmp2, 1, m_traceVn, 1,
                     m_traceVn, 1);
    }

    return m_traceVn;
}

/**
 * @brief Return the flux vector for the linear advection equation.
 *
 * @param physfield   Fields.
 * @param flux        Resulting flux.
 */
void UnsteadyAdvection::GetFluxVector(
    const Array<OneD, Array<OneD, NekDouble>> &physfield,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux)
{
    ASSERTL1(flux[0].size() == m_velocity.size(),
             "Dimension of flux array and velocity array do not match");

    const int nq = m_fields[0]->GetNpoints();

    for (int i = 0; i < flux.size(); ++i)
    {
        for (int j = 0; j < flux[0].size(); ++j)
        {
            for (int k = 0; k < nq; ++k)
            {
                // If ALE we need to take off the grid velocity
                flux[i][j][k] =
                    physfield[i][k] * (m_velocity[j][k] - m_gridVelocity[j][k]);
            }
        }
    }
}

/**
 * @brief Return the flux vector for the linear advection equation using
 * the dealiasing technique.
 *
 * @param physfield   Fields.
 * @param flux        Resulting flux.
 */
void UnsteadyAdvection::GetFluxVectorDeAlias(
    const Array<OneD, Array<OneD, NekDouble>> &physfield,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux)
{
    ASSERTL1(flux[0].size() == m_velocity.size(),
             "Dimension of flux array and velocity array do not match");

    int nq         = physfield[0].size();
    int nVariables = physfield.size();

    // Factor to rescale 1d points in dealiasing
    NekDouble OneDptscale = 2;

    Array<OneD, Array<OneD, NekDouble>> advVel_plane(m_velocity.size());

    // Get number of points to dealias a cubic non-linearity
    nq = m_fields[0]->Get1DScaledTotPoints(OneDptscale);

    // Initialisation of higher-space variables
    Array<OneD, Array<OneD, NekDouble>> physfieldInterp(nVariables);
    Array<OneD, Array<OneD, NekDouble>> velocityInterp(m_expdim);
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> fluxInterp(nVariables);

    // Interpolation to higher space of physfield
    for (int i = 0; i < nVariables; ++i)
    {
        physfieldInterp[i] = Array<OneD, NekDouble>(nq);
        fluxInterp[i]      = Array<OneD, Array<OneD, NekDouble>>(m_expdim);
        for (int j = 0; j < m_expdim; ++j)
        {
            fluxInterp[i][j] = Array<OneD, NekDouble>(nq);
        }

        m_fields[0]->PhysInterp1DScaled(OneDptscale, physfield[i],
                                        physfieldInterp[i]);
    }

    // Interpolation to higher space of velocity
    for (int j = 0; j < m_expdim; ++j)
    {
        velocityInterp[j] = Array<OneD, NekDouble>(nq);

        m_fields[0]->PhysInterp1DScaled(OneDptscale, m_velocity[j],
                                        velocityInterp[j]);
    }

    // Evaluation of flux vector in the higher space
    for (int i = 0; i < flux.size(); ++i)
    {
        for (int j = 0; j < flux[0].size(); ++j)
        {
            Vmath::Vmul(nq, physfieldInterp[i], 1, velocityInterp[j], 1,
                        fluxInterp[i][j], 1);
        }
    }

    // Galerkin project solution back to original space
    for (int i = 0; i < nVariables; ++i)
    {
        for (int j = 0; j < m_spacedim; ++j)
        {
            m_fields[0]->PhysGalerkinProjection1DScaled(
                OneDptscale, fluxInterp[i][j], flux[i][j]);
        }
    }
}

void UnsteadyAdvection::v_GenerateSummary(SolverUtils::SummaryList &s)
{
    AdvectionSystem::v_GenerateSummary(s);
    if (m_useGJPStabilisation)
    {
        SolverUtils::AddSummaryItem(
            s, "GJP Stab. Impl.    ",
            m_session->GetSolverInfo("GJPStabilisation"));
        SolverUtils::AddSummaryItem(s, "GJP Stab. JumpScale", m_GJPJumpScale);
    }
}

bool UnsteadyAdvection::v_PreIntegrate([[maybe_unused]] int step)
{
    return false;
}

void UnsteadyAdvection::v_ExtraFldOutput(
    std::vector<Array<OneD, NekDouble>> &fieldcoeffs,
    std::vector<std::string> &variables)
{
    bool extraFields;
    m_session->MatchSolverInfo("OutputExtraFields", "True", extraFields, true);

    if (extraFields && m_ALESolver)
    {
        ExtraFldOutputGridVelocity(fieldcoeffs, variables);
    }
}

void UnsteadyAdvection::v_ALEInitObject(
    int spaceDim, Array<OneD, MultiRegions::ExpListSharedPtr> &fields)
{
    if (m_projectionType == MultiRegions::eDiscontinuous)
    {
        m_spaceDim  = spaceDim;
        m_fieldsALE = fields;

        // Initialise grid velocities as 0s
        m_gridVelocity      = Array<OneD, Array<OneD, NekDouble>>(m_spaceDim);
        m_gridVelocityTrace = Array<OneD, Array<OneD, NekDouble>>(m_spaceDim);
        for (int i = 0; i < spaceDim; ++i)
        {
            m_gridVelocity[i] =
                Array<OneD, NekDouble>(fields[0]->GetTotPoints(), 0.0);
            m_gridVelocityTrace[i] = Array<OneD, NekDouble>(
                fields[0]->GetTrace()->GetTotPoints(), 0.0);
        }
    }
    ALEHelper::InitObject(spaceDim, fields);
}

} // namespace Nektar
