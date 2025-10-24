///////////////////////////////////////////////////////////////////////////////
//
// File: TransMovingWall.cpp
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
// Description: Abstract base class for Extrapolate.
//
///////////////////////////////////////////////////////////////////////////////

#include <IncNavierStokesSolver/BoundaryConditions/TransMovingWall.h>
#include <LibUtilities/Communication/Comm.h>

namespace Nektar
{

std::string TransMovingWall::className =
    GetIncBCFactory().RegisterCreatorFunction(
        "TransMovingWall", TransMovingWall::create,
        "Far field boundary condition of moving reference frame");

TransMovingWall::TransMovingWall(
    [[maybe_unused]] const LibUtilities::SessionReaderSharedPtr pSession,
    [[maybe_unused]] Array<OneD, MultiRegions::ExpListSharedPtr> pFields,
    [[maybe_unused]] Array<OneD, SpatialDomains::BoundaryConditionShPtr> cond,
    [[maybe_unused]] Array<OneD, MultiRegions::ExpListSharedPtr> exp,
    [[maybe_unused]] int nbnd, [[maybe_unused]] int spacedim,
    [[maybe_unused]] int bnddim)
    : StaticWall(pSession, pFields, cond, exp, nbnd, spacedim, bnddim)
{
    classname  = "TransMovingWall";
    m_pressure = cond.size() - 1;
    for (size_t i = 0; i < m_bnddim; ++i)
    {
        if (cond[i]->GetUserDefined() == "TimeDependent")
        {
            m_BndConds[i] = cond[i];
        }
    }
    if (cond[m_pressure]->GetUserDefined() == classname)
    {
        m_BndExp[m_pressure] = exp[m_pressure];
    }
}

void TransMovingWall::v_Initialise(
    const LibUtilities::SessionReaderSharedPtr &pSession)
{
    if (m_BndExp.empty())
    {
        return;
    }
    StaticWall::v_Initialise(pSession);
    if (pSession->DefinesParameter("TimeStep"))
    {
        m_dt = pSession->GetParameter("TimeStep");
    }
    else
    {
        m_dt = 2.E-4;
    }
    NekDouble invdt  = 1. / m_dt;
    Fourth_Coeffs[0] = 1. / 12. * invdt;
    Fourth_Coeffs[1] = -2. / 3. * invdt;
    Fourth_Coeffs[2] = 2. / 3. * invdt;
    Fourth_Coeffs[3] = -1. / 12. * invdt;
}

void TransMovingWall::v_Update(
    [[maybe_unused]] const Array<OneD, const Array<OneD, NekDouble>> &fields,
    [[maybe_unused]] const Array<OneD, const Array<OneD, NekDouble>> &Adv,
    std::map<std::string, NekDouble> &params)
{
    if (m_BndExp.empty() || fields.size() == 0)
    {
        return;
    }
    ++m_numCalls;
    int nptsPlane0 = 0;
    SetNumPointsOnPlane0(nptsPlane0);
    Array<OneD, Array<OneD, NekDouble>> rhs(m_bnddim);
    for (int i = 0; i < m_bnddim; ++i)
    {
        rhs[i] = Array<OneD, NekDouble>(m_npoints, 0.);
    }
    // add viscous term
    AddVisPressureBCs(fields, rhs, params);
    // Add DuDt
    std::map<std::string, NekDouble> transParams;
    std::vector<std::string> accStr = {"A_x", "A_y", "A_z"};
    NekDouble time = 0., dt2 = 2. * m_dt;
    if (params.find("Time") != params.end())
    {
        time = params["Time"];
    }
    std::vector<NekDouble> times = {time - dt2, time - m_dt, time + m_dt,
                                    time + dt2};
    for (int i = 0; i < m_bnddim; ++i)
    {
        if (m_BndConds.find(i) != m_BndConds.end() && nptsPlane0)
        {
            NekDouble dudt = 0.;
            LibUtilities::EquationSharedPtr equ =
                std::static_pointer_cast<
                    SpatialDomains::DirichletBoundaryCondition>(m_BndConds[i])
                    ->m_dirichletCondition;
            for (int j = 0; j < times.size(); ++j)
            {
                dudt += Fourth_Coeffs[j] * equ->Evaluate(0, 0., 0., times[j]);
            }
            transParams[accStr[i]] = dudt;
        }
    }
    AddRigidBodyAcc(rhs, transParams, nptsPlane0);
    m_BndExp[m_pressure]->NormVectorIProductWRTBase(
        rhs, m_BndExp[m_pressure]->UpdateCoeffs());
}

} // namespace Nektar
