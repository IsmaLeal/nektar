///////////////////////////////////////////////////////////////////////////////
//
// File: MovingFrameFar.cpp
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

#include <IncNavierStokesSolver/BoundaryConditions/MovingFrameFar.h>
#include <LibUtilities/Communication/Comm.h>

namespace Nektar
{

std::string MovingFrameFar::className =
    GetIncBCFactory().RegisterCreatorFunction(
        "MovingFrameFar", MovingFrameFar::create,
        "Far field boundary condition of moving reference frame");

MovingFrameFar::MovingFrameFar(
    [[maybe_unused]] const LibUtilities::SessionReaderSharedPtr pSession,
    [[maybe_unused]] Array<OneD, MultiRegions::ExpListSharedPtr> pFields,
    [[maybe_unused]] Array<OneD, SpatialDomains::BoundaryConditionShPtr> cond,
    [[maybe_unused]] Array<OneD, MultiRegions::ExpListSharedPtr> exp,
    [[maybe_unused]] int nbnd, [[maybe_unused]] int spacedim,
    [[maybe_unused]] int bnddim)
    : IncBaseCondition(pSession, pFields, cond, exp, nbnd, spacedim, bnddim)
{
    classname = "MovingFrameFar";
    for (size_t i = 0; i < m_spacedim; ++i)
    {
        m_BndConds[i] = cond[i];
        if (cond[i]->GetUserDefined() == classname)
        {
            m_BndExp[i] = exp[i];
        }
    }
}

void MovingFrameFar::v_Initialise(
    [[maybe_unused]] const LibUtilities::SessionReaderSharedPtr &pSession)
{
    IncBaseCondition::v_Initialise(pSession);
    m_definedVels.clear();
    for (int k = 0; k < m_spacedim; ++k)
    {
        m_definedVels.push_back(
            std::static_pointer_cast<
                SpatialDomains::DirichletBoundaryCondition>(m_BndConds[k])
                ->m_dirichletCondition);
    }
}

void MovingFrameFar::v_Update(
    [[maybe_unused]] const Array<OneD, const Array<OneD, NekDouble>> &fields,
    [[maybe_unused]] const Array<OneD, const Array<OneD, NekDouble>> &Adv,
    std::map<std::string, NekDouble> &params)
{
    int nptsPlane0 = 0;
    SetNumPointsOnPlane0(nptsPlane0);
    if (0 == nptsPlane0)
    {
        return;
    }
    NekDouble time = 0.;
    if (params.find("Time") != params.end())
    {
        time = params["Time"];
    }
    Array<OneD, NekDouble> vels(m_spacedim, 0.0);
    for (int i = 0; i < m_spacedim; ++i)
    {
        vels[i] = m_definedVels[i]->Evaluate(0., 0., 0., time);
    }
    if (params.find("Theta_z") != params.end())
    {
        NekDouble v0 = vels[0], v1 = vels[1];
        NekDouble c = cos(params["Theta_z"]), s = sin(params["Theta_z"]);
        vels[0] = v0 * c + v1 * s;
        vels[1] = -v0 * s + v1 * c;
    }
    for (auto &it : m_BndExp)
    {
        int k = it.first;
        if (it.second->GetExpType() == MultiRegions::e2DH1D)
        {
            Array<OneD, NekDouble> tmpvel =
                it.second->GetPlane(0)->UpdatePhys();
            Vmath::Fill(nptsPlane0, vels[k], tmpvel, 1);
            it.second->GetPlane(0)->FwdTransBndConstrained(
                tmpvel, it.second->GetPlane(0)->UpdateCoeffs());
        }
        else
        {
            Array<OneD, NekDouble> tmpvel = it.second->UpdatePhys();
            Vmath::Fill(nptsPlane0, vels[k], tmpvel, 1);
            it.second->FwdTransBndConstrained(tmpvel,
                                              it.second->UpdateCoeffs());
        }
    }
}

} // namespace Nektar
