////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessForceDecompose.cpp
//
//  For more information, please see: http://www.nektar.info/
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description: Perform force decomposition.
//
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
using namespace std;

#include <LibUtilities/BasicUtils/SharedArray.hpp>

#include "ProcessForceDecompose.h"

namespace Nektar::FieldUtils
{

ProcessForceDecompose::ProcessForceDecompose(FieldSharedPtr f)
    : ProcessModule(f)
{
}

NekDouble ProcessForceDecompose::GetViscosity()
{
    return m_f->m_session->GetParameter("Kinvis");
}

void ProcessForceDecompose::GetQ(
    const Array<OneD, MultiRegions::ExpListSharedPtr> exp,
    Array<OneD, NekDouble> &Q)
{
    int method = m_config["Q"].as<int>();
    if (method == 1)
    {
        QFromPressure(exp, Q);
    }
    else
    {
        QFromField(exp, Q);
    }
}

void ProcessForceDecompose::QFromField(
    const Array<OneD, MultiRegions::ExpListSharedPtr> exp,
    Array<OneD, NekDouble> &Q)
{
    int npoints = m_f->m_exp[0]->GetNpoints();
    int IQ      = FindVariable("Q");
    if (Q.size() < npoints)
    {
        Q = Array<OneD, NekDouble>(npoints);
    }
    Vmath::Vcopy(npoints, exp[IQ]->GetPhys(), 1, Q, 1);
}

int ProcessForceDecompose::FindVariable(const std::string &var)
{
    auto index = find(m_f->m_variables.begin(), m_f->m_variables.end(), var);
    if (index == m_f->m_variables.end())
    {
        NEKERROR(ErrorUtil::efatal, var + " field not found.");
    }
    return index - m_f->m_variables.begin();
}

void ProcessForceDecompose::QFromPressure(
    const Array<OneD, MultiRegions::ExpListSharedPtr> exp,
    Array<OneD, NekDouble> &Q)
{
    int npoints = exp[0]->GetNpoints();
    int ip      = FindVariable("p");
    if (Q.size() < npoints)
    {
        Q = Array<OneD, NekDouble>(npoints);
    }

    Array<OneD, Array<OneD, NekDouble>> grad(m_spacedim);
    Array<OneD, NekDouble> tmp(npoints, 0.);

    for (int i = 0; i < m_spacedim; ++i)
    {
        grad[i] = Array<OneD, NekDouble>(npoints);
    }
    Vmath::Zero(npoints, Q, 1);

    for (int i = 0; i < m_spacedim; ++i)
    {
        exp[ip]->PhysDeriv(i, m_f->m_exp[ip]->GetPhys(), grad[i]);
        exp[ip]->PhysDeriv(i, grad[i], tmp);
        Vmath::Vadd(npoints, tmp, 1, Q, 1, Q, 1);
    }
    Vmath::Smul(npoints, 0.5, Q, 1, Q, 1);
}

void ProcessForceDecompose::GetPhi(
    const Array<OneD, MultiRegions::ExpListSharedPtr> exp,
    Array<OneD, Array<OneD, NekDouble>> &phi, std::map<int, std::string> &Iphi)
{
    Iphi.clear();
    int npoints = exp[0]->GetNpoints();
    for (size_t i = 0; i < m_f->m_variables.size(); ++i)
    {
        if (std::string::npos != m_f->m_variables[i].find("phi"))
        {
            Iphi[i] = m_f->m_variables[i];
        }
    }
    if (phi.size() != Iphi.size())
    {
        phi = Array<OneD, Array<OneD, NekDouble>>(Iphi.size());
    }
    int i = 0;
    for (const auto &p : Iphi)
    {
        if (phi[i].size() < npoints)
        {
            phi[i] = Array<OneD, NekDouble>(npoints, 0.);
        }
        Vmath::Vcopy(npoints, exp[p.first]->GetPhys(), 1, phi[i], 1);
        ++i;
    }
}

void ProcessForceDecompose::GetInfoPhi(std::map<int, std::string> &Iphi)
{
    Iphi.clear();
    for (size_t i = 0; i < m_f->m_variables.size(); ++i)
    {
        if (std::string::npos != m_f->m_variables[i].find("phi"))
        {
            Iphi[i] = m_f->m_variables[i];
        }
    }
}

void ProcessForceDecompose::GetGradPressure(
    const Array<OneD, MultiRegions::ExpListSharedPtr> exp,
    Array<OneD, Array<OneD, NekDouble>> &gradp)
{
    int npoints = exp[0]->GetNpoints();
    if (gradp.size() < m_spacedim)
    {
        gradp = Array<OneD, Array<OneD, NekDouble>>(m_spacedim);
    }
    for (size_t i = 0; i < gradp.size(); ++i)
    {
        if (gradp[i].size() < npoints)
        {
            gradp[i] = Array<OneD, NekDouble>(npoints, 0.);
        }
    }
    int iv;
    try
    {
        iv = FindVariable("p");
    }
    catch (...)
    {
        return;
    }
    Array<OneD, NekDouble> pressure(npoints, 0.);
    Vmath::Vcopy(npoints, exp[iv]->GetPhys(), 1, pressure, 1);
    for (int i = 0; i < m_spacedim; ++i)
    {
        exp[0]->PhysDeriv(i, pressure, gradp[i]);
    }
}

void ProcessForceDecompose::GetVelocity(
    const Array<OneD, MultiRegions::ExpListSharedPtr> exp,
    Array<OneD, Array<OneD, NekDouble>> &vel)
{
    int npoints = exp[0]->GetNpoints();
    if (vel.size() < m_spacedim)
    {
        vel = Array<OneD, Array<OneD, NekDouble>>(m_spacedim);
    }
    std::vector<std::string> vars = {"u", "v", "w"};
    for (size_t i = 0; i < m_spacedim; ++i)
    {
        int iv = FindVariable(vars[i]);
        if (vel[i].size() < npoints)
        {
            vel[i] = Array<OneD, NekDouble>(npoints);
        }
        Vmath::Vcopy(npoints, exp[iv]->GetPhys(), 1, vel[i], 1);
    }
}

void ProcessForceDecompose::GetStressTensor(
    const Array<OneD, MultiRegions::ExpListSharedPtr> exp,
    Array<OneD, Array<OneD, NekDouble>> &stress)
{
    int npoints = exp[0]->GetNpoints();
    Array<OneD, Array<OneD, NekDouble>> vel;
    GetVelocity(exp, vel);
    Array<OneD, Array<OneD, NekDouble>> grad(m_spacedim * m_spacedim);
    Array<OneD, NekDouble> tmp(npoints);
    for (size_t i = 0; i < grad.size(); ++i)
    {
        grad[i] = Array<OneD, NekDouble>(npoints);
    }

    if (stress.size() < m_spacedim * m_spacedim)
    {
        stress = Array<OneD, Array<OneD, NekDouble>>(m_spacedim * m_spacedim);
    }

    for (size_t i = 0; i < stress.size(); ++i)
    {
        if (stress[i].size() < npoints)
        {
            stress[i] = Array<OneD, NekDouble>(npoints);
        }
    }

    for (int i = 0; i < m_spacedim; ++i)
    {
        for (int j = 0; j < m_spacedim; ++j)
        {
            exp[i]->PhysDeriv(j, vel[i], grad[i * m_spacedim + j]);
        }
    }

    // Compute stress component terms
    //    tau_ij = mu*(u_i,j + u_j,i) + mu*lambda*delta_ij*div(u)
    for (int i = 0; i < m_spacedim; ++i)
    {
        for (int j = i; j < m_spacedim; ++j)
        {
            Vmath::Vadd(npoints, grad[i * m_spacedim + j], 1,
                        grad[j * m_spacedim + i], 1, stress[i * m_spacedim + j],
                        1);

            Vmath::Smul(npoints, GetViscosity(), stress[i * m_spacedim + j], 1,
                        stress[i * m_spacedim + j], 1);
        }
    }
    for (int i = 0; i < m_spacedim; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            Vmath::Vcopy(npoints, stress[j * m_spacedim + i], 1,
                         stress[i * m_spacedim + j], 1);
        }
    }
}

void ProcessForceDecompose::GetLaplaceVelocity(
    const Array<OneD, MultiRegions::ExpListSharedPtr> exp,
    Array<OneD, Array<OneD, NekDouble>> &lapvel)
{
    int npoints = exp[0]->GetNpoints();
    Array<OneD, Array<OneD, NekDouble>> vel;
    GetVelocity(exp, vel);
    if (lapvel.size() != vel.size())
    {
        lapvel = Array<OneD, Array<OneD, NekDouble>>(vel.size());
    }
    for (size_t i = 0; i < lapvel.size(); ++i)
    {
        if (lapvel[i].size() < npoints)
        {
            lapvel[i] = Array<OneD, NekDouble>(npoints);
        }
    }
    Array<OneD, Array<OneD, NekDouble>> grad(m_spacedim);
    Array<OneD, NekDouble> tmp(npoints);
    for (int i = 0; i < m_spacedim; ++i)
    {
        grad[i] = Array<OneD, NekDouble>(npoints);
    }
    for (size_t i = 0; i < lapvel.size(); ++i)
    {
        Vmath::Zero(npoints, lapvel[i], 1);
        for (int j = 0; j < m_spacedim; ++j)
        {
            exp[i]->PhysDeriv(j, vel[i], grad[j]);
            exp[i]->PhysDeriv(j, grad[j], tmp);
            Vmath::Vadd(npoints, tmp, 1, lapvel[i], 1, lapvel[i], 1);
        }
    }
}

NekDouble ProcessForceDecompose::PhysIntegral(
    MultiRegions::ExpListSharedPtr exp, Array<OneD, NekDouble> value)
{
    NekDouble sum = 0.;
    for (int i = 0; i < exp->GetExpSize(); ++i)
    {
        sum += exp->GetExp(i)->Integral(value + exp->GetPhys_Offset(i));
    }
    return sum;
}

void ProcessForceDecompose::VolumeIntegrateForce(
    const MultiRegions::ExpListSharedPtr &field,
    const Array<OneD, Array<OneD, NekDouble>> &data,
    const LibUtilities::CommSharedPtr &comm, std::map<int, std::string> &Iphi,
    Array<OneD, NekDouble> &BoundBox, int dir)
{
    int Nslots   = 100;
    int ndata    = data.size();
    NekDouble dh = (BoundBox[dir * 2 + 1] - BoundBox[dir * 2]) / Nslots;
    Array<OneD, Array<OneD, NekDouble>> slots(ndata);
    for (int i = 0; i < ndata; ++i)
    {
        slots[i] = Array<OneD, NekDouble>(Nslots, 0.);
    }

    for (size_t e = 0; e < field->GetExpSize(); ++e)
    {
        LocalRegions::ExpansionSharedPtr exp = field->GetExp(e);
        SpatialDomains::Geometry *geom       = exp->GetGeom();
        int nv                               = geom->GetNumVerts();
        NekDouble gc[3]                      = {0., 0., 0.};
        NekDouble gct[3]                     = {0., 0., 0.};
        bool inside                          = false;
        for (size_t j = 0; j < nv; ++j)
        {
            SpatialDomains::PointGeom *vertex = geom->GetVertex(j);
            vertex->GetCoords(gct[0], gct[1], gct[2]);
            gc[0] += gct[0] / NekDouble(nv);
            gc[1] += gct[1] / NekDouble(nv);
            gc[2] += gct[2] / NekDouble(nv);
            if (BoundBox[0] <= gct[0] && gct[0] <= BoundBox[1] &&
                BoundBox[2] <= gct[1] && gct[1] <= BoundBox[3] &&
                BoundBox[4] <= gct[2] && gct[2] <= BoundBox[5])
            {
                inside = true;
            }
        }
        if (inside)
        {
            int index = floor((gc[dir] - BoundBox[2 * dir]) / dh);
            index     = std::max(std::min(index, Nslots - 1), 0);
            for (int j = 0; j < ndata; ++j)
            {
                slots[j][index] +=
                    exp->Integral(data[j] + field->GetPhys_Offset(e));
            }
        }
    }
    for (int j = 0; j < ndata; ++j)
    {
        comm->AllReduce(slots[j], LibUtilities::ReduceSum);
    }
    if (comm->TreatAsRankZero())
    {
        Array<OneD, NekDouble> sum(ndata, 0.);
        cout << "variables = x";
        for (const auto &phi : Iphi)
        {
            cout << ", f" + phi.second;
        }
        cout << "\n";
        for (int i = 0; i < Nslots; ++i)
        {
            cout << (BoundBox[2 * dir] + (i + 0.5) * dh) << " ";
            for (int j = 0; j < ndata; ++j)
            {
                sum[j] += slots[j][i];
                cout << sum[j] << " ";
            }
            cout << "\n";
        }
        cout << endl;
    }
}

} // namespace Nektar::FieldUtils
