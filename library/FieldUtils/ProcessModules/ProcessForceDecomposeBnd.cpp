////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessForceDecomposeBnd.cpp
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
//  Description: Computes boundary force elements using weighted pressure source
//  theory.
//
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>

#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <MultiRegions/ExpList.h>

#include "ProcessForceDecomposeBnd.h"

using namespace std;

namespace Nektar::FieldUtils
{

ModuleKey ProcessForceDecomposeBnd::className =
    GetModuleFactory().RegisterCreatorFunction(
        ModuleKey(eProcessModule, "FDecomposeBnd"),
        ProcessForceDecomposeBnd::create, "Computes boundary force elements.");

ProcessForceDecomposeBnd::ProcessForceDecomposeBnd(FieldSharedPtr f)
    : ProcessForceDecompose(f)
{
    f->m_declareExpansionAsContField = true;
    f->m_requireBoundaryExpansion    = true;
}

ProcessForceDecomposeBnd::~ProcessForceDecomposeBnd()
{
}

void ProcessForceDecomposeBnd::v_Process(po::variables_map &vm)
{
    m_f->SetUpExp(vm);

    if (m_f->m_exp[0]->GetNumElmts() != 0)
    {
        for (size_t i = 0; i < m_f->m_exp.size(); ++i)
        {
            m_f->m_exp[i]->FillBndCondFromField(m_f->m_exp[i]->GetCoeffs());
        }
    }

    int nfields = m_f->m_variables.size();
    m_expdim    = m_f->m_graph->GetSpaceDimension();
    m_spacedim  = m_expdim + m_f->m_numHomogeneousDir;
    if (m_f->m_exp[0]->GetNumElmts() == 0)
    {
        return;
    }
    if (m_spacedim == 1)
    {
        NEKERROR(
            ErrorUtil::efatal,
            "Error: Force decomposition for a 1D problem cannot be computed");
    }

    // Create map of boundary ids for partitioned domains
    int nBnds = 0, cnt = 0, Nphi = 0;
    map<int, int> BndRegionMap;
    SpatialDomains::BoundaryConditions bcs(m_f->m_session,
                                           m_f->m_exp[0]->GetGraph());
    const SpatialDomains::BoundaryRegionCollection bregions =
        bcs.GetBoundaryRegions();
    for (auto &breg_it : bregions)
    {
        nBnds                       = max(nBnds, breg_it.first);
        BndRegionMap[breg_it.first] = cnt++;
    }
    // assuming all boundary regions are consecutive number if
    // regions is one more than maximum id
    nBnds++;
    // not all partitions in parallel touch all boundaries so
    // find maximum number of boundaries
    m_f->m_comm->GetSpaceComm()->AllReduce(nBnds, LibUtilities::ReduceMax);

    std::map<int, std::string> Infophi;
    GetInfoPhi(Infophi);
    Nphi = Infophi.size();

    // Declare arrays
    Array<OneD, MultiRegions::ExpListSharedPtr> BndExp(nfields);
    Array<OneD, MultiRegions::ExpListSharedPtr> BndElmtExp(nfields);

    Array<OneD, NekDouble> friction(nBnds * m_spacedim,
                                    std::numeric_limits<NekDouble>::lowest());
    Array<OneD, NekDouble> vispress(nBnds * Nphi,
                                    std::numeric_limits<NekDouble>::lowest());
    Array<OneD, NekDouble> acceleration(
        nBnds * Nphi, std::numeric_limits<NekDouble>::lowest());

    // Loop over boundaries to Write, boundary integral
    for (const auto &bit : bregions)
    {
        int b = BndRegionMap[bit.first];
        // Get expansion list for boundary and for elements containing this bnd
        for (int i = 0; i < nfields; i++)
        {
            BndExp[i] = m_f->m_exp[i]->UpdateBndCondExpansion(b);
        }
        for (int i = 0; i < nfields; i++)
        {
            m_f->m_exp[i]->GetBndElmtExpansion(b, BndElmtExp[i]);
        }

        // Get number of points in expansions
        int nqb = BndExp[0]->GetTotPoints();

        // Initialise local arrays for the velocity gradients, and
        // stress components size of total number of quadrature
        // points for elements in this bnd
        Array<OneD, Array<OneD, NekDouble>> gradp, stress, lapvel, phi;
        GetPhi(BndElmtExp, phi, Infophi);
        GetGradPressure(BndElmtExp, gradp);
        GetStressTensor(BndElmtExp, stress);
        GetLaplaceVelocity(BndElmtExp, lapvel);

        // Get boundary values.
        Array<OneD, Array<OneD, NekDouble>> normals;
        Array<OneD, NekDouble> normLapVel(nqb, 0.), normp(nqb, 0.),
            norma(nqb, 0.), tmp(nqb, 0.);
        Array<OneD, Array<OneD, NekDouble>> bndphi(phi.size());
        Array<OneD, Array<OneD, NekDouble>> bndvalue(m_spacedim * m_spacedim);
        for (size_t i = 0; i < bndvalue.size(); ++i)
        {
            bndvalue[i] = Array<OneD, NekDouble>(nqb);
        }
        for (size_t i = 0; i < bndphi.size(); ++i)
        {
            bndphi[i] = Array<OneD, NekDouble>(nqb);
        }
        // Reverse normals, to get correct orientation for the body
        m_f->m_exp[0]->GetBoundaryNormals(b, normals);
        for (int i = 0; i < m_spacedim; ++i)
        {
            Vmath::Neg(nqb, normals[i], 1);
        }
        // phi on boundary
        for (size_t i = 0; i < phi.size(); ++i)
        {
            m_f->m_exp[0]->ExtractElmtToBndPhys(b, phi[i], bndphi[i]);
        }
        // wall shear stress
        for (size_t i = 0; i < stress.size(); ++i)
        {
            m_f->m_exp[0]->ExtractElmtToBndPhys(b, stress[i], bndvalue[i]);
        }
        for (int i = 0; i < m_spacedim; ++i)
        {
            Vmath::Zero(nqb, tmp, 1);
            for (int j = 0; j < m_expdim; ++j)
            {
                Vmath::Vvtvp(nqb, normals[j], 1, bndvalue[i * m_spacedim + j],
                             1, tmp, 1, tmp, 1);
            }
            friction[i + bit.first * m_spacedim] = BndExp[i]->Integral(tmp);
        }
        // wall pressure viscous
        for (int i = 0; i < m_expdim; ++i)
        {
            m_f->m_exp[0]->ExtractElmtToBndPhys(b, lapvel[i], bndvalue[i]);
            Vmath::Vvtvp(nqb, normals[i], 1, bndvalue[i], 1, normLapVel, 1,
                         normLapVel, 1);
        }
        Vmath::Smul(nqb, GetViscosity(), normLapVel, 1, normLapVel, 1);
        for (int i = 0; i < Nphi; ++i)
        {
            Vmath::Vmul(nqb, bndphi[i], 1, normLapVel, 1, tmp, 1);
            vispress[i + bit.first * Nphi] = BndExp[i]->Integral(tmp);
        }
        // wall acceleration
        for (int i = 0; i < m_expdim; ++i)
        {
            m_f->m_exp[0]->ExtractElmtToBndPhys(b, gradp[i], bndvalue[i]);
            Vmath::Vvtvp(nqb, normals[i], 1, bndvalue[i], 1, normp, 1, normp,
                         1);
        }
        Vmath::Vsub(nqb, normLapVel, 1, normp, 1, norma, 1);
        for (int i = 0; i < Nphi; ++i)
        {
            Vmath::Vmul(nqb, bndphi[i], 1, norma, 1, tmp, 1);
            acceleration[i + bit.first * Nphi] = -BndExp[i]->Integral(tmp);
        }
    }

    m_f->m_comm->AllReduce(friction, LibUtilities::ReduceMax);
    m_f->m_comm->AllReduce(vispress, LibUtilities::ReduceMax);
    m_f->m_comm->AllReduce(acceleration, LibUtilities::ReduceMax);

    if (m_f->m_comm->TreatAsRankZero())
    {
        cout << "Force decomposition results\n";
        cout << "number of boundaries " << nBnds << "\n";
        cout << "space dimension " << m_spacedim << "\n";
        cout << "number of phi " << Nphi << "\n";
        cout << "Friction force (x, y, ...)[" + std::to_string(m_spacedim) +
                    "] on all boundaries \n FRIC ";
        for (const auto f : friction)
        {
            cout << f << " ";
        }
        cout << "\n";
        cout << "Visouce pressure force (phi0, phi1, ...)[" +
                    std::to_string(Nphi) +
                    "] on all "
                    "boundaries \n VISP ";
        for (const auto f : vispress)
        {
            cout << f << " ";
        }
        cout << "\n";
        cout << "Wall acceleratrion (phi0, phi1, ...)[" + std::to_string(Nphi) +
                    "] on all boundaries "
                    "\n ACCF ";
        for (const auto f : acceleration)
        {
            cout << f << " ";
        }
        cout << "\n";
    }
}

} // namespace Nektar::FieldUtils
