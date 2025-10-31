////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessForceDecomposeVol.cpp
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
//  Description: Computes volume force elements using weighted pressure source
//  theory.
//
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
using namespace std;

#include <LibUtilities/BasicUtils/ParseUtils.h>
#include <LibUtilities/BasicUtils/SharedArray.hpp>

#include "ProcessForceDecomposeVol.h"

namespace Nektar::FieldUtils
{

ModuleKey ProcessForceDecomposeVol::className =
    GetModuleFactory().RegisterCreatorFunction(
        ModuleKey(eProcessModule, "FDecomposeVol"),
        ProcessForceDecomposeVol::create, "Computes volume force elements.");

ProcessForceDecomposeVol::ProcessForceDecomposeVol(FieldSharedPtr f)
    : ProcessForceDecompose(f)
{
    m_config["Q"] = ConfigOption(
        false, "0", "Method to obtain Q: 0 user provided; 1 from pressure;");
    m_config["box"]     = ConfigOption(false, "NotSet",
                                       "Bounding box of the integration domain "
                                           "box=xmin,xmax,ymin,ymax,zmin,zmax");
    m_config["scandir"] = ConfigOption(false, "NotSet", "Scan direction");
}

ProcessForceDecomposeVol::~ProcessForceDecomposeVol()
{
}

void ProcessForceDecomposeVol::v_Process(po::variables_map &vm)
{
    m_f->SetUpExp(vm);

    int nfields = m_f->m_variables.size();
    m_expdim    = m_f->m_graph->GetSpaceDimension();
    m_spacedim  = m_expdim + m_f->m_numHomogeneousDir;
    ASSERTL0(m_spacedim == 3 || m_spacedim == 2,
             "ProcessForceDecomposeVol must be computed for a 2D, quasi-3D, or "
             "3D case.");

    // Skip in case of empty partition
    if (m_f->m_exp[0]->GetNumElmts() == 0)
    {
        return;
    }

    Array<OneD, NekDouble> BoundBox;
    if (m_config["box"].as<string>().compare("NotSet") != 0)
    {
        vector<NekDouble> values;
        ParseUtils::GenerateVector(m_config["box"].as<string>(), values);

        ASSERTL0(
            values.size() == 6,
            "box string should contain 6 values xmin,xmax,ymin,ymax,zmin,zmax");
        BoundBox = Array<OneD, NekDouble>(values.size());
        for (size_t i = 0; i < values.size(); ++i)
        {
            BoundBox[i] = values[i];
        }
    }
    else
    {
        BoundBox    = Array<OneD, NekDouble>(6);
        BoundBox[0] = -3.;
        BoundBox[1] = 6.;
        BoundBox[2] = -3.;
        BoundBox[3] = 3.;
        BoundBox[4] = -1.;
        BoundBox[5] = 1.;
    }
    int scanDir = 0;
    if (m_config["scandir"].as<string>().compare("NotSet") != 0)
    {
        scanDir = m_config["scandir"].as<int>();
    }

    std::map<int, std::string> Iphi;
    for (size_t i = 0; i < m_f->m_variables.size(); ++i)
    {
        if (std::string::npos != m_f->m_variables[i].find("phi"))
        {
            Iphi[i] = m_f->m_variables[i];
        }
    }
    int addfields = Iphi.size();

    int npoints = m_f->m_exp[0]->GetNpoints();

    int nstrips;
    m_f->m_session->LoadParameter("Strip_Z", nstrips, 1);

    vector<MultiRegions::ExpListSharedPtr> Exp(nstrips * addfields);

    for (int s = 0; s < nstrips; ++s) // homogeneous strip varient
    {
        Array<OneD, MultiRegions::ExpListSharedPtr> exp(nfields);
        for (int i = 0; i < nfields; ++i)
        {
            exp[i] = m_f->m_exp[s * nfields + i];
        }
        Array<OneD, NekDouble> Q(npoints);
        GetQ(exp, Q);
        Array<OneD, Array<OneD, NekDouble>> Qphi(Iphi.size());
        int i = 0;
        for (const auto &phi : Iphi)
        {
            Qphi[i] = Array<OneD, NekDouble>(npoints, 0.);
            Vmath::Vmul(npoints, Q, 1,
                        m_f->m_exp[s * nfields + phi.first]->GetPhys(), 1,
                        Qphi[i], 1);
            Vmath::Smul(npoints, 2., Qphi[i], 1, Qphi[i], 1);

            Exp[s * addfields + i] =
                m_f->AppendExpList(m_f->m_numHomogeneousDir);
            Vmath::Vcopy(npoints, Qphi[i], 1,
                         Exp[s * addfields + i]->UpdatePhys(), 1);
            Exp[s * addfields + i]->FwdTransLocalElmt(
                Qphi[i], Exp[s * addfields + i]->UpdateCoeffs());

            auto it =
                m_f->m_exp.begin() + s * (nfields + addfields) + nfields + i;
            m_f->m_exp.insert(it, Exp[s * addfields + i]);
            ++i;
        }
        // process force
        VolumeIntegrateForce(m_f->m_exp[0], Qphi, m_f->m_comm, Iphi, BoundBox,
                             scanDir);
    }

    for (const auto &phi : Iphi)
    {
        m_f->m_variables.push_back("Q" + phi.second);
    }
}

} // namespace Nektar::FieldUtils
