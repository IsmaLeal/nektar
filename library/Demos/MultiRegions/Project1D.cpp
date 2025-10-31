///////////////////////////////////////////////////////////////////////////////
//
// File: Project1D.cpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <MultiRegions/ContField.h>
#include <SpatialDomains/MeshGraph.h>
#include <SpatialDomains/MeshGraphIO.h>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace Nektar;

// Custom MeshGraph class for programatically creating the mesh instead of
// reading from session.
class CustomMeshGraphIO : public SpatialDomains::MeshGraphIO
{
public:
    CustomMeshGraphIO(NekDouble a, NekDouble b, int nx, int nmodesx)
        : MeshGraphIO(), m_a(a), m_b(b), m_nx(nx), m_nmodesx(nmodesx)
    {
        m_meshGraph =
            MemoryManager<SpatialDomains::MeshGraph>::AllocateSharedPtr();
    }

    SpatialDomains::MeshGraphSharedPtr CreateGeometry(
        [[maybe_unused]] LibUtilities::DomainRangeShPtr rng,
        [[maybe_unused]] bool fillGraph)
    {
        int meshDimension  = 1;
        int spaceDimension = 1;
        m_meshGraph->SetMeshDimension(meshDimension);
        m_meshGraph->SetSpaceDimension(spaceDimension);

        // Create a bunch of point geometries
        for (int i = 0; i < m_nx + 1; ++i)
        {
            auto geom =
                ObjPoolManager<SpatialDomains::PointGeom>::AllocateUniquePtr(
                    spaceDimension, i, m_a + i * (m_b - m_a) / m_nx, 0.0, 0.0);
            m_meshGraph->AddGeom(i, std::move(geom));
        }

        // Create a bunch of segment geometries
        for (int i = 0; i < m_nx; ++i)
        {
            std::array<SpatialDomains::PointGeom *, 2> pts = {
                m_meshGraph->GetPointGeom(i), m_meshGraph->GetPointGeom(i + 1)};
            auto geom =
                ObjPoolManager<SpatialDomains::SegGeom>::AllocateUniquePtr(
                    i, spaceDimension, pts);
            m_meshGraph->AddGeom(i, std::move(geom));
        }

        // Set up a composite for the domain, like ReadComposites() followed by
        // ReadDomain()
        SpatialDomains::CompositeSharedPtr comp =
            MemoryManager<SpatialDomains::Composite>::AllocateSharedPtr();
        for (int i = 0; i < m_nx; ++i)
        {
            comp->m_geomVec.push_back(m_meshGraph->GetSegGeom(i));
        }
        auto &meshComposites = m_meshGraph->GetComposites();
        meshComposites[0]    = comp;
        auto &domain         = m_meshGraph->GetDomain();
        domain[0]            = meshComposites;

        // Instead of MeshGraph::ReadExpansionInfo() in MeshGraph::FillGraph(),
        // we set expansion info here:
        // First initialise expansion info arguments
        LibUtilities::ShapeType arg_shapeType =
            LibUtilities::ShapeType::eSegment;
        vector<unsigned int> arg_elementIDs;
        for (unsigned int i = 0; i < m_nx; ++i)
        {
            arg_elementIDs.push_back(i);
        };
        vector<LibUtilities::BasisType> arg_basis;
        arg_basis.push_back(LibUtilities::BasisType::eModified_A);
        bool arg_uniOrder = true;
        vector<unsigned int> arg_numModes(m_nx, m_nmodesx);
        vector<string> arg_fields(1, "u");
        // Then store them in a fielddefs object and pass it to
        // SetExpansionInfo()
        vector<LibUtilities::FieldDefinitionsSharedPtr> fielddefs;
        fielddefs.push_back(
            MemoryManager<LibUtilities::FieldDefinitions>::AllocateSharedPtr(
                arg_shapeType, arg_elementIDs, arg_basis, arg_uniOrder,
                arg_numModes, arg_fields));
        m_meshGraph->SetExpansionInfo(fielddefs);

        // 1D case from MeshGraph::FillGraph()
        for (auto [id, seg] :
             m_meshGraph->GetGeomMap<SpatialDomains::SegGeom>())
        {
            seg->Setup();
        }

        return m_meshGraph;
    }

protected:
    /// storage for endpoints
    NekDouble m_a, m_b;
    /// number of elements in x direction
    int m_nx;
    /// number of modes in x direction
    int m_nmodesx;

    // v_ReadGeometry, v_WriteGeometry and v_PartitionMesh should not be used by
    // this class
    void v_ReadGeometry([[maybe_unused]] bool fillGraph) override
    {
        NEKERROR(ErrorUtil::efatal, "Not implemented.");
    }
    void v_WriteGeometry(
        [[maybe_unused]] const string &outfilename,
        [[maybe_unused]] bool defaultExp = false,
        [[maybe_unused]] const LibUtilities::FieldMetaDataMap &metadata =
            LibUtilities::NullFieldMetaDataMap) override
    {
        NEKERROR(ErrorUtil::efatal, "Not implemented.");
    }
    void v_PartitionMesh(
        [[maybe_unused]] LibUtilities::SessionReaderSharedPtr session) override
    {
        NEKERROR(ErrorUtil::efatal, "Not implemented.");
    }
};

template <typename T> vector<NekDouble> Project(T exp)
{
    int nq = exp->GetTotPoints();

    // define coordinates and solution
    Array<OneD, NekDouble> sol, xc0, xc1, xc2;
    sol = Array<OneD, NekDouble>(nq);
    xc0 = Array<OneD, NekDouble>(nq);
    xc1 = Array<OneD, NekDouble>(nq);
    xc2 = Array<OneD, NekDouble>(nq);
    exp->GetCoords(xc0);
    Vmath::Zero(nq, &xc1[0], 1);
    Vmath::Zero(nq, &xc2[0], 1);

    // Define a function to be projected. In this case we select sin(2*pi*x)
    for (int i = 0; i < nq; ++i)
    {
        sol[i] = 0.0;
        sol[i] += sin(2 * M_PI * xc0[i]);
    }

    // Perform projection
    Array<OneD, NekDouble> approx(nq);
    exp->FwdTrans(sol, exp->UpdateCoeffs());
    exp->BwdTrans(exp->GetCoeffs(), approx);

    // Calculate L2 and Linf error
    vector<NekDouble> L_errors;
    L_errors.push_back(exp->Linf(approx, sol));
    L_errors.push_back(exp->L2(approx, sol));
    return L_errors;
}

// Perform linear least squares method to find line of best fit
vector<NekDouble> LinLeastSq(vector<NekDouble> x, vector<NekDouble> y)
{
    ASSERTL0(x.size() == y.size(), "x and y must be the same size.");
    int n = x.size();
    // Calculate sums of data needed for linear least squares formula
    NekDouble sumx  = 0;
    NekDouble sumy  = 0;
    NekDouble sumxy = 0;
    NekDouble sumxx = 0;
    NekDouble sumyy = 0;
    for (int i = 0; i < n; ++i)
    {
        sumx += x[i];
        sumy += y[i];
        sumxy += x[i] * y[i];
        sumxx += x[i] * x[i];
        sumyy += y[i] * y[i];
    };
    // Calculate gradient, intercept and correlation coefficient of fit
    NekDouble m = (n * sumxy - sumx * sumy) / (n * sumxx - sumx * sumx);
    NekDouble b = (sumy - m * sumx) / n;
    NekDouble r = (sumxy - sumx * sumy / n) /
                  sqrt((sumxx - sumx * sumx / n) * (sumyy - sumy * sumy / n));
    vector<NekDouble> result = {m, b, r};
    return result;
}

// Add command line option for local vs continuous projection
string arg1 = LibUtilities::SessionReader::RegisterCmdLineFlag(
    "local", "l",
    "Perform local projection instead of the default continuous projection");
// Add command line argument for number of modes
string arg2 = LibUtilities::SessionReader::RegisterCmdLineArgument(
    "nmodes", "n",
    "Set number of number of modes for projection, default is 6");
// Add command line argument for x-coord of left bound
string arg3 = LibUtilities::SessionReader::RegisterCmdLineArgument(
    "boundl", "", "Sets x position of left bound, default is 0");
// Add command line argument for x-coord of right bound
string arg4 = LibUtilities::SessionReader::RegisterCmdLineArgument(
    "boundr", "", "Sets x position of right bound, default is 1");

// This routine projects a polynomial using various numbers of elements and
// reports the degree of convergence.
int main(int argc, char **argv)
{
    // Read in the blank session file.
    LibUtilities::SessionReaderSharedPtr session =
        LibUtilities::SessionReader::CreateInstance(argc, argv);
    bool local = session->DefinesCmdLineArgument("local");
    cout << (local ? "Local" : "Continuous") << " projection" << endl;
    session->InitSession();

    // Choose properties; numModes and bounds can be set from command line.
    int numModes = 6;
    if (session->DefinesCmdLineArgument("nmodes"))
    {
        numModes = stoi(session->GetCmdLineArgument<string>("nmodes"));
    }
    ASSERTL0(numModes > 1, "Too few modes, must have at least 2.");
    NekDouble leftBound = 0;
    if (session->DefinesCmdLineArgument("boundl"))
    {
        leftBound = stod(session->GetCmdLineArgument<string>("boundl"));
    }
    NekDouble rightBound = 1;
    if (session->DefinesCmdLineArgument("boundr"))
    {
        rightBound = stod(session->GetCmdLineArgument<string>("boundr"));
    }
    ASSERTL0(leftBound < rightBound, "leftBound must be less than rightBound");
    vector<int> numSegmentsVec = {
        10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    };
    int numRuns = numSegmentsVec.size();
    // Initialise vector to store values of log(1/h)
    vector<NekDouble> xlog(numRuns);
    // Initialise vector to store values of log(L 2 error)
    vector<NekDouble> ylog(numRuns);
    for (int i = 0; i < numRuns; ++i)
    {
        // Generate the mesh.
        auto graphIO = MemoryManager<CustomMeshGraphIO>::AllocateSharedPtr(
            leftBound, rightBound, numSegmentsVec[i], numModes);
        SpatialDomains::MeshGraphSharedPtr graph =
            graphIO->CreateGeometry(LibUtilities::NullDomainRangeShPtr, false);

        if (i == 0)
        {
            cout << graph->GetMeshDimension() << "D, " << numModes << " modes."
                 << endl;
        }
        vector<NekDouble> L_errors;

        // Create an expansion list.
        if (local)
        {
            auto exp = MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
                session, graph);
            L_errors = Project<decltype(exp)>(exp);
        }
        else
        {
            auto exp =
                MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
                    session, graph);
            L_errors = Project<decltype(exp)>(exp);
        }

        // Store values
        xlog[i] = log(numSegmentsVec[i] / (rightBound - leftBound));
        ylog[i] = log(L_errors[1]);
    }

    // Print results
    cout << endl << "numSegments, xlog, ylog" << endl;
    for (int i = 0; i < numRuns; ++i)
    {
        cout << numSegmentsVec[i] << "   " << xlog[i] << "   " << ylog[i]
             << endl;
    }
    vector<NekDouble> linreg = LinLeastSq(xlog, ylog);
    // Make cout temporarily use more decimal places
    ios_base::fmtflags originalFlags = cout.flags();
    cout << std::setprecision(10) << fixed;
    cout << "Gradient: " << linreg[0] << endl;
    cout.flags(originalFlags);

    session->Finalise();
    return 0;
}
