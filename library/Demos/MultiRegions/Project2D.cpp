///////////////////////////////////////////////////////////////////////////////
//
// File: Project2D.cpp
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
    CustomMeshGraphIO(NekDouble a, NekDouble b, NekDouble c, NekDouble d,
                      int nx, int ny, int nmodes,
                      LibUtilities::ShapeType shapeType)
        : MeshGraphIO(), m_a(a), m_b(b), m_c(c), m_d(d), m_nx(nx), m_ny(ny),
          m_nmodes(nmodes), m_shapeType(shapeType)
    {
        m_meshGraph =
            MemoryManager<SpatialDomains::MeshGraph>::AllocateSharedPtr();
    }

    SpatialDomains::MeshGraphSharedPtr CreateGeometry(
        [[maybe_unused]] LibUtilities::DomainRangeShPtr rng,
        [[maybe_unused]] bool fillGraph)
    {
        int meshDimension  = 2;
        int spaceDimension = 2;
        m_meshGraph->SetMeshDimension(meshDimension);
        m_meshGraph->SetSpaceDimension(spaceDimension);

        // Create a bunch of point geometries
        for (int i = 0; i < m_nx + 1; ++i)
        {
            for (int j = 0; j < m_ny + 1; ++j)
            {
                auto geom = ObjPoolManager<SpatialDomains::PointGeom>::
                    AllocateUniquePtr(spaceDimension, MyPntId(i, j),
                                      m_a + i * (m_b - m_a) / m_nx,
                                      m_c + j * (m_d - m_c) / m_ny, 0.0);
                m_meshGraph->AddGeom(MyPntId(i, j), std::move(geom));
            }
        }

        // Create a bunch of segment geometries,
        // first the segments parallel to y
        for (int i = 0; i < m_nx + 1; ++i)
        {
            for (int j = 0; j < m_ny; ++j)
            {
                std::array<SpatialDomains::PointGeom *, 2> pts = {
                    m_meshGraph->GetPointGeom(MyPntId(i, j)),
                    m_meshGraph->GetPointGeom(MyPntId(i, j + 1))};
                auto geom =
                    ObjPoolManager<SpatialDomains::SegGeom>::AllocateUniquePtr(
                        MyYSegId(i, j), spaceDimension, pts);
                m_meshGraph->AddGeom(MyYSegId(i, j), std::move(geom));
            }
        }
        // then the segments parallel to x
        for (int i = 0; i < m_nx; ++i)
        {
            for (int j = 0; j < m_ny + 1; ++j)
            {
                std::array<SpatialDomains::PointGeom *, 2> pts = {
                    m_meshGraph->GetPointGeom(MyPntId(i, j)),
                    m_meshGraph->GetPointGeom(MyPntId(i + 1, j))};
                auto geom =
                    ObjPoolManager<SpatialDomains::SegGeom>::AllocateUniquePtr(
                        MyXSegId(i, j), spaceDimension, pts);
                m_meshGraph->AddGeom(MyXSegId(i, j), std::move(geom));
            }
        }
        // then the diagonal segments if triangular elements
        if (m_shapeType == LibUtilities::eTriangle)
        {
            for (int i = 0; i < m_nx; ++i)
            {
                for (int j = 0; j < m_ny; ++j)
                {
                    std::array<SpatialDomains::PointGeom *, 2> pts = {
                        m_meshGraph->GetPointGeom(MyPntId(i, j)),
                        m_meshGraph->GetPointGeom(MyPntId(i + 1, j + 1))};
                    auto geom = ObjPoolManager<SpatialDomains::SegGeom>::
                        AllocateUniquePtr(MyDiagSegId(i, j), spaceDimension,
                                          pts);
                    m_meshGraph->AddGeom(MyDiagSegId(i, j), std::move(geom));
                }
            }
        }

        switch (m_shapeType)
        {
            case LibUtilities::eTriangle:
            {
                // Create a bunch of upper and lower triangular geometries,
                // listing segments anticlockwise.
                for (int i = 0; i < m_nx; ++i)
                {
                    for (int j = 0; j < m_ny; ++j)
                    {
                        std::array<SpatialDomains::SegGeom *, 3> u_edges = {
                            m_meshGraph->GetSegGeom(MyYSegId(i, j)),
                            m_meshGraph->GetSegGeom(MyDiagSegId(i, j)),
                            m_meshGraph->GetSegGeom(MyXSegId(i, j + 1))};
                        auto tri1 = ObjPoolManager<SpatialDomains::TriGeom>::
                            AllocateUniquePtr(MyUpperTriId(i, j), u_edges);
                        m_meshGraph->AddGeom(MyUpperTriId(i, j),
                                             std::move(tri1));

                        std::array<SpatialDomains::SegGeom *, 3> l_edges = {
                            m_meshGraph->GetSegGeom(MyXSegId(i, j)),
                            m_meshGraph->GetSegGeom(MyYSegId(i + 1, j)),
                            m_meshGraph->GetSegGeom(MyDiagSegId(i, j))};
                        auto tri2 = ObjPoolManager<SpatialDomains::TriGeom>::
                            AllocateUniquePtr(MyLowerTriId(i, j), l_edges);
                        m_meshGraph->AddGeom(MyLowerTriId(i, j),
                                             std::move(tri2));
                    }
                }
            }
            break;
            case LibUtilities::eQuadrilateral:
            {
                // Create a bunch of quadrilateral geometries, listing segments
                // anticlockwise.
                for (int i = 0; i < m_nx; ++i)
                {
                    for (int j = 0; j < m_ny; ++j)
                    {
                        std::array<SpatialDomains::SegGeom *, 4> edges = {
                            m_meshGraph->GetSegGeom(MyYSegId(i, j)),
                            m_meshGraph->GetSegGeom(MyXSegId(i, j)),
                            m_meshGraph->GetSegGeom(MyYSegId(i + 1, j)),
                            m_meshGraph->GetSegGeom(MyXSegId(i, j + 1))};
                        auto quad = ObjPoolManager<SpatialDomains::QuadGeom>::
                            AllocateUniquePtr(MyQuadId(i, j), edges);
                        m_meshGraph->AddGeom(MyQuadId(i, j), std::move(quad));
                    }
                }
            }
            break;
            default:
                NEKERROR(ErrorUtil::efatal, "Shape type not implemented.");
                break;
        }

        // Set up a composite for the domain, like ReadComposites() followed by
        // ReadDomain()
        SpatialDomains::CompositeSharedPtr comp =
            MemoryManager<SpatialDomains::Composite>::AllocateSharedPtr();
        switch (m_shapeType)
        {
            case LibUtilities::eTriangle:
            {
                for (int i = 0; i < m_nx; ++i)
                {
                    for (int j = 0; j < m_ny; ++j)
                    {
                        comp->m_geomVec.push_back(
                            m_meshGraph->GetTriGeom(MyUpperTriId(i, j)));
                    }
                }
                for (int i = 0; i < m_nx; ++i)
                {
                    for (int j = 0; j < m_ny; ++j)
                    {
                        comp->m_geomVec.push_back(
                            m_meshGraph->GetTriGeom(MyLowerTriId(i, j)));
                    }
                }
            }
            break;
            case LibUtilities::eQuadrilateral:
            {
                for (int i = 0; i < m_nx; ++i)
                {
                    for (int j = 0; j < m_ny; ++j)
                    {
                        comp->m_geomVec.push_back(
                            m_meshGraph->GetQuadGeom(MyQuadId(i, j)));
                    }
                }
            }
            break;
            default:
                NEKERROR(ErrorUtil::efatal, "Shape type not implemented.");
                break;
        }
        auto &meshComposites = m_meshGraph->GetComposites();
        meshComposites[0]    = comp;
        auto &domain         = m_meshGraph->GetDomain();
        domain[0]            = meshComposites;

        // Instead of MeshGraph::ReadExpansionInfo() in MeshGraph::FillGraph(),
        // we set expansion info here:
        // First initialise expansion info arguments
        LibUtilities::ShapeType arg_shapeType = m_shapeType;
        vector<unsigned int> arg_elementIDs;
        for (unsigned int id = 0; id < comp->m_geomVec.size(); ++id)
        {
            arg_elementIDs.push_back(id);
        };
        vector<LibUtilities::BasisType> arg_basis = {LibUtilities::eModified_A};
        switch (m_shapeType)
        {
            case LibUtilities::eTriangle:
            {
                arg_basis.push_back(LibUtilities::eModified_B);
            }
            break;
            case LibUtilities::eQuadrilateral:
            {
                arg_basis.push_back(LibUtilities::eModified_A);
            }
            break;
            default:
                NEKERROR(ErrorUtil::efatal, "Shape type not implemented.");
                break;
        }
        bool arg_uniOrder = true;
        vector<unsigned int> arg_numModes(comp->m_geomVec.size(), m_nmodes);
        vector<string> arg_fields(1, "u");
        // Then store them in a fielddefs object and pass it to
        // SetExpansionInfo()
        vector<LibUtilities::FieldDefinitionsSharedPtr> fielddefs = {
            MemoryManager<LibUtilities::FieldDefinitions>::AllocateSharedPtr(
                arg_shapeType, arg_elementIDs, arg_basis, arg_uniOrder,
                arg_numModes, arg_fields)};
        m_meshGraph->SetExpansionInfo(fielddefs);

        // 2D case from MeshGraph::FillGraph()
        for (auto [id, tri] :
             m_meshGraph->GetGeomMap<SpatialDomains::TriGeom>())
        {
            tri->Setup();
        }
        for (auto [id, quad] :
             m_meshGraph->GetGeomMap<SpatialDomains::QuadGeom>())
        {
            quad->Setup();
        }

        return m_meshGraph;
    }

protected:
    /// storage for endpoints
    NekDouble m_a, m_b, m_c, m_d;
    /// number of quad elements (or pairs of triangle elements) in x and y
    /// directions
    int m_nx, m_ny;
    /// number of modes in x and y directions
    int m_nmodes;
    /// shape of element used
    LibUtilities::ShapeType m_shapeType;

    // Functions for the IDs of points, segments, quads, and triangles
    int MyPntId(int i, int j)
    {
        return j + ((m_ny + 1) * i);
    }
    int MyYSegId(int i, int j)
    {
        return j + (m_ny * i);
    }
    int MyXSegId(int i, int j)
    {
        return j + ((m_ny + 1) * i) + (m_ny * (m_nx + 1));
    }
    int MyDiagSegId(int i, int j)
    {
        return j + (m_ny * i) + ((m_ny + 1) * m_nx) + (m_ny * (m_nx + 1));
    }
    int MyQuadId(int i, int j)
    {
        return j + (m_ny * i);
    }
    int MyUpperTriId(int i, int j)
    {
        return j + (m_ny * i);
    }
    int MyLowerTriId(int i, int j)
    {
        return j + (m_ny * i) + (m_ny * m_nx);
    }

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
    exp->GetCoords(xc0, xc1);
    Vmath::Zero(nq, &xc2[0], 1);

    // Define a function to be projected. In this case we select
    // sin(2*pi*x)+sin(2*pi*y)
    for (int n = 0; n < nq; ++n)
    {
        sol[n] = 0.0;
        sol[n] += sin(2 * M_PI * xc0[n]);
        sol[n] += sin(2 * M_PI * xc1[n]);
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
string arg = LibUtilities::SessionReader::RegisterCmdLineFlag(
    "local", "l",
    "Perform local projection instead of the default continuous projection");
// Add command line option for triangular vs quad elements
string arg1 = LibUtilities::SessionReader::RegisterCmdLineFlag(
    "triangles", "t",
    "Use triangular elements instead of the default quadrilateral elements");
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
// Add command line argument for y-coord of bottom bound
string arg5 = LibUtilities::SessionReader::RegisterCmdLineArgument(
    "boundb", "", "Sets y position of bottom bound, default is 0");
// Add command line argument for y-coord of top bound
string arg6 = LibUtilities::SessionReader::RegisterCmdLineArgument(
    "boundt", "", "Sets y position of top bound, default is 1");

// This routine projects a polynomial using various numbers of elements and
// reports the degree of convergence.
int main(int argc, char **argv)
{
    // Read in the blank session file.
    LibUtilities::SessionReaderSharedPtr session =
        LibUtilities::SessionReader::CreateInstance(argc, argv);
    bool local     = session->DefinesCmdLineArgument("local");
    bool triangles = session->DefinesCmdLineArgument("triangles");
    cout << (local ? "Local" : "Continuous") << " projection" << endl;
    session->InitSession();

    // Choose properties; shapeType, numModes and bounds can be set from command
    // line.
    auto shapeType = LibUtilities::eQuadrilateral;
    if (triangles)
    {
        shapeType = LibUtilities::eTriangle;
    }
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
    NekDouble bottomBound = 0;
    if (session->DefinesCmdLineArgument("boundb"))
    {
        bottomBound = stod(session->GetCmdLineArgument<string>("boundb"));
    }
    NekDouble topBound = 1;
    if (session->DefinesCmdLineArgument("boundt"))
    {
        topBound = stod(session->GetCmdLineArgument<string>("boundt"));
    }
    ASSERTL0(bottomBound < topBound, "bottomBound must be less than topBound");
    // numXAreasVec and numXAreasVec dictate in each run how many quads or pairs
    // of triangles are in each direction
    vector<int> numXAreasVec = {
        10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    };
    vector<int> numYAreasVec = {
        10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    };
    ASSERTL0(numXAreasVec.size() == numYAreasVec.size(),
             "Conflicting numbers of runs in numXAreasVec and numYAreasVec");
    int numRuns = numXAreasVec.size();
    // Initialise vector to store values of log(1/h)
    vector<NekDouble> logNumAreas(numRuns);
    // Initialise vector to store values of log(L 2 error)
    vector<NekDouble> logL2Error(numRuns);
    for (int n = 0; n < numRuns; ++n)
    {
        // Generate the mesh.
        auto graphIO = MemoryManager<CustomMeshGraphIO>::AllocateSharedPtr(
            leftBound, rightBound, bottomBound, topBound, numXAreasVec[n],
            numYAreasVec[n], numModes, shapeType);
        SpatialDomains::MeshGraphSharedPtr graph =
            graphIO->CreateGeometry(LibUtilities::NullDomainRangeShPtr, false);

        if (n == 0)
        {
            cout << graph->GetMeshDimension() << "D "
                 << (triangles ? "triangles, " : "quads, ") << numModes
                 << " modes." << endl;
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
        logNumAreas[n] = log(numXAreasVec[n] / (rightBound - leftBound) +
                             numYAreasVec[n] / (topBound - bottomBound));
        logL2Error[n]  = log(L_errors[1]);
    }

    // Print results
    cout << endl << "numXAreas, numYAreas, logNumAreas, logL2Error" << endl;
    for (int n = 0; n < numRuns; ++n)
    {
        cout << numXAreasVec[n] << "   " << numYAreasVec[n] << "   ";
        cout << logNumAreas[n] << "   " << logL2Error[n] << endl;
    }
    vector<NekDouble> linreg = LinLeastSq(logNumAreas, logL2Error);
    // Make cout temporarily use more decimal places
    ios_base::fmtflags originalFlags = cout.flags();
    cout << std::setprecision(10) << fixed;
    cout << "Gradient: " << linreg[0] << endl;
    cout.flags(originalFlags);

    session->Finalise();
    return 0;
}
