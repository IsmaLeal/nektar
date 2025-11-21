///////////////////////////////////////////////////////////////////////////////
//
//  File: OutputNekpp.cpp
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
//  Description: Nektar++ file format output.
//
///////////////////////////////////////////////////////////////////////////////

#include <set>
#include <string>
#include <thread>

using namespace std;

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>

namespace io = boost::iostreams;

#include <LibUtilities/BasicUtils/CppCommandLine.hpp>
#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <NekMesh/MeshElements/Element.h>
#include <SpatialDomains/MeshGraph.h>
#include <SpatialDomains/MeshGraphIO.h>
#include <SpatialDomains/PointGeom.h>
#include <tinyxml.h>

#include "OutputNekpp.h"

using namespace Nektar::NekMesh;
using namespace Nektar::SpatialDomains;

namespace Nektar::NekMesh
{
ModuleKey OutputNekpp::className1 = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eOutputModule, "xml"), OutputNekpp::create,
    "Writes a Nektar++ xml file.");

ModuleKey OutputNekpp::className2 = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eOutputModule, "nekg"), OutputNekpp::create,
    "Writes a Nektar++ file with hdf5.");

OutputNekpp::OutputNekpp(MeshSharedPtr m) : OutputModule(m)
{
    m_config["chkbndcomp"] = ConfigOption(
        true, "0", "Put all undefined in a compoaite with id=9999");
    m_config["test"] = ConfigOption(
        true, "0", "Attempt to load resulting mesh and create meshgraph.");
    m_config["stats"] =
        ConfigOption(true, "0", "Print out basic mesh statistics.");
    m_config["uncompress"] = ConfigOption(true, "0", "Uncompress xml sections");
    m_config["order"] = ConfigOption(false, "-1", "Enforce a polynomial order");
    m_config["testcond"] = ConfigOption(false, "", "Test a condition.");
    m_config["varopti"] =
        ConfigOption(true, "0", "Run the variational optimser");
    m_config["orient"] = ConfigOption(true, "0", "Reorder Prisms and Tets");
}

OutputNekpp::~OutputNekpp()
{
}

template <typename T>
void TestElmts(SpatialDomains::GeomMapView<T> &geomMap,
               [[maybe_unused]] SpatialDomains::MeshGraphSharedPtr &graph,
               LibUtilities::Interpreter &strEval, int exprId, Logger &log)
{
    for (auto [id, geom] : geomMap)
    {
        geom->Setup();
        geom->FillGeom();

        if (exprId != -1)
        {
            int nq  = geom->GetXmap()->GetTotPoints();
            int dim = geom->GetCoordim();

            Array<OneD, Array<OneD, NekDouble>> coords(3);

            for (int i = 0; i < 3; ++i)
            {
                coords[i] = Array<OneD, NekDouble>(nq, 0.0);
            }

            for (int i = 0; i < dim; ++i)
            {
                geom->GetXmap()->BwdTrans(geom->GetCoeffs(i), coords[i]);
            }

            for (int i = 0; i < nq; ++i)
            {
                NekDouble output = strEval.Evaluate(
                    exprId, coords[0][i], coords[1][i], coords[2][i], 0.0);

                if (output != 1.0)
                {
                    log(FATAL) << "Output mesh failed coordinate test" << endl;
                }
            }

            // Also evaluate at mid-point to test for deformed vs. regular
            // elements.
            Array<OneD, NekDouble> eta(dim, 0.0), evalPt(3, 0.0);
            for (int i = 0; i < dim; ++i)
            {
                evalPt[i] = geom->GetXmap()->PhysEvaluate(eta, coords[i]);
            }

            NekDouble output =
                strEval.Evaluate(exprId, evalPt[0], evalPt[1], evalPt[2], 0.0);

            if (output != 1.0)
            {
                log(FATAL) << "Output mesh failed coordinate midpoint test"
                           << endl;
            }
        }
    }
}

void OutputNekpp::Process()
{
    string filename = m_config["outfile"].as<string>();

    m_log(VERBOSE) << "Writing Nektar++ file '" << filename << "'" << endl;

    // Check whether file exists.
    if (!CheckOverwrite(filename))
    {
        return;
    }

    int order = m_config["order"].as<int>();

    if (order != -1)
    {
        m_mesh->MakeOrder(order, LibUtilities::ePolyEvenlySpaced, m_log);
    }

    // Useful when doing r-adaptation
    if (m_config["varopti"].beenSet)
    {
        unsigned int np        = std::thread::hardware_concurrency();
        ModuleSharedPtr module = GetModuleFactory().CreateInstance(
            ModuleKey(eProcessModule, "varopti"), m_mesh);
        module->RegisterConfig("hyperelastic", "");
        module->RegisterConfig("numthreads", std::to_string(np));

        try
        {
            module->SetDefaults();
            module->Process();
        }
        catch (runtime_error &e)
        {
            m_log(WARNING) << "Variational optimisation has failed with "
                           << "message:" << endl;
            m_log(WARNING) << e.what() << endl;
            m_log(WARNING) << "The mesh will be written as is, it may be "
                           << "invalid" << endl;
            return;
        }
    }

    if (m_config["stats"].beenSet)
    {
        m_mesh->PrintStats(m_log);
    }

    if (m_config["orient"].beenSet)
    {
        PerMap empty;
        ReorderPrisms(empty);
    }

    // Default to compressed XML output.
    std::string type = "XmlCompressed";

    // Compress output and append .gz extension
    if (fs::path(filename).extension() == ".xml" &&
        m_config["uncompress"].beenSet)
    {
        type = "Xml";
    }
    else if (fs::path(filename).extension() == ".nekg")
    {
        type = "HDF5";
    }

    SpatialDomains::MeshGraphSharedPtr graph =
        MemoryManager<SpatialDomains::MeshGraph>::AllocateSharedPtr();
    graph->Empty(m_mesh->m_expDim, m_mesh->m_spaceDim);

    TransferVertices(graph);

    std::unordered_map<int, SegGeom *> segMap;
    TransferEdges(graph, segMap);
    TransferFaces(graph, segMap);
    TransferElements(graph);
    TransferCurves(graph);
    TransferComposites(graph);
    TransferDomain(graph);

    auto graphIO = SpatialDomains::GetMeshGraphIOFactory().CreateInstance(type);
    graphIO->SetMeshGraph(graph);
    graphIO->WriteGeometry(filename, true, m_mesh->m_metadata);

    // Test the resulting XML file (with a basic test) by loading it
    // with the session reader, generating the MeshGraph and testing if
    // each element is valid.
    if (m_config["test"].beenSet)
    {
        // Create an equation based on the test condition. Should evaluate to 1
        // or 0 using boolean logic.
        string testcond = m_config["testcond"].as<string>();
        int exprId      = -1;

        if (testcond.length() > 0)
        {
            exprId = m_strEval.DefineFunction("x y z", testcond);
        }

        vector<string> filenames(1);

        if (type == "HDF5")
        {
            vector<string> tmp;
            boost::split(tmp, filename, boost::is_any_of("."));
            filenames[0] = tmp[0] + ".xml";
        }
        else
        {
            filenames[0] = filename;
        }

        LibUtilities::CppCommandLine cmd({"NekMesh"});
        LibUtilities::SessionReaderSharedPtr vSession =
            LibUtilities::SessionReader::CreateInstance(
                1, cmd.GetArgv(), filenames, m_mesh->m_comm);
        SpatialDomains::MeshGraphSharedPtr graph =
            SpatialDomains::MeshGraphIO::Read(vSession);

        TestElmts(graph->GetGeomMap<SpatialDomains::SegGeom>(), graph,
                  m_strEval, exprId, m_log);
        TestElmts(graph->GetGeomMap<SpatialDomains::TriGeom>(), graph,
                  m_strEval, exprId, m_log);
        TestElmts(graph->GetGeomMap<SpatialDomains::QuadGeom>(), graph,
                  m_strEval, exprId, m_log);
        TestElmts(graph->GetGeomMap<SpatialDomains::TetGeom>(), graph,
                  m_strEval, exprId, m_log);
        TestElmts(graph->GetGeomMap<SpatialDomains::PrismGeom>(), graph,
                  m_strEval, exprId, m_log);
        TestElmts(graph->GetGeomMap<SpatialDomains::PyrGeom>(), graph,
                  m_strEval, exprId, m_log);
        TestElmts(graph->GetGeomMap<SpatialDomains::HexGeom>(), graph,
                  m_strEval, exprId, m_log);
    }
}

void OutputNekpp::TransferVertices(MeshGraphSharedPtr graph)
{
    for (auto &it : m_mesh->m_vertexSet)
    {
        auto vert = ObjPoolManager<PointGeom>::AllocateUniquePtr(
            m_mesh->m_spaceDim, it->m_id, it->m_x, it->m_y, it->m_z);
        graph->AddGeom(it->m_id, std::move(vert));
    }
}

void OutputNekpp::TransferEdges(MeshGraphSharedPtr graph,
                                std::unordered_map<int, SegGeom *> &edgeMap)
{
    if (m_mesh->m_expDim >= 2)
    {
        for (auto &it : m_mesh->m_edgeSet)
        {
            std::array<PointGeom *, 2> verts = {
                graph->GetPointGeom(it->m_n1->m_id),
                graph->GetPointGeom(it->m_n2->m_id)};
            SegGeomUniquePtr edge = ObjPoolManager<SegGeom>::AllocateUniquePtr(
                it->m_id, m_mesh->m_spaceDim, verts);
            edgeMap[it->m_id] = edge.get();
            graph->AddGeom(it->m_id, std::move(edge));
        }
    }
}

void OutputNekpp::TransferFaces(MeshGraphSharedPtr graph,
                                std::unordered_map<int, SegGeom *> &edgeMap)
{
    if (m_mesh->m_expDim == 3)
    {
        for (auto &it : m_mesh->m_faceSet)
        {
            if (it->m_edgeList.size() == 3)
            {
                std::array<SegGeom *, TriGeom::kNedges> edges = {
                    edgeMap[it->m_edgeList[0]->m_id],
                    edgeMap[it->m_edgeList[1]->m_id],
                    edgeMap[it->m_edgeList[2]->m_id]};

                TriGeomUniquePtr tri =
                    ObjPoolManager<TriGeom>::AllocateUniquePtr(it->m_id, edges);
                graph->AddGeom(it->m_id, std::move(tri));
            }
            else
            {
                std::array<SegGeom *, QuadGeom::kNedges> edges = {
                    edgeMap[it->m_edgeList[0]->m_id],
                    edgeMap[it->m_edgeList[1]->m_id],
                    edgeMap[it->m_edgeList[2]->m_id],
                    edgeMap[it->m_edgeList[3]->m_id]};

                QuadGeomUniquePtr quad =
                    ObjPoolManager<QuadGeom>::AllocateUniquePtr(it->m_id,
                                                                edges);
                graph->AddGeom(it->m_id, std::move(quad));
            }
        }
    }
}

void OutputNekpp::TransferElements(MeshGraphSharedPtr graph)
{
    vector<ElementSharedPtr> &elmt = m_mesh->m_element[m_mesh->m_expDim];

    for (int i = 0; i < elmt.size(); ++i)
    {
        switch (elmt[i]->GetTag()[0])
        {
            case 'S':
            {
                int id                              = elmt[i]->GetId();
                std::array<PointGeom *, 2> vertices = {
                    graph->GetPointGeom(elmt[i]->GetVertex(0)->m_id),
                    graph->GetPointGeom(elmt[i]->GetVertex(1)->m_id)};
                auto geom = ObjPoolManager<SegGeom>::AllocateUniquePtr(
                    id, m_mesh->m_spaceDim, vertices);
                graph->AddGeom(id, std::move(geom));
            }
            break;
            case 'T':
            {
                int id = elmt[i]->GetId();
                std::array<SegGeom *, TriGeom::kNedges> edges = {
                    graph->GetSegGeom(elmt[i]->GetEdge(0)->m_id),
                    graph->GetSegGeom(elmt[i]->GetEdge(1)->m_id),
                    graph->GetSegGeom(elmt[i]->GetEdge(2)->m_id)};

                auto geom =
                    ObjPoolManager<TriGeom>::AllocateUniquePtr(id, edges);
                graph->AddGeom(id, std::move(geom));
            }
            break;
            case 'Q':
            {
                int id = elmt[i]->GetId();
                std::array<SegGeom *, QuadGeom::kNedges> edges = {
                    graph->GetSegGeom(elmt[i]->GetEdge(0)->m_id),
                    graph->GetSegGeom(elmt[i]->GetEdge(1)->m_id),
                    graph->GetSegGeom(elmt[i]->GetEdge(2)->m_id),
                    graph->GetSegGeom(elmt[i]->GetEdge(3)->m_id)};

                auto geom =
                    ObjPoolManager<QuadGeom>::AllocateUniquePtr(id, edges);
                graph->AddGeom(id, std::move(geom));
            }
            break;
            case 'A':
            {
                int id = elmt[i]->GetId();
                std::array<TriGeom *, 4> tfaces;
                for (int j = 0; j < 4; ++j)
                {
                    Geometry2D *face =
                        graph->GetGeometry2D(elmt[i]->GetFace(j)->m_id);
                    tfaces[j] = static_cast<TriGeom *>(face);
                }

                auto geom =
                    ObjPoolManager<TetGeom>::AllocateUniquePtr(id, tfaces);
                graph->AddGeom(id, std::move(geom));
            }
            break;
            case 'P':
            {
                std::array<Geometry2D *, 5> faces;

                int id = elmt[i]->GetId();
                for (int j = 0; j < 5; ++j)
                {
                    Geometry2D *face =
                        graph->GetGeometry2D(elmt[i]->GetFace(j)->m_id);

                    if (face->GetShapeType() == LibUtilities::eTriangle)
                    {
                        faces[j] = static_cast<TriGeom *>(face);
                    }
                    else if (face->GetShapeType() ==
                             LibUtilities::eQuadrilateral)
                    {
                        faces[j] = static_cast<QuadGeom *>(face);
                    }
                }
                auto geom =
                    ObjPoolManager<PyrGeom>::AllocateUniquePtr(id, faces);
                graph->AddGeom(id, std::move(geom));
            }
            break;
            case 'R':
            {
                std::array<Geometry2D *, 5> faces;

                int id = elmt[i]->GetId();
                for (int j = 0; j < 5; ++j)
                {
                    Geometry2D *face =
                        graph->GetGeometry2D(elmt[i]->GetFace(j)->m_id);

                    if (face->GetShapeType() == LibUtilities::eTriangle)
                    {
                        faces[j] = static_cast<TriGeom *>(face);
                    }
                    else if (face->GetShapeType() ==
                             LibUtilities::eQuadrilateral)
                    {
                        faces[j] = static_cast<QuadGeom *>(face);
                    }
                }
                auto geom =
                    ObjPoolManager<PrismGeom>::AllocateUniquePtr(id, faces);
                graph->AddGeom(id, std::move(geom));
            }
            break;
            case 'H':
            {
                std::array<QuadGeom *, 6> faces;

                int id = elmt[i]->GetId();
                for (int j = 0; j < 6; ++j)
                {
                    Geometry2D *face =
                        graph->GetGeometry2D(elmt[i]->GetFace(j)->m_id);
                    faces[j] = static_cast<QuadGeom *>(face);
                }

                auto geom =
                    ObjPoolManager<HexGeom>::AllocateUniquePtr(id, faces);
                graph->AddGeom(id, std::move(geom));
            }
            break;
            default:
                ASSERTL1(false, "Unknown element type");
        }
    }
}

void OutputNekpp::TransferCurves(MeshGraphSharedPtr graph)
{
    CurveMap &edges  = graph->GetCurvedEdges();
    auto &curveNodes = graph->GetAllCurveNodes();

    int edgecnt = 0;

    for (auto &it : m_mesh->m_edgeSet)
    {
        if (it->m_edgeNodes.size() > 0)
        {
            CurveUniquePtr curve = ObjPoolManager<Curve>::AllocateUniquePtr(
                it->m_id, it->m_curveType);
            vector<NodeSharedPtr> ns;
            it->GetCurvedNodes(ns);
            for (int i = 0; i < ns.size(); i++)
            {
                PointGeomUniquePtr vert =
                    ObjPoolManager<PointGeom>::AllocateUniquePtr(
                        m_mesh->m_spaceDim, edgecnt, ns[i]->m_x, ns[i]->m_y,
                        ns[i]->m_z);
                curve->m_points.push_back(vert.get());
                curveNodes.push_back(std::move(vert));
            }

            edges[it->m_id] = std::move(curve);
            edgecnt++;
        }
    }

    if (m_mesh->m_expDim == 1 && m_mesh->m_spaceDim > 1)
    {
        for (int e = 0; e < m_mesh->m_element[1].size(); e++)
        {
            ElementSharedPtr el = m_mesh->m_element[1][e];
            vector<NodeSharedPtr> ns;
            el->GetCurvedNodes(ns);
            if (ns.size() > 2)
            {
                CurveUniquePtr curve = ObjPoolManager<Curve>::AllocateUniquePtr(
                    el->GetId(), el->GetCurveType());

                for (int i = 0; i < ns.size(); i++)
                {
                    PointGeomUniquePtr vert =
                        ObjPoolManager<PointGeom>::AllocateUniquePtr(
                            m_mesh->m_spaceDim, edgecnt, ns[i]->m_x, ns[i]->m_y,
                            ns[i]->m_z);
                    curve->m_points.push_back(vert.get());
                    curveNodes.push_back(std::move(vert));
                }

                edges[el->GetId()] = std::move(curve);
                edgecnt++;
            }
        }
    }

    CurveMap &faces = graph->GetCurvedFaces();

    int facecnt = 0;

    for (auto &it : m_mesh->m_faceSet)
    {
        if (it->m_faceNodes.size() > 0)
        {
            CurveUniquePtr curve = ObjPoolManager<Curve>::AllocateUniquePtr(
                it->m_id, it->m_curveType);
            vector<NodeSharedPtr> ns;
            it->GetCurvedNodes(ns);
            for (int i = 0; i < ns.size(); i++)
            {
                PointGeomUniquePtr vert =
                    ObjPoolManager<PointGeom>::AllocateUniquePtr(
                        m_mesh->m_spaceDim, facecnt, ns[i]->m_x, ns[i]->m_y,
                        ns[i]->m_z);
                curve->m_points.push_back(vert.get());
                curveNodes.push_back(std::move(vert));
            }

            faces[it->m_id] = std::move(curve);
            facecnt++;
        }
    }

    if (m_mesh->m_expDim == 2 && m_mesh->m_spaceDim == 3)
    {
        // manifold case
        for (int e = 0; e < m_mesh->m_element[2].size(); e++)
        {
            ElementSharedPtr el = m_mesh->m_element[2][e];

            if (el->GetVolumeNodes().size() > 0) // needed for extract surf case
            {
                vector<NodeSharedPtr> ns;
                el->GetCurvedNodes(ns);
                if (ns.size() > 4)
                {
                    CurveUniquePtr curve =
                        ObjPoolManager<Curve>::AllocateUniquePtr(
                            el->GetId(), el->GetCurveType());

                    for (int i = 0; i < ns.size(); i++)
                    {
                        PointGeomUniquePtr vert =
                            ObjPoolManager<PointGeom>::AllocateUniquePtr(
                                m_mesh->m_spaceDim, facecnt, ns[i]->m_x,
                                ns[i]->m_y, ns[i]->m_z);
                        curve->m_points.push_back(vert.get());
                        curveNodes.push_back(std::move(vert));
                    }

                    faces[el->GetId()] = std::move(curve);
                    facecnt++;
                }
            }
        }
    }
}

void OutputNekpp::TransferComposites(MeshGraphSharedPtr graph)
{
    SpatialDomains::CompositeMap &comps = graph->GetComposites();
    map<int, string> &compLabels        = graph->GetCompositesLabels();

    for (auto &it : m_mesh->m_composite)
    {
        if (it.second->m_items.size() > 0)
        {
            int indx = it.second->m_id;
            SpatialDomains::CompositeSharedPtr curVector =
                MemoryManager<SpatialDomains::Composite>::AllocateSharedPtr();

            if (it.second->m_label.size())
            {
                compLabels[indx] = it.second->m_label;
            }

            switch (it.second->m_tag[0])
            {
                case 'V':
                {
                    for (int i = 0; i < it.second->m_items.size(); i++)
                    {
                        auto geom =
                            graph->GetPointGeom(it.second->m_items[i]->GetId());
                        curVector->m_geomVec.push_back(geom);
                    }
                }
                break;
                case 'S':
                case 'E':
                {
                    for (int i = 0; i < it.second->m_items.size(); i++)
                    {
                        auto geom =
                            graph->GetSegGeom(it.second->m_items[i]->GetId());
                        curVector->m_geomVec.push_back(geom);
                    }
                }
                break;
                case 'Q':
                {
                    for (int i = 0; i < it.second->m_items.size(); i++)
                    {
                        auto geom =
                            graph->GetQuadGeom(it.second->m_items[i]->GetId());
                        curVector->m_geomVec.push_back(geom);
                    }
                }
                break;
                case 'T':
                {
                    for (int i = 0; i < it.second->m_items.size(); i++)
                    {
                        auto geom =
                            graph->GetTriGeom(it.second->m_items[i]->GetId());
                        curVector->m_geomVec.push_back(geom);
                    }
                }
                break;
                case 'F':
                {
                    for (int i = 0; i < it.second->m_items.size(); i++)
                    {
                        auto geom = graph->GetGeometry2D(
                            it.second->m_items[i]->GetId());
                        curVector->m_geomVec.push_back(geom);
                    }
                }
                break;
                case 'A':
                {
                    for (int i = 0; i < it.second->m_items.size(); i++)
                    {
                        auto geom =
                            graph->GetTetGeom(it.second->m_items[i]->GetId());
                        curVector->m_geomVec.push_back(geom);
                    };
                }
                break;
                case 'P':
                {
                    for (int i = 0; i < it.second->m_items.size(); i++)
                    {
                        auto geom =
                            graph->GetPyrGeom(it.second->m_items[i]->GetId());
                        curVector->m_geomVec.push_back(geom);
                    }
                }
                break;
                case 'R':
                {
                    for (int i = 0; i < it.second->m_items.size(); i++)
                    {
                        auto geom =
                            graph->GetPrismGeom(it.second->m_items[i]->GetId());
                        curVector->m_geomVec.push_back(geom);
                    }
                }
                break;
                case 'H':
                {
                    for (int i = 0; i < it.second->m_items.size(); i++)
                    {
                        auto geom =
                            graph->GetHexGeom(it.second->m_items[i]->GetId());
                        curVector->m_geomVec.push_back(geom);
                    }
                }
                break;
                default:
                    ASSERTL1(false, "Unknown element type");
            }

            comps[indx] = curVector;
        }
    }

    if (m_config["chkbndcomp"].beenSet)
    {
        if (m_mesh->m_expDim == 3)
        {
            // check to see if any boundary surfaces are not set and if so put
            // them into a set with tag 9999
            map<int, FaceSharedPtr> NotSet;

            // loop over faceset and make a map of all faces only linked
            // to one element
            for (auto &it : m_mesh->m_faceSet)
            {
                // for some reason links are defined twice so should have
                // been 1 but needs to be 2
                if (it->m_elLink.size() == 1)
                {
                    NotSet[it->m_id] = it;
                }
            }

            // reove composites and remove
            for (auto &it : m_mesh->m_element[2])
            {
                int id = it->GetId();
                if (NotSet.count(id))
                {
                    NotSet.erase(id);
                }
            }

            // Make composite of all missing faces
            if (NotSet.size())
            {
                int indx                                     = 9999;
                SpatialDomains::CompositeSharedPtr curVector = MemoryManager<
                    SpatialDomains::Composite>::AllocateSharedPtr();

                for (auto &sit : NotSet)
                {
                    curVector->m_geomVec.push_back(
                        graph->GetGeometry2D(sit.first));
                }
                comps[indx] = curVector;
            }
        }
    }
}

// @TODO: We currently lose domain information from input file here. This
// assumes
//        every composite that is of expansion dimension is a separate domain
//        and sequentially numbered. So junks multi-composite domains & IDs.
void OutputNekpp::TransferDomain(MeshGraphSharedPtr graph)
{
    std::map<int, SpatialDomains::CompositeMap> &domain = graph->GetDomain();

    int cnt = 0;
    for (auto &it : m_mesh->m_composite)
    {

        string list;
        if (it.second->m_items[0]->GetDim() == m_mesh->m_expDim)
        {
            if (list.length() > 0)
            {
                list += ",";
            }
            list += std::to_string(it.second->m_id);

            SpatialDomains::CompositeMap fullDomain;
            graph->GetCompositeList(list, fullDomain);
            domain[cnt++] = fullDomain;
        }
    }
}

} // namespace Nektar::NekMesh
