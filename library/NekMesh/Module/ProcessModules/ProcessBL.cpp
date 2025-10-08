////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessBL.cpp
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
//  Description: Refine prismatic or quadrilateral boundary layer elements.
//
////////////////////////////////////////////////////////////////////////////////

#include <string>

#include <LibUtilities/BasicUtils/ParseUtils.h>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/Foundations/BLPoints.h>
#include <LibUtilities/Foundations/ManagerAccess.h>
#include <LibUtilities/Interpreter/Interpreter.h>
#include <LocalRegions/HexExp.h>
#include <LocalRegions/PrismExp.h>
#include <LocalRegions/QuadExp.h>

#include "ProcessBL.h"
#include <NekMesh/MeshElements/Element.h>

using namespace std;
using namespace Nektar::NekMesh;

namespace Nektar::NekMesh
{
ModuleKey ProcessBL::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eProcessModule, "bl"), ProcessBL::create,
    "Refines a prismatic boundary layers. Updated version - "
    "FaceNodes + EdgeNodes");

int **helper(int lda, int arr[][2])
{
    int **ret = new int *[lda];
    for (int i = 0; i < lda; ++i)
    {
        ret[i]    = new int[2];
        ret[i][0] = arr[i][0];
        ret[i][1] = arr[i][1];
    }
    return ret;
}

int **helper(int lda, int arr[][4])
{
    int **ret = new int *[lda];
    for (int i = 0; i < lda; ++i)
    {
        ret[i]    = new int[4];
        ret[i][0] = arr[i][0];
        ret[i][1] = arr[i][1];
        ret[i][2] = arr[i][2];
        ret[i][3] = arr[i][3];
    }
    return ret;
}

int **helper(int lda, int arr[][3])
{
    int **ret = new int *[lda];
    for (int i = 0; i < lda; ++i)
    {
        ret[i]    = new int[3];
        ret[i][0] = arr[i][0];
        ret[i][1] = arr[i][1];
        ret[i][2] = arr[i][2];
    }
    return ret;
}

struct SplitMapHelper
{
    int size;
    int dir;
    int oppositeFace;
    int bfacesSize;
    int *bfaces;

    int nEdgeToSplit;
    int *edgesToSplit;
    int **edgeVert;

    int **conn;

    int nEdgeToCurve;
    int *edgesToCurve;
    int **intEdgeFace;
    int **extEdgeFace;
    int blpDir;
    int **gll;
};

ProcessBL::ProcessBL(MeshSharedPtr m) : ProcessModule(m)
{
    // BL mesh configuration.
    m_config["layers"] =
        ConfigOption(false, "2", "Number of layers to refine.");
    m_config["nq"] =
        ConfigOption(false, "5", "Number of points in high order elements.");
    m_config["surf"] =
        ConfigOption(false, "", "Tag identifying surface connected to prism.");
    m_config["r"] =
        ConfigOption(false, "2.0", "Ratio to use in geometry progression.");
}

ProcessBL::~ProcessBL()
{
}

void ProcessBL::Process()
{
    m_log(VERBOSE) << "Refining boundary layer." << endl;

    int dim = m_mesh->m_expDim;
    switch (dim)
    {
        case 2:
            BoundaryLayer2D();
            break;

        case 3:
            BoundaryLayer3D();
            break;

        default:
            m_log(FATAL) << "Only 2D and 3D meshes supported." << endl;
            break;
    }

    ProcessVertices();
    ProcessEdges();

    ProcessFaces();
    ProcessElements();
    ProcessComposites();
}

void ProcessBL::BoundaryLayer2D()
{
    // This implementation of 2D does not support bidirectional splitting

    int nodeId = m_mesh->m_vertexSet.size();
    int nl     = m_config["layers"].as<int>();
    int nq     = m_config["nq"].as<int>();

    // determine if geometric ratio is string or a constant.
    LibUtilities::Interpreter rEval;
    NekDouble r        = 1;
    int rExprId        = -1;
    bool ratioIsString = false;

    if (m_config["r"].isType<NekDouble>())
    {
        r = m_config["r"].as<NekDouble>();
    }
    else
    {
        std::string rstr = m_config["r"].as<string>();
        rExprId          = rEval.DefineFunction("x y z", rstr);
        ratioIsString    = true;
    }

    // Default PointsType.
    LibUtilities::PointsType pt = LibUtilities::eGaussLobattoLegendre;

    // Map which takes element ID to edge on surface. This enables
    // splitting to occur in either y-direction of the prism.
    map<int, int> splitEls;

    // edgeMap associates geometry edge IDs to the (nl+1) vertices which
    // are generated along that edge when a prism is split, and is used
    // to avoid generation of duplicate vertices. It is stored as an
    // unordered map for speed.
    std::unordered_map<int, vector<NodeSharedPtr>> edgeMap;

    string surf = m_config["surf"].as<string>();
    if (surf.size() > 0)
    {
        vector<unsigned int> surfs;
        ParseUtils::GenerateSeqVector(surf, surfs);
        sort(surfs.begin(), surfs.end());

        // If surface is defined, process list of elements to find those
        // that are connected to it.
        for (int i = 0; i < m_mesh->m_element[m_mesh->m_expDim].size(); ++i)
        {
            ElementSharedPtr el = m_mesh->m_element[m_mesh->m_expDim][i];
            int nSurf           = el->GetEdgeCount();

            for (int j = 0; j < nSurf; ++j)
            {
                int bl = el->GetBoundaryLink(j);
                if (bl == -1)
                {
                    continue;
                }

                ElementSharedPtr bEl =
                    m_mesh->m_element[m_mesh->m_expDim - 1][bl];
                vector<int> tags = bEl->GetTagList();
                vector<int> inter;

                sort(tags.begin(), tags.end());
                set_intersection(surfs.begin(), surfs.end(), tags.begin(),
                                 tags.end(), back_inserter(inter));
                ASSERTL0(inter.size() <= 1, "Intersection of surfaces wrong");

                if (inter.size() == 1)
                {
                    if (el->GetConf().m_e != LibUtilities::eQuadrilateral)
                    {
                        m_log(WARNING)
                            << "Found non-quad element to split in "
                            << "surface " << surf << "; ignoring" << endl;
                        continue;
                    }

                    if (splitEls.count(el->GetId()) > 0)
                    {
                        m_log(WARNING)
                            << "quad already found; ignoring" << endl;
                        continue;
                    }

                    splitEls[el->GetId()] = j;
                }
            }
        }
    }
    else
    {
        ASSERTL0(false, "Surface must be specified.");
    }

    if (splitEls.size() == 0)
    {
        m_log(WARNING) << "No elements detected to split." << endl;
        return;
    }

    // Erase all elements from the element list. Elements will be
    // re-added as they are split.
    vector<ElementSharedPtr> el = m_mesh->m_element[m_mesh->m_expDim];
    m_mesh->m_element[m_mesh->m_expDim].clear();

    // Iterate over list of elements of expansion dimension.
    for (int i = 0; i < el.size(); ++i)
    {

        if (splitEls.count(el[i]->GetId()) == 0)
        {
            m_mesh->m_element[m_mesh->m_expDim].push_back(el[i]);
            continue;
        }

        // Find other boundary faces if any
        std::map<int, int> bLink;
        for (int j = 0; j < 4; j++)
        {
            int bl = el[i]->GetBoundaryLink(j);
            if ((bl != -1) && (j != splitEls[el[i]->GetId()]))
            {
                bLink[j] = bl;
            }
        }

        SpatialDomains::EntityHolder holder;
        // Get elemental geometry object.
        auto geom = dynamic_cast<SpatialDomains::QuadGeom *>(
            el[i]->GetGeom(m_mesh->m_spaceDim, holder));

        // Determine whether to use reverse points.
        // (if edges 1 or 2 are on the surface)
        LibUtilities::PointsType t =
            ((splitEls[el[i]->GetId()] + 1) % 4) < 2
                ? LibUtilities::eBoundaryLayerPoints
                : LibUtilities::eBoundaryLayerPointsRev;

        if (ratioIsString) // determine value of r base on geom
        {
            NekDouble x, y, z;
            NekDouble x1, y1, z1;
            int nverts = geom->GetNumVerts();

            x = y = z = 0.0;

            for (int i = 0; i < nverts; ++i)
            {
                geom->GetVertex(i)->GetCoords(x1, y1, z1);
                x += x1;
                y += y1;
                z += z1;
            }
            x /= (NekDouble)nverts;
            y /= (NekDouble)nverts;
            z /= (NekDouble)nverts;
            r = rEval.Evaluate(rExprId, x, y, z, 0.0);
        }

        // Create basis.
        LibUtilities::BasisKey B0(LibUtilities::eModified_A, nq,
                                  LibUtilities::PointsKey(nq, pt));
        LibUtilities::BasisKey B1(LibUtilities::eModified_A, 2,
                                  LibUtilities::PointsKey(nl + 1, t, r));

        // Create local region.
        LocalRegions::QuadExpSharedPtr q;
        if (splitEls[el[i]->GetId()] % 2)
        {
            q = MemoryManager<LocalRegions::QuadExp>::AllocateSharedPtr(B1, B0,
                                                                        geom);
        }
        else
        {
            q = MemoryManager<LocalRegions::QuadExp>::AllocateSharedPtr(B0, B1,
                                                                        geom);
        }

        // Grab co-ordinates.
        Array<OneD, NekDouble> x(nq * (nl + 1), 0.0);
        Array<OneD, NekDouble> y(nq * (nl + 1), 0.0);
        Array<OneD, NekDouble> z(nq * (nl + 1), 0.0);
        q->GetCoords(x, y, z);

        vector<vector<NodeSharedPtr>> edgeNodes(2);

        // Loop over edges to be split.
        for (int j = 0; j < 2; ++j)
        {
            int locEdge = (splitEls[el[i]->GetId()] + 1 + 2 * j) % 4;
            int edgeId  = el[i]->GetEdge(locEdge)->m_id;

            // Determine whether we have already generated vertices
            // along this edge.
            auto eIt = edgeMap.find(edgeId);

            if (eIt == edgeMap.end())
            {
                // If not then resize storage to hold new points.
                edgeNodes[j].resize(nl + 1);

                // Re-use existing vertices at endpoints of edge to
                // avoid duplicating the existing vertices.
                edgeNodes[j][0]  = el[i]->GetVertex(locEdge);
                edgeNodes[j][nl] = el[i]->GetVertex((locEdge + 1) % 4);

                // Variable geometric ratio
                if (ratioIsString)
                {
                    NekDouble x0, y0;
                    NekDouble x1, y1;
                    NekDouble xm, ym, zm = 0.0;

                    // -> Find edge end and mid points
                    x0 = edgeNodes[j][0]->m_x;
                    y0 = edgeNodes[j][0]->m_y;

                    x1 = edgeNodes[j][nl]->m_x;
                    y1 = edgeNodes[j][nl]->m_y;

                    xm = 0.5 * (x0 + x1);
                    ym = 0.5 * (y0 + y1);

                    // evaluate r factor based on mid point value
                    NekDouble rnew;
                    rnew = rEval.Evaluate(rExprId, xm, ym, zm, 0.0);

                    // Get basis with new r;
                    t = (j == 0) ? LibUtilities::eBoundaryLayerPoints
                                 : LibUtilities::eBoundaryLayerPointsRev;
                    LibUtilities::PointsKey Pkey(nl + 1, t, rnew);
                    LibUtilities::PointsSharedPtr newP =
                        LibUtilities::PointsManager()[Pkey];

                    const Array<OneD, const NekDouble> z = newP->GetZ();

                    // Create new interior nodes based on this new blend
                    for (int k = 1; k < nl; ++k)
                    {
                        xm = 0.5 * (1 + z[k]) * (x1 - x0) + x0;
                        ym = 0.5 * (1 + z[k]) * (y1 - y0) + y0;
                        zm = 0.0;
                        edgeNodes[j][k] =
                            NodeSharedPtr(new Node(nodeId++, xm, ym, zm));
                    }
                }
                else
                {
                    // Create new interior nodes.
                    int pos = 0;
                    for (int k = 1; k < nl; ++k)
                    {
                        switch (locEdge)
                        {
                            case 0:
                                pos = k;
                                break;
                            case 1:
                                pos = nq - 1 + k * nq;
                                break;
                            case 2:
                                pos = nq * (nl + 1) - 1 - k;
                                break;
                            case 3:
                                pos = nq * nl - k * nq;
                                break;
                            default:
                                NEKERROR(ErrorUtil::efatal,
                                         "Quad edge should be < 4.");
                                break;
                        }
                        edgeNodes[j][k] = NodeSharedPtr(
                            new Node(nodeId++, x[pos], y[pos], z[pos]));
                    }
                }

                // Store these edges in edgeMap.
                edgeMap[edgeId] = edgeNodes[j];
            }
            else
            {
                // Check orientation
                if (eIt->second[0] == el[i]->GetVertex(locEdge))
                {
                    // Same orientation: copy nodes
                    edgeNodes[j] = eIt->second;
                }
                else
                {
                    // Reversed orientation: copy in reversed order
                    edgeNodes[j].resize(nl + 1);
                    for (int k = 0; k < nl + 1; ++k)
                    {
                        edgeNodes[j][k] = eIt->second[nl - k];
                    }
                }
            }
        }

        // Create element layers.
        for (int j = 0; j < nl; ++j)
        {
            // Get corner vertices.
            vector<NodeSharedPtr> nodeList(4);
            switch (splitEls[el[i]->GetId()])
            {
                case 0:
                {
                    nodeList[0] = edgeNodes[1][nl - j];
                    nodeList[1] = edgeNodes[0][j];
                    nodeList[2] = edgeNodes[0][j + 1];
                    nodeList[3] = edgeNodes[1][nl - j - 1];
                    break;
                }
                case 1:
                {
                    nodeList[0] = edgeNodes[1][j];
                    nodeList[1] = edgeNodes[1][j + 1];
                    nodeList[2] = edgeNodes[0][nl - j - 1];
                    nodeList[3] = edgeNodes[0][nl - j];
                    break;
                }
                case 2:
                {
                    nodeList[0] = edgeNodes[0][nl - j];
                    nodeList[1] = edgeNodes[1][j];
                    nodeList[2] = edgeNodes[1][j + 1];
                    nodeList[3] = edgeNodes[0][nl - j - 1];
                    break;
                }
                case 3:
                {
                    nodeList[0] = edgeNodes[0][j];
                    nodeList[1] = edgeNodes[0][j + 1];
                    nodeList[2] = edgeNodes[1][nl - j - 1];
                    nodeList[3] = edgeNodes[1][nl - j];
                    break;
                }
            }
            // Create the element.
            ElmtConfig conf(LibUtilities::eQuadrilateral, 1, true, false, true);
            ElementSharedPtr elmt = GetElementFactory().CreateInstance(
                LibUtilities::eQuadrilateral, conf, nodeList,
                el[i]->GetTagList());

            // Add high order nodes to split edges.
            for (int l = 0; l < 2; ++l)
            {
                int locEdge          = (splitEls[el[i]->GetId()] + 2 * l) % 4;
                EdgeSharedPtr HOedge = elmt->GetEdge(locEdge);
                int pos              = 0;
                for (int k = 1; k < nq - 1; ++k)
                {
                    switch (locEdge)
                    {
                        case 0:
                            pos = j * nq + k;
                            break;
                        case 1:
                            pos = j + 1 + k * (nl + 1);
                            break;
                        case 2:
                            pos = (j + 1) * nq + (nq - 1) - k;
                            break;
                        case 3:
                            pos = (nl + 1) * (nq - 1) + j - k * (nl + 1);
                            break;
                        default:
                            NEKERROR(ErrorUtil::efatal,
                                     "Quad edge should be < 4.");
                            break;
                    }
                    HOedge->m_edgeNodes.push_back(
                        NodeSharedPtr(new Node(nodeId++, x[pos], y[pos], 0.0)));
                }
                HOedge->m_curveType = pt;
            }

            // Change the elements on the boundary
            // to match the layers
            for (auto &it : bLink)
            {
                int eid = it.first;
                int bl  = it.second;

                if (j == 0)
                {
                    // For first layer reuse existing 2D element.
                    ElementSharedPtr e =
                        m_mesh->m_element[m_mesh->m_expDim - 1][bl];
                    for (int k = 0; k < 2; ++k)
                    {
                        e->SetVertex(k, nodeList[(eid + k) % 4]);
                    }
                }
                else
                {
                    // For all other layers create new element.
                    vector<NodeSharedPtr> qNodeList(2);
                    for (int k = 0; k < 2; ++k)
                    {
                        qNodeList[k] = nodeList[(eid + k) % 4];
                    }
                    vector<int> tagBE;
                    tagBE = m_mesh->m_element[m_mesh->m_expDim - 1][bl]
                                ->GetTagList();
                    ElmtConfig bconf(LibUtilities::eSegment, 1, true, true,
                                     false);
                    ElementSharedPtr boundaryElmt =
                        GetElementFactory().CreateInstance(
                            LibUtilities::eSegment, bconf, qNodeList, tagBE);
                    m_mesh->m_element[m_mesh->m_expDim - 1].push_back(
                        boundaryElmt);
                }
            }

            m_mesh->m_element[m_mesh->m_expDim].push_back(elmt);
        }
    }
}

void ProcessBL::BoundaryLayer3D()
{
    m_log(VERBOSE) << "Elements before = " << m_mesh->m_element[3].size()
                   << endl;

    // A set containing all element types which are valid.
    set<LibUtilities::ShapeType> validElTypes;
    validElTypes.insert(LibUtilities::ePrism);
    validElTypes.insert(LibUtilities::eHexahedron);

    // int nodeId = m_mesh->m_vertexSet.size();
    int nl = m_config["layers"].as<int>();
    int nq = m_config["nq"].as<int>();

    // determine if geometric ratio is string or a constant.
    LibUtilities::Interpreter rEval;
    NekDouble r = 1;
    // int rExprId        = -1;
    // bool ratioIsString = false;

    if (m_config["r"].isType<NekDouble>())
    {
        r = m_config["r"].as<NekDouble>();
    }
    else
    {
        m_log(FATAL) << "R is string only in 2D possible - give Double."
                     << endl;
        // std::string rstr = m_config["r"].as<string>();
        // rExprId          = rEval.DefineFunction("x y z", rstr);
        // ratioIsString    = true;
    }

    // Prismatic node -> face map.
    int prismFaceNodes[5][4] = {
        {0, 1, 2, 3}, {0, 1, 4, -1}, {1, 2, 5, 4}, {3, 2, 5, -1}, {0, 3, 5, 4}};
    int hexFaceNodes[6][4] = {{0, 1, 2, 3}, {0, 1, 5, 4}, {1, 2, 6, 5},
                              {3, 2, 6, 7}, {0, 3, 7, 4}, {4, 5, 6, 7}};
    map<LibUtilities::ShapeType, int **> faceNodeMap;
    faceNodeMap[LibUtilities::ePrism]      = helper(5, prismFaceNodes);
    faceNodeMap[LibUtilities::eHexahedron] = helper(6, hexFaceNodes);

    // Default PointsType.
    LibUtilities::PointsKey ekey(
        nq, LibUtilities::eGaussLobattoLegendre); // ePolyEvenlySpaced
                                                  // //eGaussLobattoLegendre
    Array<OneD, NekDouble> gll;
    LibUtilities::PointsManager()[ekey]->GetPoints(gll);

    // Default FaceNode point
    LibUtilities::PointsKey pkey(
        nq,
        LibUtilities::eNodalTriElec); // eNodalTriElec eNodalTriFekete
    // eNodalTriFekete ; eNodalTriEvenlySpaced
    Array<OneD, NekDouble> u_FaceNodes, v_FaceNodes;
    LibUtilities::PointsManager()[pkey]->GetPoints(u_FaceNodes, v_FaceNodes);

    // Map which takes element ID to face on surface. This enables
    // splitting to occur in either y-direction of the prism.
    unordered_map<int, int> splitEls;
    unordered_map<int, int>::iterator sIt;

    // Set up maps which takes an edge (in nektar++ ordering) and return
    // their offset and stride in the 3d array of collapsed quadrature
    // points. Note that this map includes only the edges that are on
    // the triangular faces as the edges in the normal direction are
    // linear.
    map<LibUtilities::ShapeType, map<int, SplitMapHelper>> splitMap;

    ////////////////////////////////////////
    // HEX DIR X
    ////////////////////////////////////////

    SplitMapHelper splitHex0;
    int splitMapBFacesHex0[4]      = {1, 2, 3, 4};
    int splitedgehex0[4]           = {4, 5, 6, 7};
    int splitHex0EdgeVert[4][2]    = {{0, 4}, {1, 5}, {2, 6}, {3, 7}};
    int splitMapConnHex0[8][2]     = {{0, 0}, {1, 0}, {2, 0}, {3, 0},
                                      {0, 1}, {1, 1}, {2, 1}, {3, 1}};
    int splitedgestocurvehex0[8]   = {0, 1, 2, 3, 8, 9, 10, 11};
    int splithex0gll[8][3]         = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1},
                                      {-1, 1, -1},  {-1, -1, 1}, {1, -1, 1},
                                      {1, 1, 1},    {-1, 1, 1}};
    int splitHex0IntEdgeFace[4][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    int splitHex0ExtEdgeFace[4][2] = {{8, 1}, {9, 2}, {10, 3}, {11, 4}};

    splitHex0.size         = 8;
    splitHex0.dir          = 0;
    splitHex0.oppositeFace = 5;
    splitHex0.nEdgeToSplit = 4;
    splitHex0.edgesToSplit = splitedgehex0;
    splitHex0.edgeVert     = helper(4, splitHex0EdgeVert);
    splitHex0.conn         = helper(8, splitMapConnHex0);
    splitHex0.bfacesSize   = 4;
    splitHex0.bfaces       = splitMapBFacesHex0;
    splitHex0.nEdgeToCurve = 8;
    splitHex0.edgesToCurve = splitedgestocurvehex0;
    splitHex0.intEdgeFace  = helper(4, splitHex0IntEdgeFace);
    splitHex0.extEdgeFace  = helper(4, splitHex0ExtEdgeFace);
    splitHex0.blpDir       = 2;
    splitHex0.gll          = helper(8, splithex0gll);

    splitMap[LibUtilities::eHexahedron][0] = splitHex0;
    int splitMapConnHex0rev[8][2]          = {{0, 1}, {1, 1}, {2, 1}, {3, 1},
                                              {0, 0}, {1, 0}, {2, 0}, {3, 0}};

    SplitMapHelper splitHex5;
    splitHex5.size                         = 8;
    splitHex5.dir                          = 1;
    splitHex5.oppositeFace                 = 0;
    splitHex5.nEdgeToSplit                 = 4;
    splitHex5.edgesToSplit                 = splitedgehex0;
    splitHex5.edgeVert                     = helper(4, splitHex0EdgeVert);
    splitHex5.conn                         = helper(8, splitMapConnHex0rev);
    splitHex5.bfacesSize                   = 4;
    splitHex5.bfaces                       = splitMapBFacesHex0;
    splitHex5.nEdgeToCurve                 = 8;
    splitHex5.edgesToCurve                 = splitedgestocurvehex0;
    splitHex5.intEdgeFace                  = helper(4, splitHex0ExtEdgeFace);
    splitHex5.extEdgeFace                  = helper(4, splitHex0IntEdgeFace);
    splitHex5.blpDir                       = 2;
    splitHex5.gll                          = helper(8, splithex0gll);
    splitMap[LibUtilities::eHexahedron][5] = splitHex5;

    ////////////////////////////////////////
    // HEX DIR Y
    ////////////////////////////////////////

    SplitMapHelper splitHex1;
    int splitMapBFacesHex1[4]      = {0, 2, 5, 4};
    int splitedgehex1[4]           = {11, 9, 1, 3};
    int splitHex1EdgeVert[4][2]    = {{4, 7}, {5, 6}, {1, 2}, {0, 3}};
    int splitMapConnHex1[8][2]     = {{3, 0}, {2, 0}, {2, 1}, {3, 1},
                                      {0, 0}, {1, 0}, {1, 1}, {0, 1}};
    int splitedgestocurvehex1[8]   = {4, 8, 5, 0, 7, 10, 6, 2};
    int splitHex1IntEdgeFace[4][2] = {{0, 0}, {4, 4}, {5, 2}, {8, 5}};
    int splitHex1ExtEdgeFace[4][2] = {{2, 0}, {6, 2}, {7, 4}, {10, 5}};

    splitHex1.size                         = 8;
    splitHex1.dir                          = 0;
    splitHex1.oppositeFace                 = 3;
    splitHex1.nEdgeToSplit                 = 4;
    splitHex1.edgesToSplit                 = splitedgehex1;
    splitHex1.edgeVert                     = helper(4, splitHex1EdgeVert);
    splitHex1.conn                         = helper(8, splitMapConnHex1);
    splitHex1.bfacesSize                   = 4;
    splitHex1.bfaces                       = splitMapBFacesHex1;
    splitHex1.nEdgeToCurve                 = 8;
    splitHex1.edgesToCurve                 = splitedgestocurvehex1;
    splitHex1.intEdgeFace                  = helper(4, splitHex1IntEdgeFace);
    splitHex1.extEdgeFace                  = helper(4, splitHex1ExtEdgeFace);
    splitHex1.blpDir                       = 1;
    splitHex1.gll                          = helper(8, splithex0gll);
    splitMap[LibUtilities::eHexahedron][1] = splitHex1;

    SplitMapHelper splitHex3;
    int splitMapConnHex1rev[8][2] = {{3, 1}, {2, 1}, {2, 0}, {3, 0},
                                     {0, 1}, {1, 1}, {1, 0}, {0, 0}};

    splitHex3.size                         = 8;
    splitHex3.dir                          = 1;
    splitHex3.oppositeFace                 = 1;
    splitHex3.nEdgeToSplit                 = 4;
    splitHex3.edgesToSplit                 = splitedgehex1;
    splitHex3.edgeVert                     = helper(4, splitHex1EdgeVert);
    splitHex3.conn                         = helper(8, splitMapConnHex1rev);
    splitHex3.bfacesSize                   = 4;
    splitHex3.bfaces                       = splitMapBFacesHex1;
    splitHex3.nEdgeToCurve                 = 8;
    splitHex3.edgesToCurve                 = splitedgestocurvehex1;
    splitHex3.intEdgeFace                  = helper(4, splitHex1ExtEdgeFace);
    splitHex3.extEdgeFace                  = helper(4, splitHex1IntEdgeFace);
    splitHex3.blpDir                       = 1;
    splitHex3.gll                          = helper(8, splithex0gll);
    splitMap[LibUtilities::eHexahedron][3] = splitHex3;

    ////////////////////////////////////////
    // HEX DIR Z
    ////////////////////////////////////////

    SplitMapHelper splitHex4;
    int splitMapBFacesHex4[4]      = {0, 1, 5, 3};
    int splitedgehex4[4]           = {8, 0, 2, 10};
    int splitHex4EdgeVert[4][2]    = {{4, 5}, {0, 1}, {3, 2}, {7, 6}};
    int splitMapConnHex4[8][2]     = {{1, 0}, {1, 1}, {2, 1}, {2, 0},
                                      {0, 0}, {0, 1}, {3, 1}, {3, 0}};
    int splitedgestocurvehex4[8]   = {4, 11, 7, 3, 5, 9, 6, 1};
    int splitHex4IntEdgeFace[4][2] = {{3, 0}, {4, 1}, {7, 3}, {11, 5}};
    int splitHex4ExtEdgeFace[4][2] = {{1, 0}, {5, 1}, {6, 3}, {9, 5}};

    splitHex4.size                         = 8;
    splitHex4.dir                          = 0;
    splitHex4.oppositeFace                 = 2;
    splitHex4.nEdgeToSplit                 = 4;
    splitHex4.edgesToSplit                 = splitedgehex4;
    splitHex4.edgeVert                     = helper(4, splitHex4EdgeVert);
    splitHex4.conn                         = helper(8, splitMapConnHex4);
    splitHex4.bfacesSize                   = 4;
    splitHex4.bfaces                       = splitMapBFacesHex4;
    splitHex4.nEdgeToCurve                 = 8;
    splitHex4.edgesToCurve                 = splitedgestocurvehex4;
    splitHex4.intEdgeFace                  = helper(4, splitHex4IntEdgeFace);
    splitHex4.extEdgeFace                  = helper(4, splitHex4ExtEdgeFace);
    splitHex4.blpDir                       = 0;
    splitHex4.gll                          = helper(8, splithex0gll);
    splitMap[LibUtilities::eHexahedron][4] = splitHex4;

    SplitMapHelper splitHex2;
    int splitMapConnHex4rev[8][2] = {{1, 1}, {1, 0}, {2, 0}, {2, 1},
                                     {0, 1}, {0, 0}, {3, 0}, {3, 1}};

    splitHex2.size                         = 8;
    splitHex2.dir                          = 1;
    splitHex2.oppositeFace                 = 4;
    splitHex2.nEdgeToSplit                 = 4;
    splitHex2.edgesToSplit                 = splitedgehex4;
    splitHex2.edgeVert                     = helper(4, splitHex4EdgeVert);
    splitHex2.conn                         = helper(8, splitMapConnHex4rev);
    splitHex2.bfacesSize                   = 4;
    splitHex2.bfaces                       = splitMapBFacesHex4;
    splitHex2.nEdgeToCurve                 = 8;
    splitHex2.edgesToCurve                 = splitedgestocurvehex4;
    splitHex2.intEdgeFace                  = helper(4, splitHex4ExtEdgeFace);
    splitHex2.extEdgeFace                  = helper(4, splitHex4IntEdgeFace);
    splitHex2.blpDir                       = 0;
    splitHex2.gll                          = helper(8, splithex0gll);
    splitMap[LibUtilities::eHexahedron][2] = splitHex2;

    ////////////////////////////////////////
    // PRISM DIR Y
    ////////////////////////////////////////

    SplitMapHelper splitprism1;
    int splitMapBFacesPrism1[3]      = {0, 2, 4};
    int splitedgeprism1[3]           = {3, 1, 8};
    int splitPrism1EdgeVert[3][2]    = {{0, 3}, {1, 2}, {4, 5}};
    int splitMapConnPrism1[6][2]     = {{0, 0}, {1, 0}, {1, 1},
                                        {0, 1}, {2, 0}, {2, 1}};
    int splitedgestocurveprism1[6]   = {0, 4, 5, 2, 6, 7};
    int splitprism1gll[6][3]         = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1},
                                        {-1, 1, -1},  {-1, -1, 1}, {-1, 1, 1}};
    int splitPrism1IntEdgeFace[3][2] = {{0, 0}, {4, 4}, {5, 2}};
    int splitPrism1ExtEdgeFace[3][2] = {{2, 0}, {6, 2}, {7, 4}};

    splitprism1.size                  = 6;
    splitprism1.dir                   = 0;
    splitprism1.oppositeFace          = 3;
    splitprism1.nEdgeToSplit          = 3;
    splitprism1.edgesToSplit          = splitedgeprism1;
    splitprism1.edgeVert              = helper(3, splitPrism1EdgeVert);
    splitprism1.conn                  = helper(6, splitMapConnPrism1);
    splitprism1.bfacesSize            = 3;
    splitprism1.bfaces                = splitMapBFacesPrism1;
    splitprism1.nEdgeToCurve          = 6;
    splitprism1.edgesToCurve          = splitedgestocurveprism1;
    splitprism1.intEdgeFace           = helper(3, splitPrism1IntEdgeFace);
    splitprism1.extEdgeFace           = helper(3, splitPrism1ExtEdgeFace);
    splitprism1.blpDir                = 1;
    splitprism1.gll                   = helper(8, splitprism1gll);
    splitMap[LibUtilities::ePrism][1] = splitprism1;

    SplitMapHelper splitprism3;
    int splitMapConnPrism1rev[6][2]   = {{0, 1}, {1, 1}, {1, 0},
                                         {0, 0}, {2, 1}, {2, 0}};
    splitprism3.size                  = 6;
    splitprism3.dir                   = 1;
    splitprism3.oppositeFace          = 1;
    splitprism3.nEdgeToSplit          = 3;
    splitprism3.edgesToSplit          = splitedgeprism1;
    splitprism3.edgeVert              = helper(3, splitPrism1EdgeVert);
    splitprism3.conn                  = helper(6, splitMapConnPrism1rev);
    splitprism3.bfacesSize            = 3;
    splitprism3.bfaces                = splitMapBFacesPrism1;
    splitprism3.nEdgeToCurve          = 6;
    splitprism3.edgesToCurve          = splitedgestocurveprism1;
    splitprism3.intEdgeFace           = helper(3, splitPrism1ExtEdgeFace);
    splitprism3.extEdgeFace           = helper(3, splitPrism1IntEdgeFace);
    splitprism3.blpDir                = 1;
    splitprism3.gll                   = helper(8, splitprism1gll);
    splitMap[LibUtilities::ePrism][3] = splitprism3;

    // edgeMap associates geometry edge IDs to the (nl+1) vertices which are
    // generated along that edge when a prism is split, and is used to avoid
    // generation of duplicate vertices. It is stored as an unordered map for
    // speed.
    unordered_map<int, vector<NodeSharedPtr>> edgeMap;
    unordered_map<int, vector<NodeSharedPtr>>::iterator eIt;

    string surf = m_config["surf"].as<string>();
    if (surf.size() == 0)
    {
        m_log(WARNING) << "No surfaces detected to split, continuing." << endl;
        return;
    }
    vector<unsigned int> surfs;
    ParseUtils::GenerateSeqVector(surf.c_str(), surfs);
    sort(surfs.begin(), surfs.end());

    // If surface is defined, process list of elements to find those
    // that are connected to it.
    for (int i = 0; i < m_mesh->m_element[m_mesh->m_expDim].size(); ++i)
    {
        ElementSharedPtr el = m_mesh->m_element[m_mesh->m_expDim][i];
        int nSurf           = el->GetFaceCount();

        for (int j = 0; j < nSurf; ++j)
        {
            int bl = el->GetBoundaryLink(j);
            if (bl == -1)
            {
                continue;
            }

            ElementSharedPtr bEl = m_mesh->m_element[m_mesh->m_expDim - 1][bl];
            vector<int> tags     = bEl->GetTagList();
            vector<int> inter;

            sort(tags.begin(), tags.end());
            set_intersection(surfs.begin(), surfs.end(), tags.begin(),
                             tags.end(), back_inserter(inter));
            ASSERTL0(inter.size() <= 1, "Intersection of surfaces wrong");

            if (inter.size() == 1)
            {
                if (el->GetConf().m_e == LibUtilities::eHexahedron)
                {
                    map<int, SplitMapHelper>::iterator f =
                        splitMap[LibUtilities::eHexahedron].find(j);
                    if (f == splitMap[LibUtilities::eHexahedron].end())
                    {
                        m_log(WARNING) << "Splitting hex on face " << j
                                       << " unsupported" << endl;
                        continue;
                    }

                    if (splitEls.count(el->GetId()) > 0)
                    {
                        m_log(WARNING) << "Hex already found; "
                                       << "ignoring" << endl;
                    }

                    splitEls[el->GetId()] = j;
                }
                else if (el->GetConf().m_e == LibUtilities::ePrism)
                {
                    map<int, SplitMapHelper>::iterator f =
                        splitMap[LibUtilities::ePrism].find(j);
                    if (f == splitMap[LibUtilities::ePrism].end())
                    {
                        m_log(WARNING) << "Splitting prism on face " << j
                                       << " unsupported" << endl;
                        continue;
                    }

                    if (splitEls.count(el->GetId()) > 0)
                    {
                        m_log(WARNING) << "Prism already found; "
                                       << "ignoring" << endl;
                    }

                    splitEls[el->GetId()] = j;
                }
                else if (validElTypes.count(el->GetConf().m_e) == 0)
                {
                    m_log(WARNING) << "Unsupported element type "
                                   << "found in surface " << j << "; "
                                   << "ignoring" << endl;
                    continue;
                }
            }
        }
    }

    if (splitEls.size() == 0)
    {
        m_log(WARNING) << "No elements detected to split; continuing." << endl;
        return;
    }

    // Erase all elements from the element list. Elements will be
    // re-added as they are split.
    vector<ElementSharedPtr> el = m_mesh->m_element[m_mesh->m_expDim];
    m_mesh->m_element[m_mesh->m_expDim].clear();

    // Start of the new module, improving surface accuracy from here !!!
    map<int, ElementSharedPtr> FaceIDtoEL2D;
    map<int, SpatialDomains::Geometry3D *> geomMap;
    map<int, SpatialDomains::SegGeom *> edgeGeomMap;
    SpatialDomains::EntityHolder holder;

    for (int i = 0; i < el.size(); ++i)
    {
        const int elId = el[i]->GetId();
        sIt            = splitEls.find(elId);
        if (sIt == splitEls.end())
        {
            continue;
        }

        // Get elemental geometry object and put into map.
        geomMap[elId] = dynamic_cast<SpatialDomains::Geometry3D *>(
            el[i]->GetGeom(m_mesh->m_spaceDim, holder));

        // Get all edge geometry too for evaluations.
        for (int j = 0; j < el[i]->GetEdgeCount(); j++)
        {
            EdgeSharedPtr e = el[i]->GetEdge(j);
            auto f          = edgeGeomMap.find(e->m_id);
            if (f == edgeGeomMap.end())
            {
                edgeGeomMap[e->m_id] = e->GetGeom(m_mesh->m_spaceDim, holder);
            }
        }

        // KK add connectivity between element[2] vs element[3] in a map<el[2]
        // index, index-el[3]>
        for (int j = 0; j < el[i]->GetFaceCount(); j++)
        {
            if (el[i]->GetFace(j)->m_elLink.size() == 1)
            {
                // find the element 2D layer
                int ElIndex2D = el[i]->GetBoundaryLink(j);

                FaceIDtoEL2D[el[i]->GetFace(j)->m_id] =
                    m_mesh->m_element[2][ElIndex2D];
            }
        }
    }

    // node id to element id to para
    map<int, map<int, NekDouble>> paraElm; // parametric WRT to element
    map<int, map<int, NekDouble>> paraEdg; // parametric WRT edge

    map<int, vector<ElementSharedPtr>> elToStack;

    // 1. Create a copy vector of the edges inside the PrismLayer.
    vector<EdgeSharedPtr> PrismEdgesCopy;
    vector<int> PrismEdgesCopyID;

    for (int i = 0; i < el.size(); ++i)
    {
        const int elId = el[i]->GetId();
        sIt            = splitEls.find(elId);

        // 1.1. Any elements that are not boundary prisms marked for splitting
        // are ignored.
        if (sIt == splitEls.end())
        {
            // m_mesh->m_element[m_mesh->m_expDim].push_back(el[i]);
            continue;
        }

        // 1/2. Insert a copy of the edges (unique)
        for (auto edge : el[i]->GetEdgeList())
        {
            auto it = find(PrismEdgesCopyID.begin(), PrismEdgesCopyID.end(),
                           edge->m_id);
            if (it == PrismEdgesCopyID.end())
            {

                // 1.2.1 Create a new edge with the same m_n1 and m_n2
                EdgeSharedPtr edgeCopy =
                    std::shared_ptr<Edge>(new Edge(edge->m_n1, edge->m_n2));

                // 1.2.2 Put the same id of the new edge
                edgeCopy->m_id = edge->m_id;
                // 1.2.3 Add the same element links
                edgeCopy->m_elLink = edge->m_elLink;

                //  1.2.4 Add the ID and the Edge pointer to the vector
                PrismEdgesCopyID.push_back(edgeCopy->m_id);
                PrismEdgesCopy.push_back(edgeCopy);
            }
        }
    }

    // 2. Create the edge splitting and fill the PrismEdgesCopy with BL
    // distribution with correct directionality Grab the boundary layer points
    // distributions.
    LibUtilities::PointsKey bkey(nl + 1, LibUtilities::eBoundaryLayerPoints, r);
    Array<OneD, NekDouble> blp;
    LibUtilities::PointsManager()[bkey]->GetPoints(blp);

    // Reversed BL points in -1 1
    LibUtilities::PointsKey brkey(nl + 1, LibUtilities::eBoundaryLayerPointsRev,
                                  r);
    Array<OneD, NekDouble> blpr;
    LibUtilities::PointsManager()[brkey]->GetPoints(blpr);

    Nektar::Array<Nektar::OneD, Nektar::NekDouble> BLPoints(blp.size());

    for (int i = 0; i < el.size();
         ++i) // this might be speeded up, if in 1. a new list of pointer
              // elements is created (it will be more RAM costly)
    {
        const int elId = el[i]->GetId();
        sIt            = splitEls.find(elId);

        // 2.1. Any elements that are not boundary prisms marked for splitting
        // are ignored.
        if (sIt == splitEls.end())
        {
            continue;
        }

        // 2.2 Identify Edges to be split - only Prisms supported (Fill
        // EdgesToSplit with the copy of the edges)
        if (el[i]->GetShapeType() != 7)
        {
            m_log(FATAL)
                << "KK-This module is WIP and supports only prism elements at "
                   "the moment. The Legacy ProcessBL supported Tets & Hexes as "
                   "well. Check for pyramids in the mesh."
                << endl;
        }

        Array<OneD, EdgeSharedPtr> EdgesToSplit(3);
        EdgesToSplit[0] = el[i]->GetEdge(1);
        EdgesToSplit[1] = el[i]->GetEdge(3);
        EdgesToSplit[2] = el[i]->GetEdge(8);

        // 2.3 Identify the starting vertices/face in the BL

        Array<OneD, NodeSharedPtr> SurfaceVertices(3);
        if (el[i]->GetFace(3)->m_elLink.size() == 1 &&
            el[i]->GetFace(1)->m_elLink.size() == 2)
        {
            SurfaceVertices[0] = el[i]->GetFace(3)->m_vertexList[0];
            SurfaceVertices[1] = el[i]->GetFace(3)->m_vertexList[1];
            SurfaceVertices[2] = el[i]->GetFace(3)->m_vertexList[2];
        }
        else if (el[i]->GetFace(1)->m_elLink.size() == 1 &&
                 el[i]->GetFace(3)->m_elLink.size() == 2)
        {
            // else if(el[i]->GetFace(1)->m_faceNodes.size()!=0)

            SurfaceVertices[0] = el[i]->GetFace(1)->m_vertexList[0];
            SurfaceVertices[1] = el[i]->GetFace(1)->m_vertexList[1];
            SurfaceVertices[2] = el[i]->GetFace(1)->m_vertexList[2];
        }
        else
        {
            m_log(FATAL) << "Neither Triangular Faces of the Prism is "
                            "associated with Boundary. "
                         << endl;
        }

        // 2.4 Loop over the Edges to be split -> need to populate EdgeNodes of
        // the CopyEdges !
        for (int j = 0; j < EdgesToSplit.size(); j++)
        {
            EdgeSharedPtr edge = EdgesToSplit[j];

            auto EdgeID = find(PrismEdgesCopyID.begin(), PrismEdgesCopyID.end(),
                               edge->m_id);

            EdgeSharedPtr edgeCopy =
                PrismEdgesCopy[EdgeID - PrismEdgesCopyID.begin()];

            // 2.4.1 Skip all edge copies that already have the BL distribution
            // as EdgeNodes
            if (edgeCopy->m_edgeNodes.size() != 0)
            {
                continue;
            }

            // 2.4.2 Find orientation of the edge
            auto it = find(SurfaceVertices.begin(), SurfaceVertices.end(),
                           edge->m_n1);
            if (it != SurfaceVertices.end())
            {
                BLPoints = blp;
            }
            else
            {
                BLPoints = blpr;
            }

            // 2.4.3 Create the EDGE expansion (Important to do it on the
            // original Curved edge to keep the VarOpti curvature in the Edge
            // 1/3/8 of the prism )
            auto geom = edge->GetGeom(m_mesh->m_spaceDim, holder);
            geom->FillGeom();

            StdRegions::StdExpansionSharedPtr xmap = geom->GetXmap();
            Array<OneD, NekDouble> coeffs0         = geom->GetCoeffs(0);
            Array<OneD, NekDouble> coeffs1         = geom->GetCoeffs(1);
            Array<OneD, NekDouble> coeffs2         = geom->GetCoeffs(2);
            Array<OneD, NekDouble> xc(xmap->GetTotPoints());
            Array<OneD, NekDouble> yc(xmap->GetTotPoints());
            Array<OneD, NekDouble> zc(xmap->GetTotPoints());
            xmap->BwdTrans(coeffs0, xc);
            xmap->BwdTrans(coeffs1, yc);
            xmap->BwdTrans(coeffs2, zc);

            // 2.4.4 Generate the new BL vertices, add them to
            // m_mesh->VertexList, add them as EdgeNodes to the CopyEdge
            for (int k = 1; k < nl; k++)
            {
                Array<OneD, NekDouble> xp(1);
                xp[0] = BLPoints[k];

                std::array<NekDouble, 3> loc;
                loc[0] = xmap->PhysEvaluate(xp, xc);
                loc[1] = xmap->PhysEvaluate(xp, yc);
                loc[2] = xmap->PhysEvaluate(xp, zc);

                NodeSharedPtr NewBLVertex = NodeSharedPtr(new Node(
                    m_mesh->m_vertexSet.size(), loc[0], loc[1], loc[2]));

                // If this edge is attached to some CAD, then perform a
                // reverse projection to determine parametrisation on the
                // CAD curve/surface.
                if (edge->m_parentCAD)
                {
                    if (edge->m_parentCAD->GetType() == CADType::eCurve)
                    {
                        CADCurveSharedPtr c =
                            std::dynamic_pointer_cast<CADCurve>(
                                edge->m_parentCAD);
                        NekDouble t;
                        c->loct(loc, t);
                        NewBLVertex->SetCADCurve(c, t);
                    }
                    else if (edge->m_parentCAD->GetType() == CADType::eSurf)
                    {
                        CADSurfSharedPtr s = std::dynamic_pointer_cast<CADSurf>(
                            edge->m_parentCAD);
                        auto uv = s->locuv(loc);
                        NewBLVertex->SetCADSurf(s, uv);
                    }
                }

                // Add the Vertex to m_mesh and to the edgeCopy edgenodes
                m_mesh->m_vertexSet.insert(NewBLVertex);

                edgeCopy->m_edgeNodes.push_back(NewBLVertex);
            }
        }
    }

    // 3.Split Elements High-order-> Create edges ; Create New Elements ;
    // Translate the FaceNodes !!!

    for (int i = 0; i < el.size(); ++i)
    {
        const int elId = el[i]->GetId();
        sIt            = splitEls.find(elId);

        // 3.0. Any elements that are not boundary prisms marked for splitting
        // are ignored.
        if (sIt == splitEls.end())
        {
            m_mesh->m_element[m_mesh->m_expDim].push_back(el[i]);
            continue;
        }

        ElementSharedPtr el_macro = el[i];

        // 3.1 Check the faces with boundary for the element[2]
        map<int, ElementSharedPtr> LocalFaceIDtoBoundaryEl_2D_MAP;
        for (int j = 0; j < el_macro->GetFaceCount(); j++)
        {
            // create a vector with Macro Face Ids, that will loop for every new
            // element and will create the element[2] with the same config
            if (el_macro->GetBoundaryLink(j) != -1)
            {
                LocalFaceIDtoBoundaryEl_2D_MAP[j] =
                    FaceIDtoEL2D[el_macro->GetFace(j)->m_id];
            }
        }

        // 3.2. Create the expansion of the Macro Prism Element
        auto geom = el_macro->GetGeom(3, holder);
        geom->FillGeom();

        StdRegions::StdExpansionSharedPtr xmap = geom->GetXmap();
        Array<OneD, NekDouble> coeffs0         = geom->GetCoeffs(0);
        Array<OneD, NekDouble> coeffs1         = geom->GetCoeffs(1);
        Array<OneD, NekDouble> coeffs2         = geom->GetCoeffs(2);

        Array<OneD, NekDouble> xc(xmap->GetTotPoints());
        Array<OneD, NekDouble> yc(xmap->GetTotPoints());
        Array<OneD, NekDouble> zc(xmap->GetTotPoints());

        // Does Backwards transform from Std to Cartesian based on quadrature
        // nodes
        xmap->BwdTrans(coeffs0, xc); // uses the expansion coeffs in STD to fill
                                     // all Element nodes in Physical spaces
        xmap->BwdTrans(coeffs1, yc);
        xmap->BwdTrans(coeffs2, zc);

        // 3.3. Create Orientation Markers
        int FaceOrient = 0;
        Array<OneD, NodeSharedPtr> SurfaceVertices(3);

        if (el_macro->GetFace(1)->m_elLink.size() == 2 &&
            el_macro->GetFace(3)->m_elLink.size() == 1)
        { // Curved ->Face(3)
            FaceOrient         = -1;
            SurfaceVertices[0] = el[i]->GetFace(3)->m_vertexList[0];
            SurfaceVertices[1] = el[i]->GetFace(3)->m_vertexList[1];
            SurfaceVertices[2] = el[i]->GetFace(3)->m_vertexList[2];
        }
        else if (el_macro->GetFace(1)->m_elLink.size() == 1 &&
                 el_macro->GetFace(3)->m_elLink.size() == 2)
        { // Curved Face(1)
            FaceOrient         = 1;
            SurfaceVertices[0] = el[i]->GetFace(1)->m_vertexList[0];
            SurfaceVertices[1] = el[i]->GetFace(1)->m_vertexList[1];
            SurfaceVertices[2] = el[i]->GetFace(1)->m_vertexList[2];
        }
        else
        {
            m_log(FATAL)
                << "KK Development - orientation of the Prism faces failed - "
                   "likely Prism that has CAD on both triangular faces. "
                << endl;
        }

        // 3.4 Create Vector of Vertices for the Copy Edges - always starting
        // from the Boundary/Curved Face ! 3.4.1 Find the corresponding
        // copyedges //

        // Create a MAP instead of vector
        EdgeSharedPtr EdgeCopy_1 =
            PrismEdgesCopy[(find(PrismEdgesCopyID.begin(),
                                 PrismEdgesCopyID.end(),
                                 el_macro->GetEdge(1)->m_id)) -
                           PrismEdgesCopyID.begin()];
        EdgeSharedPtr EdgeCopy_3 =
            PrismEdgesCopy[(find(PrismEdgesCopyID.begin(),
                                 PrismEdgesCopyID.end(),
                                 el_macro->GetEdge(3)->m_id)) -
                           PrismEdgesCopyID.begin()];
        EdgeSharedPtr EdgeCopy_8 =
            PrismEdgesCopy[(find(PrismEdgesCopyID.begin(),
                                 PrismEdgesCopyID.end(),
                                 el_macro->GetEdge(8)->m_id)) -
                           PrismEdgesCopyID.begin()];

        // Add a check that EdgeCopy1->m_n2 EdgeCopy3->m_n2 and EdgeCopy8->m_n2
        // are Surface edges together else needs rearangment

        // 3.4.2 Rearange the vectors with BL vertices, so we always start the
        // splitting from the Boundary/Curved Face
        vector<NodeSharedPtr> EdgeBL_1, EdgeBL_3, EdgeBL_8;
        auto it1 = find(SurfaceVertices.begin(), SurfaceVertices.end(),
                        EdgeCopy_1->m_n1);
        auto it3 = find(SurfaceVertices.begin(), SurfaceVertices.end(),
                        EdgeCopy_3->m_n1);
        auto it8 = find(SurfaceVertices.begin(), SurfaceVertices.end(),
                        EdgeCopy_8->m_n1);

        // int MacroEdgeOrient = 0 ;
        if (it1 != SurfaceVertices.end() && it3 != SurfaceVertices.end() &&
            it8 != SurfaceVertices.end())
        {
            BLPoints = blp;
            //  MacroEdgeOrient = 1 ;
        }
        else if (it1 == SurfaceVertices.end() && it3 == SurfaceVertices.end() &&
                 it8 == SurfaceVertices.end())
        {
            BLPoints = blpr;
            // MacroEdgeOrient = -1 ;
        }
        else
        {
            m_log(FATAL) << "KK Development - PrismBL orientation is mixed, "
                            "need to rearange so it start from the surface !!! "
                         << endl;
        }

        EdgeBL_1 = EdgeCopy_1->m_edgeNodes;
        EdgeBL_1.insert(EdgeBL_1.begin(), EdgeCopy_1->m_n1);
        EdgeBL_1.push_back(
            EdgeCopy_1
                ->m_n2); // Is EdgeCopy_1->m_n1 same as el_macro->GetEdge(1)

        EdgeBL_3 = EdgeCopy_3->m_edgeNodes;
        EdgeBL_3.insert(EdgeBL_3.begin(), EdgeCopy_3->m_n1);
        EdgeBL_3.push_back(EdgeCopy_3->m_n2);

        EdgeBL_8 = EdgeCopy_8->m_edgeNodes;
        EdgeBL_8.insert(EdgeBL_8.begin(), EdgeCopy_8->m_n1);
        EdgeBL_8.push_back(EdgeCopy_8->m_n2);

        // 3.5 Create HO Elements starting from m_n1  !!!
        vector<NodeSharedPtr> NodeList(el_macro->GetVertexCount());
        LibUtilities::ShapeType elType = el_macro->GetConf().m_e;

        for (int j = 0; j < nl; j++)
        {
            // 3.5.1 Create the NodeList
            NodeList[0] = EdgeBL_3[j];
            NodeList[1] = EdgeBL_1[j];
            NodeList[2] = EdgeBL_1[j + 1];
            NodeList[3] = EdgeBL_3[j + 1];
            NodeList[4] = EdgeBL_8[j];
            NodeList[5] = EdgeBL_8[j + 1];

            // 3.5.2 Create the Linear Element
            ElmtConfig conf(elType, 1, false, false, false);
            ElementSharedPtr elmt_new = GetElementFactory().CreateInstance(
                elType, conf, NodeList, el_macro->GetTagList());
            elmt_new->m_parentCAD = el_macro->m_parentCAD;

            // 3.5.3 Copy CAD Dependencies

            // 3.5.4 For the inner layers create HO EdgeNodes
            EdgeSharedPtr EdgeNew_2, EdgeNew_6, EdgeNew_7;
            int LocalFaceToCurve;
            Array<OneD, NekDouble> xp(3);
            if (FaceOrient == -1)
            {
                // FACE 3 curved
                EdgeNew_2        = elmt_new->GetEdge(2);
                EdgeNew_6        = elmt_new->GetEdge(6);
                EdgeNew_7        = elmt_new->GetEdge(7);
                xp[1]            = BLPoints[j + 1];
                LocalFaceToCurve = 3;
            }
            else
            {
                // FACE 1 curved
                EdgeNew_2        = elmt_new->GetEdge(0);
                EdgeNew_6        = elmt_new->GetEdge(5);
                EdgeNew_7        = elmt_new->GetEdge(4);
                xp[1]            = BLPoints[j];
                LocalFaceToCurve = 1;
            }

            for (int k = 1; k < nq - 1; k++)
            {
                NekDouble tb = -1.0;
                NekDouble te = 1.0;
                // EdgeNew_2 - assumer V2-V3 orientation
                if (EdgeNew_2->m_edgeNodes.size() != gll.size() - 2)
                {
                    // tb =-1;
                    // te= 1 ;

                    xp[0] = tb * (1.0 - gll[k]) / 2.0 +
                            te * (1.0 + gll[k]) /
                                2.0; // gll[k] ; // tb * (1.0 - gll[k]) / 2.0 +
                                     // te * (1.0 + gll[k]) / 2.0;
                    xp[2] = -1;

                    Array<OneD, NekDouble> loc(3);
                    loc[0] = xmap->PhysEvaluate(xp, xc);
                    loc[1] = xmap->PhysEvaluate(xp, yc);
                    loc[2] = xmap->PhysEvaluate(xp, zc);

                    EdgeNew_2->m_curveType =
                        LibUtilities::PointsManager()[ekey]->GetPointsType();
                    EdgeNew_2->m_edgeNodes.push_back(
                        NodeSharedPtr(new Node(0, loc[0], loc[1], loc[2])));
                }

                // EdgeNew_7 - assumer V3-V5 orientation
                if (EdgeNew_7->m_edgeNodes.size() != gll.size() - 2)
                {
                    xp[0] = -1;
                    xp[2] =
                        tb * (1.0 - gll[k]) / 2.0 + te * (1.0 + gll[k]) / 2.0;

                    Array<OneD, NekDouble> loc(3);
                    loc[0] = xmap->PhysEvaluate(xp, xc);
                    loc[1] = xmap->PhysEvaluate(xp, yc);
                    loc[2] = xmap->PhysEvaluate(xp, zc);

                    EdgeNew_7->m_curveType =
                        LibUtilities::PointsManager()[ekey]->GetPointsType();
                    EdgeNew_7->m_edgeNodes.push_back(
                        NodeSharedPtr(new Node(0, loc[0], loc[1], loc[2])));
                }

                // EdgeNew_6 - assumer V3-V5 orientation
                if (EdgeNew_6->m_edgeNodes.size() != gll.size() - 2)
                {
                    xp[0] = -1 * (tb * (1.0 - gll[k]) / 2.0 +
                                  te * (1.0 + gll[k]) / 2.0);
                    xp[2] =
                        (tb * (1.0 - gll[k]) / 2.0 + te * (1.0 + gll[k]) / 2.0);

                    Array<OneD, NekDouble> loc(3);
                    loc[0] = xmap->PhysEvaluate(xp, xc);
                    loc[1] = xmap->PhysEvaluate(xp, yc);
                    loc[2] = xmap->PhysEvaluate(xp, zc);

                    EdgeNew_6->m_curveType =
                        LibUtilities::PointsManager()[ekey]->GetPointsType();
                    EdgeNew_6->m_edgeNodes.push_back(
                        NodeSharedPtr(new Node(0, loc[0], loc[1], loc[2])));
                }
            }

            // 3.5.6 Add  the FaceNodes to and FaceNodes as a translation of the
            // BL distribution in Xi2 direction

            if (elmt_new->GetFace(LocalFaceToCurve)->m_faceNodes.size() == 0)
            {
                // int faceNodeId= 0 ;
                for (int k = 3 * (nq - 1), id = 0; k < u_FaceNodes.size();
                     k++, id++)
                {

                    xp[0] = u_FaceNodes[k];
                    // xp[1] =  BLPoints[j+1];
                    xp[2] = v_FaceNodes[k];

                    Array<OneD, NekDouble> loc(3);
                    loc[0] = xmap->PhysEvaluate(xp, xc);
                    loc[1] = xmap->PhysEvaluate(xp, yc);
                    loc[2] = xmap->PhysEvaluate(xp, zc);

                    NodeSharedPtr new_faceNode =
                        NodeSharedPtr(new Node(id, loc[0], loc[1], loc[2]));

                    elmt_new->GetFace(LocalFaceToCurve)
                        ->m_faceNodes.push_back(new_faceNode);
                }

                elmt_new->GetFace(LocalFaceToCurve)->m_curveType =
                    LibUtilities::PointsManager()[pkey]->GetPointsType();
            }

            // 3.5.7 For the first or last layer try to reuse edges and Face
            // directly from el_macro
            if (j == 0 || j == nl - 1) // FaceOrient1 ->
            {
                // Edges check within the macro element
                vector<EdgeSharedPtr> el_macroEdges = el_macro->GetEdgeList();
                for (int k = 0; k < elmt_new->GetEdgeCount(); ++k)
                {
                    EdgeSharedPtr edge = elmt_new->GetEdge(k);
                    auto edge_it =
                        find(el_macroEdges.begin(), el_macroEdges.end(), edge);

                    if (edge_it != el_macroEdges.end())
                    {
                        // elmt_new->SetEdge(k, *edge_it);
                    }
                }

                // Faces check within the macro element
                vector<FaceSharedPtr> el_macroFaces = el_macro->GetFaceList();
                for (int k = 0; k < elmt_new->GetFaceCount(); k++)
                {
                    FaceSharedPtr face = elmt_new->GetFace(k);
                    auto face_it =
                        find(el_macroFaces.begin(), el_macroFaces.end(), face);
                    if (face_it != el_macroFaces.end())
                    {
                        // elmt_new->SetFace(k, *face_it);
                    }
                }
            }

            // 3.5.8 Create the Boundary Faces [ element[2]]
            // for orientation use  FaceOrient = 1 (Face 1 Curve -> Face 3
            // Linear)

            for (int k = 0; k < el_macro->GetFaceCount(); k++)
            {
                // check if the face has a boundary macro face
                if (LocalFaceIDtoBoundaryEl_2D_MAP[k])
                {

                    vector<int> tagBE =
                        LocalFaceIDtoBoundaryEl_2D_MAP[k]->GetTagList();

                    // Define the new element face index
                    int j_boundary;
                    if (FaceOrient == 1)
                    {
                        j_boundary = 0;
                    }
                    else
                    {
                        j_boundary = nl - 1;
                    }

                    // Reconstruct the triangular surface !!!
                    if (LocalFaceIDtoBoundaryEl_2D_MAP[k]->GetConf().m_e ==
                        LibUtilities::eTriangle)
                    {
                        if (j == j_boundary)
                        {
                            vector<int> tagBE;

                            tagBE =
                                m_mesh
                                    ->m_element[m_mesh->m_expDim - 1]
                                               [el_macro->GetBoundaryLink(k)]
                                    ->GetTagList();
                            ElmtConfig bconf(LibUtilities::eTriangle, 1, false,
                                             false, false);

                            vector<NodeSharedPtr> qNodeList =
                                elmt_new->GetFace(k)->m_vertexList;

                            ElementSharedPtr boundaryElmt =
                                GetElementFactory().CreateInstance(
                                    LibUtilities::eTriangle, bconf, qNodeList,
                                    tagBE);

                            m_mesh->m_element[2][el_macro->GetBoundaryLink(k)] =
                                boundaryElmt;

                            continue;
                        }
                        else
                        {
                            continue; // skip
                        }
                    }

                    // Reconstruct all Quadrilateral surfaces

                    if (LocalFaceIDtoBoundaryEl_2D_MAP[k]->GetConf().m_e !=
                        LibUtilities::eQuadrilateral)
                    {
                        m_log(WARNING)
                            << "Found non-quad element to split in "
                            << "surface " << surf << "; ignoring" << endl;
                        continue;
                    }

                    vector<NodeSharedPtr> qNodeList =
                        elmt_new->GetFace(k)->m_vertexList;
                    ElmtConfig bconf(LibUtilities::eQuadrilateral, 1, false,
                                     false, false);
                    ElementSharedPtr boundaryElmt =
                        GetElementFactory().CreateInstance(
                            LibUtilities::eQuadrilateral, bconf, qNodeList,
                            tagBE);

                    // Overwrite first layer boundary element with new
                    // boundary element, otherwise push this back to end of
                    // the boundary list
                    if (j == j_boundary)
                    {
                        m_mesh->m_element[2][el_macro->GetBoundaryLink(k)] =
                            boundaryElmt;
                    }
                    else
                    {
                        m_mesh->m_element[2].push_back(boundaryElmt);
                    }
                }
            }

            // 3.5.9 Add the element to m_mesh
            m_mesh->m_element[m_mesh->m_expDim].push_back(elmt_new);
        }

        // // 3.6 Delete the macro element 2D Boundary Faces and substitute them
        // with new for(int k = 0 ; k < el_macro->GetFaceCount() ; k++)
        // {
        //     FaceSharedPtr face = el_macro->GetFace(k) ;

        //     // Avoid all internal faces
        //     if(face->m_elLink.size() != 1 )
        //     {
        //         continue ;
        //     }
        // }
    }

    // boost::ignore_unused(nodeId, rExprId, ratioIsString);

    m_log(VERBOSE) << "Elements after  = " << m_mesh->m_element[3].size()
                   << endl;

    // ClearElementLinks();
    ProcessVertices();
    ProcessEdges();

    ProcessFaces();
    ProcessElements();
    ProcessComposites();
}
} // namespace Nektar::NekMesh
