////////////////////////////////////////////////////////////////////////////////
//
//  File: InputCGNS.cpp
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
//  Description: CGNS converter.
//
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

#include "InputCGNS.h"
#include <NekMesh/MeshElements/Element.h>
#include <SpatialDomains/MeshGraphIO.h>

namespace Nektar::NekMesh
{

using namespace Nektar::NekMesh;

ModuleKey InputCGNS::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eInputModule, "cgns"), InputCGNS::create, "Reads CGNS file.");
/**
 * @brief Set up InputCGNS object.
 */
InputCGNS::InputCGNS(MeshSharedPtr m) : InputModule(m)
{
    m_config["processall"] =
        ConfigOption(true, "0", "Process edges, faces as well as composites");
}

InputCGNS::~InputCGNS()
{
}

/**
 * @brief Add node to m_node and m_vertexSet
 */
void InputCGNS::SaveNode(int id, NekDouble x, NekDouble y, NekDouble z)
{
    NodeSharedPtr newNode = std::make_shared<Node>(id, x, y, z);
    m_mesh->m_node.push_back(newNode);
}

/**
 * CGNS file can be read using the CGNS Mid-Level Library
 * https://cgns.github.io/CGNS_docs_current/midlevel/index.html
 * Process() converts this information into .xml format for use in Nektar++
 */
void InputCGNS::Process()
{
    SetupElements();

    ProcessEdges();
    ProcessFaces();
    ProcessElements();
    ProcessComposites();

    PrintSummary();
}

void InputCGNS::SetupElements()
{
    string fname = m_config["infile"].as<string>();
    cgsize_t
        sizes[9]; // 9 is the max number required (for a 3D structured mesh):
    // [NVertexI, NVertexJ, NVertexK, NCellI, NCellJ, NCellK, NBoundVertexI,
    // NBoundVertexJ, NBoundVertexK]).
    // https://cgns.github.io/CGNS_docs_current/midlevel/structural.html#zone

    char zoneName[33]; // 33 is the max set by convention

    if (cg_open(fname.c_str(), CG_MODE_READ, &m_fileIndex))
    {
        m_log(FATAL) << "Error opening file: '" << fname << "'" << endl;
    }
    else
    {
        m_log(VERBOSE) << "Reading CGNS file: '" << fname << "'" << endl;
    }

    char baseName[33];
    int expDim, spaceDim;

    // Read the base information
    if (cg_base_read(m_fileIndex, m_baseIndex, baseName, &expDim, &spaceDim))
    {
        m_log(FATAL) << "Error reading base information" << endl;
        cg_close(m_fileIndex);
    }
    else
    {
        m_mesh->m_expDim   = expDim;
        m_mesh->m_spaceDim = spaceDim;
        m_log(VERBOSE) << "Read base information" << endl;
        m_log(VERBOSE) << "Base name    : " << baseName << endl;
        m_log(VERBOSE) << "Element dim  : " << m_mesh->m_expDim << endl;
        m_log(VERBOSE) << "Physical dim : " << m_mesh->m_spaceDim << endl;
    }

    ZoneType_t zoneType;

    // Read the zone type
    if (cg_zone_type(m_fileIndex, m_baseIndex, m_zoneIndex, &zoneType))
    {
        m_log(FATAL) << "Error reading zone type" << endl;
        cg_close(m_fileIndex);
    }
    else
    {
        ASSERTL0(zoneType == CGNS_ENUMV(Unstructured),
                 "Only unstructured meshes supported");
    }

    int nCoords = 0, n3DElems = 0;

    // Read the zone information
    if (cg_zone_read(m_fileIndex, m_baseIndex, m_zoneIndex, zoneName, sizes))
    {
        m_log(FATAL) << "Error reading zone information" << endl;
        cg_close(m_fileIndex);
    }
    else
    {
        m_log(VERBOSE) << "Read zone information" << endl;
        m_log(VERBOSE) << "Zone name   : " << zoneName << endl;
        // NOTE: the following two lines only work for unstructured meshes
        nCoords  = sizes[0];
        n3DElems = sizes[1];
        m_log(VERBOSE) << "Grid Points : " << nCoords << endl;
        m_log(VERBOSE) << "No of elmts : " << n3DElems << endl;
    }

    // Read in coordinates from the mesh
    m_x.resize(nCoords);
    m_y.resize(nCoords);
    m_z.resize(nCoords);

    cgsize_t rmin[3] = {1, 1, 1};
    cgsize_t rmax[3] = {nCoords, 1, 1}; // NOTE: for unstructured meshes

    if (cg_coord_read(m_fileIndex, m_baseIndex, m_zoneIndex, "CoordinateX",
                      RealDouble, rmin, rmax, m_x.data()) ||
        cg_coord_read(m_fileIndex, m_baseIndex, m_zoneIndex, "CoordinateY",
                      RealDouble, rmin, rmax, m_y.data()) ||
        cg_coord_read(m_fileIndex, m_baseIndex, m_zoneIndex, "CoordinateZ",
                      RealDouble, rmin, rmax, m_z.data()))
    {
        cg_close(m_fileIndex);
        m_log(FATAL) << "Error reading coordinates" << endl;
    }
    else
    {
        m_log(VERBOSE) << "Read coordinates" << endl;
    }

    // Create a node for each coord
    for (uint i{0}; i < nCoords; i++)
    {
        SaveNode(i, m_x[i], m_y[i], m_z[i]);
    }

    // Read number of sections
    int nSections;
    if (cg_nsections(m_fileIndex, m_baseIndex, m_zoneIndex, &nSections))
    {
        m_log(FATAL) << "Error reading number of sections." << endl;
        cg_close(m_fileIndex);
    }
    else
    {
        m_log(VERBOSE) << "Number of sections: " << nSections << endl;
    }

    // each pair contains the element type and the element's node indices
    vector<pair<ElementType_t, vector<int>>> elemInfo;
    map<string, pair<int, int>> SectionNameToRange;

    // Get the element vertices for each section
    for (int sectionInd = 1; sectionInd <= nSections; ++sectionInd)
    {
        char sectionName[33];
        ElementType_t elemType;
        cgsize_t start, end;
        int nbndry, parentFlag;

        if (cg_section_read(m_fileIndex, m_baseIndex, m_zoneIndex, sectionInd,
                            sectionName, &elemType, &start, &end, &nbndry,
                            &parentFlag))
        {
            m_log(FATAL) << "Error reading section information." << std::endl;
            cg_close(m_fileIndex);
        }
        else
        {
            int elemSize = end - start + 1;
            m_log(VERBOSE) << "Reading section data: " << endl;
            m_log(VERBOSE) << "        section name: " << sectionName << endl;
            m_log(VERBOSE) << "        section type: "
                           << ElementTypeName[elemType] << endl;
            m_log(VERBOSE) << "        section count: " << elemSize << endl;

            cgsize_t elemDataSize;

            cg_ElementDataSize(m_fileIndex, m_baseIndex, m_zoneIndex,
                               sectionInd, &elemDataSize);

            // store section name and range for boundary composite processing
            // if shape is specficied as seperate type
            LibUtilities::ShapeType shapeType = m_elemType2ShapeType[elemType];
            if (shapeType == LibUtilities::eTriangle ||
                shapeType == LibUtilities::eQuadrilateral)
            {
                SectionNameToRange[sectionName] =
                    pair<int, int>(elemInfo.size(), elemInfo.size() + elemSize);
            }

            if (elemType == CGNS_ENUMV(ElementTypeNull) ||
                elemType == CGNS_ENUMV(ElementTypeUserDefined) ||
                elemType == CGNS_ENUMV(NGON_n) ||
                elemType == CGNS_ENUMV(NFACE_n))
            {
                m_log(FATAL) << "Section type " << ElementTypeName[elemType]
                             << " not supported" << endl;
            }
            else if (elemType == CGNS_ENUMV(MIXED))
            {
                ExtractMixedElemInfo(elemDataSize, elemSize, sectionInd,
                                     elemInfo);
            }
            else
            {
                ExtractSeparatedElemInfo(elemDataSize, elemSize, elemType,
                                         sectionInd, elemInfo);
            }
        }
    }

    // read zone connectivity grid for periodic BCs if exist
    int nConns;
    map<string, set<int>> ConnNameToFaces;
    if (cg_nconns(m_fileIndex, m_baseIndex, m_zoneIndex, &nConns))
    {
        m_log(FATAL) << "Error reading number of connectivities." << std::endl;
        cg_close(m_fileIndex);
    }
    else
    {
        m_log(VERBOSE) << "Number of connectivities: " << nConns << std::endl;

        for (int connInd = 1; connInd <= nConns; ++connInd)
        {

            char connName[33], donorName[33];
            GridLocation_t location;
            GridConnectivityType_t connType;
            PointSetType_t ptsetType, donorPtsetType;
            cgsize_t npnts, ndataDonor;
            ZoneType_t donorZoneType;
            DataType_t donorDataType;

            if (cg_conn_info(m_fileIndex, m_baseIndex, m_zoneIndex, connInd,
                             connName, &location, &connType, &ptsetType, &npnts,
                             donorName, &donorZoneType, &donorPtsetType,
                             &donorDataType, &ndataDonor))
            {
                m_log(FATAL) << "Error reading connectivity info in  "
                             << connInd << std::endl;
                cg_close(m_fileIndex);
            }
            else
            {
                m_log(VERBOSE) << "Reading connectivity info" << endl;
            }
            // check to see points are given as a cell list if not give error
            if (donorPtsetType != CGNS_ENUMV(PointListDonor))
            {
                m_log(FATAL) << "Periodic boudnary curretly needs to be provide"
                                " as a point-list rather than a cell list"
                             << std::endl;
            }

            float RotCenter[3], RotAngle[3], Translation[3];

            // read connectivity transation information to add to name
            cg_conn_periodic_read(m_fileIndex, m_baseIndex, m_zoneIndex,
                                  connInd, RotCenter, RotAngle, Translation);

            string NameFull =
                string(connName) +
                ":Trans=" + boost::lexical_cast<string>(Translation[0]) + "," +
                boost::lexical_cast<string>(Translation[1]) + "," +
                boost::lexical_cast<string>(Translation[2]);

            cgsize_t *pnts = new cgsize_t[npnts];
            if (cg_conn_read_short(m_fileIndex, m_baseIndex, m_zoneIndex,
                                   connInd, pnts))
            {
                m_log(FATAL) << "Error reading connectivity section " << connInd
                             << std::endl;
                cg_close(m_fileIndex);
            }
            else
            {
                for (unsigned n = 0; n < npnts; ++n)
                {
                    ConnNameToFaces[NameFull].insert(pnts[n]);
                }

                m_log(VERBOSE) << "Added connectivity list for " << connName
                               << " using num faces: " << npnts << std::endl;
            }
        }
    }

    // create FaceNodes, ElementFaces and BndElemenFaces from elemInfo
    std::unordered_map<int, vector<int>> FaceNodes;
    Array<OneD, vector<int>> ElementFaces(n3DElems);
    vector<int> BoundaryElems;
    vector<int> VolumeElems;
    ReadFaces(elemInfo, FaceNodes, ElementFaces, BoundaryElems, VolumeElems);

    // Close the CGNS file
    if (cg_close(m_fileIndex))
    {
        m_log(FATAL) << "Error closing CGNS file." << std::endl;
    }
    else
    {
        m_log(VERBOSE) << "Closing CGNS file." << std::endl;
    }

    // 3D Zone
    // Reset node ordering so that all prism faces have
    // consistent numbering for singular vertex re-ordering
    ResetNodes(m_mesh->m_node, ElementFaces, FaceNodes, elemInfo, VolumeElems);

    int nComposite = 0, i = 0;
    int nelements = ElementFaces.size();
    int cnt;

    // create Prisms first
    cnt = 0;
    for (i = 0; i < nelements; ++i)
    {
        Array<OneD, int> Nodes =
            SortFaceNodes(m_mesh->m_node, ElementFaces[i], FaceNodes);
        if (ElementFaces[i].size() == 5 && Nodes.size() == 6) // is a prism
        {
            GenElement3D(m_mesh->m_node, elemInfo[VolumeElems[i]].second,
                         ElementFaces[i], FaceNodes,
                         elemInfo[VolumeElems[i]].first, nComposite, true);
            ++cnt;
        }
    }
    m_log(VERBOSE) << "  - # of prisms: " << cnt << endl;

    if (cnt)
    {
        nComposite++;
    }

    // create Pyras second
    cnt = 0;
    for (i = 0; i < nelements; ++i)
    {
        Array<OneD, int> Nodes =
            SortFaceNodes(m_mesh->m_node, ElementFaces[i], FaceNodes);
        if (ElementFaces[i].size() == 5 && Nodes.size() == 5) // is a pyra
        {
            GenElement3D(m_mesh->m_node, elemInfo[VolumeElems[i]].second,
                         ElementFaces[i], FaceNodes,
                         elemInfo[VolumeElems[i]].first, nComposite, true);
            ++cnt;
        }
    }
    m_log(VERBOSE) << "  - # of pyramids: " << cnt << endl;

    if (cnt)
    {
        nComposite++;
    }

    // create Tets third
    cnt = 0;
    for (i = 0; i < nelements; ++i)
    {
        if (ElementFaces[i].size() == 4) // is a tetra
        {
            GenElement3D(m_mesh->m_node, elemInfo[VolumeElems[i]].second,
                         ElementFaces[i], FaceNodes,
                         elemInfo[VolumeElems[i]].first, nComposite, true);
            ++cnt;
        }
    }
    m_log(VERBOSE) << "  - # of tetrahedra: " << cnt << endl;

    if (cnt)
    {
        nComposite++;
    }

    // create Hexes fourth
    cnt = 0;
    for (i = 0; i < nelements; ++i)
    {
        if (ElementFaces[i].size() == 6) // is a hexa
        {
            GenElement3D(m_mesh->m_node, elemInfo[VolumeElems[i]].second,
                         ElementFaces[i], FaceNodes,
                         elemInfo[VolumeElems[i]].first, nComposite, true);
            ++cnt;
        }
    }
    m_log(VERBOSE) << "  - # of hexa: " << cnt << endl;
    if (cnt)
    {
        nComposite++;
    }

    // Insert vertices into map.
    for (auto &node : m_mesh->m_node)
    {
        m_mesh->m_vertexSet.insert(node);
    }

    // create Boundary elements last
    for (auto sec : SectionNameToRange)
    {
        cnt = 0;
        for (int i = sec.second.first; i < sec.second.second; ++i)
        {
            GenElement2D(m_mesh->m_node, elemInfo[i].second, elemInfo[i].first,
                         nComposite, true);
            ++cnt;
        }

        if (cnt)
        {
            m_mesh->m_faceLabels[nComposite] = sec.first;
            m_log(VERBOSE) << "  - # of bndry elmts in " + sec.first + " :"
                           << cnt << endl;
            nComposite++;
        }
    }

    // create periodic boundary elements  if present

    // set up list of faces that are not shared by two elements
    // and so are an external face
    set<FaceSharedPtr> facelist;
    if (ConnNameToFaces.size())
    {
        // process faces to make a unique list in m_mesh->m_faceSet
        ProcessFaces();

        for (auto &it : m_mesh->m_faceSet)
        {
            // if only has one element link and no tag id store element shared
            // ptr
            if (it->m_elLink.size() == 1)
            {
                facelist.insert(it);
            }
        }
    }

    for (auto &con : ConnNameToFaces)
    {
        // find faces with nodes along periodic points.
        cnt = 0;
        for (auto &it : facelist)
        {
            bool OnBoundary = true;
            for (int i = 0; i < it->m_vertexList.size(); ++i)
            {
                if (con.second.count(it->m_vertexList[i]->m_id + 1) == 0)
                {
                    OnBoundary = false;
                }
            }

            if (OnBoundary)
            {
                // using the face information generate a 2D element and add to
                // m_mesh_m_element

                vector<int> tags;
                tags.push_back(nComposite);

                bool faceNodes, volumeNodes;
                LibUtilities::ShapeType shapeType =
                    (it->m_edgeList.size() == 3) ? LibUtilities::eTriangle
                                                 : LibUtilities::eQuadrilateral;
                faceNodes   = it->m_faceNodes.size() ? true : false;
                volumeNodes = false;
                LibUtilities::PointsType edgeCurveType =
                    LibUtilities::ePolyEvenlySpaced;
                LibUtilities::PointsType faceCurveType =
                    (shapeType == LibUtilities::eTriangle)
                        ? LibUtilities::eNodalTriEvenlySpaced
                        : LibUtilities::ePolyEvenlySpaced;
                int order = it->m_edgeList[0]->m_edgeNodes.size() + 1;

                ElmtConfig conf(shapeType, order, faceNodes, volumeNodes, false,
                                edgeCurveType, faceCurveType);

                vector<NodeSharedPtr> nodeList;
                for (int i = 0; i < it->m_vertexList.size(); ++i)
                {
                    nodeList.push_back(it->m_vertexList[i]);
                }
                for (int i = 0; i < it->m_edgeList.size(); ++i)
                {
                    for (int j = 0; j < it->m_edgeList[i]->m_edgeNodes.size();
                         ++j)
                    {
                        nodeList.push_back(it->m_edgeList[i]->m_edgeNodes[j]);
                    }
                }
                for (int i = 0; i < it->m_faceNodes.size(); ++i)
                {
                    nodeList.push_back(it->m_faceNodes[i]);
                }

                ElementSharedPtr E = GetElementFactory().CreateInstance(
                    shapeType, conf, nodeList, tags);

                m_mesh->m_element[E->GetDim()].push_back(E);

                ++cnt;
            }
        }

        if (cnt)
        {
            m_mesh->m_faceLabels[nComposite] = con.first;
            m_log(VERBOSE) << "  - # of bndry elmts in " + con.first + " :"
                           << cnt << endl;
            nComposite++;
        }
    }
}

void InputCGNS::ExtractSeparatedElemInfo(
    cgsize_t elemDataSize, int elemSize, ElementType_t elemType, int sectionInd,
    vector<pair<ElementType_t, vector<int>>> &elemInfo)
{
    cgsize_t *ielem = new cgsize_t[elemDataSize];
    cgsize_t iparentdata;

    int nElemNodes;

    if (cg_elements_read(m_fileIndex, m_baseIndex, m_zoneIndex, sectionInd,
                         ielem, &iparentdata))
    {
        m_log(FATAL) << "Error reading element type " << elemType << endl;
    }
    else
    {
        m_log(VERBOSE) << "Read element type "
                       << ElementTypeName[m_ind2ElemType[elemType]] << endl;
    }

    // Get the number of nodes
    cg_npe(elemType, &nElemNodes);

    for (uint sectionElemIndex = 0; sectionElemIndex < elemSize;
         sectionElemIndex++)
    {
        elemInfo.push_back(pair<ElementType_t, vector<int>>(elemType, {}));

        for (uint vertIndex = sectionElemIndex * nElemNodes;
             vertIndex < (sectionElemIndex + 1) * nElemNodes; vertIndex++)
        {
            // -1 is to convert between index conventions (CGNS starts at 1)
            elemInfo[elemInfo.size() - 1].second.push_back(ielem[vertIndex] -
                                                           1);
        }
    }

    delete[] ielem;
}

void InputCGNS::ExtractMixedElemInfo(
    cgsize_t elemDataSize, int elemSize, int sectionInd,
    vector<pair<ElementType_t, vector<int>>> &elemInfo)
{

    if (elemSize == 0)
    {
        return;
    }

    cgsize_t *ielem      = new cgsize_t[elemDataSize];
    cgsize_t *offsetData = new cgsize_t[elemSize + 1];
    cgsize_t iparentdata;

    // The data array ElementStartOffset contains the starting
    // positions of each element in the ElementConnectivity data array
    // and its last value corresponds to the ElementConnectivity total
    // size.
    // https://cgns.github.io/CGNS_docs_current/sids/gridflow.html#Elements
    if (cg_poly_elements_read(m_fileIndex, m_baseIndex, m_zoneIndex, sectionInd,
                              ielem, offsetData, &iparentdata))
    {
        m_log(FATAL) << "Error reading element type MIXED" << endl;
    }
    else
    {
        m_log(VERBOSE) << "Read element type MIXED" << endl;
    }

    ElementType_t elemType;

    for (uint sectionElemIndex = 0; sectionElemIndex < elemSize;
         sectionElemIndex++)
    {
        elemType = m_ind2ElemType[ielem[offsetData[sectionElemIndex]]];

        elemInfo.push_back(pair<ElementType_t, vector<int>>(elemType, {}));

        for (uint vertIndex = offsetData[sectionElemIndex] + 1;
             vertIndex < offsetData[sectionElemIndex + 1]; vertIndex++)
        {
            elemInfo[elemInfo.size() - 1].second.push_back(ielem[vertIndex] -
                                                           1);
        }
    }

    delete[] ielem;
    delete[] offsetData;
}

/**
 * In the CGNS format, each element is defined by a list of vertices in a
 * certain order
 * To use ResetNodes, we need to create...
 * FaceNodes: the nodes that form each face (tri or quad)
 * ElementFaces: the faces that form each element (tet, pyra, prism, hex)
 * BoundaryElems: the faces that form the boundary
 */
void InputCGNS::ReadFaces(vector<pair<ElementType_t, vector<int>>> &elemInfo,
                          std::unordered_map<int, vector<int>> &FaceNodes,
                          Array<OneD, vector<int>> &ElementFaces,
                          vector<int> &BoundaryElems, vector<int> &VolumeElems)
{
    set<int> internalFaces = {};

    map<set<int>, int> faceSets;
    int volElemIdx = 0;

    for (int elemIdx = 0; elemIdx < elemInfo.size(); elemIdx++)
    {
        ElementType_t elemType            = elemInfo[elemIdx].first;
        vector<int> elemNodes             = elemInfo[elemIdx].second;
        LibUtilities::ShapeType shapeType = m_elemType2ShapeType[elemType];
        if (shapeType == LibUtilities::eTriangle ||
            shapeType == LibUtilities::eQuadrilateral)
        {
            BoundaryElems.push_back(elemIdx);
            continue;
        }

        VolumeElems.push_back(elemIdx);

        ASSERTL0((shapeType == LibUtilities::eTetrahedron ||
                  shapeType == LibUtilities::eHexahedron ||
                  shapeType == LibUtilities::ePyramid ||
                  shapeType == LibUtilities::ePrism),
                 "shapeType not recognised");

        // loop over each face
        for (vector<int> localFaceNodes : m_shapeType2LocElemNodes[shapeType])
        {
            // the set is used to check for uniqueness
            set<int> globalFaceNodeSet;
            // the vector is appended to FaceNodes
            vector<int> globalFaceNodeVector;

            for (int i : localFaceNodes)
            {
                globalFaceNodeSet.insert(elemNodes[i]);
                globalFaceNodeVector.push_back(elemNodes[i]);
            }

            auto testIns =
                faceSets.insert({globalFaceNodeSet, faceSets.size()});

            int faceIdx = testIns.first->second;
            ElementFaces[volElemIdx].push_back(faceIdx);

            if (testIns.second) // if the face doesn't already exist in set
            {
                FaceNodes[faceIdx] = globalFaceNodeVector;
            }
        }
        volElemIdx++;
    }
}

static void PrismLineFaces(int prismid, map<int, int> &facelist,
                           vector<vector<int>> &FaceToPrisms,
                           vector<vector<int>> &PrismToFaces,
                           vector<bool> &PrismDone)
{
    if (PrismDone[prismid] == false)
    {
        PrismDone[prismid] = true;

        // Add faces0
        int face       = PrismToFaces[prismid][0];
        facelist[face] = face;
        for (int i = 0; i < FaceToPrisms[face].size(); ++i)
        {
            PrismLineFaces(FaceToPrisms[face][i], facelist, FaceToPrisms,
                           PrismToFaces, PrismDone);
        }

        // Add faces1
        face           = PrismToFaces[prismid][1];
        facelist[face] = face;
        for (int i = 0; i < FaceToPrisms[face].size(); ++i)
        {
            PrismLineFaces(FaceToPrisms[face][i], facelist, FaceToPrisms,
                           PrismToFaces, PrismDone);
        }
    }
}

/**
 * Reorder the node IDs to set the orientation of the elments
 */
void InputCGNS::ResetNodes(vector<NodeSharedPtr> &Vnodes,
                           Array<OneD, vector<int>> &ElementFaces,
                           std::unordered_map<int, vector<int>> &FaceNodes,
                           vector<pair<ElementType_t, vector<int>>> &elemInfo,
                           vector<int> &VolumeElems)
{
    int i, j;
    vector<int> NodeReordering(Vnodes.size(), -1);
    int face1_map[3] = {0, 1, 4};
    int face3_map[3] = {3, 2, 5};
    int nodeid       = 0;
    int revNodeid    = Vnodes.size() - 1;
    map<int, bool> FacesRenumbered;

    // Determine Prism triangular face connectivity.
    vector<vector<int>> FaceToPrisms(FaceNodes.size());
    vector<vector<int>> PrismToFaces(ElementFaces.size());
    map<int, int> Prisms;

    map<int, int> Pyras;

    // the global tri/quad face IDs for each prism/pyra
    vector<vector<int>> GlobTriFaces(ElementFaces.size());
    vector<vector<int>> GlobQuadFaces(ElementFaces.size());

    // generate map of prism-faces to prisms and prism to
    // triangular-faces as well as ids of each prism.
    for (i = 0; i < ElementFaces.size(); ++i)
    {
        // Find Prism (and pyramids!).
        if (ElementFaces[i].size() == 5)
        {
            vector<int> LocTriFaces;
            // Find tri and quad faces faces
            for (j = 0; j < ElementFaces[i].size(); ++j)
            {
                if (FaceNodes[ElementFaces[i][j]].size() == 3)
                {
                    LocTriFaces.push_back(j);
                    GlobTriFaces[i].push_back(ElementFaces[i][j]);
                }
                else
                {
                    GlobQuadFaces[i].push_back(ElementFaces[i][j]);
                }
            }

            if (LocTriFaces.size() == 2) // prism otherwise a pyramid
            {
                Prisms[i] = i;

                PrismToFaces[i].push_back(ElementFaces[i][LocTriFaces[0]]);
                PrismToFaces[i].push_back(ElementFaces[i][LocTriFaces[1]]);

                FaceToPrisms[ElementFaces[i][LocTriFaces[0]]].push_back(i);
                FaceToPrisms[ElementFaces[i][LocTriFaces[1]]].push_back(i);
            }
            else if (LocTriFaces.size() == 4)
            {
                Pyras[i] = i;
            }
            else
            {
                ASSERTL0(
                    false,
                    "Not set up for elements which are not Prism or Pyramid");
            }
        }
    }

    // set the apex node as node in each pyra and ensure any neighbouring
    // prisms align
    for (auto &PyraIt : Pyras)
    {
        // find the ID of the apex node (the node in any of the tri faces which
        // is not also in the quad (base) face)
        int apexID            = -1;
        int pyraElemIdx       = PyraIt.second;
        vector<int> baseNodes = FaceNodes[GlobQuadFaces[pyraElemIdx][0]];

        bool pyramidShielded = false;

        // assert that none of the base nodes have already been ID'd
        for (int id : baseNodes)
        {
            if (NodeReordering[id] == -1)
            {
                continue;
            }

            // attempt to resolve issues with impossible meshes BUT this is
            // ideally resolved in the mesh generator
            PyramidShielding(Vnodes, ElementFaces, FaceNodes, elemInfo,
                             VolumeElems, NodeReordering, pyraElemIdx);

            pyramidShielded = true;
            break;
        }

        // skip the rest of ResetNodes for this pyramid
        if (pyramidShielded)
        {
            break;
        }

        // the choice of the 0th tri face is arbitrary
        for (int id : FaceNodes[GlobTriFaces[pyraElemIdx][0]])
        {
            // if a node `id` is not in the base, it must be the apex
            if (find(baseNodes.begin(), baseNodes.end(), id) == baseNodes.end())
            {
                apexID = id;
                break;
            }
        }

        ASSERTL0(apexID != -1, "Apex node not found in pyramid");

        // if apex node hasn't already been given an ID (because of sharing it
        // with another pyramid)
        if (NodeReordering[apexID] == -1)
        {
            NodeReordering[apexID] = revNodeid--;
        }

        // traverse along each prism line that exists out of the pyra
        for (int faceId : GlobTriFaces[pyraElemIdx]) // for each tri face
        {
            TraversePyraPrismLine(pyraElemIdx, faceId, apexID, FaceToPrisms,
                                  GlobTriFaces, Vnodes, ElementFaces, FaceNodes,
                                  NodeReordering, revNodeid);
        }
    }

    // For every prism find the list of prismatic elements
    // that represent an aligned block of cells. Then renumber
    // these blocks consecutively
    vector<bool> FacesDone(FaceNodes.size(), false);
    vector<bool> PrismDone(ElementFaces.size(), false);

    for (auto &PrismIt : Prisms)
    {
        int elmtid = PrismIt.first;
        map<int, int> facelist;

        if (PrismDone[elmtid])
        {
            continue;
        }
        else
        {
            // Generate list of faces in list
            PrismLineFaces(elmtid, facelist, FaceToPrisms, PrismToFaces,
                           PrismDone);
        }
        // loop over faces and number vertices of associated prisms.
        for (auto &faceIt : facelist)
        {
            int faceid = faceIt.second;

            for (i = 0; i < FaceToPrisms[faceid].size(); ++i)
            {
                int prismid = FaceToPrisms[faceid][i];

                if ((FacesDone[PrismToFaces[prismid][0]] == true) &&
                    (FacesDone[PrismToFaces[prismid][1]] == true))
                {
                    continue;
                }

                Array<OneD, int> Nodes =
                    SortFaceNodes(Vnodes, ElementFaces[prismid], FaceNodes);

                if ((FacesDone[PrismToFaces[prismid][0]] == false) &&
                    (FacesDone[PrismToFaces[prismid][1]] == false))
                {
                    // number all nodes consecutive since already correctly
                    // re-arranged.
                    for (i = 0; i < 3; ++i)
                    {
                        if (NodeReordering[Nodes[face1_map[i]]] == -1)
                        {
                            NodeReordering[Nodes[face1_map[i]]] = nodeid++;
                        }
                    }

                    for (i = 0; i < 3; ++i)
                    {
                        if (NodeReordering[Nodes[face3_map[i]]] == -1)
                        {
                            NodeReordering[Nodes[face3_map[i]]] = nodeid++;
                        }
                    }
                }
                else if ((FacesDone[PrismToFaces[prismid][0]] == false) &&
                         (FacesDone[PrismToFaces[prismid][1]] == true))
                {
                    // find node of highest id
                    int max_id1, max_id2;

                    max_id1 = (NodeReordering[Nodes[face3_map[0]]] <
                               NodeReordering[Nodes[face3_map[1]]])
                                  ? 1
                                  : 0;
                    max_id2 = (NodeReordering[Nodes[face3_map[max_id1]]] <
                               NodeReordering[Nodes[face3_map[2]]])
                                  ? 2
                                  : max_id1;

                    // add numbering according to the order of
                    int id0 = (max_id1 == 1) ? 0 : 1;

                    if (NodeReordering[Nodes[face1_map[id0]]] == -1)
                    {
                        NodeReordering[Nodes[face1_map[id0]]] = nodeid++;
                    }

                    if (NodeReordering[Nodes[face1_map[max_id1]]] == -1)
                    {
                        NodeReordering[Nodes[face1_map[max_id1]]] = nodeid++;
                    }

                    if (NodeReordering[Nodes[face1_map[max_id2]]] == -1)
                    {
                        NodeReordering[Nodes[face1_map[max_id2]]] = nodeid++;
                    }
                }
                else if ((FacesDone[PrismToFaces[prismid][0]] == true) &&
                         (FacesDone[PrismToFaces[prismid][1]] == false))
                {
                    // find node of highest id
                    int max_id1, max_id2;

                    max_id1 = (NodeReordering[Nodes[face1_map[0]]] <
                               NodeReordering[Nodes[face1_map[1]]])
                                  ? 1
                                  : 0;
                    max_id2 = (NodeReordering[Nodes[face1_map[max_id1]]] <
                               NodeReordering[Nodes[face1_map[2]]])
                                  ? 2
                                  : max_id1;

                    // add numbering according to order of
                    int id0 = (max_id1 == 1) ? 0 : 1;

                    if (NodeReordering[Nodes[face3_map[id0]]] == -1)
                    {
                        NodeReordering[Nodes[face3_map[id0]]] = nodeid++;
                    }

                    if (NodeReordering[Nodes[face3_map[max_id1]]] == -1)
                    {
                        NodeReordering[Nodes[face3_map[max_id1]]] = nodeid++;
                    }

                    if (NodeReordering[Nodes[face3_map[max_id2]]] == -1)
                    {
                        NodeReordering[Nodes[face3_map[max_id2]]] = nodeid++;
                    }
                }
            }
        }
    }

    // fill in any unset nodes from other shapes
    for (i = 0; i < NodeReordering.size(); ++i)
    {
        if (NodeReordering[i] == -1)
        {
            NodeReordering[i] = nodeid++;
        }
    }

    ASSERTL1(nodeid == revNodeid + 1, "Have not renumbered all nodes");

    // Renumbering successfull so reset nodes and faceNodes;
    for (auto &it : FaceNodes)
    {
        for (j = 0; j < it.second.size(); ++j)
        {
            it.second[j] = NodeReordering[it.second[j]];
        }
    }

    vector<NodeSharedPtr> save(Vnodes);
    for (i = 0; i < Vnodes.size(); ++i)
    {
        Vnodes[NodeReordering[i]] = save[i];
        Vnodes[NodeReordering[i]]->SetID(NodeReordering[i]);
    }

    for (auto &ei : elemInfo)
    {
        for (int vertIdx = 0; vertIdx < ei.second.size(); vertIdx++)
        {
            ei.second[vertIdx] = NodeReordering[ei.second[vertIdx]];
        }
    }
}

/**
 * Replace a 'problematic' pyramid with a smaller pyramid and four tets.
 * The tets shield the apex and pyramid tri-faces from neighbouring
 * elements.
 */
void InputCGNS::PyramidShielding(
    vector<NodeSharedPtr> &Vnodes, Array<OneD, vector<int>> &ElementFaces,
    std::unordered_map<int, vector<int>> &FaceNodes,
    vector<pair<ElementType_t, vector<int>>> &elemInfo,
    vector<int> &VolumeElems, vector<int> &NodeReordering, int pyraElemIdx)
{
    m_log(WARNING) << "Impossible mesh: cannot enforce apex as "
                      "collapsed point. Will attempt pyramid shielding"
                   << endl;

    ElementType_t pyraType = elemInfo[VolumeElems[pyraElemIdx]].first;
    int order              = m_elemType2Order[pyraType];
    int n                  = order - 1;
    bool hasFaceNodes      = m_elemType2FaceNodes[pyraType];
    bool hasVolNodes       = m_elemType2VolNodes[pyraType];

    vector<int> oldPyra = elemInfo[VolumeElems[pyraElemIdx]].second;

    // 1. Create the new nodes
    // define the pyramid vertices
    vector<NekVector<NekDouble>> oldVerts;

    for (int vertIdx = 0; vertIdx < 5; vertIdx++)
    {
        NekDouble vArray[] = {Vnodes[oldPyra[vertIdx]]->m_x,
                              Vnodes[oldPyra[vertIdx]]->m_y,
                              Vnodes[oldPyra[vertIdx]]->m_z};
        NekVector<NekDouble> v(3, vArray);
        oldVerts.push_back(v);
    }

    // the new apex is the geometric mean of the old apex and the base centre
    // this is somewhat arbitrary
    NekVector<NekDouble> newApex = (4 * oldVerts[4] + oldVerts[0] +
                                    oldVerts[1] + oldVerts[2] + oldVerts[3]) /
                                   8;

    // Create nodes
    // a) new apex node
    int newApexIdx = Vnodes.size();
    SaveNode(newApexIdx, newApex[0], newApex[1], newApex[2]);

    vector<vector<int>> newEdgeNodesIdx(5);
    vector<vector<int>> baseEdgeNodesIdx(4);
    vector<vector<int>> sideEdgeNodesIdx(4);

    vector<vector<vector<int>>> newFaceNodesIdx(8);
    vector<vector<vector<int>>> sideFaceNodesIdx(4);

    vector<int> newPyraVolNodesIdx; // the volume nodes of the new pyra
    vector<int> newTetVolNodesIdx;  // the volume node for each tet
                                    // (if TETRA_35, empty otherwise)

    // b) mid-edge nodes
    if (order > 1)
    {
        NekVector<NekDouble> tmp;
        double frac;
        int newNodeIdx;

        // i) pyramid mid-edge nodes
        for (int edgeIdx = 0; edgeIdx < 4; edgeIdx++)
        {
            for (int nodeIdx = 0; nodeIdx < n; nodeIdx++)
            {
                // linearly interpolate between v[i] (base vert) and newApex
                frac = (nodeIdx + 1.0) / (n + 1.0);
                tmp  = oldVerts[edgeIdx] + frac * (newApex - oldVerts[edgeIdx]);
                newNodeIdx = Vnodes.size();
                SaveNode(newNodeIdx, tmp[0], tmp[1], tmp[2]);
                newEdgeNodesIdx[edgeIdx].push_back(newNodeIdx);
            }
        }

        // ii) tet mid-edge nodes
        for (int nodeIdx = 0; nodeIdx < n; nodeIdx++)
        {
            // linearly interpolate between the new and old apices
            frac       = (nodeIdx + 1.0) / (n + 1.0);
            tmp        = newApex + frac * (oldVerts[4] - newApex);
            newNodeIdx = Vnodes.size();
            SaveNode(newNodeIdx, tmp[0], tmp[1], tmp[2]);
            newEdgeNodesIdx[4].push_back(newNodeIdx);
        }

        // iii) old pyra base mid-edge nodes
        for (int edgeIdx = 0; edgeIdx < 4; edgeIdx++)
        {
            baseEdgeNodesIdx[edgeIdx].assign(oldPyra.begin() + 5 + edgeIdx * n,
                                             oldPyra.begin() + 5 +
                                                 (edgeIdx + 1) * n);
        }

        // iv) old pyra side mid-edge nodes
        for (int edgeIdx = 0; edgeIdx < 4; edgeIdx++)
        {
            sideEdgeNodesIdx[edgeIdx].assign(
                oldPyra.begin() + 5 + (edgeIdx + 4) * n,
                oldPyra.begin() + 5 + (edgeIdx + 5) * n);
        }
    }

    /** eg. if the 0th face represents...
     *            ^
     *           / \
     *          /   \
     *         /  f  \
     *        / d  e  \
     *       / a  b  c \
     *      /___________\
     *
     * newFaceNodesIdx[0] = {{a, b, c},
     *                       {d, e   },
     *                       {f      }}
     * We split the face by rows to make it easier to reverse the
     * orientation
     */

    // c) mid-face nodes
    if (hasFaceNodes)
    {
        NekVector<NekDouble> tmp;
        double frac_v, frac_u;
        int newNodeIdx;

        // i) pyramid mid-face nodes
        for (int faceIdx = 0; faceIdx < 4; faceIdx++)
        {
            for (int j = 0; j < n - 1; j++)
            {
                frac_v = (j + 1.0) / (n + 1); // 1.0 to convert to double
                newFaceNodesIdx[faceIdx].push_back({});

                for (double i = 0; i < n - 1 - j; i++)
                {
                    /**  tmp = a + f_u * (b-a) + f_v * (c-a)
                     *
                     *       i ^
                     *           |
                     *           c
                     *           | \
                     *           a - b -> i
                     *
                     *   where a = oldVert[i]        (base node)
                     *         b = oldVert[(i+1)%4]  (next base node)
                     *         c = new apex
                     */

                    frac_u = (i + 1) / (n + 1);

                    tmp = oldVerts[faceIdx] +
                          frac_u * (oldVerts[(faceIdx + 1) % 4] -
                                    oldVerts[faceIdx]) +
                          frac_v * (newApex - oldVerts[faceIdx]);

                    newNodeIdx = Vnodes.size();
                    SaveNode(newNodeIdx, tmp[0], tmp[1], tmp[2]);
                    newFaceNodesIdx[faceIdx][j].push_back(newNodeIdx);
                }
            }
        }

        // ii) tet mid-face nodes
        for (int faceIdx = 0; faceIdx < 4; faceIdx++)
        {
            for (int j = 0; j < n - 1; j++)
            {
                frac_v = (j + 1.0) / (n + 1); // 1.0 to convert to double
                newFaceNodesIdx[4 + faceIdx].push_back({});

                for (double i = 0; i < n - 1 - j; i++)
                {
                    /** tmp = a + f_u * (b-a) + f_v * (c-a)
                     *
                     *       i ^
                     *           |
                     *           c
                     *           | \
                     *           a - b -> i
                     *
                     *   tmp = a + f_u * (b-a) + f_v * (c-a)
                     *   where a = oldVert[i]  (base node)
                     *         b = new apex
                     *         c = old apex
                     */

                    frac_u = (i + 1) / (n + 1);

                    tmp = oldVerts[faceIdx] +
                          frac_u * (newApex - oldVerts[faceIdx]) +
                          frac_v * (oldVerts[4] - oldVerts[faceIdx]);

                    newNodeIdx = Vnodes.size();
                    SaveNode(newNodeIdx, tmp[0], tmp[1], tmp[2]);
                    newFaceNodesIdx[4 + faceIdx][j].push_back(newNodeIdx);
                }
            }
        }

        // iii) leave the base mid-face nodes

        // iv) old pyra triangular mid-face nodes
        int i = 5 + 8 * n + n * n; // index of the first tri mid-face node

        for (int faceIdx = 0; faceIdx < 4; faceIdx++)
        {
            for (int j = 0; j < n - 1; j++)
            {

                sideFaceNodesIdx[faceIdx].push_back({});
                sideFaceNodesIdx[faceIdx][j].assign(
                    oldPyra.begin() + i, oldPyra.begin() + i + n - 1 - j);

                i += n - 1 - j;
            }
        }
    }

    // d) mid-vol nodes
    if (hasVolNodes)
    {
        NekVector<NekDouble> tmp;
        int newNodeIdx;

        // i) pyramid mid-vol nodes
        for (int k = 0; k < n - 1; k++)
        {
            double frac_w = (k + 1.0) / (n + 1.0);

            if ((n - 1 - k) > 2)
            {
                m_log(WARNING)
                    << "Too many mid-volume nodes in this cross-section";
            }
            else if ((n - 1 - k) == 1)
            {
                tmp = oldVerts[0] + 0.5 * (oldVerts[1] - oldVerts[0]) +
                      0.5 * (oldVerts[3] - oldVerts[0]) +
                      frac_w * (newApex - oldVerts[0]);
                newNodeIdx = Vnodes.size();
                SaveNode(newNodeIdx, tmp[0], tmp[1], tmp[2]);
                newPyraVolNodesIdx.push_back(newNodeIdx);
            }
            else if ((n - 1 - k) == 2)
            {
                vector<int> spiral_i = {0, 1, 1, 0};
                vector<int> spiral_j = {0, 0, 1, 1};
                int i, j;

                for (int nodeIdx = 0; nodeIdx < 4; nodeIdx++)
                {
                    i = spiral_i[nodeIdx];
                    j = spiral_j[nodeIdx];

                    double frac_v = (j + 1.0) / (n - k);
                    double frac_u = (i + 1.0) / (n - k);

                    tmp = oldVerts[0] + frac_u * (oldVerts[1] - oldVerts[0]) +
                          frac_v * (oldVerts[3] - oldVerts[0]) +
                          frac_w * (newApex - oldVerts[0]);
                    newNodeIdx = Vnodes.size();
                    SaveNode(newNodeIdx, tmp[0], tmp[1], tmp[2]);
                    newPyraVolNodesIdx.push_back(newNodeIdx);
                }
            }
        }

        // ii) tet mid-vol nodes
        if (order == 4)
        {
            for (int tetIdx = 0; tetIdx < 4; tetIdx++)
            {
                tmp = (oldVerts[tetIdx] + oldVerts[(tetIdx + 1) % 4] +
                       oldVerts[4] + newApex) /
                      4;
                newNodeIdx = Vnodes.size();
                SaveNode(newNodeIdx, tmp[0], tmp[1], tmp[2]);
                newTetVolNodesIdx.push_back(newNodeIdx);
            }
        }
    }

    // 2 a). UPDATE ElemInfo for pyra
    vector<int> &pyraNodes = elemInfo[VolumeElems[pyraElemIdx]].second;

    // i) leave the base edge nodes as is

    // ii) change the apex
    pyraNodes[4] = newApexIdx;

    // iii) change the side edge nodes
    for (int edgeIdx = 0; edgeIdx < 4; edgeIdx++)
    {
        // copy from the new edge nodes to pyraNodes
        copy(newEdgeNodesIdx[edgeIdx].begin(), newEdgeNodesIdx[edgeIdx].end(),
             pyraNodes.begin() + 5 + (edgeIdx + 4) * n);
    }

    // iv) leave the base face nodes

    // v) change the tri face nodes
    int i =
        5 + 8 * n + hasFaceNodes * n * n; // index of first tri mid-face node

    for (int faceIdx = 0; faceIdx < 4; faceIdx++)
    {
        for (vector<int> &row : newFaceNodesIdx[faceIdx])
        {
            copy(row.begin(), row.end(), pyraNodes.begin() + i);
            i += row.size();
        }
    }

    // vi) change the volume nodes
    copy(newPyraVolNodesIdx.begin(), newPyraVolNodesIdx.end(),
         pyraNodes.begin() + i);
    i += newPyraVolNodesIdx.size();
    ASSERTL0(i == pyraNodes.size(),
             "The wrong number of nodes were changed in the process of "
             "PyramidShielding");

    // 2 b). UPDATE ElemInfo for tets
    for (int tetIdx = elemInfo.size(); tetIdx < elemInfo.size() + 4; tetIdx++)
    {
        VolumeElems.push_back(tetIdx);
    }

    ElementType_t tetType = m_pyrShieldingTetType[pyraType];

    for (int tetIdx = 0; tetIdx < 4; tetIdx++)
    {
        // i) allocate memory
        // vector<int> tetNodes(how many?); // TODO: pre-allocate for efficiency
        vector<int> tetNodes;

        // ii) add vertices
        tetNodes.push_back(oldPyra[tetIdx]);
        tetNodes.push_back(oldPyra[(tetIdx + 1) % 4]);
        tetNodes.push_back(newApexIdx);
        tetNodes.push_back(oldPyra[4]);

        // iii) define and insert edges
        if (order > 1)
        {
            vector<int> &temp0 = baseEdgeNodesIdx[tetIdx];
            vector<int> &temp1 = newEdgeNodesIdx[(tetIdx + 1) % 4];
            vector<int> &temp2 = newEdgeNodesIdx[tetIdx];
            vector<int> &temp3 = sideEdgeNodesIdx[tetIdx];
            vector<int> &temp4 = sideEdgeNodesIdx[(tetIdx + 1) % 4];
            vector<int> &temp5 = newEdgeNodesIdx[4];

            tetNodes.insert(tetNodes.end(), temp0.begin(), temp0.end());
            tetNodes.insert(tetNodes.end(), temp1.begin(), temp1.end());
            tetNodes.insert(tetNodes.end(), temp2.begin(), temp2.end());
            tetNodes.insert(tetNodes.end(), temp3.begin(), temp3.end());
            tetNodes.insert(tetNodes.end(), temp4.begin(), temp4.end());
            tetNodes.insert(tetNodes.end(), temp5.begin(), temp5.end());
        }

        // iv) define and insert faces
        if (hasFaceNodes)
        {
            vector<vector<vector<int>>> tetFaces0to2 = {
                newFaceNodesIdx[tetIdx], sideFaceNodesIdx[tetIdx],
                newFaceNodesIdx[4 + (tetIdx + 1) % 4]};

            vector<vector<int>> &tetFace3 = newFaceNodesIdx[4 + tetIdx];

            // append faces 0-2
            for (vector<vector<int>> &face : tetFaces0to2)
            {
                for (vector<int> &row : face)
                {
                    tetNodes.insert(tetNodes.end(), row.begin(), row.end());
                }
            }

            // append face 3 (reversed)
            for (vector<int> &row : tetFace3)
            {
                tetNodes.insert(tetNodes.end(), row.rbegin(), row.rend());
            }
        }

        if (hasVolNodes)
        {
            tetNodes.push_back(newTetVolNodesIdx[tetIdx]);
        }
        elemInfo.push_back({tetType, tetNodes});
    }

    // 3. ADD faces to FaceNodes and UPDATE ElementFaces
    // create new pyra faces
    int newPyraFace0 = FaceNodes.size();
    for (int faceIdx = 0; faceIdx < 4; faceIdx++)
    {
        FaceNodes[newPyraFace0 + faceIdx] = {
            oldPyra[faceIdx], oldPyra[(faceIdx + 1) % 4], newApexIdx};
    }

    // update ElementFaces for the shielded pyra
    ElementFaces[pyraElemIdx] = {newPyraFace0, newPyraFace0 + 1,
                                 newPyraFace0 + 2, newPyraFace0 + 3,
                                 ElementFaces[pyraElemIdx][4]};

    // add tet faces
    int tetFace0 = newPyraFace0 + 4;
    for (int faceIdx = 0; faceIdx < 4; faceIdx++)
    {
        FaceNodes[tetFace0 + faceIdx] = {oldPyra[faceIdx], oldPyra[4],
                                         newApexIdx};
    }

    // update ElementFaces with new tets
    int tet0 = ElementFaces.size();
    Array<OneD, vector<int>> NewElementFaces(tet0 + 4);
    for (int elemIdx = 0; elemIdx < tet0; elemIdx++)
    {
        NewElementFaces[elemIdx] = ElementFaces[elemIdx];
    }

    for (int elemIdx = 0; elemIdx < 4; elemIdx++)
    {
        NewElementFaces[tet0 + elemIdx] = {
            newPyraFace0 + elemIdx, tetFace0 + (elemIdx + 1) % 4,
            ElementFaces[pyraElemIdx][elemIdx], tetFace0 + elemIdx};
    }

    ElementFaces = NewElementFaces;

    // 4. Add to NodeReordering
    int numNodesCreated =
        1                                                     // 1 node created
        + 5 * n                                               // 5 edges
        + (8 * n * (n - 1) / 2) * hasFaceNodes                // 8 tri faces
        + ((n - 1) * n * (2 * (n + 1) - 3) / 6) * hasVolNodes // 1 pyra
        + (order == 4 ? 4 : 0) * hasVolNodes;                 // 4 tets
    for (int nodeIdx = 0; nodeIdx < numNodesCreated; nodeIdx++)
    {
        NodeReordering.push_back(NodeReordering.size());
    }
}

/**
 * Starting from a pyramid apex, traverse along a prism line (a line of
 * prisms with their tri faces touching) recursively Ensure that the
 * collapsed points for each of the tri faces match up with each other and
 * with the pyramid apex - this is done by IDing these points with
 * revNodeId--, ensuring that they have a higher index than the other tri
 * nodes
 */
void InputCGNS::TraversePyraPrismLine(
    int currElemId, int currFaceId, int currApexNode,
    std::vector<std::vector<int>> FaceToPrisms,
    std::vector<std::vector<int>> GlobTriFaces,
    std::vector<NodeSharedPtr> &Vnodes,
    Array<OneD, std::vector<int>> &ElementFaces,
    std::unordered_map<int, std::vector<int>> &FaceNodes,
    std::vector<int> &NodeReordering, int &revNodeid)
{
    // these convert between the node ordering generated by OrderFaceNodes()
    // and the node ordering for the two triangular faces
    int face1_map[3] = {0, 1, 4};
    int face3_map[3] = {3, 2, 5};

    // iterate over the one/two prisms that connect the face
    for (int nextElemId : FaceToPrisms[currFaceId])
    {
        // only carry out the rest of the function if travelling in the right
        // direction
        if (nextElemId == currElemId)
        {
            continue;
        }

        // iterate over the two tri faces (one at each end)
        for (int nextFaceId : GlobTriFaces[nextElemId])
        {
            // ensure that we are travelling in the right direction
            if (nextFaceId == currFaceId)
            {
                continue;
            }

            // reorder the nodes
            Array<OneD, int> Nodes =
                SortFaceNodes(Vnodes, ElementFaces[nextElemId], FaceNodes);

            int nextApexNode = -1; // the ID of the opposite collapsed point
            int currFace     = -1; // either 1 or 3

            // highest ID for each face
            int highestId1 = -1;
            int highestId3 = -1;

            // find the orientation of the prism (face 1 vs 3)
            for (int i = 0; i < 3; ++i)
            {
                highestId1 = (NodeReordering[Nodes[face1_map[i]]] > highestId1)
                                 ? NodeReordering[Nodes[face1_map[i]]]
                                 : highestId1;

                highestId3 = (NodeReordering[Nodes[face3_map[i]]] > highestId3)
                                 ? NodeReordering[Nodes[face3_map[i]]]
                                 : highestId3;

                if (Nodes[face1_map[i]] == currApexNode)
                {
                    nextApexNode = Nodes[face3_map[i]];
                    currFace     = 1;
                }

                if (Nodes[face3_map[i]] == currApexNode)
                {
                    nextApexNode = Nodes[face1_map[i]];
                    currFace     = 3;
                }
            }

            if (currFace == 1)
            {
                ASSERTL0(highestId3 == NodeReordering[nextApexNode],
                         "Error meshing: incompatible collapsed point");
            }
            else if (currFace == 3)
            {
                ASSERTL0(highestId1 == NodeReordering[nextApexNode],
                         "Error meshing: incompatible collapsed point");
            }
            else
            {
                m_log(FATAL) << "Error meshing: collapsed point not found";
            }

            if (NodeReordering[nextApexNode] == -1)
            {
                NodeReordering[nextApexNode] = revNodeid--;
            }

            // continue traversing along the line
            TraversePyraPrismLine(nextElemId, nextFaceId, nextApexNode,
                                  FaceToPrisms, GlobTriFaces, Vnodes,
                                  ElementFaces, FaceNodes, NodeReordering,
                                  revNodeid);
        }
    }
}

void InputCGNS::GenElement2D(vector<NodeSharedPtr> &VertNodes,
                             vector<int> elemNodes, ElementType_t elemType,
                             int nComposite, bool DoOrient)
{
    int order                         = m_elemType2Order[elemType];
    LibUtilities::ShapeType shapeType = m_elemType2ShapeType[elemType];
    LibUtilities::PointsType edgeCurveType;
    LibUtilities::PointsType faceCurveType;

    // Create element tags
    vector<int> tags;
    tags.push_back(nComposite);

    // make unique node list
    vector<NodeSharedPtr> nodeList;

    // Look up reordering.
    auto oIt = m_orderingMap.find(elemType);

    // If it's not created, then create it.
    if (oIt == m_orderingMap.end())
    {
        oIt = m_orderingMap
                  .insert(make_pair(elemType, CGNSReordering(shapeType, order)))
                  .first;
    }

    for (int i = 0; i < elemNodes.size(); i++)
    {
        nodeList.push_back((oIt->second.size())
                               ? VertNodes[elemNodes[oIt->second[i]]]
                               : VertNodes[elemNodes[i]]);
    }

    bool faceNodes, volumeNodes;
    faceNodes     = m_elemType2FaceNodes[elemType];
    volumeNodes   = false;
    edgeCurveType = LibUtilities::ePolyEvenlySpaced;
    faceCurveType = (shapeType == LibUtilities::eTriangle)
                        ? LibUtilities::eNodalTriEvenlySpaced
                        : LibUtilities::ePolyEvenlySpaced;

    ElmtConfig conf(shapeType, order, faceNodes, volumeNodes, DoOrient,
                    edgeCurveType, faceCurveType);
    ElementSharedPtr E =
        GetElementFactory().CreateInstance(shapeType, conf, nodeList, tags);

    m_mesh->m_element[E->GetDim()].push_back(E);
}

void InputCGNS::GenElement3D(vector<NodeSharedPtr> &VertNodes,
                             vector<int> elemNodes, vector<int> &ElementFace,
                             std::unordered_map<int, vector<int>> &FaceNodes,
                             ElementType_t elemType, int nComposite,
                             bool DoOrient)
{
    int order                         = m_elemType2Order[elemType];
    LibUtilities::ShapeType shapeType = m_elemType2ShapeType[elemType];
    LibUtilities::PointsType edgeCurveType;
    LibUtilities::PointsType faceCurveType;

    // Create element tags
    vector<int> tags;
    tags.push_back(nComposite);

    // make unique node list
    vector<NodeSharedPtr> nodeList;

    if (order == 1)
    {
        Array<OneD, int> Nodes =
            SortFaceNodes(VertNodes, ElementFace, FaceNodes);

        for (int i = 0; i < Nodes.size(); i++)
        {
            nodeList.push_back(VertNodes[Nodes[i]]);
        }
    }
    else
    {
        // Look up reordering.
        auto oIt = m_orderingMap.find(elemType);

        // If it's not created, then create it.
        if (oIt == m_orderingMap.end())
        {
            oIt = m_orderingMap
                      .insert(
                          make_pair(elemType, CGNSReordering(shapeType, order)))
                      .first;
        }

        for (int i = 0; i < elemNodes.size(); i++)
        {
            nodeList.push_back((oIt->second.size())
                                   ? VertNodes[elemNodes[oIt->second[i]]]
                                   : VertNodes[elemNodes[i]]);
        }
    }

    bool faceNodes, volumeNodes;
    faceNodes     = m_elemType2FaceNodes[elemType];
    volumeNodes   = m_elemType2VolNodes[elemType];
    edgeCurveType = LibUtilities::ePolyEvenlySpaced;
    faceCurveType = (shapeType == LibUtilities::eTetrahedron)
                        ? LibUtilities::eNodalTriEvenlySpaced
                        : LibUtilities::ePolyEvenlySpaced;

    ElmtConfig conf(shapeType, order, faceNodes, volumeNodes, DoOrient,
                    edgeCurveType, faceCurveType);
    ElementSharedPtr E =
        GetElementFactory().CreateInstance(shapeType, conf, nodeList, tags);

    m_mesh->m_element[E->GetDim()].push_back(E);
}

vector<int> InputCGNS::CGNSReordering(LibUtilities::ShapeType shapeType,
                                      int order)
{
    vector<int> cgns2gmsh, gmsh2nek, mapping;

    switch (order)
    {
        case 1:    // 1st order
            break; // no reordering needed

        case 2: // 2nd order
            switch (shapeType)
            {
                case LibUtilities::eTriangle:
                    break;

                case LibUtilities::eQuadrilateral:
                    break;

                case LibUtilities::eTetrahedron:
                    break;

                case LibUtilities::ePyramid:
                    break;

                case LibUtilities::ePrism:
                    mapping = {3, 4,  1,  0, 5, 2,  12, 10, 6,
                               9, 14, 13, 7, 8, 11, 15, 16, 17};
                    break;

                case LibUtilities::eHexahedron:
                    break;

                default:
                    m_log(WARNING)
                        << "Shape type " << shapeType << " not recognised";
                    break;
            }
            break;

        case 3: // 3rd order
            switch (shapeType)
            {
                case LibUtilities::eTriangle:
                    break;

                case LibUtilities::eQuadrilateral:
                    mapping = {0, 1, 2,  3,  4,  5,  6,  7,
                               8, 9, 10, 11, 12, 13, 15, 14};
                    break;

                case LibUtilities::eTetrahedron:
                    mapping = {0,  1,  2,  3,  4,  5,  6,  7,  9,  8,
                               10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
                    break;

                case LibUtilities::ePyramid:
                    mapping = {0,  1,  2,  3,  4,  5,  6,  7,  8,  10,
                               9,  12, 11, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 24, 23, 25, 26, 27, 28, 29};
                    break;

                case LibUtilities::ePrism:
                    mapping = {3,  4,  1,  0,  5,  2,  18, 19, 15, 14,
                               6,  7,  13, 12, 23, 22, 20, 21, 8,  9,
                               11, 10, 17, 16, 28, 27, 25, 26, 37, 32,
                               29, 31, 30, 24, 35, 34, 36, 33, 38, 39};
                    break;

                case LibUtilities::eHexahedron:
                    mapping = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                               11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                               22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                               33, 35, 34, 36, 37, 39, 38, 40, 41, 43, 42,
                               45, 44, 46, 47, 49, 48, 50, 51, 52, 53, 55,
                               54, 56, 57, 59, 58, 60, 61, 63, 62};
                    break;

                default:
                    m_log(WARNING)
                        << "Shape type " << shapeType << " not recognised";
                    break;
            }
            break;

        case 4: // 4th order
            switch (shapeType)
            {
                case LibUtilities::eTriangle:
                    break;

                case LibUtilities::eQuadrilateral:
                    mapping = {0,  1,  2,  3,  4,  5,  6,  7,  8,
                               9,  10, 11, 12, 13, 14, 15, 16, 17,
                               18, 23, 24, 19, 22, 21, 20};
                    break;

                case LibUtilities::eTetrahedron:
                    mapping = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  12, 11,
                               10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 26, 27, 28, 29, 30, 32, 31, 33, 34};
                    break;

                case LibUtilities::ePyramid:
                    mapping = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                               13, 12, 11, 16, 15, 14, 17, 18, 19, 20, 21,
                               22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 36,
                               37, 32, 35, 34, 33, 38, 39, 40, 41, 42, 43,
                               45, 44, 46, 48, 47, 49, 50, 51, 53, 52, 54};
                    break;

                case LibUtilities::ePrism:
                    mapping = {3,  4,  1,  0,  5,  2,  24, 25, 26, 20, 19,
                               18, 6,  7,  8,  17, 16, 15, 32, 31, 30, 27,
                               28, 29, 9,  10, 11, 14, 13, 12, 23, 22, 21,
                               42, 41, 40, 43, 44, 39, 36, 37, 38, 63, 64,
                               65, 51, 52, 45, 50, 53, 46, 49, 48, 47, 33,
                               34, 35, 58, 57, 56, 59, 62, 55, 60, 61, 54,
                               72, 69, 66, 73, 70, 67, 74, 71, 68};
                    break;

                case LibUtilities::eHexahedron:
                    mapping = {0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
                               10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
                               20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
                               30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
                               40,  41,  42,  43,  44,  45,  46,  51,  52,  47,
                               50,  49,  48,  53,  54,  55,  60,  61,  56,  59,
                               58,  57,  62,  63,  64,  69,  70,  65,  68,  67,
                               66,  73,  72,  71,  74,  79,  78,  75,  76,  77,
                               82,  81,  80,  83,  88,  87,  84,  85,  86,  89,
                               90,  91,  96,  97,  92,  95,  94,  93,  98,  99,
                               100, 105, 106, 101, 104, 103, 102, 107, 108, 109,
                               114, 115, 110, 113, 112, 111, 116, 117, 118, 123,
                               124, 119, 122, 121, 120};
                    break;

                default:
                    m_log(WARNING)
                        << "Shape type " << shapeType << " not recognised";
                    break;
            }
            break;

        default:
            m_log(FATAL) << "Order " << order << " not recognised" << endl;
            break;
    }

    return mapping;
}

Array<OneD, int> InputCGNS::SortEdgeNodes(vector<NodeSharedPtr> &Vnodes,
                                          vector<int> &FaceNodes)
{
    Array<OneD, int> returnval;

    if (FaceNodes.size() == 3) // Triangle
    {
        returnval = Array<OneD, int>(3);

        returnval[0] = FaceNodes[0];
        returnval[1] = FaceNodes[1];
        returnval[2] = FaceNodes[2];
    }
    else if (FaceNodes.size() == 4) // quadrilateral
    {
        returnval = Array<OneD, int>(4);

        int indx0 = FaceNodes[0];
        int indx1 = FaceNodes[1];
        int indx2 = FaceNodes[2];
        int indx3 = FaceNodes[3];

        // calculate 0-1,
        Node a = *(Vnodes[indx1]) - *(Vnodes[indx0]);
        // calculate 0-2,
        Node b      = *(Vnodes[indx2]) - *(Vnodes[indx0]);
        Node acurlb = a.curl(b);

        // calculate 2-1,
        Node c = *(Vnodes[indx1]) - *(Vnodes[indx2]);
        // calculate 3-2,
        Node d      = *(Vnodes[indx3]) - *(Vnodes[indx2]);
        Node acurld = a.curl(d);

        NekDouble acurlb_dot_acurld = acurlb.dot(acurld);
        if (acurlb_dot_acurld > 0.0)
        {
            returnval[0] = indx0;
            returnval[1] = indx1;
            returnval[2] = indx2;
            returnval[3] = indx3;
        }
        else
        {
            returnval[0] = indx0;
            returnval[1] = indx1;
            returnval[2] = indx3;
            returnval[3] = indx2;
        }
    }

    return returnval;
}

Array<OneD, int> InputCGNS::SortFaceNodes(
    vector<NodeSharedPtr> &Vnodes, vector<int> &ElementFaces,
    std::unordered_map<int, vector<int>> &FaceNodes)
{
    int i, j;
    Array<OneD, int> returnval;

    if (ElementFaces.size() == 4) // Tetrahedron
    {
        ASSERTL1(FaceNodes[ElementFaces[0]].size() == 3,
                 "Face is not triangular");

        returnval = Array<OneD, int>(4);

        auto it   = FaceNodes.find(ElementFaces[0]);
        int indx0 = it->second[0];
        int indx1 = it->second[1];
        int indx2 = it->second[2];
        int indx3 = -1;

        // calculate 0-1,
        Node a = *(Vnodes[indx1]) - *(Vnodes[indx0]);
        // calculate 0-2,
        Node b = *(Vnodes[indx2]) - *(Vnodes[indx0]);

        // Find fourth node index;
        ASSERTL1(FaceNodes[ElementFaces[1]].size() == 3,
                 "Face is not triangular");

        auto it2 = FaceNodes.find(ElementFaces[1]);
        for (i = 0; i < 3; ++i)
        {
            if ((it2->second[i] != indx0) && (it2->second[i] != indx1) &&
                (it2->second[i] != indx2))
            {
                indx3 = it2->second[i];
                break;
            }
        }

        // calculate 0-3,
        Node c      = *(Vnodes[indx3]) - *(Vnodes[indx0]);
        Node acurlb = a.curl(b);

        NekDouble acurlb_dotc = acurlb.dot(c);
        if (acurlb_dotc < 0.0)
        {
            returnval[0] = indx0;
            returnval[1] = indx1;
            returnval[2] = indx2;
            returnval[3] = indx3;
        }
        else
        {
            returnval[0] = indx1;
            returnval[1] = indx0;
            returnval[2] = indx2;
            returnval[3] = indx3;
        }
    }
    else if (ElementFaces.size() == 5) // prism or pyramid
    {
        int triface0, triface1, triface2, triface3;
        int quadface0, quadface1, quadface2;
        bool isPrism = true;

        // find ids of tri faces and first quad face
        triface0 = triface1 = triface2 = triface3 = -1;
        quadface0 = quadface1 = quadface2 = -1;
        for (i = 0; i < 5; ++i)
        {
            auto it = FaceNodes.find(ElementFaces[i]);
            if (it->second.size() == 3)
            {
                if (triface0 == -1)
                {
                    triface0 = i;
                }
                else if (triface1 == -1)
                {
                    triface1 = i;
                }
                else if (triface2 == -1)
                {
                    triface2 = i;
                    isPrism  = false;
                }
                else if (triface3 == -1)
                {
                    triface3 = i;
                }
            }

            if (it->second.size() == 4)
            {
                if (quadface0 == -1)
                {
                    quadface0 = i;
                }
                else if (quadface1 == -1)
                {
                    quadface1 = i;
                }
                else if (quadface2 == -1)
                {
                    quadface2 = i;
                }
            }
        }

        if (isPrism) // Prism
        {
            returnval = Array<OneD, int>(6);
            ASSERTL1(quadface0 != -1, "Quad face 0 not found");
            ASSERTL1(quadface1 != -1, "Quad face 1 not found");
            ASSERTL1(quadface2 != -1, "Quad face 2 not found");
            ASSERTL1(triface0 != -1, "Tri face 0 not found");
            ASSERTL1(triface1 != -1, "Tri face 1 not found");
        }
        else // Pyramid
        {
            returnval = Array<OneD, int>(5);
            ASSERTL1(quadface0 != -1, "Quad face 0 not found");
            ASSERTL1(triface0 != -1, "Tri face 0 not found");
            ASSERTL1(triface1 != -1, "Tri face 1 not found");
            ASSERTL1(triface2 != -1, "Tri face 2 not found");
            ASSERTL1(triface3 != -1, "Tri face 3 not found");
        }

        // find matching nodes between triface0 and triquad0
        int indx0, indx1, indx2, indx3, indx4;

        indx0 = indx1 = indx2 = indx3 = indx4 = -1;
        // Loop over all quad nodes and if they match any
        // triangular nodes If they do set these to indx0 and
        // indx1 and if not set it to indx2, indx3

        auto &triface0_vec  = FaceNodes.find(ElementFaces[triface0])->second;
        auto &quadface0_vec = FaceNodes.find(ElementFaces[quadface0])->second;
        for (i = 0; i < 4; ++i)
        {
            for (j = 0; j < 3; ++j)
            {
                if (triface0_vec[j] == quadface0_vec[i])
                {
                    break; // same node break
                }
            }

            if (j == 3) // Vertex not in quad face
            {
                if (indx2 == -1)
                {
                    indx2 = quadface0_vec[i];
                }
                else if (indx3 == -1)
                {
                    indx3 = quadface0_vec[i];
                }
                else
                {
                    m_log(FATAL) << "More than two vertices do not match "
                                 << "triangular face" << endl;
                }
            }
            else // if found match then set indx0,indx1;
            {
                if (indx0 == -1)
                {
                    indx0 = quadface0_vec[i];
                }
                else
                {
                    indx1 = quadface0_vec[i];
                }
            }
        }

        // Finally check for top vertex
        for (int i = 0; i < 3; ++i)
        {
            if (triface0_vec[i] != indx0 && triface0_vec[i] != indx1 &&
                triface0_vec[i] != indx2)
            {
                indx4 = triface0_vec[i];
                break;
            }
        }

        // calculate 0-1,
        Node a = *(Vnodes[indx1]) - *(Vnodes[indx0]);
        // calculate 0-4,
        Node b = *(Vnodes[indx4]) - *(Vnodes[indx0]);
        // calculate 0-2,
        Node c      = *(Vnodes[indx2]) - *(Vnodes[indx0]);
        Node acurlb = a.curl(b);

        NekDouble acurlb_dotc = acurlb.dot(c);
        if (acurlb_dotc < 0.0)
        {
            returnval[0] = indx0;
            returnval[1] = indx1;
            returnval[4] = indx4;
        }
        else
        {
            returnval[0] = indx1;
            returnval[1] = indx0;
            returnval[4] = indx4;
        }

        // check to see if two vertices are shared between one of the other
        // faces to define which is indx2 and indx3
        if (isPrism == true)
        {
            auto &quadface1_vec =
                FaceNodes.find(ElementFaces[quadface1])->second;
            auto &quadface2_vec =
                FaceNodes.find(ElementFaces[quadface2])->second;
            int cnt = 0;
            for (int i = 0; i < 4; ++i)
            {
                if (quadface1_vec[i] == returnval[1] ||
                    quadface1_vec[i] == indx2)
                {
                    cnt++;
                }
            }

            if (cnt == 2) // have two matching vertices
            {
                returnval[2] = indx2;
                returnval[3] = indx3;
            }
            else
            {
                cnt = 0;
                for (int i = 0; i < 4; ++i)
                {
                    if (quadface2_vec[i] == returnval[1] ||
                        quadface2_vec[i] == indx2)
                    {
                        cnt++;
                    }
                }

                if (cnt != 2) // neither of the other faces has two matching
                              // nodes so reverse
                {
                    returnval[2] = indx3;
                    returnval[3] = indx2;
                }
                else // have two matching vertices
                {
                    returnval[2] = indx2;
                    returnval[3] = indx3;
                }
            }
            // finally need to find last vertex from second triangular face.
            auto &triface1_vec = FaceNodes.find(ElementFaces[triface1])->second;
            for (int i = 0; i < 3; ++i)
            {
                if (triface1_vec[i] != indx2 && triface1_vec[i] != indx3)
                {
                    returnval[5] = triface1_vec[i];
                    break;
                }
            }
        }
        else
        {
            vector<int> trifaceid;
            trifaceid.push_back(triface1);
            trifaceid.push_back(triface2);
            trifaceid.push_back(triface3);
            int cnt = 0;
            for (auto id : trifaceid)
            {
                auto &triface_vec = FaceNodes.find(ElementFaces[id])->second;
                cnt               = 0;
                for (int i = 0; i < 3; ++i)
                {
                    if (triface_vec[i] == returnval[1] ||
                        triface_vec[i] == indx2)
                    {
                        cnt++;
                    }
                }

                if (cnt == 2) // have two matching vertices
                {
                    returnval[2] = indx2;
                    returnval[3] = indx3;
                    break;
                }
            }
            if (cnt != 2)
            {
                returnval[2] = indx3;
                returnval[3] = indx2;
            }
        }
    }
    else if (ElementFaces.size() == 6) // hexahedron
    {
        ASSERTL1(FaceNodes[ElementFaces[0]].size() == 4 &&
                     FaceNodes[ElementFaces[1]].size() == 4 &&
                     FaceNodes[ElementFaces[2]].size() == 4 &&
                     FaceNodes[ElementFaces[3]].size() == 4 &&
                     FaceNodes[ElementFaces[4]].size() == 4 &&
                     FaceNodes[ElementFaces[5]].size() == 4,
                 "Shape not recognised");

        returnval = Array<OneD, int>(8);

        // neighbours map contains a set of each neighbouring node for every
        // node
        map<int, std::unordered_set<int>> neighbours;

        // populate neighbours
        for (int faceId : ElementFaces)
        {
            vector<int> nodes = FaceNodes[faceId];
            neighbours[nodes[0]].insert({nodes[1], nodes[3]});
            neighbours[nodes[1]].insert({nodes[2], nodes[0]});
            neighbours[nodes[2]].insert({nodes[3], nodes[1]});
            neighbours[nodes[3]].insert({nodes[0], nodes[2]});
        }

        // the bottom face is chosen arbitrarily as the 0th indexed one
        vector<int> bottomFace = FaceNodes[ElementFaces[0]];
        std::unordered_set<int> bottomFaceSet(bottomFace.begin(),
                                              bottomFace.end());

        // the top face is the face opposite, sharing no edges or vertices
        vector<int> topFace;

        // find the top face
        for (int i = 1; i < 6; i++)
        {
            bool sharedEdge = false;
            for (int id : FaceNodes[ElementFaces[i]])
            {
                if (bottomFaceSet.count(id)) // if node @ id is in the
                                             // bottom face
                {
                    sharedEdge = true;
                    break;
                }
            }
            if (sharedEdge)
            {
                continue;
            }
            topFace = FaceNodes[ElementFaces[i]];
        }

        ASSERTL0(topFace.size() == 4, "Top face of hex not found");

        // insert the bottom face nodes into returnval[0:4]
        // insert the top face nodes into returnval[4:]
        // returnval[i+4] is the node opposite returnval[i]
        for (int i = 0; i < 4; i++)
        {
            returnval[i]     = bottomFace[i];
            bool oppositeSet = false;

            for (int id : neighbours[bottomFace[i]])
            {
                if (!bottomFaceSet.count(id))
                {
                    returnval[i + 4] = id;
                    oppositeSet      = true;
                    break;
                }
            }
            ASSERTL0(oppositeSet, "Node in the top face of hex not numbered");
        }
    }
    else
    {
        m_log(FATAL) << "SortFaceNodes not set up for this number of faces"
                     << endl;
    }

    return returnval;
}
} // namespace Nektar::NekMesh

/*  TODO:
    - improve ResetNodes
        - test

    - figure out if part labels can be read
        (eg. 'inlet', 'outlet', ...)

    - add support for the structured mesh type?
*/
