////////////////////////////////////////////////////////////////////////////////
//
//  File: MeshGraphIO.cpp
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
//  Description:
//
////////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/ParseUtils.h>
#include <SpatialDomains/MeshGraph.h>
#include <SpatialDomains/MeshGraphIO.h>

namespace Nektar::SpatialDomains
{

/**
 * Returns an instance of the MeshGraphIO factory, held as a singleton.
 */
MeshGraphIOFactory &GetMeshGraphIOFactory()
{
    static MeshGraphIOFactory instance;
    return instance;
}

MeshGraphSharedPtr MeshGraphIO::Read(
    const LibUtilities::SessionReaderSharedPtr session,
    LibUtilities::DomainRangeShPtr rng, bool fillGraph,
    SpatialDomains::MeshGraphSharedPtr partitionedGraph)
{
    LibUtilities::CommSharedPtr comm = session->GetComm();
    ASSERTL0(comm.get(), "Communication not initialised.");

    // Populate SessionReader. This should be done only on the root process so
    // that we can partition appropriately without all processes having to read
    // in the input file.
    const bool isRoot = comm->TreatAsRankZero();
    std::string geomType;

    if (isRoot)
    {
        // Parse the XML document.
        session->InitSession();

        // Get geometry type, i.e. XML (compressed/uncompressed) or HDF5.
        geomType = session->GetGeometryType();

        // Convert to a vector of chars so that we can broadcast.
        std::vector<char> v(geomType.c_str(),
                            geomType.c_str() + geomType.length());

        size_t length = v.size();
        comm->Bcast(length, 0);
        comm->Bcast(v, 0);
    }
    else
    {
        size_t length;
        comm->Bcast(length, 0);

        std::vector<char> v(length);
        comm->Bcast(v, 0);

        geomType = std::string(v.begin(), v.end());
    }

    // Every process then creates a mesh. Partitioning logic takes place inside
    // the PartitionMesh function so that we can support different options for
    // XML and HDF5.
    MeshGraphIOSharedPtr meshIO =
        GetMeshGraphIOFactory().CreateInstance(geomType);
    meshIO->m_meshGraph =
        MemoryManager<SpatialDomains::MeshGraph>::AllocateSharedPtr();
    meshIO->m_meshGraph->SetSession(session);

    // For Parallel-in-Time
    //    In contrast to XML, a pre-partitioned mesh directory (_xml) is not
    //    produced when partitionning the mesh for the fine solver when using
    //    HDF5. In order to guarantee the same partition on all time level,
    //    the fine mesh partition has to be copied explicitly.
    if (partitionedGraph && geomType == "HDF5")
    {
        meshIO->m_meshGraph->SetPartition(partitionedGraph);
        meshIO->m_meshPartitioned = true;
    }

    // set up domain range info before partitioning
    meshIO->m_meshGraph->SetDomainRange(rng);

    meshIO->PartitionMesh(session);

    // Finally, read the geometry information.
    meshIO->ReadGeometry(fillGraph);

    return meshIO->m_meshGraph;
}

/**
 * @brief Create mesh entities for this graph.
 *
 * This function will create a map of all mesh entities of the current graph,
 * which can then be used within the mesh partitioner to construct an
 * appropriate partitioning.
 */
std::map<int, MeshEntity> MeshGraphIO::CreateMeshEntities()
{
    std::map<int, MeshEntity> elements;
    switch (m_meshGraph->GetMeshDimension())
    {
        case 1:
        {
            for (auto [id, geom] : m_meshGraph->GetGeomMap<SegGeom>())
            {
                MeshEntity e;
                e.id = e.origId = id;
                e.list.push_back(geom->GetVertex(0)->GetGlobalID());
                e.list.push_back(geom->GetVertex(1)->GetGlobalID());
                e.ghost        = false;
                elements[e.id] = e;
            }
        }
        break;
        case 2:
        {
            for (auto [id, geom] : m_meshGraph->GetGeomMap<TriGeom>())
            {
                if (m_meshGraph->CheckRange(*geom))
                {
                    MeshEntity e;
                    e.id = e.origId = id;
                    e.list.push_back(geom->GetEdge(0)->GetGlobalID());
                    e.list.push_back(geom->GetEdge(1)->GetGlobalID());
                    e.list.push_back(geom->GetEdge(2)->GetGlobalID());
                    e.ghost        = false;
                    elements[e.id] = e;
                }
            }
            for (auto [id, geom] : m_meshGraph->GetGeomMap<QuadGeom>())
            {
                if (m_meshGraph->CheckRange(*geom))
                {
                    MeshEntity e;
                    e.id = e.origId = id;
                    e.list.push_back(geom->GetEdge(0)->GetGlobalID());
                    e.list.push_back(geom->GetEdge(1)->GetGlobalID());
                    e.list.push_back(geom->GetEdge(2)->GetGlobalID());
                    e.list.push_back(geom->GetEdge(3)->GetGlobalID());
                    e.ghost        = false;
                    elements[e.id] = e;
                }
            }
        }
        break;
        case 3:
        {
            for (auto [id, geom] : m_meshGraph->GetGeomMap<TetGeom>())
            {
                if (m_meshGraph->CheckRange(*geom))
                {
                    MeshEntity e;
                    e.id = e.origId = id;
                    e.list.push_back(geom->GetFace(0)->GetGlobalID());
                    e.list.push_back(geom->GetFace(1)->GetGlobalID());
                    e.list.push_back(geom->GetFace(2)->GetGlobalID());
                    e.list.push_back(geom->GetFace(3)->GetGlobalID());
                    e.ghost        = false;
                    elements[e.id] = e;
                }
            }
            for (auto [id, geom] : m_meshGraph->GetGeomMap<PyrGeom>())
            {
                if (m_meshGraph->CheckRange(*geom))
                {
                    MeshEntity e;
                    e.id = e.origId = id;
                    e.list.push_back(geom->GetFace(0)->GetGlobalID());
                    e.list.push_back(geom->GetFace(1)->GetGlobalID());
                    e.list.push_back(geom->GetFace(2)->GetGlobalID());
                    e.list.push_back(geom->GetFace(3)->GetGlobalID());
                    e.list.push_back(geom->GetFace(4)->GetGlobalID());
                    e.ghost        = false;
                    elements[e.id] = e;
                }
            }
            for (auto [id, geom] : m_meshGraph->GetGeomMap<PrismGeom>())
            {
                if (m_meshGraph->CheckRange(*geom))
                {
                    MeshEntity e;
                    e.id = e.origId = id;
                    e.list.push_back(geom->GetFace(0)->GetGlobalID());
                    e.list.push_back(geom->GetFace(1)->GetGlobalID());
                    e.list.push_back(geom->GetFace(2)->GetGlobalID());
                    e.list.push_back(geom->GetFace(3)->GetGlobalID());
                    e.list.push_back(geom->GetFace(4)->GetGlobalID());
                    e.ghost        = false;
                    elements[e.id] = e;
                }
            }
            for (auto [id, geom] : m_meshGraph->GetGeomMap<HexGeom>())
            {
                if (m_meshGraph->CheckRange(*geom))
                {
                    MeshEntity e;
                    e.id = e.origId = id;
                    e.list.push_back(geom->GetFace(0)->GetGlobalID());
                    e.list.push_back(geom->GetFace(1)->GetGlobalID());
                    e.list.push_back(geom->GetFace(2)->GetGlobalID());
                    e.list.push_back(geom->GetFace(3)->GetGlobalID());
                    e.list.push_back(geom->GetFace(4)->GetGlobalID());
                    e.list.push_back(geom->GetFace(5)->GetGlobalID());
                    e.ghost        = false;
                    elements[e.id] = e;
                }
            }
        }
        break;
    }

    return elements;
}

CompositeDescriptor MeshGraphIO::CreateCompositeDescriptor()
{
    auto meshComposites = &m_meshGraph->GetComposites();
    CompositeDescriptor ret;

    for (auto &comp : *meshComposites)
    {
        std::pair<LibUtilities::ShapeType, std::vector<int>> tmp;
        tmp.first = comp.second->m_geomVec[0]->GetShapeType();

        tmp.second.resize(comp.second->m_geomVec.size());
        for (size_t i = 0; i < tmp.second.size(); ++i)
        {
            tmp.second[i] = comp.second->m_geomVec[i]->GetGlobalID();
        }

        ret[comp.first] = tmp;
    }

    return ret;
}

/**
 * @brief Returns a string representation of a composite.
 */
std::string MeshGraphIO::GetCompositeString(CompositeSharedPtr comp)
{
    if (comp->m_geomVec.size() == 0)
    {
        return "";
    }

    // Create a map that gets around the issue of mapping faces -> F and edges
    // -> E inside the tag.
    std::map<LibUtilities::ShapeType, std::pair<std::string, std::string>>
        compMap;
    compMap[LibUtilities::ePoint]         = std::make_pair("V", "V");
    compMap[LibUtilities::eSegment]       = std::make_pair("S", "E");
    compMap[LibUtilities::eQuadrilateral] = std::make_pair("Q", "F");
    compMap[LibUtilities::eTriangle]      = std::make_pair("T", "F");
    compMap[LibUtilities::eTetrahedron]   = std::make_pair("A", "A");
    compMap[LibUtilities::ePyramid]       = std::make_pair("P", "P");
    compMap[LibUtilities::ePrism]         = std::make_pair("R", "R");
    compMap[LibUtilities::eHexahedron]    = std::make_pair("H", "H");

    std::stringstream s;

    Geometry *firstGeom = comp->m_geomVec[0];
    int shapeDim        = firstGeom->GetShapeDim();
    std::string tag     = (shapeDim < m_meshGraph->GetMeshDimension())
                              ? compMap[firstGeom->GetShapeType()].second
                              : compMap[firstGeom->GetShapeType()].first;

    std::vector<unsigned int> idxList;
    std::transform(comp->m_geomVec.begin(), comp->m_geomVec.end(),
                   std::back_inserter(idxList),
                   [](Geometry *geom) { return geom->GetGlobalID(); });

    s << " " << tag << "[" << ParseUtils::GenerateSeqString(idxList) << "] ";
    return s.str();
}

} // namespace Nektar::SpatialDomains
