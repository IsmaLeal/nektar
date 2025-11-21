////////////////////////////////////////////////////////////////////////////////
//
//  File: MeshGraphIOHDF5.cpp
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
//  Description: HDF5-based mesh format for Nektar++.
//
////////////////////////////////////////////////////////////////////////////////

#include <boost/algorithm/string.hpp>
#include <tinyxml.h>
#include <type_traits>

#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <LibUtilities/BasicUtils/ParseUtils.h>
#include <LibUtilities/BasicUtils/Timer.h>
#include <SpatialDomains/MeshGraphIOHDF5.h>
#include <SpatialDomains/MeshPartition.h>
#include <SpatialDomains/Movement/Movement.h>

#define TIME_RESULT(verb, msg, timer)                                          \
    if (verb)                                                                  \
    {                                                                          \
        std::cout << "  - " << msg << ": " << timer.TimePerTest(1) << "\n"     \
                  << std::endl;                                                \
    }

using namespace Nektar::LibUtilities;

namespace Nektar::SpatialDomains
{

/// Version of the Nektar++ HDF5 geometry format, which is embedded into the
/// main NEKTAR/GEOMETRY group as an attribute.
const unsigned int MeshGraphIOHDF5::FORMAT_VERSION = 2;

std::string MeshGraphIOHDF5::className =
    GetMeshGraphIOFactory().RegisterCreatorFunction(
        "HDF5", MeshGraphIOHDF5::create, "IO with HDF5 geometry");

void MeshGraphIOHDF5::v_ReadGeometry(bool fillGraph)
{
    ReadComposites();
    ReadDomain();

    m_meshGraph->ReadExpansionInfo();

    // Close up shop.
    m_mesh->Close();
    m_maps->Close();
    m_file->Close();

    if (fillGraph)
    {
        m_meshGraph->FillGraph();
    }
}

/**
 * @brief Utility function to split a vector equally amongst a number of
 * processors.
 *
 * @param vecsize  Size of the total amount of work
 * @param rank     Rank of this process
 * @param nprocs   Number of processors in the group
 *
 * @return A pair with the offset this process should occupy, along with the
 *         count for the amount of work.
 */
std::pair<size_t, size_t> SplitWork(size_t vecsize, int rank, int nprocs)
{
    size_t div = vecsize / nprocs;
    size_t rem = vecsize % nprocs;
    if (rank < rem)
    {
        return std::make_pair(rank * (div + 1), div + 1);
    }
    else
    {
        return std::make_pair((rank - rem) * div + rem * (div + 1), div);
    }
}

template <class T, typename std::enable_if<T::kDim == 0, int>::type = 0>
inline int GetGeomDataDim([[maybe_unused]] GeomMapView<T> &geomMap)
{
    return 3;
}

template <class T, typename std::enable_if<T::kDim == 1, int>::type = 0>
inline int GetGeomDataDim([[maybe_unused]] GeomMapView<T> &geomMap)
{
    return T::kNverts;
}

template <class T, typename std::enable_if<T::kDim == 2, int>::type = 0>
inline int GetGeomDataDim([[maybe_unused]] GeomMapView<T> &geomMap)
{
    return T::kNedges;
}

template <class T, typename std::enable_if<T::kDim == 3, int>::type = 0>
inline int GetGeomDataDim([[maybe_unused]] GeomMapView<T> &geomMap)
{
    return T::kNfaces;
}

template <class... T>
inline void UniqueValues([[maybe_unused]] std::unordered_set<int> &unique)
{
}

template <class... T>
inline void UniqueValues(std::unordered_set<int> &unique,
                         const std::vector<int> &input, T &...args)
{
    for (auto i : input)
    {
        unique.insert(i);
    }

    UniqueValues(unique, args...);
}

std::string MeshGraphIOHDF5::cmdSwitch =
    LibUtilities::SessionReader::RegisterCmdLineFlag(
        "use-hdf5-node-comm", "",
        "Use a per-node communicator for HDF5 partitioning.");

/**
 * @brief Partition the mesh
 */
void MeshGraphIOHDF5::v_PartitionMesh(
    LibUtilities::SessionReaderSharedPtr session)

{
    LibUtilities::Timer all;
    all.Start();
    int err;
    LibUtilities::CommSharedPtr comm     = session->GetComm();
    LibUtilities::CommSharedPtr commMesh = comm->GetRowComm();
    const bool isRoot                    = comm->TreatAsRankZero();

    // By default, only the root process will have read the session file, which
    // is done to avoid every process needing to read the XML file. For HDF5, we
    // don't care about this, so just have every process parse the session file.
    if (!isRoot)
    {
        session->InitSession();
    }

    // We use the XML geometry to find information about the HDF5 file.
    m_session            = session;
    m_xmlGeom            = session->GetElement("NEKTAR/GEOMETRY");
    TiXmlAttribute *attr = m_xmlGeom->FirstAttribute();
    int meshDimension    = 3;
    int spaceDimension   = 3;

    while (attr)
    {
        std::string attrName(attr->Name());
        if (attrName == "DIM")
        {
            err = attr->QueryIntValue(&meshDimension);
            ASSERTL0(err == TIXML_SUCCESS, "Unable to read mesh dimension.");
        }
        else if (attrName == "SPACE")
        {
            err = attr->QueryIntValue(&spaceDimension);
            ASSERTL0(err == TIXML_SUCCESS, "Unable to read space dimension.");
        }
        else if (attrName == "PARTITION")
        {
            ASSERTL0(false,
                     "PARTITION parameter should only be used in XML meshes");
        }
        else if (attrName == "HDF5FILE")
        {
            m_hdf5Name = attr->Value();
        }
        else if (attrName == "PARTITIONED")
        {
            ASSERTL0(false,
                     "PARTITIONED parameter should only be used in XML meshes");
        }
        else
        {
            std::string errstr("Unknown attribute: ");
            errstr += attrName;
            ASSERTL1(false, errstr.c_str());
        }
        // Get the next attribute.
        attr = attr->Next();
    }

    ASSERTL0(m_hdf5Name.size() > 0, "Unable to obtain mesh file name.");
    ASSERTL0(meshDimension <= spaceDimension,
             "Mesh dimension greater than space dimension.");

    m_meshGraph->SetMeshDimension(meshDimension);
    m_meshGraph->SetSpaceDimension(spaceDimension);

    // Open handle to the HDF5 mesh
    LibUtilities::H5::PListSharedPtr parallelProps = H5::PList::Default();
    m_readPL                                       = H5::PList::Default();

    if (commMesh->GetSize() > 1)
    {
        // Use MPI/O to access the file
        parallelProps = H5::PList::FileAccess();
        parallelProps->SetMpio(commMesh);
        // Use collective IO
        m_readPL = H5::PList::DatasetXfer();
        m_readPL->SetDxMpioCollective();
    }

    m_file = H5::File::Open(m_hdf5Name, H5F_ACC_RDONLY, parallelProps);

    auto root = m_file->OpenGroup("NEKTAR");
    ASSERTL0(root, "Cannot find NEKTAR group in HDF5 file.");

    auto root2 = root->OpenGroup("GEOMETRY");
    ASSERTL0(root2, "Cannot find NEKTAR/GEOMETRY group in HDF5 file.");

    // Check format version
    H5::Group::AttrIterator attrIt  = root2->attr_begin();
    H5::Group::AttrIterator attrEnd = root2->attr_end();
    for (; attrIt != attrEnd; ++attrIt)
    {
        if (*attrIt == "FORMAT_VERSION")
        {
            break;
        }
    }
    ASSERTL0(attrIt != attrEnd,
             "Unable to determine Nektar++ geometry HDF5 file version.");
    root2->GetAttribute("FORMAT_VERSION", m_inFormatVersion);

    ASSERTL0(m_inFormatVersion <= FORMAT_VERSION,
             "File format in " + m_hdf5Name +
                 " is higher than supported in "
                 "this version of Nektar++");

    m_mesh = root2->OpenGroup("MESH");
    ASSERTL0(m_mesh, "Cannot find NEKTAR/GEOMETRY/MESH group in HDF5 file.");
    m_maps = root2->OpenGroup("MAPS");
    ASSERTL0(m_mesh, "Cannot find NEKTAR/GEOMETRY/MAPS group in HDF5 file.");

    if (m_meshPartitioned)
    {
        return;
    }

    if (m_meshGraph->GetDomainRange() &&
        m_meshGraph->GetDomainRange()->m_compElmts)
    {
        m_meshGraph->GetDomainRange()->m_compElmts = meshDimension;
    }

    SetupCompositeRange(m_meshGraph->GetDomainRange());

    m_meshPartitioned = true;
    m_meshGraph->SetMeshPartitioned(true);

    // Depending on dimension, read element IDs.
    std::map<int,
             std::vector<std::tuple<std::string, int, LibUtilities::ShapeType>>>
        dataSets;

    dataSets[1] = {{"SEG", 2, LibUtilities::eSegment}};
    dataSets[2] = {{"TRI", 3, LibUtilities::eTriangle},
                   {"QUAD", 4, LibUtilities::eQuadrilateral}};
    dataSets[3] = {{"TET", 4, LibUtilities::eTetrahedron},
                   {"PYR", 5, LibUtilities::ePyramid},
                   {"PRISM", 5, LibUtilities::ePrism},
                   {"HEX", 6, LibUtilities::eHexahedron}};

    const bool verbRoot = isRoot && session->DefinesCmdLineArgument("verbose");

    if (verbRoot)
    {
        std::cout << "Reading HDF5 geometry..." << std::endl;
    }

    // If we want to use an multi-level communicator, then split the
    // communicator at this point. We set by default the inter-node communicator
    // to be the normal communicator: this way if the multi-level partitioning
    // is disabled we proceed as 'normal'.
    LibUtilities::CommSharedPtr innerComm, interComm = comm;
    int innerRank = 0, innerSize = 1,
        interRank = interComm->GetRowComm()->GetRank(),
        interSize = interComm->GetRowComm()->GetSize();

    if (session->DefinesCmdLineArgument("use-hdf5-node-comm"))
    {
        ASSERTL0(comm->GetSize() == commMesh->GetSize(),
                 "--use-hdf5-node-comm not available with Parallel-in-Time")

        auto splitComm = comm->SplitCommNode();
        innerComm      = splitComm.first;
        interComm      = splitComm.second;
        innerRank      = innerComm->GetRank();
        innerSize      = innerComm->GetSize();

        if (innerRank == 0)
        {
            interRank = interComm->GetRank();
            interSize = interComm->GetSize();
        }
    }

    // Unordered set of rows of the dataset this process needs to read for the
    // elements of dimension meshDimension.
    std::unordered_set<int> toRead;

    // Calculate reasonably even distribution of processors for calling
    // ptScotch. We'll do this work on only one process per node.
    LibUtilities::Timer t;
    t.Start();

    if (verbRoot)
    {
        std::cout << "  - beginning partitioning" << std::endl;
    }

    // Perform initial read if either (a) we are on all ranks and multi-level
    // partitioning is not enabled; or (b) rank 0 of all nodes if it is.
    if (innerRank == 0)
    {
        LibUtilities::Timer t2;
        t2.Start();

        const bool verbRoot2 =
            isRoot && session->DefinesCmdLineArgument("verbose");

        // Read IDs for partitioning purposes
        std::vector<int> ids;

        // Map from element ID to 'row' which is a contiguous ordering required
        // for parallel partitioning.
        std::vector<MeshEntity> elmts;
        std::unordered_map<int, int> row2id, id2row;

        LibUtilities::H5::FileSharedPtr file    = m_file;
        LibUtilities::H5::PListSharedPtr readPL = m_readPL;
        LibUtilities::H5::GroupSharedPtr mesh = m_mesh, maps = m_maps;

        if (innerComm)
        {
            // For per-node partitioning, create a temporary reader (otherwise
            // communicators are inconsistent).
            auto parallelProps = H5::PList::FileAccess();
            parallelProps->SetMpio(interComm);

            // Use collective IO
            readPL = H5::PList::DatasetXfer();
            readPL->SetDxMpioCollective();
            file = H5::File::Open(m_hdf5Name, H5F_ACC_RDONLY, parallelProps);

            auto root  = file->OpenGroup("NEKTAR");
            auto root2 = root->OpenGroup("GEOMETRY");
            mesh       = root2->OpenGroup("MESH");
            maps       = root2->OpenGroup("MAPS");
        }

        int rowCount = 0;
        for (auto &it : dataSets[meshDimension])
        {
            std::string ds = std::get<0>(it);

            if (!mesh->ContainsDataSet(ds))
            {
                continue;
            }

            // Open metadata dataset
            H5::DataSetSharedPtr data    = mesh->OpenDataSet(ds);
            H5::DataSpaceSharedPtr space = data->GetSpace();
            std::vector<hsize_t> dims    = space->GetDims();

            H5::DataSetSharedPtr mdata    = maps->OpenDataSet(ds);
            H5::DataSpaceSharedPtr mspace = mdata->GetSpace();
            std::vector<hsize_t> mdims    = mspace->GetDims();

            // TODO: This could perhaps be done more intelligently; reads all
            // IDs for the top-level elements so that we can construct the dual
            // graph of the mesh.
            std::vector<int> tmpElmts, tmpIds;
            mdata->Read(tmpIds, mspace, readPL);
            data->Read(tmpElmts, space, readPL);

            const int nGeomData = std::get<1>(it);

            if (m_meshGraph->GetDomainRange() ==
                LibUtilities::NullDomainRangeShPtr)
            {
                // avoid range checking on larger meshes if not required
                for (int i = 0, cnt = 0; i < tmpIds.size(); ++i, ++rowCount)
                {
                    MeshEntity e;
                    row2id[rowCount]  = tmpIds[i];
                    id2row[tmpIds[i]] = row2id[rowCount];
                    e.id              = rowCount;
                    e.origId          = tmpIds[i];
                    e.ghost           = false;
                    e.list            = std::vector<unsigned int>(
                        &tmpElmts[cnt], &tmpElmts[cnt + nGeomData]);
                    elmts.push_back(e);
                    cnt += nGeomData;
                }
            }
            else
            {
                // avoid range checking on larger meshes if not required
                for (int i = 0, cnt = 0; i < tmpIds.size(); ++i, ++rowCount)
                {
                    MeshEntity e;
                    row2id[rowCount]  = tmpIds[i];
                    id2row[tmpIds[i]] = row2id[rowCount];
                    e.id              = rowCount;
                    e.origId          = tmpIds[i];
                    e.ghost           = false;
                    e.list            = std::vector<unsigned int>(
                        &tmpElmts[cnt], &tmpElmts[cnt + nGeomData]);
                    if (m_meshGraph->CheckRange(e))
                    {
                        elmts.push_back(e);
                    }
                    cnt += nGeomData;
                }
            }
        }

        interComm->GetRowComm()->Block();

        t2.Stop();
        TIME_RESULT(verbRoot2, "  - initial read", t2);
        t2.Start();

        // Do not partition in serial since postprocessing may not
        // lead to very suitable element distribution that then leads
        // to challenges with Scotch
        if (commMesh->GetSize() > 1)
        {
            // Check to see we have at least as many processors as elements.
            size_t numElmt = elmts.size();
            ASSERTL0(commMesh->GetSize() <= numElmt,
                     "This mesh has more processors than elements!");

            auto elRange = SplitWork(numElmt, interRank, interSize);

            // Construct map of element entities for partitioner.
            std::map<int, MeshEntity> partElmts;
            std::unordered_set<int> facetIDs;

            int vcnt = 0;

            for (int el = elRange.first; el < elRange.first + elRange.second;
                 ++el, ++vcnt)
            {
                MeshEntity elmt    = elmts[el];
                elmt.ghost         = false;
                partElmts[elmt.id] = elmt;

                for (auto &facet : elmt.list)
                {
                    facetIDs.insert(facet);
                }
            }

            // Now identify ghost vertices for the graph. This could also
            // probably be improved.
            int nLocal = vcnt;
            for (int i = 0; i < numElmt; ++i)
            {
                // Ignore anything we already read.
                if (i >= elRange.first && i < elRange.first + elRange.second)
                {
                    continue;
                }

                MeshEntity elmt = elmts[i];
                bool insert     = false;

                // Check for connections to local elements.
                for (auto &eId : elmt.list)
                {
                    if (facetIDs.find(eId) != facetIDs.end())
                    {
                        insert = true;
                        break;
                    }
                }

                if (insert)
                {
                    elmt.ghost         = true;
                    partElmts[elmt.id] = elmt;
                }
            }

            // Create partitioner. Default partitioner to use is PtScotch. Use
            // ParMetis as default if it is installed. Override default with
            // command-line flags if they are set.
            std::string partitionerName =
                commMesh->GetSize() > 1 ? "PtScotch" : "Scotch";
            if (GetMeshPartitionFactory().ModuleExists("ParMetis"))
            {
                partitionerName = "ParMetis";
            }
            if (session->DefinesCmdLineArgument("use-parmetis"))
            {
                partitionerName = "ParMetis";
            }
            if (session->DefinesCmdLineArgument("use-ptscotch"))
            {
                partitionerName = "PtScotch";
            }

            MeshPartitionSharedPtr partitioner =
                GetMeshPartitionFactory().CreateInstance(
                    partitionerName, session, interComm, meshDimension,
                    partElmts, CreateCompositeDescriptor(id2row));

            t2.Stop();
            TIME_RESULT(verbRoot2, "  - partitioner setup", t2);
            t2.Start();

            partitioner->PartitionMesh(interSize, true, false, nLocal);
            t2.Stop();
            TIME_RESULT(verbRoot2, "  - partitioning", t2);
            t2.Start();

            // Now construct a second graph that is partitioned in serial by
            // this rank.
            std::vector<unsigned int> nodeElmts;
            partitioner->GetElementIDs(interRank, nodeElmts);

            if (innerSize > 1)
            {
                // Construct map of element entities for partitioner.
                std::map<int, MeshEntity> partElmts;
                std::unordered_map<int, int> row2elmtid, elmtid2row;

                int vcnt = 0;

                // We need to keep track of which elements in the new partition
                // correspond to elemental IDs for later (in a similar manner to
                // row2id).
                for (auto &elmtRow : nodeElmts)
                {
                    row2elmtid[vcnt]                  = elmts[elmtRow].origId;
                    elmtid2row[elmts[elmtRow].origId] = vcnt;
                    MeshEntity elmt                   = elmts[elmtRow];
                    elmt.ghost                        = false;
                    partElmts[vcnt++]                 = elmt;
                }

                // Create temporary serial communicator for serial partitioning.
                auto tmpComm = LibUtilities::GetCommFactory().CreateInstance(
                    "Serial", 0, 0);

                MeshPartitionSharedPtr partitioner =
                    GetMeshPartitionFactory().CreateInstance(
                        "Scotch", session, tmpComm, meshDimension, partElmts,
                        CreateCompositeDescriptor(elmtid2row));

                t2.Stop();
                TIME_RESULT(verbRoot2, "  - inner partition setup", t2);
                t2.Start();

                partitioner->PartitionMesh(innerSize, true, false, 0);

                t2.Stop();
                TIME_RESULT(verbRoot2, "  - inner partitioning", t2);
                t2.Start();

                // Send contributions to remaining processors.
                for (int i = 1; i < innerSize; ++i)
                {
                    std::vector<unsigned int> tmp;
                    partitioner->GetElementIDs(i, tmp);
                    size_t tmpsize = tmp.size();
                    for (int j = 0; j < tmpsize; ++j)
                    {
                        tmp[j] = row2elmtid[tmp[j]];
                    }
                    innerComm->Send(i, tmpsize);
                    innerComm->Send(i, tmp);
                }

                t2.Stop();
                TIME_RESULT(verbRoot2, "  - inner partition scatter", t2);

                std::vector<unsigned int> tmp;
                partitioner->GetElementIDs(0, tmp);

                for (auto &tmpId : tmp)
                {
                    toRead.insert(row2elmtid[tmpId]);
                }
            }
            else
            {
                for (auto &tmpId : nodeElmts)
                {
                    toRead.insert(row2id[tmpId]);
                }
            }
        }
        else // Serial: Fill toRead with Elmt.origId
        {
            for (auto &tmpId : elmts)
            {
                toRead.insert(tmpId.origId);
            }
        }
    }
    else
    {
        // For multi-level partitioning, the innermost rank receives its
        // partitions from rank 0 on each node.
        size_t tmpSize;
        innerComm->Recv(0, tmpSize);
        std::vector<unsigned int> tmp(tmpSize);
        innerComm->Recv(0, tmp);

        for (auto &tmpId : tmp)
        {
            toRead.insert(tmpId);
        }
    }

    t.Stop();
    TIME_RESULT(verbRoot, "partitioning total", t);

    // Since objects are going to be constructed starting from vertices, we
    // now need to recurse down the geometry facet dimensions to figure out
    // which rows to read from each dataset.
    std::vector<int> vertIDs, segIDs, triIDs, quadIDs;
    std::vector<int> tetIDs, prismIDs, pyrIDs, hexIDs;
    std::vector<int> segData, triData, quadData, tetData;
    std::vector<int> prismData, pyrData, hexData;
    std::vector<NekDouble> vertData;

    auto &vertSet     = m_meshGraph->GetGeomMap<PointGeom>();
    auto &segGeoms    = m_meshGraph->GetGeomMap<SegGeom>();
    auto &triGeoms    = m_meshGraph->GetGeomMap<TriGeom>();
    auto &quadGeoms   = m_meshGraph->GetGeomMap<QuadGeom>();
    auto &hexGeoms    = m_meshGraph->GetGeomMap<HexGeom>();
    auto &pyrGeoms    = m_meshGraph->GetGeomMap<PyrGeom>();
    auto &prismGeoms  = m_meshGraph->GetGeomMap<PrismGeom>();
    auto &tetGeoms    = m_meshGraph->GetGeomMap<TetGeom>();
    auto &curvedEdges = m_meshGraph->GetCurvedEdges();
    auto &curvedFaces = m_meshGraph->GetCurvedFaces();

    if (meshDimension == 3)
    {
        t.Start();
        // Read 3D data
        ReadGeometryData(hexGeoms, "HEX", toRead, hexIDs, hexData);
        ReadGeometryData(pyrGeoms, "PYR", toRead, pyrIDs, pyrData);
        ReadGeometryData(prismGeoms, "PRISM", toRead, prismIDs, prismData);
        ReadGeometryData(tetGeoms, "TET", toRead, tetIDs, tetData);

        toRead.clear();
        UniqueValues(toRead, hexData, pyrData, prismData, tetData);
        t.Stop();
        TIME_RESULT(verbRoot, "read 3D elements", t);
    }

    if (meshDimension >= 2)
    {
        t.Start();
        // Read 2D data
        ReadGeometryData(triGeoms, "TRI", toRead, triIDs, triData);
        ReadGeometryData(quadGeoms, "QUAD", toRead, quadIDs, quadData);

        toRead.clear();
        UniqueValues(toRead, triData, quadData);
        t.Stop();
        TIME_RESULT(verbRoot, "read 2D elements", t);
    }

    if (meshDimension >= 1)
    {
        t.Start();
        // Read 1D data
        ReadGeometryData(segGeoms, "SEG", toRead, segIDs, segData);

        toRead.clear();
        UniqueValues(toRead, segData);
        t.Stop();
        TIME_RESULT(verbRoot, "read 1D elements", t);
    }

    t.Start();
    ReadGeometryData(vertSet, "VERT", toRead, vertIDs, vertData);
    t.Stop();
    TIME_RESULT(verbRoot, "read 0D elements", t);

    // Now start to construct geometry objects, starting from vertices
    // upwards.
    t.Start();
    FillGeomMap(vertSet, CurveMap(), vertIDs, vertData);
    t.Stop();
    TIME_RESULT(verbRoot, "construct 0D elements", t);

    if (meshDimension >= 1)
    {
        // Read curves
        toRead.clear();
        for (auto &edge : segIDs)
        {
            toRead.insert(edge);
        }
        ReadCurveMap(curvedEdges, "CURVE_EDGE", toRead);

        t.Start();
        FillGeomMap(segGeoms, curvedEdges, segIDs, segData);
        t.Stop();
        TIME_RESULT(verbRoot, "construct 1D elements", t);
    }

    if (meshDimension >= 2)
    {
        // Read curves
        toRead.clear();
        for (auto &face : triIDs)
        {
            toRead.insert(face);
        }
        for (auto &face : quadIDs)
        {
            toRead.insert(face);
        }
        ReadCurveMap(curvedFaces, "CURVE_FACE", toRead);

        t.Start();
        FillGeomMap(triGeoms, curvedFaces, triIDs, triData);
        FillGeomMap(quadGeoms, curvedFaces, quadIDs, quadData);
        t.Stop();
        TIME_RESULT(verbRoot, "construct 2D elements", t);
    }

    if (meshDimension >= 3)
    {
        t.Start();
        FillGeomMap(hexGeoms, CurveMap(), hexIDs, hexData);
        FillGeomMap(prismGeoms, CurveMap(), prismIDs, prismData);
        FillGeomMap(pyrGeoms, CurveMap(), pyrIDs, pyrData);
        FillGeomMap(tetGeoms, CurveMap(), tetIDs, tetData);
        t.Stop();
        TIME_RESULT(verbRoot, "construct 3D elements", t);
    }

    // Populate m_bndRegOrder.
    if (session->DefinesElement("NEKTAR/CONDITIONS"))
    {
        std::set<int> vBndRegionIdList;
        TiXmlElement *vConditions =
            new TiXmlElement(*session->GetElement("Nektar/Conditions"));
        TiXmlElement *vBndRegions =
            vConditions->FirstChildElement("BOUNDARYREGIONS");
        // Use fine-level for mesh partition (Parallel-in-Time)
        LibUtilities::SessionReader::GetXMLElementTimeLevel(vBndRegions, 0);
        TiXmlElement *vItem;

        if (vBndRegions)
        {
            auto &graph_bndRegOrder = m_meshGraph->GetBndRegionOrdering();
            vItem                   = vBndRegions->FirstChildElement();
            while (vItem)
            {
                std::string vSeqStr = vItem->FirstChild()->ToText()->Value();
                std::string::size_type indxBeg = vSeqStr.find_first_of('[') + 1;
                std::string::size_type indxEnd = vSeqStr.find_last_of(']') - 1;
                vSeqStr = vSeqStr.substr(indxBeg, indxEnd - indxBeg + 1);

                std::vector<unsigned int> vSeq;
                ParseUtils::GenerateSeqVector(vSeqStr.c_str(), vSeq);

                int p                = atoi(vItem->Attribute("ID"));
                m_bndRegOrder[p]     = vSeq;
                graph_bndRegOrder[p] = vSeq;
                vItem                = vItem->NextSiblingElement();
            }
        }
    }

    all.Stop();
    TIME_RESULT(verbRoot, "total time", all);
}

template <class T, typename DataType>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<T> &geomMap, [[maybe_unused]] int id,
    [[maybe_unused]] DataType *data, [[maybe_unused]] Curve *curve)
{
}

template <>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<PointGeom> &geomMap, int id, NekDouble *data,
    [[maybe_unused]] Curve *curve)
{
    m_meshGraph->CreatePointGeom(m_meshGraph->GetSpaceDimension(), id, data[0],
                                 data[1], data[2]);
}

template <>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<SegGeom> &geomMap, int id, int *data,
    Curve *curve)
{
    std::array<PointGeom *, 2> pts = {m_meshGraph->GetPointGeom(data[0]),
                                      m_meshGraph->GetPointGeom(data[1])};

    m_meshGraph->CreateSegGeom(id, m_meshGraph->GetSpaceDimension(), pts,
                               curve);
}

template <>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<TriGeom> &geomMap, int id, int *data,
    Curve *curve)
{
    std::array<SegGeom *, 3> segs = {m_meshGraph->GetSegGeom(data[0]),
                                     m_meshGraph->GetSegGeom(data[1]),
                                     m_meshGraph->GetSegGeom(data[2])};
    m_meshGraph->CreateTriGeom(id, segs, curve);
}

template <>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<QuadGeom> &geomMap, int id, int *data,
    Curve *curve)
{
    std::array<SegGeom *, 4> segs = {
        m_meshGraph->GetSegGeom(data[0]), m_meshGraph->GetSegGeom(data[1]),
        m_meshGraph->GetSegGeom(data[2]), m_meshGraph->GetSegGeom(data[3])};
    m_meshGraph->CreateQuadGeom(id, segs, curve);
}

template <>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<TetGeom> &geomMap, int id, int *data,
    [[maybe_unused]] Curve *curve)
{
    std::array<TriGeom *, 4> faces = {
        m_meshGraph->GetTriGeom(data[0]), m_meshGraph->GetTriGeom(data[1]),
        m_meshGraph->GetTriGeom(data[2]), m_meshGraph->GetTriGeom(data[3])};

    auto geom = m_meshGraph->CreateTetGeom(id, faces);
    m_meshGraph->PopulateFaceToElMap(geom, TetGeom::kNfaces);
}

template <>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<PyrGeom> &geomMap, int id, int *data,
    [[maybe_unused]] Curve *curve)
{
    std::array<Geometry2D *, 5> faces = {m_meshGraph->GetGeometry2D(data[0]),
                                         m_meshGraph->GetGeometry2D(data[1]),
                                         m_meshGraph->GetGeometry2D(data[2]),
                                         m_meshGraph->GetGeometry2D(data[3]),
                                         m_meshGraph->GetGeometry2D(data[4])};

    auto geom = m_meshGraph->CreatePyrGeom(id, faces);
    m_meshGraph->PopulateFaceToElMap(geom, PyrGeom::kNfaces);
}

template <>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<PrismGeom> &geomMap, int id, int *data,
    [[maybe_unused]] Curve *curve)
{
    std::array<Geometry2D *, 5> faces = {m_meshGraph->GetGeometry2D(data[0]),
                                         m_meshGraph->GetGeometry2D(data[1]),
                                         m_meshGraph->GetGeometry2D(data[2]),
                                         m_meshGraph->GetGeometry2D(data[3]),
                                         m_meshGraph->GetGeometry2D(data[4])};

    auto geom = m_meshGraph->CreatePrismGeom(id, faces);
    m_meshGraph->PopulateFaceToElMap(geom, PrismGeom::kNfaces);
}

template <>
void MeshGraphIOHDF5::ConstructGeomObject(
    [[maybe_unused]] GeomMapView<HexGeom> &geomMap, int id, int *data,
    [[maybe_unused]] Curve *curve)
{
    std::array<QuadGeom *, 6> faces = {
        m_meshGraph->GetQuadGeom(data[0]), m_meshGraph->GetQuadGeom(data[1]),
        m_meshGraph->GetQuadGeom(data[2]), m_meshGraph->GetQuadGeom(data[3]),
        m_meshGraph->GetQuadGeom(data[4]), m_meshGraph->GetQuadGeom(data[5])};

    auto geom = m_meshGraph->CreateHexGeom(id, faces);
    m_meshGraph->PopulateFaceToElMap(geom, HexGeom::kNfaces);
}

template <class T, typename DataType>
void MeshGraphIOHDF5::FillGeomMap(GeomMapView<T> &geomMap,
                                  const CurveMap &curveMap,
                                  std::vector<int> &ids,
                                  std::vector<DataType> &geomData)
{
    const int nGeomData = GetGeomDataDim(geomMap);
    const int nRows     = geomData.size() / nGeomData;
    Curve *empty        = nullptr;

    // Construct geometry object.
    if (curveMap.size() > 0)
    {
        for (int i = 0, cnt = 0; i < nRows; i++, cnt += nGeomData)
        {
            auto cIt = curveMap.find(ids[i]);
            ConstructGeomObject(geomMap, ids[i], &geomData[cnt],
                                cIt == curveMap.end() ? empty
                                                      : cIt->second.get());
        }
    }
    else
    {
        for (int i = 0, cnt = 0; i < nRows; i++, cnt += nGeomData)
        {
            ConstructGeomObject(geomMap, ids[i], &geomData[cnt], empty);
        }
    }
}

template <class T, typename DataType>
void MeshGraphIOHDF5::ReadGeometryData(GeomMapView<T> &geomMap,
                                       std::string dataSet,
                                       const std::unordered_set<int> &readIds,
                                       std::vector<int> &ids,
                                       std::vector<DataType> &geomData)
{
    if (!m_mesh->ContainsDataSet(dataSet))
    {
        return;
    }

    // Open mesh dataset
    H5::DataSetSharedPtr data    = m_mesh->OpenDataSet(dataSet);
    H5::DataSpaceSharedPtr space = data->GetSpace();
    std::vector<hsize_t> dims    = space->GetDims();

    // Open metadata dataset
    H5::DataSetSharedPtr mdata    = m_maps->OpenDataSet(dataSet);
    H5::DataSpaceSharedPtr mspace = mdata->GetSpace();
    std::vector<hsize_t> mdims    = mspace->GetDims();

    ASSERTL0(mdims[0] == dims[0], "map and data set lengths do not match");

    const int nGeomData = GetGeomDataDim(geomMap);

    // Read all IDs
    std::vector<int> allIds;
    mdata->Read(allIds, mspace);

    // Selective reading; clear data space range so that we can select
    // certain rows from the datasets.
    space->ClearRange();

    int i = 0;
    std::vector<hsize_t> coords;
    for (auto &id : allIds)
    {
        if (readIds.find(id) != readIds.end())
        {
            for (int j = 0; j < nGeomData; ++j)
            {
                coords.push_back(i);
                coords.push_back(j);
            }
            ids.push_back(id);
        }
        ++i;
    }

    space->SetSelection(coords.size() / 2, coords);

    // Read selected data.
    data->Read(geomData, space, m_readPL);
}

void MeshGraphIOHDF5::ReadCurveMap(CurveMap &curveMap, std::string dsName,
                                   const std::unordered_set<int> &readIds)
{
    auto &curveNodes = m_meshGraph->GetAllCurveNodes();

    // If dataset does not exist, exit.
    if (!m_mesh->ContainsDataSet(dsName))
    {
        return;
    }

    // Open up curve map data.
    H5::DataSetSharedPtr curveData    = m_mesh->OpenDataSet(dsName);
    H5::DataSpaceSharedPtr curveSpace = curveData->GetSpace();

    // Open up ID data set.
    H5::DataSetSharedPtr idData    = m_maps->OpenDataSet(dsName);
    H5::DataSpaceSharedPtr idSpace = idData->GetSpace();

    // Read all IDs and clear data space.
    std::vector<int> ids, newIds;
    idData->Read(ids, idSpace);
    curveSpace->ClearRange();

    // Search IDs to figure out which curves to read.
    std::vector<hsize_t> curveSel;

    int cnt = 0;
    for (auto &id : ids)
    {
        if (readIds.find(id) != readIds.end())
        {
            curveSel.push_back(cnt);
            curveSel.push_back(0);
            curveSel.push_back(cnt);
            curveSel.push_back(1);
            curveSel.push_back(cnt);
            curveSel.push_back(2);
            newIds.push_back(id);
        }

        ++cnt;
    }

    // Check to see whether any processor will read anything
    auto toRead = newIds.size();
    m_session->GetComm()->GetRowComm()->AllReduce(toRead,
                                                  LibUtilities::ReduceSum);

    if (toRead == 0)
    {
        return;
    }

    // Now read curve map and read data.
    std::vector<int> curveInfo;
    curveSpace->SetSelection(curveSel.size() / 2, curveSel);
    curveData->Read(curveInfo, curveSpace, m_readPL);

    curveSel.clear();

    std::unordered_map<int, int> curvePtOffset;

    // Construct curves. We'll populate nodes in a minute!
    for (int i = 0, cnt = 0, cnt2 = 0; i < curveInfo.size() / 3; ++i, cnt += 3)
    {
        CurveUniquePtr curve = ObjPoolManager<Curve>::AllocateUniquePtr(
            newIds[i], (LibUtilities::PointsType)curveInfo[cnt + 1]);

        curve->m_points.resize(curveInfo[cnt]);

        const int ptOffset = curveInfo[cnt + 2];

        for (int j = 0; j < curveInfo[cnt]; ++j)
        {
            // ptoffset gives us the row, multiply by 3 for number of
            // coordinates.
            curveSel.push_back(ptOffset + j);
            curveSel.push_back(0);
            curveSel.push_back(ptOffset + j);
            curveSel.push_back(1);
            curveSel.push_back(ptOffset + j);
            curveSel.push_back(2);
        }

        // Store the offset so we know to come back later on to fill in
        // these points.
        curvePtOffset[newIds[i]] = 3 * cnt2;
        cnt2 += curveInfo[cnt];

        curveMap[newIds[i]] = std::move(curve);
    }

    curveInfo.clear();

    // Open node data spacee.
    H5::DataSetSharedPtr nodeData    = m_mesh->OpenDataSet("CURVE_NODES");
    H5::DataSpaceSharedPtr nodeSpace = nodeData->GetSpace();

    nodeSpace->ClearRange();
    nodeSpace->SetSelection(curveSel.size() / 2, curveSel);

    std::vector<NekDouble> nodeRawData;
    nodeData->Read(nodeRawData, nodeSpace, m_readPL);

    // Go back and populate data from nodes.
    for (auto &cIt : curvePtOffset)
    {
        Curve *curve = curveMap[cIt.first].get();

        // Create nodes.
        int cnt = cIt.second;
        for (int i = 0; i < curve->m_points.size(); ++i, cnt += 3)
        {
            curveNodes.emplace_back(
                ObjPoolManager<PointGeom>::AllocateUniquePtr(
                    0, m_meshGraph->GetSpaceDimension(), nodeRawData[cnt],
                    nodeRawData[cnt + 1], nodeRawData[cnt + 2]));
            curve->m_points[i] = curveNodes.back().get();
        }
    }
}

void MeshGraphIOHDF5::ReadDomain()
{
    auto &domain = m_meshGraph->GetDomain();

    if (m_inFormatVersion == 1)
    {
        std::map<int, CompositeSharedPtr> fullDomain;
        H5::DataSetSharedPtr dst     = m_mesh->OpenDataSet("DOMAIN");
        H5::DataSpaceSharedPtr space = dst->GetSpace();

        std::vector<std::string> data;
        dst->ReadVectorString(data, space, m_readPL);
        m_meshGraph->GetCompositeList(data[0], fullDomain);
        domain[0] = fullDomain;

        return;
    }

    std::vector<CompositeMap> fullDomain;
    H5::DataSetSharedPtr dst     = m_mesh->OpenDataSet("DOMAIN");
    H5::DataSpaceSharedPtr space = dst->GetSpace();

    std::vector<std::string> data;
    dst->ReadVectorString(data, space, m_readPL);
    for (auto &dIt : data)
    {
        fullDomain.push_back(CompositeMap());
        m_meshGraph->GetCompositeList(dIt, fullDomain.back());
    }

    H5::DataSetSharedPtr mdata    = m_maps->OpenDataSet("DOMAIN");
    H5::DataSpaceSharedPtr mspace = mdata->GetSpace();

    std::vector<int> ids;
    mdata->Read(ids, mspace);

    for (int i = 0; i < ids.size(); ++i)
    {
        domain[ids[i]] = fullDomain[i];
    }
}

void MeshGraphIOHDF5::SetupCompositeRange(LibUtilities::DomainRangeShPtr &rng)
{
    if (!rng || rng->m_compElmts == false)
    {
        return; // composite range not being used.
    }

    std::string nm = "COMPOSITE";

    H5::DataSetSharedPtr data    = m_mesh->OpenDataSet(nm);
    H5::DataSpaceSharedPtr space = data->GetSpace();
    std::vector<hsize_t> dims    = space->GetDims();

    std::vector<std::string> comps;
    data->ReadVectorString(comps, space);

    H5::DataSetSharedPtr mdata    = m_maps->OpenDataSet(nm);
    H5::DataSpaceSharedPtr mspace = mdata->GetSpace();
    std::vector<hsize_t> mdims    = mspace->GetDims();

    std::vector<int> ids;
    mdata->Read(ids, mspace);

    for (int i = 0; i < dims[0]; i++)
    {

        if (rng->m_comps.count(ids[i]))
        {

            std::string compStr = comps[i];

            char type;
            std::istringstream strm(compStr);

            strm >> type;

            CompositeSharedPtr comp =
                MemoryManager<Composite>::AllocateSharedPtr();

            std::string::size_type indxBeg = compStr.find_first_of('[') + 1;
            std::string::size_type indxEnd = compStr.find_last_of(']') - 1;

            std::string indxStr =
                compStr.substr(indxBeg, indxEnd - indxBeg + 1);
            std::vector<unsigned int> seqVector;

            ParseUtils::GenerateSeqVector(indxStr, seqVector);

            for (auto it : seqVector) // add ids to Traceid
            {
                rng->m_traceIDs.insert(it);
            }
        }
    }
}

void MeshGraphIOHDF5::ReadComposites()
{
    auto &vertSet                = m_meshGraph->GetGeomMap<PointGeom>();
    auto &segGeoms               = m_meshGraph->GetGeomMap<SegGeom>();
    auto &triGeoms               = m_meshGraph->GetGeomMap<TriGeom>();
    auto &quadGeoms              = m_meshGraph->GetGeomMap<QuadGeom>();
    auto &tetGeoms               = m_meshGraph->GetGeomMap<TetGeom>();
    auto &pyrGeoms               = m_meshGraph->GetGeomMap<PyrGeom>();
    auto &prismGeoms             = m_meshGraph->GetGeomMap<PrismGeom>();
    auto &hexGeoms               = m_meshGraph->GetGeomMap<HexGeom>();
    CompositeMap &meshComposites = m_meshGraph->GetComposites();

    std::string nm = "COMPOSITE";

    H5::DataSetSharedPtr data    = m_mesh->OpenDataSet(nm);
    H5::DataSpaceSharedPtr space = data->GetSpace();
    std::vector<hsize_t> dims    = space->GetDims();

    std::vector<std::string> comps;
    data->ReadVectorString(comps, space);

    H5::DataSetSharedPtr mdata    = m_maps->OpenDataSet(nm);
    H5::DataSpaceSharedPtr mspace = mdata->GetSpace();
    std::vector<hsize_t> mdims    = mspace->GetDims();

    std::vector<int> ids;
    mdata->Read(ids, mspace);

    auto &graph_compOrder = m_meshGraph->GetCompositeOrdering();
    for (int i = 0; i < dims[0]; i++)
    {
        std::string compStr = comps[i];

        char type;
        std::istringstream strm(compStr);

        strm >> type;

        CompositeSharedPtr comp = MemoryManager<Composite>::AllocateSharedPtr();

        std::string::size_type indxBeg = compStr.find_first_of('[') + 1;
        std::string::size_type indxEnd = compStr.find_last_of(']') - 1;

        std::string indxStr = compStr.substr(indxBeg, indxEnd - indxBeg + 1);
        std::vector<unsigned int> seqVector;

        ParseUtils::GenerateSeqVector(indxStr, seqVector);
        m_compOrder[ids[i]]     = seqVector;
        graph_compOrder[ids[i]] = seqVector;

        switch (type)
        {
            case 'V':
                for (auto &i : seqVector)
                {
                    auto it = vertSet.find(i);
                    if (it != vertSet.end())
                    {
                        comp->m_geomVec.push_back((*it).second);
                    }
                }
                break;
            case 'S':
            case 'E':
                for (auto &i : seqVector)
                {
                    auto it = segGeoms.find(i);
                    if (it != segGeoms.end())
                    {
                        comp->m_geomVec.push_back((*it).second);
                    }
                }
                break;
            case 'Q':
                for (auto &i : seqVector)
                {
                    auto it = quadGeoms.find(i);
                    if (it != quadGeoms.end())
                    {
                        if (m_meshGraph->CheckRange(*(*it).second))
                        {
                            comp->m_geomVec.push_back((*it).second);
                        }
                    }
                }
                break;
            case 'T':
                for (auto &i : seqVector)
                {
                    auto it = triGeoms.find(i);
                    if (it != triGeoms.end())
                    {
                        if (m_meshGraph->CheckRange(*(*it).second))
                        {
                            comp->m_geomVec.push_back((*it).second);
                        }
                    }
                }
                break;
            case 'F':
                for (auto &i : seqVector)
                {
                    auto it1 = quadGeoms.find(i);
                    if (it1 != quadGeoms.end())
                    {
                        if (m_meshGraph->CheckRange(*(*it1).second))
                        {
                            comp->m_geomVec.push_back((*it1).second);
                        }
                    }
                    auto it2 = triGeoms.find(i);
                    if (it2 != triGeoms.end())
                    {
                        if (m_meshGraph->CheckRange(*(*it2).second))
                        {
                            comp->m_geomVec.push_back((*it2).second);
                        }
                    }
                }
                break;
            case 'A':
                for (auto &i : seqVector)
                {
                    auto it = tetGeoms.find(i);
                    if (it != tetGeoms.end())
                    {
                        if (m_meshGraph->CheckRange(*(*it).second))
                        {
                            comp->m_geomVec.push_back((*it).second);
                        }
                    }
                }
                break;
            case 'P':
                for (auto &i : seqVector)
                {
                    auto it = pyrGeoms.find(i);
                    if (it != pyrGeoms.end())
                    {
                        if (m_meshGraph->CheckRange(*(*it).second))
                        {
                            comp->m_geomVec.push_back((*it).second);
                        }
                    }
                }
                break;
            case 'R':
                for (auto &i : seqVector)
                {
                    auto it = prismGeoms.find(i);
                    if (it != prismGeoms.end())
                    {
                        if (m_meshGraph->CheckRange(*(*it).second))
                        {
                            comp->m_geomVec.push_back((*it).second);
                        }
                    }
                }
                break;
            case 'H':
                for (auto &i : seqVector)
                {
                    auto it = hexGeoms.find(i);
                    if (it != hexGeoms.end())
                    {
                        if (m_meshGraph->CheckRange(*(*it).second))
                        {
                            comp->m_geomVec.push_back((*it).second);
                        }
                    }
                }
                break;
        }

        if (comp->m_geomVec.size() > 0)
        {
            meshComposites[ids[i]] = comp;
        }
    }
}

CompositeDescriptor MeshGraphIOHDF5::CreateCompositeDescriptor(
    std::unordered_map<int, int> &id2row)
{
    CompositeDescriptor ret;

    std::string nm = "COMPOSITE";

    H5::DataSetSharedPtr data    = m_mesh->OpenDataSet(nm);
    H5::DataSpaceSharedPtr space = data->GetSpace();
    std::vector<hsize_t> dims    = space->GetDims();

    std::vector<std::string> comps;
    data->ReadVectorString(comps, space);

    H5::DataSetSharedPtr mdata    = m_maps->OpenDataSet(nm);
    H5::DataSpaceSharedPtr mspace = mdata->GetSpace();
    std::vector<hsize_t> mdims    = mspace->GetDims();

    std::vector<int> ids;
    mdata->Read(ids, mspace);

    for (int i = 0; i < dims[0]; i++)
    {
        std::string compStr = comps[i];

        char type;
        std::istringstream strm(compStr);

        strm >> type;

        std::string::size_type indxBeg = compStr.find_first_of('[') + 1;
        std::string::size_type indxEnd = compStr.find_last_of(']') - 1;

        std::string indxStr = compStr.substr(indxBeg, indxEnd - indxBeg + 1);
        std::vector<unsigned int> seqVector;
        ParseUtils::GenerateSeqVector(indxStr, seqVector);

        LibUtilities::ShapeType shapeType = eNoShapeType;

        switch (type)
        {
            case 'V':
                shapeType = LibUtilities::ePoint;
                break;
            case 'S':
            case 'E':
                shapeType = LibUtilities::eSegment;
                break;
            case 'Q':
            case 'F':
                // Note that for HDF5, the composite descriptor is only used
                // for partitioning purposes so 'F' tag is not really going
                // to be critical in this context.
                shapeType = LibUtilities::eQuadrilateral;
                break;
            case 'T':
                shapeType = LibUtilities::eTriangle;
                break;
            case 'A':
                shapeType = LibUtilities::eTetrahedron;
                break;
            case 'P':
                shapeType = LibUtilities::ePyramid;
                break;
            case 'R':
                shapeType = LibUtilities::ePrism;
                break;
            case 'H':
                shapeType = LibUtilities::eHexahedron;
                break;
        }

        ASSERTL0(shapeType != eNoShapeType, "Invalid shape.");

        std::vector<int> filteredVector;
        for (auto &compElmt : seqVector)
        {
            if (id2row.find(compElmt) == id2row.end())
            {
                continue;
            }

            filteredVector.push_back(compElmt);
        }

        if (filteredVector.size() == 0)
        {
            continue;
        }

        ret[ids[i]] = std::make_pair(shapeType, filteredVector);
    }

    return ret;
}

template <class T, typename std::enable_if<T::kDim == 0, int>::type = 0>
inline NekDouble GetGeomData(T *geom, int i)
{
    return (*geom)(i);
}

template <class T, typename std::enable_if<T::kDim == 1, int>::type = 0>
inline int GetGeomData(T *geom, int i)
{
    return geom->GetVid(i);
}

template <class T, typename std::enable_if<T::kDim == 2, int>::type = 0>
inline int GetGeomData(T *geom, int i)
{
    return geom->GetEid(i);
}

template <class T, typename std::enable_if<T::kDim == 3, int>::type = 0>
inline int GetGeomData(T *geom, int i)
{
    return geom->GetFid(i);
}

template <class T>
void MeshGraphIOHDF5::WriteGeometryMap(GeomMapView<T> &geomMap,
                                       std::string datasetName)
{
    typedef typename std::conditional<std::is_same_v<T, PointGeom>, NekDouble,
                                      int>::type DataType;

    const int nGeomData = GetGeomDataDim(geomMap);
    const size_t nGeom  = geomMap.size();

    if (nGeom == 0)
    {
        return;
    }

    // Construct a map storing IDs
    std::vector<int> idMap(nGeom);
    std::vector<DataType> data(nGeom * nGeomData);

    int cnt1 = 0, cnt2 = 0;
    for (auto [id, geom] : geomMap)
    {
        idMap[cnt1++] = id;

        for (int j = 0; j < nGeomData; ++j)
        {
            data[cnt2 + j] = GetGeomData(geom, j);
        }

        cnt2 += nGeomData;
    }

    std::vector<hsize_t> dims = {static_cast<hsize_t>(nGeom),
                                 static_cast<hsize_t>(nGeomData)};
    H5::DataTypeSharedPtr tp  = H5::DataType::OfObject(data[0]);
    H5::DataSpaceSharedPtr ds =
        std::shared_ptr<H5::DataSpace>(new H5::DataSpace(dims));
    H5::DataSetSharedPtr dst = m_mesh->CreateDataSet(datasetName, tp, ds);
    dst->Write(data, ds);

    tp   = H5::DataType::OfObject(idMap[0]);
    dims = {nGeom};
    ds   = std::shared_ptr<H5::DataSpace>(new H5::DataSpace(dims));
    dst  = m_maps->CreateDataSet(datasetName, tp, ds);
    dst->Write(idMap, ds);
}

void MeshGraphIOHDF5::WriteCurveMap(CurveMap &curves, std::string dsName,
                                    MeshCurvedPts &curvedPts, int &ptOffset,
                                    int &newIdx)
{
    std::vector<int> data, map;

    // Compile curve data.
    for (auto &c : curves)
    {
        map.push_back(c.first);
        data.push_back(c.second->m_points.size());
        data.push_back(c.second->m_ptype);
        data.push_back(ptOffset);

        ptOffset += c.second->m_points.size();

        for (auto &pt : c.second->m_points)
        {
            MeshVertex v;
            v.id = newIdx;
            pt->GetCoords(v.x, v.y, v.z);
            curvedPts.pts.push_back(v);
            curvedPts.index.push_back(newIdx++);
        }
    }

    // Write data.
    std::vector<hsize_t> dims = {data.size() / 3, 3};
    H5::DataTypeSharedPtr tp  = H5::DataType::OfObject(data[0]);
    H5::DataSpaceSharedPtr ds =
        std::shared_ptr<H5::DataSpace>(new H5::DataSpace(dims));
    H5::DataSetSharedPtr dst = m_mesh->CreateDataSet(dsName, tp, ds);
    dst->Write(data, ds);

    tp   = H5::DataType::OfObject(map[0]);
    dims = {map.size()};
    ds   = std::shared_ptr<H5::DataSpace>(new H5::DataSpace(dims));
    dst  = m_maps->CreateDataSet(dsName, tp, ds);
    dst->Write(map, ds);
}

void MeshGraphIOHDF5::WriteCurvePoints(MeshCurvedPts &curvedPts)
{
    std::vector<double> vertData(curvedPts.pts.size() * 3);

    int cnt = 0;
    for (auto &pt : curvedPts.pts)
    {
        vertData[cnt++] = pt.x;
        vertData[cnt++] = pt.y;
        vertData[cnt++] = pt.z;
    }

    std::vector<hsize_t> dims = {curvedPts.pts.size(), 3};
    H5::DataTypeSharedPtr tp  = H5::DataType::OfObject(vertData[0]);
    H5::DataSpaceSharedPtr ds =
        std::shared_ptr<H5::DataSpace>(new H5::DataSpace(dims));
    H5::DataSetSharedPtr dst = m_mesh->CreateDataSet("CURVE_NODES", tp, ds);
    dst->Write(vertData, ds);
}

void MeshGraphIOHDF5::WriteComposites(CompositeMap &composites)
{
    std::vector<std::string> comps;

    // dont need location map only a id map
    // will filter the composites per parition on read, its easier
    // composites do not need to be written in paralell.
    std::vector<int> c_map;

    for (auto &cIt : composites)
    {
        if (cIt.second->m_geomVec.size() == 0)
        {
            continue;
        }

        comps.push_back(GetCompositeString(cIt.second));
        c_map.push_back(cIt.first);
    }

    H5::DataTypeSharedPtr tp  = H5::DataType::String();
    H5::DataSpaceSharedPtr ds = H5::DataSpace::OneD(comps.size());
    H5::DataSetSharedPtr dst  = m_mesh->CreateDataSet("COMPOSITE", tp, ds);
    dst->WriteVectorString(comps, ds, tp);

    tp  = H5::DataType::OfObject(c_map[0]);
    ds  = H5::DataSpace::OneD(c_map.size());
    dst = m_maps->CreateDataSet("COMPOSITE", tp, ds);
    dst->Write(c_map, ds);
}

void MeshGraphIOHDF5::WriteDomain(std::map<int, CompositeMap> &domain)
{
    // dont need location map only a id map
    // will filter the composites per parition on read, its easier
    // composites do not need to be written in paralell.
    std::vector<int> d_map;
    std::vector<std::vector<unsigned int>> idxList;

    int cnt = 0;
    for (auto &dIt : domain)
    {
        idxList.push_back(std::vector<unsigned int>());
        for (auto cIt = dIt.second.begin(); cIt != dIt.second.end(); ++cIt)
        {
            idxList[cnt].push_back(cIt->first);
        }

        ++cnt;
        d_map.push_back(dIt.first);
    }

    std::stringstream domString;
    std::vector<std::string> doms;
    for (auto &cIt : idxList)
    {
        doms.push_back(ParseUtils::GenerateSeqString(cIt));
    }

    H5::DataTypeSharedPtr tp  = H5::DataType::String();
    H5::DataSpaceSharedPtr ds = H5::DataSpace::OneD(doms.size());
    H5::DataSetSharedPtr dst  = m_mesh->CreateDataSet("DOMAIN", tp, ds);
    dst->WriteVectorString(doms, ds, tp);

    tp  = H5::DataType::OfObject(d_map[0]);
    ds  = H5::DataSpace::OneD(d_map.size());
    dst = m_maps->CreateDataSet("DOMAIN", tp, ds);
    dst->Write(d_map, ds);
}

void MeshGraphIOHDF5::v_WriteGeometry(
    const std::string &outfilename, bool defaultExp,
    [[maybe_unused]] const LibUtilities::FieldMetaDataMap &metadata)
{
    auto &vertSet                = m_meshGraph->GetGeomMap<PointGeom>();
    auto &segGeoms               = m_meshGraph->GetGeomMap<SegGeom>();
    auto &triGeoms               = m_meshGraph->GetGeomMap<TriGeom>();
    auto &quadGeoms              = m_meshGraph->GetGeomMap<QuadGeom>();
    auto &hexGeoms               = m_meshGraph->GetGeomMap<HexGeom>();
    auto &pyrGeoms               = m_meshGraph->GetGeomMap<PyrGeom>();
    auto &prismGeoms             = m_meshGraph->GetGeomMap<PrismGeom>();
    auto &tetGeoms               = m_meshGraph->GetGeomMap<TetGeom>();
    CurveMap &curvedEdges        = m_meshGraph->GetCurvedEdges();
    CurveMap &curvedFaces        = m_meshGraph->GetCurvedFaces();
    CompositeMap &meshComposites = m_meshGraph->GetComposites();
    auto domain                  = m_meshGraph->GetDomain();

    std::vector<std::string> tmp;
    boost::split(tmp, outfilename, boost::is_any_of("."));
    std::string filenameXml  = tmp[0] + ".xml";
    std::string filenameHdf5 = tmp[0] + ".nekg";

    //////////////////
    // XML part
    //////////////////

    // Check to see if a xml of the same name exists
    // if might have boundary conditions etc, we will just alter the
    // geometry tag if needed
    TiXmlDocument *doc = new TiXmlDocument;
    TiXmlElement *root;
    TiXmlElement *geomTag;

    if (fs::exists(filenameXml.c_str()))
    {
        std::ifstream file(filenameXml.c_str());
        file >> (*doc);
        TiXmlHandle docHandle(doc);
        root = docHandle.FirstChildElement("NEKTAR").Element();
        ASSERTL0(root, "Unable to find NEKTAR tag in file.");
        geomTag    = root->FirstChildElement("GEOMETRY");
        defaultExp = false;
    }
    else
    {
        TiXmlDeclaration *decl = new TiXmlDeclaration("1.0", "utf-8", "");
        doc->LinkEndChild(decl);
        root = new TiXmlElement("NEKTAR");
        doc->LinkEndChild(root);

        geomTag = new TiXmlElement("GEOMETRY");
        root->LinkEndChild(geomTag);
    }

    // Update attributes with dimensions.
    geomTag->SetAttribute("DIM", m_meshGraph->GetMeshDimension());
    geomTag->SetAttribute("SPACE", m_meshGraph->GetSpaceDimension());
    geomTag->SetAttribute("HDF5FILE", filenameHdf5);

    geomTag->Clear();

    if (defaultExp)
    {
        TiXmlElement *expTag = new TiXmlElement("EXPANSIONS");

        for (auto it = meshComposites.begin(); it != meshComposites.end(); it++)
        {
            if (it->second->m_geomVec[0]->GetShapeDim() ==
                m_meshGraph->GetMeshDimension())
            {
                TiXmlElement *exp = new TiXmlElement("E");
                exp->SetAttribute("COMPOSITE",
                                  "C[" + std::to_string(it->first) + "]");
                exp->SetAttribute("NUMMODES", 4);
                exp->SetAttribute("TYPE", "MODIFIED");
                exp->SetAttribute("FIELDS", "u");

                expTag->LinkEndChild(exp);
            }
        }
        root->LinkEndChild(expTag);
    }

    auto movement = m_meshGraph->GetMovement();
    if (movement)
    {
        movement->WriteMovement(root);
    }

    doc->SaveFile(filenameXml);

    //////////////////
    // HDF5 part
    //////////////////

    // This is serial IO so we will just override any existing file.
    m_file        = H5::File::Create(filenameHdf5, H5F_ACC_TRUNC);
    auto hdfRoot  = m_file->CreateGroup("NEKTAR");
    auto hdfRoot2 = hdfRoot->CreateGroup("GEOMETRY");

    // Write format version.
    hdfRoot2->SetAttribute("FORMAT_VERSION", FORMAT_VERSION);

    // Create main groups.
    m_mesh = hdfRoot2->CreateGroup("MESH");
    m_maps = hdfRoot2->CreateGroup("MAPS");

    WriteGeometryMap(vertSet, "VERT");
    WriteGeometryMap(segGeoms, "SEG");
    if (m_meshGraph->GetMeshDimension() > 1)
    {
        WriteGeometryMap(triGeoms, "TRI");
        WriteGeometryMap(quadGeoms, "QUAD");
    }
    if (m_meshGraph->GetMeshDimension() > 2)
    {
        WriteGeometryMap(tetGeoms, "TET");
        WriteGeometryMap(pyrGeoms, "PYR");
        WriteGeometryMap(prismGeoms, "PRISM");
        WriteGeometryMap(hexGeoms, "HEX");
    }

    // Write curves
    int ptOffset = 0, newIdx = 0;
    MeshCurvedPts curvePts;
    WriteCurveMap(curvedEdges, "CURVE_EDGE", curvePts, ptOffset, newIdx);
    WriteCurveMap(curvedFaces, "CURVE_FACE", curvePts, ptOffset, newIdx);
    WriteCurvePoints(curvePts);

    // Write composites and domain.
    WriteComposites(meshComposites);
    WriteDomain(domain);
}

} // namespace Nektar::SpatialDomains
