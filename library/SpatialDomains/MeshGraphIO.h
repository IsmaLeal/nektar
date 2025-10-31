////////////////////////////////////////////////////////////////////////////////
//
//  File: MeshGraphIO.h
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

#ifndef NEKTAR_SPATIALDOMAINS_MGIO_H
#define NEKTAR_SPATIALDOMAINS_MGIO_H

#include <LibUtilities/BasicConst/NektarUnivTypeDefs.hpp>
#include <SpatialDomains/MeshEntities.hpp>
#include <SpatialDomains/MeshGraph.h>

namespace Nektar::SpatialDomains
{

class MeshGraphIO;
typedef std::shared_ptr<MeshGraphIO> MeshGraphIOSharedPtr;

class MeshGraphIO
{
public:
    SPATIAL_DOMAINS_EXPORT virtual ~MeshGraphIO() = default;

    SPATIAL_DOMAINS_EXPORT static MeshGraphSharedPtr Read(
        const LibUtilities::SessionReaderSharedPtr pSession,
        LibUtilities::DomainRangeShPtr rng = LibUtilities::NullDomainRangeShPtr,
        bool fillGraph                     = true,
        SpatialDomains::MeshGraphSharedPtr partitionedGraph = nullptr);

    SPATIAL_DOMAINS_EXPORT void SetMeshGraph(MeshGraphSharedPtr &meshGraph)
    {
        m_meshGraph = meshGraph;
    }

    SPATIAL_DOMAINS_EXPORT void WriteGeometry(
        const std::string &outfilename, bool defaultExp = false,
        const LibUtilities::FieldMetaDataMap &metadata =
            LibUtilities::NullFieldMetaDataMap)
    {
        v_WriteGeometry(outfilename, defaultExp, metadata);
    }

    /*an inital read which loads a very light weight data structure*/
    SPATIAL_DOMAINS_EXPORT void ReadGeometry(bool fillGraph)
    {
        v_ReadGeometry(fillGraph);
    }

    SPATIAL_DOMAINS_EXPORT void PartitionMesh(
        LibUtilities::SessionReaderSharedPtr session)
    {
        v_PartitionMesh(session);
    }

    SPATIAL_DOMAINS_EXPORT std::map<int, MeshEntity> CreateMeshEntities();
    SPATIAL_DOMAINS_EXPORT CompositeDescriptor CreateCompositeDescriptor();

protected:
    LibUtilities::SessionReaderSharedPtr m_session;
    MeshGraphSharedPtr m_meshGraph;
    int m_partition;
    bool m_meshPartitioned = false;
    CompositeOrdering m_compOrder;
    BndRegionOrdering m_bndRegOrder;

    std::string GetCompositeString(CompositeSharedPtr comp);

    SPATIAL_DOMAINS_EXPORT MeshGraphIO() = default;

    SPATIAL_DOMAINS_EXPORT virtual void v_WriteGeometry(
        const std::string &outfilename, bool defaultExp = false,
        const LibUtilities::FieldMetaDataMap &metadata =
            LibUtilities::NullFieldMetaDataMap) = 0;

    SPATIAL_DOMAINS_EXPORT virtual void v_ReadGeometry(bool fillGraph) = 0;

    SPATIAL_DOMAINS_EXPORT virtual void v_PartitionMesh(
        LibUtilities::SessionReaderSharedPtr session) = 0;
};

typedef LibUtilities::NekFactory<std::string, MeshGraphIO> MeshGraphIOFactory;
SPATIAL_DOMAINS_EXPORT MeshGraphIOFactory &GetMeshGraphIOFactory();

} // namespace Nektar::SpatialDomains

#endif // NEKTAR_MESHGRAPHIO_H
