////////////////////////////////////////////////////////////////////////////////
//
//  File: MeshGraphIOHDF5.h
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

#ifndef NEKTAR_SPATIALDOMAINS_MGIOHDF5_H
#define NEKTAR_SPATIALDOMAINS_MGIOHDF5_H

#include <LibUtilities/BasicUtils/H5.h>
#include <SpatialDomains/MeshGraphIO.h>
#include <SpatialDomains/MeshPartition.h>

namespace Nektar::SpatialDomains
{

class MeshGraphIOHDF5 : public MeshGraphIO
{
public:
    friend class MemoryManager<MeshGraphIOHDF5>;

    static MeshGraphIOSharedPtr create()
    {
        return MemoryManager<MeshGraphIOHDF5>::AllocateSharedPtr();
    }

    static std::string className, cmdSwitch;

protected:
    TiXmlElement *m_xmlGeom{};

    MeshGraphIOHDF5()           = default;
    ~MeshGraphIOHDF5() override = default;

    // some of these functions are going to be virtual because they will be
    // inherited by the XmlCompressed version

    SPATIAL_DOMAINS_EXPORT void v_WriteGeometry(
        const std::string &outfilename, bool defaultExp = false,
        const LibUtilities::FieldMetaDataMap &metadata =
            LibUtilities::NullFieldMetaDataMap) final;
    SPATIAL_DOMAINS_EXPORT void v_ReadGeometry(bool fillGraph) final;
    SPATIAL_DOMAINS_EXPORT void v_PartitionMesh(
        LibUtilities::SessionReaderSharedPtr session) final;

private:
    void ReadCurveMap(CurveMap &curveMap, std::string dsName,
                      const std::unordered_set<int> &readIds);
    void ReadDomain();
    void ReadComposites();
    void SetupCompositeRange(LibUtilities::DomainRangeShPtr &rng);

    template <class T>
    void WriteGeometryMap(GeomMapView<T> &geomMap, std::string datasetName);

    template <class T, typename DataType>
    void ReadGeometryData(GeomMapView<T> &geomMap, std::string dataSet,
                          const std::unordered_set<int> &readIds,
                          std::vector<int> &ids,
                          std::vector<DataType> &geomData);
    template <class T, typename DataType>
    void FillGeomMap(GeomMapView<T> &geomMap, const CurveMap &curveMap,
                     std::vector<int> &ids, std::vector<DataType> &geomData);
    template <class T, typename DataType>
    void ConstructGeomObject(GeomMapView<T> &geomMap, int id, DataType *data,
                             CurveSharedPtr curve);

    CompositeDescriptor CreateCompositeDescriptor(
        std::unordered_map<int, int> &id2row);

    void WriteCurveMap(CurveMap &curves, std::string dsName,
                       MeshCurvedPts &curvedPts, int &ptOffset, int &newIdx);
    void WriteCurvePoints(MeshCurvedPts &curvedPts);

    void WriteComposites(CompositeMap &comps);
    void WriteDomain(std::map<int, CompositeMap> &domain);

    std::string m_hdf5Name;
    LibUtilities::H5::FileSharedPtr m_file;
    LibUtilities::H5::PListSharedPtr m_readPL;
    LibUtilities::H5::GroupSharedPtr m_mesh;
    LibUtilities::H5::GroupSharedPtr m_maps;
    unsigned int m_inFormatVersion;

    static const unsigned int FORMAT_VERSION;
};

} // namespace Nektar::SpatialDomains

#endif
