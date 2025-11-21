////////////////////////////////////////////////////////////////////////////////
//
//  File: MeshGraphIOXmlCompressed.h
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

#ifndef NEKTAR_SPATIALDOMAINS_MGIOXMLCOMP_H
#define NEKTAR_SPATIALDOMAINS_MGIOXMLCOMP_H

#include "MeshGraphIOXml.h"

namespace Nektar::SpatialDomains
{

class MeshGraphIOXmlCompressed : public MeshGraphIOXml
{
public:
    friend class MemoryManager<MeshGraphIOXmlCompressed>;

    static MeshGraphIOSharedPtr create()
    {
        return MemoryManager<MeshGraphIOXmlCompressed>::AllocateSharedPtr();
    }

    static std::string className;

protected:
    MeshGraphIOXmlCompressed()           = default;
    ~MeshGraphIOXmlCompressed() override = default;

    void v_ReadVertices() override;
    void v_ReadCurves() override;

    void v_ReadEdges() override;
    void v_ReadFaces() override;

    void v_ReadElements1D() override;
    void v_ReadElements2D() override;
    void v_ReadElements3D() override;

    // *keysToWrite specifies a subset of GeomMap to write during partitioning.
    void v_WriteVertices(
        TiXmlElement *geomTag,
        std::vector<int> keysToWrite = std::vector<int>()) override;
    void v_WriteEdges(TiXmlElement *geomTag, std::vector<int> keysToWrite =
                                                 std::vector<int>()) override;
    void v_WriteTris(TiXmlElement *faceTag, std::vector<int> keysToWrite =
                                                std::vector<int>()) override;
    void v_WriteQuads(TiXmlElement *faceTag, std::vector<int> keysToWrite =
                                                 std::vector<int>()) override;
    void v_WriteHexs(TiXmlElement *elmtTag, std::vector<int> keysToWrite =
                                                std::vector<int>()) override;
    void v_WritePrisms(TiXmlElement *elmtTag, std::vector<int> keysToWrite =
                                                  std::vector<int>()) override;
    void v_WritePyrs(TiXmlElement *elmtTag, std::vector<int> keysToWrite =
                                                std::vector<int>()) override;
    void v_WriteTets(TiXmlElement *elmtTag, std::vector<int> keysToWrite =
                                                std::vector<int>()) override;
    void v_WriteCurves(TiXmlElement *geomTag, CurveMap &edges, CurveMap &faces,
                       std::vector<int> *keysToWriteEdges = nullptr,
                       std::vector<int> *keysToWriteFaces = nullptr) override;
};

} // namespace Nektar::SpatialDomains

#endif
