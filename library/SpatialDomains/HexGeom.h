////////////////////////////////////////////////////////////////////////////////
//
//  File: HexGeom.h
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
//  Description: Hexahedral geometry information.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SPATIALDOMAINS_HEXGEOM_H
#define NEKTAR_SPATIALDOMAINS_HEXGEOM_H

#include <SpatialDomains/Geometry3D.h>
#include <SpatialDomains/QuadGeom.h>
#include <SpatialDomains/SpatialDomainsDeclspec.h>

namespace Nektar::SpatialDomains
{

class HexGeom : public Geometry3D
{
public:
    SPATIAL_DOMAINS_EXPORT HexGeom();
    SPATIAL_DOMAINS_EXPORT HexGeom(int id, QuadGeom *faces[]);

    SPATIAL_DOMAINS_EXPORT static const int kNverts  = 8;
    SPATIAL_DOMAINS_EXPORT static const int kNedges  = 12;
    SPATIAL_DOMAINS_EXPORT static const int kNqfaces = 6;
    SPATIAL_DOMAINS_EXPORT static const int kNtfaces = 0;
    SPATIAL_DOMAINS_EXPORT static const int kNfaces  = kNqfaces + kNtfaces;
    SPATIAL_DOMAINS_EXPORT static const int kNfacets = kNfaces;
    SPATIAL_DOMAINS_EXPORT static const std::string XMLElementType;

    SPATIAL_DOMAINS_EXPORT HexGeom(int id,
                                   std::array<QuadGeom *, kNfaces> faces);
    SPATIAL_DOMAINS_EXPORT ~HexGeom() override = default;

protected:
    GeomType v_CalcGeomType() override;
    GeomFactorsUniquePtr v_GenGeomFactors(
        LibUtilities::PointsKeyVector &keyTgt) override;
    int v_GetVertexEdgeMap(const int i, const int j) const override;
    int v_GetVertexFaceMap(const int i, const int j) const override;
    int v_GetEdgeFaceMap(const int i, const int j) const override;
    int v_GetEdgeNormalToFaceVert(const int i, const int j) const override;
    int v_GetDir(const int faceidx, const int facedir) const override;
    void v_Reset(CurveMap &curvedEdges, CurveMap &curvedFaces) override;
    void v_Setup() override;
    void v_FillGeom() override;

    inline int v_GetNumVerts() const final
    {
        return kNverts;
    }

    inline int v_GetNumEdges() const final
    {
        return kNedges;
    }

    inline int v_GetNumFaces() const final
    {
        return kNfaces;
    }

    inline PointGeom *v_GetVertex(const int i) const final
    {
        return m_verts[i];
    }

    inline Geometry1D *v_GetEdge(const int i) const final
    {
        return static_cast<Geometry1D *>(m_edges[i]);
    }

    inline Geometry2D *v_GetFace(const int i) const final
    {
        return static_cast<Geometry2D *>(m_faces[i]);
    }

    inline StdRegions::Orientation v_GetEorient(const int i) const final
    {
        return m_eorient[i];
    }

    inline StdRegions::Orientation v_GetForient(const int i) const final
    {
        return m_forient[i];
    }

    std::array<PointGeom *, kNverts> m_verts;
    std::array<SegGeom *, kNedges> m_edges;
    std::array<QuadGeom *, kNfaces> m_faces;
    std::array<StdRegions::Orientation, kNedges> m_eorient;
    std::array<StdRegions::Orientation, kNfaces> m_forient;

private:
    void SetUpLocalEdges();
    void SetUpLocalVertices();
    void SetUpEdgeOrientation();
    void SetUpFaceOrientation();
    void SetUpXmap();

    static const unsigned int VertexEdgeConnectivity[8][3];
    static const unsigned int VertexFaceConnectivity[8][3];
    static const unsigned int EdgeFaceConnectivity[12][2];
    static const unsigned int EdgeNormalToFaceVert[6][4];
};

} // namespace Nektar::SpatialDomains

#endif
