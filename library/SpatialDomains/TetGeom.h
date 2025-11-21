////////////////////////////////////////////////////////////////////////////////
//
//  File: TetGeom.h
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
//  Description: Tetrahedral geometry information.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SPATIALDOMAINS_TETGEOM
#define NEKTAR_SPATIALDOMAINS_TETGEOM

#include <SpatialDomains/Geometry3D.h>
#include <SpatialDomains/SpatialDomainsDeclspec.h>
#include <SpatialDomains/TriGeom.h>

namespace Nektar::SpatialDomains
{

class TetGeom : public Geometry3D
{
public:
    SPATIAL_DOMAINS_EXPORT TetGeom();
    SPATIAL_DOMAINS_EXPORT TetGeom(int id, TriGeom *faces[]);

    SPATIAL_DOMAINS_EXPORT static const int kNverts  = 4;
    SPATIAL_DOMAINS_EXPORT static const int kNedges  = 6;
    SPATIAL_DOMAINS_EXPORT static const int kNqfaces = 0;
    SPATIAL_DOMAINS_EXPORT static const int kNtfaces = 4;
    SPATIAL_DOMAINS_EXPORT static const int kNfaces  = kNqfaces + kNtfaces;
    SPATIAL_DOMAINS_EXPORT static const int kNfacets = kNfaces;
    SPATIAL_DOMAINS_EXPORT static const std::string XMLElementType;

    SPATIAL_DOMAINS_EXPORT TetGeom(int id,
                                   std::array<TriGeom *, kNfaces> faces);
    SPATIAL_DOMAINS_EXPORT ~TetGeom() override = default;

protected:
    int v_GetVertexEdgeMap(const int i, const int j) const override;
    int v_GetVertexFaceMap(const int i, const int j) const override;
    int v_GetEdgeFaceMap(const int i, const int j) const override;
    int v_GetEdgeNormalToFaceVert(const int i, const int j) const override;
    int v_GetDir(const int faceidx, const int facedir) const override;
    void v_Reset(CurveMap &curvedEdges, CurveMap &curvedFaces) override;
    void v_Setup() override;
    GeomType v_CalcGeomType() override;
    GeomFactorsUniquePtr v_GenGeomFactors(
        LibUtilities::PointsKeyVector &keyTgt) override;
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
    std::array<TriGeom *, kNfaces> m_faces;
    std::array<StdRegions::Orientation, kNedges> m_eorient;
    std::array<StdRegions::Orientation, kNfaces> m_forient;

private:
    void SetUpLocalEdges();
    void SetUpLocalVertices();
    void SetUpEdgeOrientation();
    void SetUpFaceOrientation();
    void SetUpXmap();

    static const unsigned int VertexEdgeConnectivity[4][3];
    static const unsigned int VertexFaceConnectivity[4][3];
    static const unsigned int EdgeFaceConnectivity[6][2];
    static const unsigned int EdgeNormalToFaceVert[4][3];
};

} // namespace Nektar::SpatialDomains

#endif // NEKTAR_SPATIALDOMAINS_TETGEOM
