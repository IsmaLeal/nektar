////////////////////////////////////////////////////////////////////////////////
//
//  File: PyrGeom.h
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
//  Description: Pyramidic geometry information.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SPATIALDOMAINS_PYRGEOM_H
#define NEKTAR_SPATIALDOMAINS_PYRGEOM_H

#include <SpatialDomains/Geometry3D.h>
#include <SpatialDomains/SegGeom.h>
#include <SpatialDomains/SpatialDomainsDeclspec.h>

namespace Nektar::SpatialDomains
{

class PyrGeom : public Geometry3D
{
public:
    SPATIAL_DOMAINS_EXPORT PyrGeom();
    SPATIAL_DOMAINS_EXPORT PyrGeom(int id, Geometry2D *faces[]);

    SPATIAL_DOMAINS_EXPORT static const int kNverts  = 5;
    SPATIAL_DOMAINS_EXPORT static const int kNedges  = 8;
    SPATIAL_DOMAINS_EXPORT static const int kNqfaces = 1;
    SPATIAL_DOMAINS_EXPORT static const int kNtfaces = 4;
    SPATIAL_DOMAINS_EXPORT static const int kNfaces  = kNqfaces + kNtfaces;
    SPATIAL_DOMAINS_EXPORT static const int kNfacets = kNfaces;
    SPATIAL_DOMAINS_EXPORT static const std::string XMLElementType;

    SPATIAL_DOMAINS_EXPORT PyrGeom(int id,
                                   std::array<Geometry2D *, kNfaces> faces);
    SPATIAL_DOMAINS_EXPORT ~PyrGeom() override = default;

protected:
    GeomType v_CalcGeomType() override;
    GeomFactorsUniquePtr v_GenGeomFactors(
        LibUtilities::PointsKeyVector &keyTgt) override;
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
    std::array<Geometry2D *, kNfaces> m_faces;
    std::array<StdRegions::Orientation, kNedges> m_eorient;
    std::array<StdRegions::Orientation, kNfaces> m_forient;

private:
    void SetUpLocalEdges();
    void SetUpLocalVertices();
    void SetUpEdgeOrientation();
    void SetUpFaceOrientation();
    void SetUpXmap();

    static const unsigned int EdgeNormalToFaceVert[5][4];
};

} // namespace Nektar::SpatialDomains

#endif // NEKTAR_SPATIALDOMAINS_PYRGEOM_H
