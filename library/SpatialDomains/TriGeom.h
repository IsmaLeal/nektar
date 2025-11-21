////////////////////////////////////////////////////////////////////////////////
//
//  File: TriGeom.h
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

#ifndef NEKTAR_SPATIALDOMAINS_TRIGEOM_H
#define NEKTAR_SPATIALDOMAINS_TRIGEOM_H

#include <SpatialDomains/Geometry2D.h>
#include <SpatialDomains/PointGeom.h>
#include <SpatialDomains/SegGeom.h>
#include <SpatialDomains/SpatialDomainsDeclspec.h>
#include <StdRegions/StdRegions.hpp>

#include <SpatialDomains/GeomFactors.h>
#include <StdRegions/StdTriExp.h>

namespace Nektar::SpatialDomains
{

class TriGeom;
class SegGeom;
struct Curve;

class TriGeom : public Geometry2D
{
public:
    SPATIAL_DOMAINS_EXPORT static const int kNedges  = 3;
    SPATIAL_DOMAINS_EXPORT static const int kNverts  = 3;
    SPATIAL_DOMAINS_EXPORT static const int kNfacets = kNedges;

    SPATIAL_DOMAINS_EXPORT TriGeom();
    SPATIAL_DOMAINS_EXPORT TriGeom(const TriGeom &in);
    SPATIAL_DOMAINS_EXPORT TriGeom(const int id,
                                   std::array<SegGeom *, kNedges> edges,
                                   Curve *curve = nullptr);
    SPATIAL_DOMAINS_EXPORT ~TriGeom() override = default;

    /// Get the orientation of face1.
    SPATIAL_DOMAINS_EXPORT static StdRegions::Orientation GetFaceOrientation(
        const TriGeom &face1, const TriGeom &face2, bool doRot, int dir,
        NekDouble angle, NekDouble tol);

    SPATIAL_DOMAINS_EXPORT static StdRegions::Orientation GetFaceOrientation(
        std::array<PointGeom *, kNedges> face1,
        std::array<PointGeom *, kNedges> face2, bool doRot, int dir,
        NekDouble angle, NekDouble tol);

protected:
    SPATIAL_DOMAINS_EXPORT NekDouble v_GetCoord(
        const int i, const Array<OneD, const NekDouble> &Lcoord) override;
    GeomType v_CalcGeomType() override;
    GeomFactorsUniquePtr v_GenGeomFactors(
        LibUtilities::PointsKeyVector &keyTgt) override;
    SPATIAL_DOMAINS_EXPORT void v_FillGeom() override;
    SPATIAL_DOMAINS_EXPORT int v_GetDir(const int faceidx,
                                        const int facedir) const override;
    SPATIAL_DOMAINS_EXPORT void v_Reset(CurveMap &curvedEdges,
                                        CurveMap &curvedFaces) override;

    SPATIAL_DOMAINS_EXPORT void v_Setup() override;

    int v_AllLeftCheck(const Array<OneD, const NekDouble> &gloCoord) override;

    inline int v_GetNumVerts() const final
    {
        return kNverts;
    }

    inline int v_GetNumEdges() const final
    {
        return kNedges;
    }

    inline PointGeom *v_GetVertex(const int i) const final
    {
        return m_verts[i];
    }

    inline Geometry1D *v_GetEdge(const int i) const final
    {
        return static_cast<Geometry1D *>(m_edges[i]);
    }

    inline StdRegions::Orientation v_GetEorient(const int i) const final
    {
        return m_eorient[i];
    }

    std::array<PointGeom *, kNverts> m_verts;
    std::array<SegGeom *, kNedges> m_edges;
    std::array<StdRegions::Orientation, kNedges> m_eorient;

private:
    void SetUpXmap();
};

} // namespace Nektar::SpatialDomains

#endif
