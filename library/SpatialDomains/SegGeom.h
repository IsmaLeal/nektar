////////////////////////////////////////////////////////////////////////////////
//
//  File: SegGeom.h
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
//  Description: Segment geometry information
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SPATIALDOMAINS_SEGGEOM_H
#define NEKTAR_SPATIALDOMAINS_SEGGEOM_H

#include <LibUtilities/Foundations/Basis.h>
#include <LibUtilities/Memory/ObjectPool.hpp>
#include <SpatialDomains/Curve.hpp>
#include <SpatialDomains/Geometry1D.h>
#include <SpatialDomains/PointGeom.h>
#include <SpatialDomains/SpatialDomainsDeclspec.h>
#include <StdRegions/StdRegions.hpp>

namespace Nektar::SpatialDomains
{

class SegGeom;
typedef unique_ptr_objpool<SegGeom> SegGeomUniquePtr;
typedef unique_ptr_objpool<PointGeom> PointGeomUniquePtr;
class EntityHolder1D
{
public:
    std::vector<PointGeomUniquePtr> m_pointVec;
    std::vector<SegGeomUniquePtr> m_segVec;
};

class SegGeom : public Geometry1D
{
public:
    SPATIAL_DOMAINS_EXPORT static const int kNverts  = 2;
    SPATIAL_DOMAINS_EXPORT static const int kNfacets = kNverts;

    SPATIAL_DOMAINS_EXPORT SegGeom();
    SPATIAL_DOMAINS_EXPORT SegGeom(int id, int coordim,
                                   std::array<PointGeom *, kNverts> vertex,
                                   Curve *curve = nullptr);

    SPATIAL_DOMAINS_EXPORT SegGeom(const SegGeom &in);

    SPATIAL_DOMAINS_EXPORT SegGeomUniquePtr
    GenerateOneSpaceDimGeom(EntityHolder1D &holder);

    SPATIAL_DOMAINS_EXPORT ~SegGeom() override = default;

    SPATIAL_DOMAINS_EXPORT static StdRegions::Orientation GetEdgeOrientation(
        const SegGeom &edge1, const SegGeom &edge2);

    inline SPATIAL_DOMAINS_EXPORT Curve *GetCurve()
    {
        return m_curve;
    }
    inline SPATIAL_DOMAINS_EXPORT void SetCurve(Curve *curvePtr)
    {
        m_curve = curvePtr;
    }

protected:
    std::array<SpatialDomains::PointGeom *, kNverts> m_verts;
    StdRegions::Orientation m_porient[kNverts];

    PointGeom *v_GetVertex(const int i) const override;
    virtual LibUtilities::ShapeType v_GetShapeType() const;
    GeomType v_CalcGeomType() override;
    GeomFactorsUniquePtr v_GenGeomFactors(
        LibUtilities::PointsKeyVector &keyTgt) override;
    void v_FillGeom() override;
    void v_Reset(CurveMap &curvedEdges, CurveMap &curvedFaces) override;
    void v_Setup() override;
    NekDouble v_GetCoord(const int i,
                         const Array<OneD, const NekDouble> &Lcoord) override;
    int v_GetNumVerts() const override;
    NekDouble v_FindDistance(const Array<OneD, const NekDouble> &xs,
                             Array<OneD, NekDouble> &xi) override;

private:
    /// Boolean indicating whether object owns the data
    Curve *m_curve = nullptr;

    void SetUpXmap();
};

} // namespace Nektar::SpatialDomains

#endif // NEKTAR_SPATIALDOMAINS_SEGGEOM_H
