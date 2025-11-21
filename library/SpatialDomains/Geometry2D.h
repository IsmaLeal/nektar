////////////////////////////////////////////////////////////////////////////////
//
//  File: Geometry2D.h
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
//  Description: 2D geometry information
//
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SPATIALDOMAINS_GEOMETRY2D_H
#define NEKTAR_SPATIALDOMAINS_GEOMETRY2D_H

#include <StdRegions/StdExpansion2D.h>
#include <StdRegions/StdRegions.hpp>

#include <SpatialDomains/Geometry.h>
#include <SpatialDomains/SpatialDomainsDeclspec.h>

namespace Nektar::SpatialDomains
{

/// 2D geometry information
class Geometry2D : public Geometry
{
public:
    SPATIAL_DOMAINS_EXPORT Geometry2D();
    SPATIAL_DOMAINS_EXPORT Geometry2D(const int coordim, Curve *curve);
    SPATIAL_DOMAINS_EXPORT ~Geometry2D() override = default;

    SPATIAL_DOMAINS_EXPORT static const int kDim = 2;
    SPATIAL_DOMAINS_EXPORT Curve *GetCurve()
    {
        return m_curve;
    }
    SPATIAL_DOMAINS_EXPORT void SetCurve(Curve *curvePtr)
    {
        m_curve = curvePtr;
    }

protected:
    Curve *m_curve;
    Array<OneD, int> m_manifold;
    Array<OneD, Array<OneD, NekDouble>> m_edgeNormal;

    SPATIAL_DOMAINS_EXPORT NekDouble
    v_GetLocCoords(const Array<OneD, const NekDouble> &coords,
                   Array<OneD, NekDouble> &Lcoords) override;

    void NewtonIterationForLocCoord(const Array<OneD, const NekDouble> &coords,
                                    const Array<OneD, const NekDouble> &ptsx,
                                    const Array<OneD, const NekDouble> &ptsy,
                                    Array<OneD, NekDouble> &Lcoords,
                                    NekDouble &dist);
    void SolveStraightEdgeQuad(const Array<OneD, const NekDouble> &coords,
                               Array<OneD, NekDouble> &Lcoords);
    void v_CalculateInverseIsoParam() override;

    //---------------------------------------
    // Helper functions
    //---------------------------------------
    int v_GetShapeDim() const override;
    NekDouble v_FindDistance(const Array<OneD, const NekDouble> &xs,
                             Array<OneD, NekDouble> &xi) override;
};

} // namespace Nektar::SpatialDomains

#endif // NEKTAR_SPATIALDOMAINS_GEOMETRY2D_H
