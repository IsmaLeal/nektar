////////////////////////////////////////////////////////////////////////////////
//
//  File: Geometry.cpp
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
//  Description:  This file contains the base class implementation for the
//                Geometry class.
//
//
////////////////////////////////////////////////////////////////////////////////

#include <SpatialDomains/Geometry.h>
#include <SpatialDomains/Geometry1D.h>
#include <SpatialDomains/Geometry2D.h>

namespace Nektar::SpatialDomains
{

/**
 * @brief Default constructor.
 */
Geometry::Geometry()
    : m_coordim(0), m_state(eNotFilled), m_setupState(false),
      m_shapeType(LibUtilities::eNoShapeType), m_globalID(-1), m_straightEdge(0)
{
}

/**
 * @brief Constructor when supplied a coordinate dimension.
 */
Geometry::Geometry(const int coordim)
    : m_coordim(coordim), m_state(eNotFilled), m_setupState(false),
      m_shapeType(LibUtilities::eNoShapeType), m_globalID(-1), m_straightEdge(0)
{
}

bool SortByGlobalId(const Geometry *&lhs, const Geometry *&rhs)
{
    return lhs->GetGlobalID() < rhs->GetGlobalID();
}

bool GlobalIdEquality(const Geometry *&lhs, const Geometry *&rhs)
{
    return lhs->GetGlobalID() == rhs->GetGlobalID();
}

/**
 * @brief Get the ID of vertex @p i of this object.
 */
int Geometry::v_GetVid(int i) const
{
    return GetVertex(i)->GetGlobalID();
}

/**
 * @brief Get the ID of edge @p i of this object.
 */
int Geometry::GetEid(int i) const
{
    return GetEdge(i)->GetGlobalID();
}

/**
 * @brief Get the ID of face @p i of this object.
 */
int Geometry::GetFid(int i) const
{
    return GetFace(i)->GetGlobalID();
}

/**
 * @copydoc Geometry::GetVertex()
 */
PointGeom *Geometry::v_GetVertex([[maybe_unused]] int i) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for shape type geometries");
    return nullptr;
}

/**
 * @copydoc Geometry::GetEdge()
 */
Geometry1D *Geometry::v_GetEdge([[maybe_unused]] int i) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for shape type geometries");
    return nullptr;
}

/**
 * @copydoc Geometry::GetFace()
 */
Geometry2D *Geometry::v_GetFace([[maybe_unused]] int i) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for shape type geometries");
    return nullptr;
}

/**
 * @copydoc Geometry::GetNumVerts()
 */
int Geometry::v_GetNumVerts() const
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for shape type geometries");
    return 0;
}

/**
 * @copydoc Geometry::GetEorient()
 */
StdRegions::Orientation Geometry::v_GetEorient(
    [[maybe_unused]] const int i) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function is not valid for this geometry.");
    return StdRegions::eForwards;
}

/**
 * @copydoc Geometry::GetForient()
 */
StdRegions::Orientation Geometry::v_GetForient(
    [[maybe_unused]] const int i) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function is not valid for this geometry.");
    return StdRegions::eFwd;
}

/**
 * @copydoc Geometry::GetNumEdges()
 */
int Geometry::v_GetNumEdges() const
{
    return 0;
}

/**
 * @copydoc Geometry::GetNumFaces()
 */
int Geometry::v_GetNumFaces() const
{
    return 0;
}

/**
 * @copydoc Geometry::GetShapeDim()
 */
int Geometry::v_GetShapeDim() const
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for shape type geometries");
    return 0;
}

/**
 * Calculates the GeomType (deformed, regular etc).
 */
GeomType Geometry::v_CalcGeomType()
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for shape type geometries");
    return eNoGeomType;
}

/**
 * @copydoc Geometry::GenGeomFactors()
 */
GeomFactorsUniquePtr Geometry::v_GenGeomFactors(
    [[maybe_unused]] LibUtilities::PointsKeyVector &keyTgt)
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for shape type geometries");
    return GeomFactorsUniquePtr();
}

int Geometry::v_AllLeftCheck(
    [[maybe_unused]] const Array<OneD, const NekDouble> &gloCoord)
{
    return 0;
}

void Geometry::v_CalculateInverseIsoParam()
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for shape type geometries");
}

/**
 * @copydoc Geometry::GetXmap()
 */
StdRegions::StdExpansionSharedPtr Geometry::v_GetXmap() const
{
    return m_xmap;
}

/**
 * @copydoc Geometry::ContainsPoint(
 *     const Array<OneD, const NekDouble> &, Array<OneD, NekDouble> &,
 *     NekDouble, NekDouble&)
 * dist is assigned value for curved elements
 */
bool Geometry::v_ContainsPoint(const Array<OneD, const NekDouble> &gloCoord,
                               Array<OneD, NekDouble> &locCoord, NekDouble tol,
                               NekDouble &dist)
{
    int inside = PreliminaryCheck(gloCoord);
    if (inside == -1)
    {
        dist = std::numeric_limits<double>::max();
        return false;
    }
    dist = GetLocCoords(gloCoord, locCoord);
    if (inside == 1)
    {
        dist = 0.;
        return true;
    }
    else
    {
        Array<OneD, NekDouble> eta(GetShapeDim(), 0.);
        m_xmap->LocCoordToLocCollapsed(locCoord, eta);
        if (ClampLocCoords(eta, tol))
        {
            if (CalcGeomType() == eRegular)
            {
                dist = std::numeric_limits<double>::max();
            }
            return false;
        }
        return 3 != m_coordim ||
               (LibUtilities::eTriangle != m_shapeType &&
                LibUtilities::eQuadrilateral != m_shapeType) ||
               dist <= tol;
    }
}

NekDouble Geometry::v_FindDistance(
    [[maybe_unused]] const Array<OneD, const NekDouble> &xs,
    [[maybe_unused]] Array<OneD, NekDouble> &xi)
{
    NEKERROR(ErrorUtil::efatal,
             "This function has not been defined for this geometry");
    return false;
}

/**
 * @copydoc Geometry::GetVertexEdgeMap()
 */
int Geometry::v_GetVertexEdgeMap([[maybe_unused]] const int i,
                                 [[maybe_unused]] const int j) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function has not been defined for this geometry");
    return 0;
}

/**
 * @copydoc Geometry::GetVertexFaceMap()
 */
int Geometry::v_GetVertexFaceMap([[maybe_unused]] const int i,
                                 [[maybe_unused]] const int j) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function has not been defined for this geometry");
    return 0;
}

/**
 * @copydoc Geometry::GetEdgeFaceMap()
 */
int Geometry::v_GetEdgeFaceMap([[maybe_unused]] const int i,
                               [[maybe_unused]] const int j) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function has not been defined for this geometry");
    return 0;
}

/**
 * @copydoc Geometry::GetEdgeNormalToFaceVert()
 */
int Geometry::v_GetEdgeNormalToFaceVert([[maybe_unused]] const int i,
                                        [[maybe_unused]] const int j) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function has not been defined for this geometry");
    return 0;
}

/**
 * @copydoc Geometry::GetDir()
 */
int Geometry::v_GetDir([[maybe_unused]] const int i,
                       [[maybe_unused]] const int j) const
{
    NEKERROR(ErrorUtil::efatal,
             "This function has not been defined for this geometry");
    return 0;
}

/**
 * @copydoc Geometry::GetCoord()
 */
NekDouble Geometry::v_GetCoord(
    [[maybe_unused]] const int i,
    [[maybe_unused]] const Array<OneD, const NekDouble> &Lcoord)
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for expansion type geometries");
    return 0.0;
}

/**
 * @copydoc Geometry::GetLocCoords()
 */
NekDouble Geometry::v_GetLocCoords(
    [[maybe_unused]] const Array<OneD, const NekDouble> &coords,
    [[maybe_unused]] Array<OneD, NekDouble> &Lcoords)
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for expansion type geometries");
    return 0.0;
}

/**
 * @copydoc Geometry::FillGeom()
 */
void Geometry::v_FillGeom()
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for expansion type geometries");
}

/**
 * @copydoc Geometry::Reset()
 */
void Geometry::v_Reset([[maybe_unused]] CurveMap &curvedEdges,
                       [[maybe_unused]] CurveMap &curvedFaces)
{
    // Reset state
    m_state = eNotFilled;
}

void Geometry::v_Setup()
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for expansion type geometries");
}

/**
 * @brief Generates the bounding box for the element.
 *
 * For regular elements, the vertices are sufficient to define the extent of
 * the bounding box. For non-regular elements, the extremes of the quadrature
 * point coordinates are used. A 10% margin is added around this computed
 * region to account for convex hull elements where the true extent of the
 * element may extend slightly beyond the quadrature points.
 */
std::array<NekDouble, 6> Geometry::GetBoundingBox()
{
    if (m_boundingBox.size() == 6)
    {
        return {{m_boundingBox[0], m_boundingBox[1], m_boundingBox[2],
                 m_boundingBox[3], m_boundingBox[4], m_boundingBox[5]}};
    }
    // NekDouble minx, miny, minz, maxx, maxy, maxz;
    Array<OneD, NekDouble> min(3), max(3);

    // Always get vertexes min/max
    PointGeom *p = GetVertex(0);
    Array<OneD, NekDouble> x(3, 0.0);
    p->GetCoords(x[0], x[1], x[2]);
    for (int j = 0; j < 3; ++j)
    {
        min[j] = x[j];
        max[j] = x[j];
    }
    for (int i = 1; i < GetNumVerts(); ++i)
    {
        p = GetVertex(i);
        p->GetCoords(x[0], x[1], x[2]);
        for (int j = 0; j < 3; ++j)
        {
            min[j] = (x[j] < min[j] ? x[j] : min[j]);
            max[j] = (x[j] > max[j] ? x[j] : max[j]);
        }
    }
    // If element is deformed loop over quadrature points
    NekDouble marginFactor = NekConstants::kGeomFactorsTol;
    if (CalcGeomType() != eRegular)
    {
        marginFactor = 0.1;
        const int nq = GetXmap()->GetTotPoints();
        Array<OneD, Array<OneD, NekDouble>> xvec(3);
        for (int j = 0; j < 3; ++j)
        {
            xvec[j] = Array<OneD, NekDouble>(nq, 0.0);
        }
        for (int j = 0; j < GetCoordim(); ++j)
        {
            GetXmap()->BwdTrans(m_coeffs[j], xvec[j]);
        }
        for (int j = 0; j < 3; ++j)
        {
            for (int i = 0; i < nq; ++i)
            {
                min[j] = (xvec[j][i] < min[j] ? xvec[j][i] : min[j]);
                max[j] = (xvec[j][i] > max[j] ? xvec[j][i] : max[j]);
            }
        }
    }
    // Add margin to bounding box, in order to
    // return the nearest element
    for (int j = 0; j < 3; ++j)
    {
        NekDouble margin =
            marginFactor * (max[j] - min[j]) + NekConstants::kFindDistanceMin;
        min[j] -= margin;
        max[j] += margin;
    }

    // save bounding box
    m_boundingBox = Array<OneD, NekDouble>(6);
    for (int j = 0; j < 3; ++j)
    {
        m_boundingBox[j]     = min[j];
        m_boundingBox[j + 3] = max[j];
    }
    // Return bounding box
    return {{min[0], min[1], min[2], max[0], max[1], max[2]}};
}

void Geometry::ClearBoundingBox()
{
    m_boundingBox = {};
}

/**
 * @brief A fast and robust check if a given global coord is outside of a
 * deformed element. For regular elements, this check is unnecessary.
 *
 * @param coords   Input Cartesian global coordinates
 *
 * @return 1 is inside of the element.
 *         0 maybe inside
 *        -1 outside of the element
 */
int Geometry::PreliminaryCheck(const Array<OneD, const NekDouble> &gloCoord)
{
    // bounding box check
    if (!MinMaxCheck(gloCoord))
    {
        return -1;
    }

    // regular element check
    if (CalcGeomType() == eRegular)
    {
        return 0;
    }

    // All left check for straight edges/plane surfaces
    return v_AllLeftCheck(gloCoord);
}

/**
 * @brief Check if given global coord is within the BoundingBox of the element.
 *
 * @param coords   Input Cartesian global coordinates
 *
 * @return True if within distance or False otherwise.
 */
bool Geometry::MinMaxCheck(const Array<OneD, const NekDouble> &gloCoord)
{
    // Validation checks
    ASSERTL1(gloCoord.size() >= m_coordim,
             "Expects number of global coordinates supplied to be greater than "
             "or equal to the mesh dimension.");

    std::array<NekDouble, 6> minMax = GetBoundingBox();
    for (int i = 0; i < m_coordim; ++i)
    {
        if ((gloCoord[i] < minMax[i]) || (gloCoord[i] > minMax[i + 3]))
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief Clamp local coords to be within standard regions [-1, 1]^dim.
 *
 * @param Lcoords  Corresponding local coordinates
 */
bool Geometry::ClampLocCoords(Array<OneD, NekDouble> &locCoord, NekDouble tol)
{
    // Validation checks
    ASSERTL1(locCoord.size() >= GetShapeDim(),
             "Expects local coordinates to be same or "
             "larger than shape dimension.");

    // If out of range clamp locCoord to be within [-1,1]^dim
    // since any larger value will be very oscillatory if
    // called by 'returnNearestElmt' option in
    // ExpList::GetExpIndex
    bool clamp = false;
    for (int i = 0; i < GetShapeDim(); ++i)
    {
        if (!std::isfinite(locCoord[i]))
        {
            locCoord[i] = 0.;
            clamp       = true;
        }
        else if (locCoord[i] < -(1. + tol))
        {
            locCoord[i] = -(1. + tol);
            clamp       = true;
        }
        else if (locCoord[i] > (1. + tol))
        {
            locCoord[i] = 1. + tol;
            clamp       = true;
        }
    }
    return clamp;
}

} // namespace Nektar::SpatialDomains
