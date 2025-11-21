////////////////////////////////////////////////////////////////////////////////
//
//  File: Geometry.h
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
//  Description:  This file contains the base class specification for the
//                Geometry class.
//
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SPATIALDOMAINS_GEOMETRY_H
#define NEKTAR_SPATIALDOMAINS_GEOMETRY_H

#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/Memory/ObjectPool.hpp>
#include <SpatialDomains/GeomFactors.h>
#include <SpatialDomains/SpatialDomainsDeclspec.h>

#include <array>
#include <unordered_map>

namespace Nektar
{
// Forward declarations for allocation pools that are defined within
// MeshGraph.cpp compilation unit.
template <>
PoolAllocator<SpatialDomains::GeomFactors>
    ObjPoolManager<SpatialDomains::GeomFactors>::m_alloc;
} // namespace Nektar

namespace Nektar::SpatialDomains
{

class Geometry; // Forward declaration for typedef.
typedef unique_ptr_objpool<Geometry> GeometryUniquePtr;
typedef unique_ptr_objpool<GeomFactors> GeomFactorsUniquePtr;

class Geometry1D;
class Geometry2D;

class PointGeom;

struct Curve;
typedef unique_ptr_objpool<Curve> CurveUniquePtr;
typedef std::map<int, CurveUniquePtr> CurveMap;
// static CurveMap NullCurveMap;

/// \brief Less than operator to sort Geometry objects by global id when sorting
/// STL containers.
SPATIAL_DOMAINS_EXPORT bool SortByGlobalId(const Geometry *&lhs,
                                           const Geometry *&rhs);

SPATIAL_DOMAINS_EXPORT bool GlobalIdEquality(const Geometry *&lhs,
                                             const Geometry *&rhs);

/// Base class for shape geometry information
class Geometry
{
public:
    SPATIAL_DOMAINS_EXPORT Geometry();
    SPATIAL_DOMAINS_EXPORT Geometry(int coordim);
    SPATIAL_DOMAINS_EXPORT virtual ~Geometry() = default;

    //---------------------------------------
    // Helper functions
    //---------------------------------------

    SPATIAL_DOMAINS_EXPORT inline int GetCoordim() const;
    SPATIAL_DOMAINS_EXPORT inline void SetCoordim(int coordim);
    SPATIAL_DOMAINS_EXPORT GeomFactorsUniquePtr
    GenGeomFactors(LibUtilities::PointsKeyVector &keyTgt);
    SPATIAL_DOMAINS_EXPORT LibUtilities::ShapeType GetShapeType(void);

    //---------------------------------------
    // Set and get ID
    //---------------------------------------
    SPATIAL_DOMAINS_EXPORT inline int GetGlobalID(void) const;
    SPATIAL_DOMAINS_EXPORT inline void SetGlobalID(int globalid);

    //---------------------------------------
    // Vertex, edge and face access
    //---------------------------------------
    SPATIAL_DOMAINS_EXPORT inline int GetVid(int i) const;
    SPATIAL_DOMAINS_EXPORT int GetEid(int i) const;
    SPATIAL_DOMAINS_EXPORT int GetFid(int i) const;
    SPATIAL_DOMAINS_EXPORT inline int GetTid(int i) const;
    SPATIAL_DOMAINS_EXPORT inline PointGeom *GetVertex(int i) const;
    SPATIAL_DOMAINS_EXPORT inline Geometry1D *GetEdge(int i) const;
    SPATIAL_DOMAINS_EXPORT inline Geometry2D *GetFace(int i) const;
    SPATIAL_DOMAINS_EXPORT inline StdRegions::Orientation GetEorient(
        const int i) const;
    SPATIAL_DOMAINS_EXPORT inline StdRegions::Orientation GetForient(
        const int i) const;
    SPATIAL_DOMAINS_EXPORT inline int GetNumVerts() const;
    SPATIAL_DOMAINS_EXPORT inline int GetNumEdges() const;
    SPATIAL_DOMAINS_EXPORT inline int GetNumFaces() const;
    SPATIAL_DOMAINS_EXPORT inline int GetShapeDim() const;

    //---------------------------------------
    // \chi mapping access
    //---------------------------------------
    SPATIAL_DOMAINS_EXPORT inline StdRegions::StdExpansionSharedPtr GetXmap()
        const;
    SPATIAL_DOMAINS_EXPORT inline const Array<OneD, const NekDouble> &GetCoeffs(
        const int i) const;
    SPATIAL_DOMAINS_EXPORT inline void FillGeom();

    //---------------------------------------
    // Point lookups
    //---------------------------------------
    SPATIAL_DOMAINS_EXPORT std::array<NekDouble, 6> GetBoundingBox();
    SPATIAL_DOMAINS_EXPORT void ClearBoundingBox();

    SPATIAL_DOMAINS_EXPORT inline bool ContainsPoint(
        const Array<OneD, const NekDouble> &gloCoord, NekDouble tol = 0.0);
    SPATIAL_DOMAINS_EXPORT inline bool ContainsPoint(
        const Array<OneD, const NekDouble> &gloCoord,
        Array<OneD, NekDouble> &locCoord, NekDouble tol);
    SPATIAL_DOMAINS_EXPORT inline bool ContainsPoint(
        const Array<OneD, const NekDouble> &gloCoord,
        Array<OneD, NekDouble> &locCoord, NekDouble tol, NekDouble &dist);
    SPATIAL_DOMAINS_EXPORT inline NekDouble GetLocCoords(
        const Array<OneD, const NekDouble> &coords,
        Array<OneD, NekDouble> &Lcoords);
    SPATIAL_DOMAINS_EXPORT inline NekDouble GetCoord(
        const int i, const Array<OneD, const NekDouble> &Lcoord);
    SPATIAL_DOMAINS_EXPORT int PreliminaryCheck(
        const Array<OneD, const NekDouble> &gloCoord);
    SPATIAL_DOMAINS_EXPORT bool MinMaxCheck(
        const Array<OneD, const NekDouble> &gloCoord);
    SPATIAL_DOMAINS_EXPORT bool ClampLocCoords(
        Array<OneD, NekDouble> &locCoord,
        NekDouble tol = std::numeric_limits<NekDouble>::epsilon());
    SPATIAL_DOMAINS_EXPORT inline NekDouble FindDistance(
        const Array<OneD, const NekDouble> &xs, Array<OneD, NekDouble> &xi);

    //---------------------------------------
    // Misc. helper functions
    //---------------------------------------
    SPATIAL_DOMAINS_EXPORT inline int GetVertexEdgeMap(int i, int j) const;
    SPATIAL_DOMAINS_EXPORT inline int GetVertexFaceMap(int i, int j) const;
    SPATIAL_DOMAINS_EXPORT inline int GetEdgeFaceMap(int i, int j) const;
    SPATIAL_DOMAINS_EXPORT inline int GetEdgeNormalToFaceVert(int i,
                                                              int j) const;
    SPATIAL_DOMAINS_EXPORT inline int GetDir(const int i,
                                             const int j = 0) const;
    SPATIAL_DOMAINS_EXPORT inline GeomType CalcGeomType();

    SPATIAL_DOMAINS_EXPORT inline void Reset(CurveMap &curvedEdges,
                                             CurveMap &curvedFaces);
    SPATIAL_DOMAINS_EXPORT inline void ResetNonRecursive(CurveMap &curvedEdges,
                                                         CurveMap &curvedFaces);

    SPATIAL_DOMAINS_EXPORT inline void Setup();

protected:
    /// Coordinate dimension of this geometry object.
    int m_coordim;
    /// \f$\chi\f$ mapping containing isoparametric transformation.
    StdRegions::StdExpansionSharedPtr m_xmap;
    /// Enumeration to dictate whether coefficients are filled.
    GeomState m_state;
    /// Wether or not the setup routines have been run
    bool m_setupState;
    /// Type of shape.
    LibUtilities::ShapeType m_shapeType;
    /// Global ID
    int m_globalID;
    /// Array containing expansion coefficients of @p m_xmap
    std::vector<Array<OneD, NekDouble>> m_coeffs;
    /// Array containing bounding box
    Array<OneD, NekDouble> m_boundingBox;
    Array<OneD, Array<OneD, NekDouble>> m_isoParameter;
    Array<OneD, Array<OneD, NekDouble>> m_invIsoParam;
    int m_straightEdge;

    //---------------------------------------
    // Helper functions
    //---------------------------------------
    virtual int v_GetVid(int i) const;
    virtual PointGeom *v_GetVertex(int i) const;
    virtual Geometry1D *v_GetEdge(int i) const;
    virtual Geometry2D *v_GetFace(int i) const;
    virtual StdRegions::Orientation v_GetEorient(const int i) const;
    virtual StdRegions::Orientation v_GetForient(const int i) const;
    virtual int v_GetNumVerts() const;
    virtual int v_GetNumEdges() const;
    virtual int v_GetNumFaces() const;
    virtual int v_GetShapeDim() const;

    virtual GeomFactorsUniquePtr v_GenGeomFactors(
        LibUtilities::PointsKeyVector &keyTgt);
    virtual StdRegions::StdExpansionSharedPtr v_GetXmap() const;
    virtual void v_FillGeom();

    virtual bool v_ContainsPoint(const Array<OneD, const NekDouble> &gloCoord,
                                 Array<OneD, NekDouble> &locCoord,
                                 NekDouble tol, NekDouble &dist);
    virtual int v_AllLeftCheck(const Array<OneD, const NekDouble> &gloCoord);

    virtual NekDouble v_GetCoord(const int i,
                                 const Array<OneD, const NekDouble> &Lcoord);
    virtual NekDouble v_GetLocCoords(const Array<OneD, const NekDouble> &coords,
                                     Array<OneD, NekDouble> &Lcoords);
    virtual NekDouble v_FindDistance(const Array<OneD, const NekDouble> &xs,
                                     Array<OneD, NekDouble> &xi);

    virtual int v_GetVertexEdgeMap(int i, int j) const;
    virtual int v_GetVertexFaceMap(int i, int j) const;
    virtual int v_GetEdgeFaceMap(int i, int j) const;
    virtual int v_GetEdgeNormalToFaceVert(const int i, const int j) const;
    virtual int v_GetDir(const int faceidx, const int facedir) const;

    virtual GeomType v_CalcGeomType();
    virtual void v_Reset(CurveMap &curvedEdges, CurveMap &curvedFaces);

    virtual void v_Setup();

    inline void SetUpCoeffs(const int nCoeffs);
    virtual void v_CalculateInverseIsoParam();
}; // class Geometry

/**
 * @brief Unary function that constructs a hash of a Geometry object, based on
 * the vertex IDs.
 */
struct GeometryHash
{
    std::size_t operator()(GeometryUniquePtr const &p) const
    {
        int i;
        size_t seed = 0;
        int nVert   = p->GetNumVerts();
        std::vector<unsigned int> ids(nVert);

        for (i = 0; i < nVert; ++i)
        {
            ids[i] = p->GetVid(i);
        }
        std::sort(ids.begin(), ids.end());
        hash_range(seed, ids.begin(), ids.end());

        return seed;
    }
};

/**
 * @brief Return the coordinate dimension of this object (i.e. the dimension of
 * the space in which this object is embedded).
 */
inline int Geometry::GetCoordim() const
{
    return m_coordim;
}

/**
 * @brief Sets the coordinate dimension of this object (i.e. the dimension of
 * the space in which this object is embedded).
 */
inline void Geometry::SetCoordim(int dim)
{
    m_coordim = dim;
}

/**
 * @brief Get the geometric shape type of this object.
 */
inline LibUtilities::ShapeType Geometry::GetShapeType()
{
    return m_shapeType;
}

/**
 * A geometric shape is considered regular if it has constant geometric
 * information, and deformed if this information changes throughout the
 * shape.
 * @returns             The type of geometry.
 * @see GeomType
 */
inline GeomType Geometry::CalcGeomType()
{
    return v_CalcGeomType();
}

/**
 * @brief Get the ID of this object.
 */
inline int Geometry::GetGlobalID(void) const
{
    return m_globalID;
}

/**
 * @brief Set the ID of this object.
 */
inline void Geometry::SetGlobalID(int globalid)
{
    m_globalID = globalid;
}

/**
 * @brief Get the ID of trace @p i of this object.
 *
 * The trace element is the facet one dimension lower than the object; for
 * example, a quadrilateral has four trace segments forming its boundary.
 */
inline int Geometry::GetTid(int i) const
{
    const int nDim = GetShapeDim();
    return nDim == 1   ? GetVid(i)
           : nDim == 2 ? GetEid(i)
           : nDim == 3 ? GetFid(i)
                       : 0;
}

/**
 * @brief Returns global id of vertex @p i of this object.
 */
inline int Geometry::GetVid(int i) const
{
    return v_GetVid(i);
}

/**
 * @brief Returns vertex @p i of this object.
 */
inline PointGeom *Geometry::GetVertex(int i) const
{
    return v_GetVertex(i);
}

/**
 * @brief Returns edge @p i of this object.
 */
inline Geometry1D *Geometry::GetEdge(int i) const
{
    return v_GetEdge(i);
}

/**
 * @brief Returns face @p i of this object.
 */
inline Geometry2D *Geometry::GetFace(int i) const
{
    return v_GetFace(i);
}

/**
 * @brief Returns the orientation of edge @p i with respect to the ordering of
 * edges in the standard element.
 */
inline StdRegions::Orientation Geometry::GetEorient(const int i) const
{
    return v_GetEorient(i);
}

/**
 * @brief Returns the orientation of face @p i with respect to the ordering of
 * faces in the standard element.
 */
inline StdRegions::Orientation Geometry::GetForient(const int i) const
{
    return v_GetForient(i);
}

/**
 * @brief Get the number of vertices of this object.
 */
inline int Geometry::GetNumVerts() const
{
    return v_GetNumVerts();
}

/**
 * @brief Get the number of edges of this object.
 */
inline int Geometry::GetNumEdges() const
{
    return v_GetNumEdges();
}

/**
 * @brief Get the number of faces of this object.
 */
inline int Geometry::GetNumFaces() const
{
    return v_GetNumFaces();
}

/**
 * @brief Get the object's shape dimension.
 *
 * For example, a segment is one dimensional and quadrilateral is two
 * dimensional.
 */
inline int Geometry::GetShapeDim() const
{
    return v_GetShapeDim();
}

/**
 * @brief Used by Expansion to generate associated GeomFactors.
 */
inline GeomFactorsUniquePtr Geometry::GenGeomFactors(
    LibUtilities::PointsKeyVector &keyTgt)
{
    return v_GenGeomFactors(keyTgt);
}

/**
 * @brief Return the mapping object Geometry::m_xmap that represents the
 * coordinate transformation from standard element to physical element.
 */
inline StdRegions::StdExpansionSharedPtr Geometry::GetXmap() const
{
    return v_GetXmap();
}

/**
 * @brief Return the coefficients of the transformation Geometry::m_xmap in
 * coordinate direction @p i.
 */
inline const Array<OneD, const NekDouble> &Geometry::GetCoeffs(
    const int i) const
{
    return m_coeffs[i];
}

/**
 * @brief Populate the coordinate mapping Geometry::m_coeffs information from
 * any children geometry elements.
 *
 * @see v_FillGeom()
 */
inline void Geometry::FillGeom()
{
    v_FillGeom();
}

/**
 * @brief Determine whether an element contains a particular Cartesian
 * coordinate \f$(x,y,z)\f$.
 *
 * @see Geometry::ContainsPoint
 */
inline bool Geometry::ContainsPoint(
    const Array<OneD, const NekDouble> &gloCoord, NekDouble tol)
{
    Array<OneD, NekDouble> locCoord(GetCoordim(), 0.0);
    NekDouble dist;
    return v_ContainsPoint(gloCoord, locCoord, tol, dist);
}

/**
 * @brief Determine whether an element contains a particular Cartesian
 * coordinate \f$(x,y,z)\f$.
 *
 * @see Geometry::ContainsPoint
 */
inline bool Geometry::ContainsPoint(
    const Array<OneD, const NekDouble> &gloCoord,
    Array<OneD, NekDouble> &locCoord, NekDouble tol)
{
    NekDouble dist;
    return v_ContainsPoint(gloCoord, locCoord, tol, dist);
}

/**
 * @brief Determine whether an element contains a particular Cartesian
 * coordinate \f$\vec{x} = (x,y,z)\f$.
 *
 * For curvilinear and non-affine elements (i.e. where the Jacobian varies as a
 * function of the standard element coordinates), this is a non-linear
 * optimisation problem that requires the use of a Newton iteration. Note
 * therefore that this can be an expensive operation.
 *
 * The parameter @p tol which is by default 0, can be used to expand the
 * coordinate range of the standard element from \f$[-1,1]^d\f$ to
 * \f$[-1-\epsilon,1+\epsilon\f$ to handle challenging edge cases. The function
 * also returns the local coordinates corresponding to @p gloCoord that can be
 * used to speed up subsequent searches.
 *
 * @param gloCoord  The coordinate \f$ (x,y,z) \f$.
 * @param locCoord  On exit, this is the local coordinate \f$\vec{\xi}\f$ such
 *                  that \f$\chi(\vec{\xi}) = \vec{x}\f$.
 * @param tol       The tolerance used to dictate the bounding box of the
 *                  standard coordinates \f$\vec{\xi}\f$.
 * @param dist      On exit, returns the minimum distance between @p gloCoord
 *                  and the quadrature points inside the element.
 *
 * @return `true` if the coordinate @p gloCoord is contained in the element;
 *         false otherwise.
 *
 * @see Geometry::GetLocCoords.
 */
inline bool Geometry::ContainsPoint(
    const Array<OneD, const NekDouble> &gloCoord,
    Array<OneD, NekDouble> &locCoord, NekDouble tol, NekDouble &dist)
{
    return v_ContainsPoint(gloCoord, locCoord, tol, dist);
}

/**
 * @brief Determine the local collapsed coordinates that correspond to a
 * given Cartesian coordinate for this geometry object.
 *
 * For curvilinear and non-affine elements (i.e. where the Jacobian varies as a
 * function of the standard element coordinates), this is a non-linear
 * optimisation problem that requires the use of a Newton iteration. Note
 * therefore that this can be an expensive operation.
 *
 * Note that, clearly, the provided Cartesian coordinate lie outside the
 * element. The function therefore returns the minimum distance from some
 * position in the element to . @p Lcoords will also be constrained to fit
 * within the range \f$[-1,1]^d\f$ where \f$ d \f$ is the dimension of the
 * element.
 *
 * @param coords   Input Cartesian global coordinates
 * @param Lcoords  Corresponding local coordinates
 *
 * @return Distance between obtained coordinates and provided ones.
 */
inline NekDouble Geometry::GetLocCoords(
    const Array<OneD, const NekDouble> &coords, Array<OneD, NekDouble> &Lcoords)
{
    return v_GetLocCoords(coords, Lcoords);
}

/**
 * @brief Given local collapsed coordinate @p Lcoord, return the value of
 * physical coordinate in direction @p i.
 */
inline NekDouble Geometry::GetCoord(const int i,
                                    const Array<OneD, const NekDouble> &Lcoord)
{
    return v_GetCoord(i, Lcoord);
}

inline NekDouble Geometry::FindDistance(const Array<OneD, const NekDouble> &xs,
                                        Array<OneD, NekDouble> &xi)
{
    return v_FindDistance(xs, xi);
}

/**
 * @brief Returns the standard element edge IDs that are connected to a given
 * vertex.
 *
 * For example, on a prism, vertex 0 is connnected to edges 0, 3, and 4;
 * `GetVertexEdgeMap(0,j)` would therefore return the values 0, 1 and 4
 * respectively. We assume that @p j runs between 0 and 2 inclusive, which is
 * true for every 3D element asides from the pyramid.
 *
 * This function is used in the construction of the low-energy preconditioner.
 *
 * @param i  The vertex to query connectivity for.
 * @param j  The local edge index between 0 and 2 connected to this element.
 *
 * @todo Expand to work with pyramid elements.
 * @see MultiRegions::PreconditionerLowEnergy
 */
inline int Geometry::GetVertexEdgeMap(int i, int j) const
{
    return v_GetVertexEdgeMap(i, j);
}

/**
 * @brief Returns the standard element face IDs that are connected to a given
 * vertex.
 *
 * For example, on a hexahedron, vertex 0 is connnected to faces 0, 1, and 4;
 * `GetVertexFaceMap(0,j)` would therefore return the values 0, 1 and 4
 * respectively. We assume that @p j runs between 0 and 2 inclusive, which is
 * true for every 3D element asides from the pyramid.
 *
 * This is used in the construction of the low-energy preconditioner.
 *
 * @param i  The vertex to query connectivity for.
 * @param j  The local face index between 0 and 2 connected to this element.
 *
 * @todo Expand to work with pyramid elements.
 * @see MultiRegions::PreconditionerLowEnergy
 */
inline int Geometry::GetVertexFaceMap(int i, int j) const
{
    return v_GetVertexFaceMap(i, j);
}

/**
 * @brief Returns the standard element edge IDs that are connected to a given
 * face.
 *
 * For example, on a prism, edge 0 is connnected to faces 0 and 1;
 * `GetEdgeFaceMap(0,j)` would therefore return the values 0 and 1
 * respectively. We assume that @p j runs between 0 and 1 inclusive, since every
 * face is connected to precisely two faces for all 3D elements.
 *
 * This function is used in the construction of the low-energy preconditioner.
 *
 * @param i  The edge to query connectivity for.
 * @param j  The local face index between 0 and 1 connected to this element.
 *
 * @see MultiRegions::PreconditionerLowEnergy
 */
inline int Geometry::GetEdgeFaceMap(int i, int j) const
{
    return v_GetEdgeFaceMap(i, j);
}

/**
 * @brief Returns the standard lement edge IDs that are normal to a given face
 * vertex.
 *
 * For example, on a hexahedron, on face 0 at vertices 0,1,2,3 the
 * edges normal to that face are 4,5,6,7, ; so
 * `GetEdgeNormalToFaceVert(0,j)` would therefore return the values 4,
 * 5, 6 and 7 respectively. We assume that @p j runs between 0 and 3
 * inclusive on a quadrilateral face and between 0 and 2 inclusive on
 * a triangular face.
 *
 * This is used to help set up a length scale normal to an face
 *
 * @param i  The face to query for the normal edge
 * @param j  The local vertex index between 0 and nverts on this face
 *
 */
inline int Geometry::GetEdgeNormalToFaceVert(int i, int j) const
{
    return v_GetEdgeNormalToFaceVert(i, j);
}

/**
 * @brief Returns the element coordinate direction corresponding to a given face
 * coordinate direction
 */
inline int Geometry::GetDir(const int faceidx, const int facedir) const
{
    return v_GetDir(faceidx, facedir);
}

/**
 * @brief Reset this geometry object: unset the current state, zero
 * Geometry::m_coeffs and remove allocated GeomFactors.
 */
inline void Geometry::Reset(CurveMap &curvedEdges, CurveMap &curvedFaces)
{
    v_Reset(curvedEdges, curvedFaces);
}

/**
 * @brief Reset this geometry object non-recursively: unset the current state,
 * zero Geometry::m_coeffs and remove allocated GeomFactors.
 */
inline void Geometry::ResetNonRecursive(CurveMap &curvedEdges,
                                        CurveMap &curvedFaces)
{
    Geometry::v_Reset(curvedEdges, curvedFaces);
}

inline void Geometry::Setup()
{
    v_Setup();
}

/**
 * @brief Initialise the Geometry::m_coeffs array.
 */
inline void Geometry::SetUpCoeffs(const int nCoeffs)
{
    m_coeffs = std::vector<Array<OneD, NekDouble>>(m_coordim);

    for (int i = 0; i < m_coordim; ++i)
    {
        m_coeffs[i] = Array<OneD, NekDouble>(nCoeffs, 0.0);
    }
}

} // namespace Nektar::SpatialDomains

#endif // NEKTAR_SPATIALDOMAINS_GEOMETRY_H
