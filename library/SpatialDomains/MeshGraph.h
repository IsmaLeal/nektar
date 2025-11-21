////////////////////////////////////////////////////////////////////////////////
//
//  File: MeshGraph.h
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

#ifndef NEKTAR_SPATIALDOMAINS_MESHGRAPH_H
#define NEKTAR_SPATIALDOMAINS_MESHGRAPH_H

#include <unordered_map>

#include <LibUtilities/BasicUtils/DomainRange.h>
#include <LibUtilities/BasicUtils/FieldIO.h>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/Memory/ObjectPool.hpp>

#include <SpatialDomains/HexGeom.h>
#include <SpatialDomains/MeshEntities.hpp>
#include <SpatialDomains/PrismGeom.h>
#include <SpatialDomains/PyrGeom.h>
#include <SpatialDomains/QuadGeom.h>
#include <SpatialDomains/SegGeom.h>
#include <SpatialDomains/TetGeom.h>
#include <SpatialDomains/TriGeom.h>

#include <SpatialDomains/Curve.hpp>
#include <SpatialDomains/SpatialDomainsDeclspec.h>

class TiXmlDocument;

namespace Nektar
{

template <>
PoolAllocator<SpatialDomains::PointGeom>
    ObjPoolManager<SpatialDomains::PointGeom>::m_alloc;
template <>
PoolAllocator<SpatialDomains::SegGeom>
    ObjPoolManager<SpatialDomains::SegGeom>::m_alloc;
template <>
PoolAllocator<SpatialDomains::TriGeom>
    ObjPoolManager<SpatialDomains::TriGeom>::m_alloc;
template <>
PoolAllocator<SpatialDomains::QuadGeom>
    ObjPoolManager<SpatialDomains::QuadGeom>::m_alloc;
template <>
PoolAllocator<SpatialDomains::TetGeom>
    ObjPoolManager<SpatialDomains::TetGeom>::m_alloc;
template <>
PoolAllocator<SpatialDomains::PyrGeom>
    ObjPoolManager<SpatialDomains::PyrGeom>::m_alloc;
template <>
PoolAllocator<SpatialDomains::PrismGeom>
    ObjPoolManager<SpatialDomains::PrismGeom>::m_alloc;
template <>
PoolAllocator<SpatialDomains::HexGeom>
    ObjPoolManager<SpatialDomains::HexGeom>::m_alloc;
template <>
PoolAllocator<SpatialDomains::GeomFactors>
    ObjPoolManager<SpatialDomains::GeomFactors>::m_alloc;
template <>
PoolAllocator<SpatialDomains::Curve>
    ObjPoolManager<SpatialDomains::Curve>::m_alloc;

namespace SpatialDomains
{

template <typename T> using GeomMap = std::map<int, unique_ptr_objpool<T>>;

// Point geom type defs
typedef unique_ptr_objpool<PointGeom> PointGeomUniquePtr;
typedef std::shared_ptr<PointGeom>
    PointGeomSharedPtr; // @TODO: Remove once fixed curves

// Geometry typedefs
typedef unique_ptr_objpool<SegGeom> SegGeomUniquePtr;
typedef unique_ptr_objpool<TriGeom> TriGeomUniquePtr;
typedef unique_ptr_objpool<QuadGeom> QuadGeomUniquePtr;
typedef unique_ptr_objpool<TetGeom> TetGeomUniquePtr;
typedef unique_ptr_objpool<PyrGeom> PyrGeomUniquePtr;
typedef unique_ptr_objpool<PrismGeom> PrismGeomUniquePtr;
typedef unique_ptr_objpool<HexGeom> HexGeomUniquePtr;
typedef unique_ptr_objpool<Geometry2D> Geometry2DUniquePtr;
typedef unique_ptr_objpool<Geometry3D> Geometry3DUniquePtr;
typedef unique_ptr_objpool<GeomFactors> GeomFactorsUniquePtr;
typedef unique_ptr_objpool<Curve> CurveUniquePtr;

// Minimal owner of Geom objects not owned by a MeshGraph.
class EntityHolder
{
public:
    std::vector<PointGeomUniquePtr> m_pointVec;
    std::vector<SegGeomUniquePtr> m_segVec;
    std::vector<TriGeomUniquePtr> m_triVec;
    std::vector<QuadGeomUniquePtr> m_quadVec;
    std::vector<TetGeomUniquePtr> m_tetVec;
    std::vector<PyrGeomUniquePtr> m_pyrVec;
    std::vector<PrismGeomUniquePtr> m_prismVec;
    std::vector<HexGeomUniquePtr> m_hexVec;
    std::vector<CurveUniquePtr> m_curveVec;
};

// Composite type def
typedef std::map<int, std::pair<LibUtilities::ShapeType, std::vector<int>>>
    CompositeDescriptor;

enum ExpansionType
{
    eNoExpansionType,
    eModified,
    eModifiedQuadPlus1,
    eModifiedQuadPlus2,
    eModifiedGLLRadau10,
    eOrthogonal,
    eGLL_Lagrange,
    eGLL_Lagrange_SEM,
    eGauss_Lagrange,
    eGauss_Lagrange_SEM,
    eFourier,
    eFourierSingleMode,
    eFourierHalfModeRe,
    eFourierHalfModeIm,
    eChebyshev,
    eFourierChebyshev,
    eChebyshevFourier,
    eFourierModified,
    eExpansionTypeSize
};

// Keep this consistent with the enums in ExpansionType.
// This is used in the BC file to specify the expansion type.
const std::string kExpansionTypeStr[] = {"NOTYPE",
                                         "MODIFIED",
                                         "MODIFIEDQUADPLUS1",
                                         "MODIFIEDQUADPLUS2",
                                         "MODIFIEDGLLRADAU10",
                                         "ORTHOGONAL",
                                         "GLL_LAGRANGE",
                                         "GLL_LAGRANGE_SEM",
                                         "GAUSS_LAGRANGE",
                                         "GAUSS_LAGRANGE_SEM",
                                         "FOURIER",
                                         "FOURIERSINGLEMODE",
                                         "FOURIERHALFMODERE",
                                         "FOURIERHALFMODEIM",
                                         "CHEBYSHEV",
                                         "FOURIER-CHEBYSHEV",
                                         "CHEBYSHEV-FOURIER",
                                         "FOURIER-MODIFIED"};

typedef std::map<int, std::vector<unsigned int>> CompositeOrdering;
typedef std::map<int, std::vector<unsigned int>> BndRegionOrdering;

struct Composite
{
    std::vector<Geometry *> m_geomVec;
};

typedef std::shared_ptr<Composite> CompositeSharedPtr;
typedef std::map<int, CompositeSharedPtr> CompositeMap;

struct ExpansionInfo;

typedef std::shared_ptr<ExpansionInfo> ExpansionInfoShPtr;
typedef std::map<int, ExpansionInfoShPtr> ExpansionInfoMap;

typedef std::shared_ptr<ExpansionInfoMap> ExpansionInfoMapShPtr;
typedef std::map<std::string, ExpansionInfoMapShPtr> ExpansionInfoMapShPtrMap;

struct ExpansionInfo
{
    ExpansionInfo(Geometry *geomPtr,
                  const LibUtilities::BasisKeyVector basiskeyvec)
        : m_geomPtr(geomPtr), m_basisKeyVector(basiskeyvec)
    {
    }

    ExpansionInfo(ExpansionInfoShPtr ExpInfo)
        : m_geomPtr(ExpInfo->m_geomPtr),
          m_basisKeyVector(ExpInfo->m_basisKeyVector)
    {
    }

    Geometry *m_geomPtr;
    LibUtilities::BasisKeyVector m_basisKeyVector;
};

typedef std::map<std::string, std::string> GeomInfoMap;
typedef std::shared_ptr<std::vector<std::pair<Geometry *, int>>>
    GeometryLinkSharedPtr;

// Forward declaration
class RefRegion;

typedef std::map<std::string, std::string> MeshMetaDataMap;

class MeshGraph;
typedef std::shared_ptr<MeshGraph> MeshGraphSharedPtr;

class Movement;
typedef std::shared_ptr<Movement> MovementSharedPtr;

template <typename T> class GeomMapView
{
public:
    class Iterator
    {
        typename GeomMap<T>::const_iterator m_it;

    public:
        using value_type = std::pair<int, T *>;

        explicit Iterator(typename GeomMap<T>::const_iterator it) : m_it(it)
        {
        }

        value_type operator*() const
        {
            return {m_it->first, m_it->second.get()};
        }
        Iterator &operator++()
        {
            ++m_it;
            return *this;
        }
        bool operator!=(const Iterator &other) const
        {
            return m_it != other.m_it;
        }
        bool operator==(const Iterator &other) const
        {
            return m_it == other.m_it;
        }
    };

    explicit GeomMapView(const GeomMap<T> &map) : m_map(map)
    {
    }

    Iterator begin() const
    {
        return Iterator(m_map.begin());
    }
    Iterator end() const
    {
        return Iterator(m_map.end());
    }

    std::size_t size() const
    {
        return m_map.size();
    }

    Iterator find(int id) const
    {
        return Iterator(m_map.find(id));
    }

    T *at(int id) const
    {
        auto it = m_map.find(id);
        return it != m_map.end() ? it->second.get() : nullptr;
    }

private:
    const GeomMap<T> &m_map;
};

/// Base class for a spectral/hp element mesh.
class MeshGraph
{
public:
    SPATIAL_DOMAINS_EXPORT MeshGraph();
    SPATIAL_DOMAINS_EXPORT virtual ~MeshGraph();

    void Empty(int dim, int space)
    {
        m_meshDimension  = dim;
        m_spaceDimension = space;
    }

    /*transfers the minial data structure to full meshgraph*/
    SPATIAL_DOMAINS_EXPORT void FillGraph();

    SPATIAL_DOMAINS_EXPORT void FillBoundingBoxTree();

    SPATIAL_DOMAINS_EXPORT std::vector<int> GetElementsContainingPoint(
        PointGeom *p);

    ////////////////////
    SPATIAL_DOMAINS_EXPORT void ReadExpansionInfo();

    /// Read refinement info.
    SPATIAL_DOMAINS_EXPORT void ReadRefinementInfo();

    /* ---- Helper functions ---- */
    /// Dimension of the mesh (can be a 1D curve in 3D space).
    int GetMeshDimension()
    {
        return m_meshDimension;
    }

    /// Dimension of the space (can be a 1D curve in 3D space).
    int GetSpaceDimension()
    {
        return m_spaceDimension;
    }

    void SetMeshDimension(int dim)
    {
        m_meshDimension = dim;
    }

    void SetSpaceDimension(int dim)
    {
        m_spaceDimension = dim;
    }

    /* Range definitions for postprorcessing */
    SPATIAL_DOMAINS_EXPORT void SetDomainRange(
        NekDouble xmin, NekDouble xmax,
        NekDouble ymin = NekConstants::kNekUnsetDouble,
        NekDouble ymax = NekConstants::kNekUnsetDouble,
        NekDouble zmin = NekConstants::kNekUnsetDouble,
        NekDouble zmax = NekConstants::kNekUnsetDouble);

    SPATIAL_DOMAINS_EXPORT void SetDomainRange(
        LibUtilities::DomainRangeShPtr rng);

    /// Check if goemetry is in range definition if activated
    SPATIAL_DOMAINS_EXPORT bool CheckRange(Geometry2D &geom);

    /// Check if goemetry is in range definition if activated
    SPATIAL_DOMAINS_EXPORT bool CheckRange(Geometry3D &geom);

    /// Check if goemetry is in range definition if activated
    SPATIAL_DOMAINS_EXPORT bool CheckRange(MeshEntity &e);

    /* ---- Composites and Domain ---- */
    CompositeSharedPtr GetComposite(int whichComposite)
    {
        ASSERTL0(m_meshComposites.find(whichComposite) !=
                     m_meshComposites.end(),
                 "Composite not found.");
        return m_meshComposites.find(whichComposite)->second;
    }

    SPATIAL_DOMAINS_EXPORT Geometry *GetCompositeItem(int whichComposite,
                                                      int whichItem);

    SPATIAL_DOMAINS_EXPORT void GetCompositeList(
        const std::string &compositeStr, CompositeMap &compositeVector) const;

    std::map<int, CompositeSharedPtr> &GetComposites()
    {
        return m_meshComposites;
    }

    std::map<int, std::string> &GetCompositesLabels()
    {
        return m_compositesLabels;
    }

    std::map<int, std::map<int, CompositeSharedPtr>> &GetDomain()
    {
        return m_domain;
    }

    std::map<int, CompositeSharedPtr> &GetDomain(int domain)
    {
        ASSERTL1(m_domain.count(domain),
                 "Request for domain which does not exist");
        return m_domain[domain];
    }

    LibUtilities::DomainRangeShPtr &GetDomainRange()
    {
        return m_domainRange;
    }

    SPATIAL_DOMAINS_EXPORT const ExpansionInfoMap &GetExpansionInfo(
        const std::string variable = "DefaultVar");

    SPATIAL_DOMAINS_EXPORT ExpansionInfoShPtr
    GetExpansionInfo(Geometry *geom, const std::string variable = "DefaultVar");

    /// Sets expansions given field definitions
    SPATIAL_DOMAINS_EXPORT void SetExpansionInfo(
        std::vector<LibUtilities::FieldDefinitionsSharedPtr> &fielddef);

    /// Sets expansions given field definition, quadrature points.
    SPATIAL_DOMAINS_EXPORT void SetExpansionInfo(
        std::vector<LibUtilities::FieldDefinitionsSharedPtr> &fielddef,
        std::vector<std::vector<LibUtilities::PointsType>> &pointstype);

    /// Sets expansions to have equispaced points
    SPATIAL_DOMAINS_EXPORT void SetExpansionInfoToEvenlySpacedPoints(
        int npoints = 0);

    /// Reset expansion to have specified polynomial order \a nmodes
    SPATIAL_DOMAINS_EXPORT void SetExpansionInfoToNumModes(int nmodes);

    /// Reset expansion to have specified point order \a
    /// npts
    SPATIAL_DOMAINS_EXPORT void SetExpansionInfoToPointOrder(int npts);
    /// This function sets the expansion #exp in map with
    /// entry #variable

    /// Set refinement info.
    SPATIAL_DOMAINS_EXPORT void SetRefinementInfo(
        ExpansionInfoMapShPtr &expansionMap);

    /// Perform the p-refinement in the selected elements
    SPATIAL_DOMAINS_EXPORT void PRefinementElmts(
        ExpansionInfoMapShPtr &expansionMap, RefRegion *&region,
        Geometry *geomVecIter);

    inline void SetExpansionInfo(const std::string variable,
                                 ExpansionInfoMapShPtr &exp);

    inline void SetSession(LibUtilities::SessionReaderSharedPtr pSession);

    /// Sets the basis key for all expansions of the given shape.
    SPATIAL_DOMAINS_EXPORT void SetBasisKey(LibUtilities::ShapeType shape,
                                            LibUtilities::BasisKeyVector &keys,
                                            std::string var = "DefaultVar");

    SPATIAL_DOMAINS_EXPORT void ResetExpansionInfoToBasisKey(
        ExpansionInfoMapShPtr &expansionMap, LibUtilities::ShapeType shape,
        LibUtilities::BasisKeyVector &keys);

    inline bool SameExpansionInfo(const std::string var1,
                                  const std::string var2);

    inline bool ExpansionInfoDefined(const std::string var);

    inline bool CheckForGeomInfo(std::string parameter);

    inline const std::string GetGeomInfo(std::string parameter);

    SPATIAL_DOMAINS_EXPORT static LibUtilities::BasisKeyVector
    DefineBasisKeyFromExpansionType(Geometry *in, ExpansionType type,
                                    const int order);

    SPATIAL_DOMAINS_EXPORT LibUtilities::BasisKeyVector
    DefineBasisKeyFromExpansionTypeHomo(Geometry *in, ExpansionType type_x,
                                        ExpansionType type_y,
                                        ExpansionType type_z,
                                        const int nummodes_x,
                                        const int nummodes_y,
                                        const int nummodes_z);

    /* ---- Manipulation of mesh ---- */
    int GetNvertices()
    {
        return m_pointGeoms.size();
    }

    /**
     * @brief Returns vertex @p id from the MeshGraph.
     */
    [[deprecated("since 5.8.0, use GetPointGeom() instead")]] PointGeom *GetVertex(
        int id)
    {
        return this->GetGeom(id, m_pointGeoms);
    }

    /**
     * @brief Returns vertex @p id from the MeshGraph.
     */
    PointGeom *GetPointGeom(int id)
    {
        return this->GetGeom(id, m_pointGeoms);
    }

    /**
     * @brief Returns segment @p id from the MeshGraph.
     */
    SegGeom *GetSegGeom(int id)
    {
        return this->GetGeom(id, m_segGeoms);
    }

    /**
     * @brief Returns triangle @p id from the MeshGraph.
     */
    TriGeom *GetTriGeom(int id)
    {
        return this->GetGeom(id, m_triGeoms);
    }

    /**
     * @brief Returns quadrilateral @p id from the MeshGraph.
     */
    QuadGeom *GetQuadGeom(int id)
    {
        return this->GetGeom(id, m_quadGeoms);
    }

    /**
     * @brief Returns tetrahedron @p id from the MeshGraph.
     */
    TetGeom *GetTetGeom(int id)
    {
        return this->GetGeom(id, m_tetGeoms);
    }

    /**
     * @brief Returns pyramid @p id from the MeshGraph.
     */
    PyrGeom *GetPyrGeom(int id)
    {
        return this->GetGeom(id, m_pyrGeoms);
    }

    /**
     * @brief Returns prism @p id from the MeshGraph.
     */
    PrismGeom *GetPrismGeom(int id)
    {
        return this->GetGeom(id, m_prismGeoms);
    }

    /**
     * @brief Returns hex @p id from the MeshGraph.
     */
    HexGeom *GetHexGeom(int id)
    {
        return this->GetGeom(id, m_hexGeoms);
    }

    /**
     * @brief Convenience function to add a geometry @p geom to the MeshGraph
     * with geometry ID @p id. Retains ownership of the passed unique_ptr.
     *
     * @p id    Geometry ID
     * @p geom  unique_ptr to geometry object.
     */
    template <typename T> void AddGeom(int id, unique_ptr_objpool<T> geom)
    {
        if constexpr (std::is_same_v<T, PointGeom>)
        {
            m_pointGeoms.insert(std::make_pair(id, std::move(geom)));
        }
        else if constexpr (std::is_same_v<T, SegGeom>)
        {
            m_segGeoms.insert(std::make_pair(id, std::move(geom)));
        }
        else if constexpr (std::is_same_v<T, QuadGeom>)
        {
            m_quadGeoms.insert(std::make_pair(id, std::move(geom)));
        }
        else if constexpr (std::is_same_v<T, TriGeom>)
        {
            m_triGeoms.insert(std::make_pair(id, std::move(geom)));
        }
        else if constexpr (std::is_same_v<T, TetGeom>)
        {
            m_tetGeoms.insert(std::make_pair(id, std::move(geom)));
        }
        else if constexpr (std::is_same_v<T, PyrGeom>)
        {
            m_pyrGeoms.insert(std::make_pair(id, std::move(geom)));
        }
        else if constexpr (std::is_same_v<T, PrismGeom>)
        {
            m_prismGeoms.insert(std::make_pair(id, std::move(geom)));
        }
        else if constexpr (std::is_same_v<T, HexGeom>)
        {
            m_hexGeoms.insert(std::make_pair(id, std::move(geom)));
        }
        else
        {
            ASSERTL0(false, "Unknown geometry type");
        }
    }

    SPATIAL_DOMAINS_EXPORT PointGeom *CreatePointGeom(const int coordim,
                                                      const int vid,
                                                      NekDouble x, NekDouble y,
                                                      NekDouble z)
    {
        auto geom =
            ObjPoolManager<PointGeom>::AllocateUniquePtr(coordim, vid, x, y, z);
        auto ret = geom.get();
        AddGeom(vid, std::move(geom));
        return ret;
    }

    SPATIAL_DOMAINS_EXPORT SegGeom *CreateSegGeom(
        int id, int coordim, std::array<PointGeom *, SegGeom::kNverts> vertex,
        Curve *curve = nullptr)
    {
        auto geom = ObjPoolManager<SegGeom>::AllocateUniquePtr(id, coordim,
                                                               vertex, curve);
        auto ret  = geom.get();
        AddGeom(id, std::move(geom));
        return ret;
    }

    SPATIAL_DOMAINS_EXPORT QuadGeom *CreateQuadGeom(
        int id, std::array<SegGeom *, QuadGeom::kNedges> edges,
        Curve *curve = nullptr)
    {
        auto geom =
            ObjPoolManager<QuadGeom>::AllocateUniquePtr(id, edges, curve);
        auto ret = geom.get();
        AddGeom(id, std::move(geom));
        return ret;
    }

    SPATIAL_DOMAINS_EXPORT TriGeom *CreateTriGeom(
        int id, std::array<SegGeom *, TriGeom::kNedges> edges,
        Curve *curve = nullptr)
    {
        auto geom =
            ObjPoolManager<TriGeom>::AllocateUniquePtr(id, edges, curve);
        auto ret = geom.get();
        AddGeom(id, std::move(geom));
        return ret;
    }

    SPATIAL_DOMAINS_EXPORT TetGeom *CreateTetGeom(
        int id, std::array<TriGeom *, TetGeom::kNfaces> faces)
    {
        auto geom = ObjPoolManager<TetGeom>::AllocateUniquePtr(id, faces);
        auto ret  = geom.get();
        AddGeom(id, std::move(geom));
        return ret;
    }

    SPATIAL_DOMAINS_EXPORT HexGeom *CreateHexGeom(
        int id, std::array<QuadGeom *, HexGeom::kNfaces> faces)
    {
        auto geom = ObjPoolManager<HexGeom>::AllocateUniquePtr(id, faces);
        auto ret  = geom.get();
        AddGeom(id, std::move(geom));
        return ret;
    }

    SPATIAL_DOMAINS_EXPORT PrismGeom *CreatePrismGeom(
        int id, std::array<Geometry2D *, PrismGeom::kNfaces> faces)
    {
        auto geom = ObjPoolManager<PrismGeom>::AllocateUniquePtr(id, faces);
        auto ret  = geom.get();
        AddGeom(id, std::move(geom));
        return ret;
    }

    SPATIAL_DOMAINS_EXPORT PyrGeom *CreatePyrGeom(
        int id, std::array<Geometry2D *, PyrGeom::kNfaces> faces)
    {
        auto geom = ObjPoolManager<PyrGeom>::AllocateUniquePtr(id, faces);
        auto ret  = geom.get();
        AddGeom(id, std::move(geom));
        return ret;
    }

    SPATIAL_DOMAINS_EXPORT CurveMap &GetCurvedEdges()
    {
        return m_curvedEdges;
    }

    SPATIAL_DOMAINS_EXPORT CurveMap &GetCurvedFaces()
    {
        return m_curvedFaces;
    }

    template <typename T> GeomMapView<T> &GetGeomMap()
    {
        if constexpr (std::is_same_v<T, PointGeom>)
        {
            return m_pointMapView;
        }
        else if constexpr (std::is_same_v<T, SegGeom>)
        {
            return m_segMapView;
        }
        else if constexpr (std::is_same_v<T, TriGeom>)
        {
            return m_triMapView;
        }
        else if constexpr (std::is_same_v<T, QuadGeom>)
        {
            return m_quadMapView;
        }
        else if constexpr (std::is_same_v<T, TetGeom>)
        {
            return m_tetMapView;
        }
        else if constexpr (std::is_same_v<T, PrismGeom>)
        {
            return m_prismMapView;
        }
        else if constexpr (std::is_same_v<T, PyrGeom>)
        {
            return m_pyrMapView;
        }
        else if constexpr (std::is_same_v<T, HexGeom>)
        {
            return m_hexMapView;
        }

        ASSERTL0(false,
                 "MeshGraph does not support the supplied geometry type.");
    }

    SPATIAL_DOMAINS_EXPORT std::unordered_map<int, GeometryLinkSharedPtr> &
    GetAllFaceToElMap()
    {
        return m_faceToElMap;
    }

    SPATIAL_DOMAINS_EXPORT std::vector<PointGeomUniquePtr> &GetAllCurveNodes()
    {
        return m_nodeSet;
    }

    SPATIAL_DOMAINS_EXPORT int GetNumElements();

    Geometry2D *GetGeometry2D(int gID)
    {
        auto it1 = m_triGeoms.find(gID);
        if (it1 != m_triGeoms.end())
        {
            return it1->second.get();
        }

        auto it2 = m_quadGeoms.find(gID);
        if (it2 != m_quadGeoms.end())
        {
            return it2->second.get();
        }

        return nullptr;
    };

    SPATIAL_DOMAINS_EXPORT GeometryLinkSharedPtr
    GetElementsFromEdge(Geometry1D *edge);

    SPATIAL_DOMAINS_EXPORT GeometryLinkSharedPtr
    GetElementsFromFace(Geometry2D *face);

    void SetPartition(SpatialDomains::MeshGraphSharedPtr graph);

    CompositeOrdering &GetCompositeOrdering()
    {
        return m_compOrder;
    }

    void SetCompositeOrdering(CompositeOrdering p_compOrder)
    {
        m_compOrder = p_compOrder;
    }

    BndRegionOrdering &GetBndRegionOrdering()
    {
        return m_bndRegOrder;
    }

    void SetBndRegionOrdering(BndRegionOrdering p_bndRegOrder)
    {
        m_bndRegOrder = p_bndRegOrder;
    }

    SPATIAL_DOMAINS_EXPORT std::map<int, MeshEntity> CreateMeshEntities();
    SPATIAL_DOMAINS_EXPORT CompositeDescriptor CreateCompositeDescriptor();

    SPATIAL_DOMAINS_EXPORT inline MovementSharedPtr &GetMovement()
    {
        return m_movement;
    }

    void Clear();

    void PopulateFaceToElMap(Geometry3D *element, int kNfaces);

    void SetMeshPartitioned(bool meshPartitioned)
    {
        m_meshPartitioned = meshPartitioned;
    }

private:
    /**
     * @brief Helper function for geometry lookups
     */
    template <typename T> T *GetGeom(int id, GeomMap<T> &geomMap)
    {
        auto it = geomMap.find(id);
        ASSERTL0(it != geomMap.end(),
                 "Unable to find geometry with ID " + std::to_string(id));
        return it->second.get();
    }

protected:
    ExpansionInfoMapShPtr SetUpExpansionInfoMap();
    std::string GetCompositeString(CompositeSharedPtr comp);

    LibUtilities::SessionReaderSharedPtr m_session;

    CurveMap m_curvedEdges;
    CurveMap m_curvedFaces;

    GeomMap<PointGeom> m_pointGeoms;
    GeomMap<SegGeom> m_segGeoms;
    GeomMap<TriGeom> m_triGeoms;
    GeomMap<QuadGeom> m_quadGeoms;
    GeomMap<TetGeom> m_tetGeoms;
    GeomMap<PyrGeom> m_pyrGeoms;
    GeomMap<PrismGeom> m_prismGeoms;
    GeomMap<HexGeom> m_hexGeoms;

    GeomMapView<PointGeom> m_pointMapView;
    GeomMapView<SegGeom> m_segMapView;
    GeomMapView<TriGeom> m_triMapView;
    GeomMapView<QuadGeom> m_quadMapView;
    GeomMapView<TetGeom> m_tetMapView;
    GeomMapView<PyrGeom> m_pyrMapView;
    GeomMapView<PrismGeom> m_prismMapView;
    GeomMapView<HexGeom> m_hexMapView;

    /// Vector of all unique curve nodes, not including vertices
    std::vector<PointGeomUniquePtr> m_nodeSet;

    int m_meshDimension;
    int m_spaceDimension;
    int m_partition;
    bool m_meshPartitioned = false;
    bool m_useExpansionType;

    // Refinement attributes (class members)
    /// Link the refinement id with the composites
    std::map<int, CompositeMap> m_refComposite;
    // std::map<int, LibUtilities::BasisKeyVector> m_refBasis;
    /// Link the refinement id with the surface region data
    std::map<int, RefRegion *> m_refRegion;
    bool m_refFlag = false;

    CompositeMap m_meshComposites;
    std::map<int, std::string> m_compositesLabels;
    std::map<int, CompositeMap> m_domain;
    LibUtilities::DomainRangeShPtr m_domainRange;

    ExpansionInfoMapShPtrMap m_expansionMapShPtrMap;

    std::unordered_map<int, GeometryLinkSharedPtr> m_faceToElMap;

    TiXmlElement *m_xmlGeom;

    CompositeOrdering m_compOrder;
    BndRegionOrdering m_bndRegOrder;

    struct GeomRTree;
    std::unique_ptr<GeomRTree> m_boundingBoxTree;
    MovementSharedPtr m_movement;
};

typedef std::shared_ptr<MeshGraph> MeshGraphSharedPtr;
typedef LibUtilities::NekFactory<std::string, MeshGraph> MeshGraphFactory;

SPATIAL_DOMAINS_EXPORT MeshGraphFactory &GetMeshGraphFactory();

/**
 *
 */
void MeshGraph::SetExpansionInfo(const std::string variable,
                                 ExpansionInfoMapShPtr &exp)
{
    if (m_expansionMapShPtrMap.count(variable) != 0)
    {
        ASSERTL0(
            false,
            (std::string("ExpansionInfo field is already set for variable ") +
             variable)
                .c_str());
    }
    else
    {
        m_expansionMapShPtrMap[variable] = exp;
    }
}

/**
 *
 */
void MeshGraph::SetSession(LibUtilities::SessionReaderSharedPtr pSession)
{
    m_session = pSession;
}

/**
 *
 */
inline bool MeshGraph::SameExpansionInfo(const std::string var1,
                                         const std::string var2)
{
    ExpansionInfoMapShPtr expVec1 = m_expansionMapShPtrMap.find(var1)->second;
    ExpansionInfoMapShPtr expVec2 = m_expansionMapShPtrMap.find(var2)->second;

    if (expVec1.get() == expVec2.get())
    {
        return true;
    }

    return false;
}

/**
 *
 */
inline bool MeshGraph::ExpansionInfoDefined(const std::string var)
{
    return m_expansionMapShPtrMap.count(var);
}

} // namespace SpatialDomains

} // namespace Nektar

#endif
