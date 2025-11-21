////////////////////////////////////////////////////////////////////////////////
//
//  File: MeshGraph.cpp
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

#include <LibUtilities/BasicUtils/CompressData.h>
#include <LibUtilities/BasicUtils/Equation.h>
#include <LibUtilities/BasicUtils/FieldIOXml.h>
#include <LibUtilities/BasicUtils/ParseUtils.h>
#include <SpatialDomains/MeshGraph.h>
#include <StdRegions/StdPrismExp.h>
#include <StdRegions/StdPyrExp.h>
#include <StdRegions/StdTetExp.h>
#include <StdRegions/StdTriExp.h>

#include <cstring>
#include <iomanip>
#include <sstream>

#include <SpatialDomains/MeshGraph.h>
#include <SpatialDomains/Movement/Movement.h>
#include <SpatialDomains/RefRegion.h>
#include <SpatialDomains/RefRegionCylinder.h>
#include <SpatialDomains/RefRegionLine.h>
#include <SpatialDomains/RefRegionParallelogram.h>
#include <SpatialDomains/RefRegionSphere.h>

#include <boost/geometry/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;

namespace Nektar::SpatialDomains
{

struct MeshGraph::GeomRTree
{
    typedef bg::model::point<NekDouble, 3, bg::cs::cartesian> BgPoint;
    typedef bg::model::box<BgPoint> BgBox;
    typedef std::pair<BgBox, int> BgRtreeValue;

    bg::index::rtree<BgRtreeValue, bg::index::rstar<16, 4>> m_bgTree;

    void InsertGeom(Geometry *const &geom)
    {
        std::array<NekDouble, 6> minMax = geom->GetBoundingBox();
        BgPoint ptMin(minMax[0], minMax[1], minMax[2]);
        BgPoint ptMax(minMax[3], minMax[4], minMax[5]);
        m_bgTree.insert(
            std::make_pair(BgBox(ptMin, ptMax), geom->GetGlobalID()));
    }
};

/**
 * Returns an instance of the MeshGraph factory, held as a singleton.
 */
MeshGraphFactory &GetMeshGraphFactory()
{
    static MeshGraphFactory instance;
    return instance;
}

MeshGraph::MeshGraph()
    : m_pointMapView(m_pointGeoms), m_segMapView(m_segGeoms),
      m_triMapView(m_triGeoms), m_quadMapView(m_quadGeoms),
      m_tetMapView(m_tetGeoms), m_pyrMapView(m_pyrGeoms),
      m_prismMapView(m_prismGeoms), m_hexMapView(m_hexGeoms)
{
    m_boundingBoxTree =
        std::unique_ptr<MeshGraph::GeomRTree>(new MeshGraph::GeomRTree());
    m_movement = std::make_shared<Movement>();
}

MeshGraph::~MeshGraph()
{
}

void MeshGraph::SetPartition(SpatialDomains::MeshGraphSharedPtr graph)
{
    m_meshPartitioned = true;

    m_meshDimension  = graph->GetMeshDimension();
    m_spaceDimension = graph->GetSpaceDimension();

    m_pointGeoms  = std::move(graph->m_pointGeoms);
    m_curvedFaces = std::move(graph->GetCurvedFaces());
    m_curvedEdges = std::move(graph->GetCurvedEdges());

    m_segGeoms   = std::move(graph->m_segGeoms);
    m_triGeoms   = std::move(graph->m_triGeoms);
    m_quadGeoms  = std::move(graph->m_quadGeoms);
    m_hexGeoms   = std::move(graph->m_hexGeoms);
    m_prismGeoms = std::move(graph->m_prismGeoms);
    m_pyrGeoms   = std::move(graph->m_pyrGeoms);
    m_tetGeoms   = std::move(graph->m_tetGeoms);

    // m_pointMapView = std::move(graph->m_pointMapView);
    // m_segMapView = std::move(graph->m_segMapView);
    // m_triMapView = std::move(graph->m_pointMapView);

    m_faceToElMap = graph->GetAllFaceToElMap();
}

void MeshGraph::FillGraph()
{
    ReadExpansionInfo();

    switch (m_meshDimension)
    {
        case 3:
        {
            for (auto &x : m_pyrGeoms)
            {
                x.second->Setup();
            }
            for (auto &x : m_prismGeoms)
            {
                x.second->Setup();
            }
            for (auto &x : m_tetGeoms)
            {
                x.second->Setup();
            }
            for (auto &x : m_hexGeoms)
            {
                x.second->Setup();
            }
        }
        break;
        case 2:
        {
            for (auto &x : m_triGeoms)
            {
                x.second->Setup();
            }
            for (auto &x : m_quadGeoms)
            {
                x.second->Setup();
            }
        }
        break;
        case 1:
        {
            for (auto &x : m_segGeoms)
            {
                x.second->Setup();
            }
        }
        break;
    }

    // Populate the movement object
    m_movement = MemoryManager<SpatialDomains::Movement>::AllocateSharedPtr(
        m_session, this);
}

void MeshGraph::FillBoundingBoxTree()
{
    m_boundingBoxTree->m_bgTree.clear();
    switch (m_meshDimension)
    {
        case 1:
            for (auto &x : m_segGeoms)
            {
                m_boundingBoxTree->InsertGeom(x.second.get());
            }
            break;
        case 2:
            for (auto &x : m_triGeoms)
            {
                m_boundingBoxTree->InsertGeom(x.second.get());
            }
            for (auto &x : m_quadGeoms)
            {
                m_boundingBoxTree->InsertGeom(x.second.get());
            }
            break;
        case 3:
            for (auto &x : m_tetGeoms)
            {
                m_boundingBoxTree->InsertGeom(x.second.get());
            }
            for (auto &x : m_prismGeoms)
            {
                m_boundingBoxTree->InsertGeom(x.second.get());
            }
            for (auto &x : m_pyrGeoms)
            {
                m_boundingBoxTree->InsertGeom(x.second.get());
            }
            for (auto &x : m_hexGeoms)
            {
                m_boundingBoxTree->InsertGeom(x.second.get());
            }
            break;
        default:
            ASSERTL0(false, "Unknown dim");
    }
}

std::vector<int> MeshGraph::GetElementsContainingPoint(PointGeom *p)
{
    if (m_boundingBoxTree->m_bgTree.empty())
    {
        FillBoundingBoxTree();
    }

    NekDouble x = 0.0;
    NekDouble y = 0.0;
    NekDouble z = 0.0;
    std::vector<GeomRTree::BgRtreeValue> matches;

    p->GetCoords(x, y, z);

    GeomRTree::BgBox b(GeomRTree::BgPoint(x, y, z),
                       GeomRTree::BgPoint(x, y, z));

    m_boundingBoxTree->m_bgTree.query(bg::index::intersects(b),
                                      std::back_inserter(matches));

    std::vector<int> vals(matches.size());

    for (int i = 0; i < matches.size(); ++i)
    {
        vals[i] = matches[i].second;
    }

    return vals;
}

int MeshGraph::GetNumElements()
{
    switch (m_meshDimension)
    {
        case 1:
        {
            return m_segGeoms.size();
        }
        break;
        case 2:
        {
            return m_triGeoms.size() + m_quadGeoms.size();
        }
        break;
        case 3:
        {
            return m_tetGeoms.size() + m_pyrGeoms.size() + m_prismGeoms.size() +
                   m_hexGeoms.size();
        }
    }

    return 0;
}

bool MeshGraph::CheckRange(Geometry2D &geom)
{
    bool returnval = true;

    if (m_domainRange != LibUtilities::NullDomainRangeShPtr)
    {
        int nverts  = geom.GetNumVerts();
        int coordim = geom.GetCoordim();

        // exclude elements outside x range if all vertices not in region
        if (m_domainRange->m_doXrange)
        {
            int ncnt_low = 0;
            int ncnt_up  = 0;
            for (int i = 0; i < nverts; ++i)
            {
                NekDouble xval = (*geom.GetVertex(i))[0];
                if (xval < m_domainRange->m_xmin)
                {
                    ncnt_low++;
                }

                if (xval > m_domainRange->m_xmax)
                {
                    ncnt_up++;
                }
            }

            // check for all verts to be less or greater than
            // range so that if element spans thin range then
            // it is still included
            if ((ncnt_up == nverts) || (ncnt_low == nverts))
            {
                returnval = false;
            }
        }

        // exclude elements outside y range if all vertices not in region
        if (m_domainRange->m_doYrange)
        {
            int ncnt_low = 0;
            int ncnt_up  = 0;
            for (int i = 0; i < nverts; ++i)
            {
                NekDouble yval = (*geom.GetVertex(i))[1];
                if (yval < m_domainRange->m_ymin)
                {
                    ncnt_low++;
                }

                if (yval > m_domainRange->m_ymax)
                {
                    ncnt_up++;
                }
            }

            // check for all verts to be less or greater than
            // range so that if element spans thin range then
            // it is still included
            if ((ncnt_up == nverts) || (ncnt_low == nverts))
            {
                returnval = false;
            }
        }

        if (coordim > 2)
        {
            // exclude elements outside z range if all vertices not in region
            if (m_domainRange->m_doZrange)
            {
                int ncnt_low = 0;
                int ncnt_up  = 0;

                for (int i = 0; i < nverts; ++i)
                {
                    NekDouble zval = (*geom.GetVertex(i))[2];

                    if (zval < m_domainRange->m_zmin)
                    {
                        ncnt_low++;
                    }

                    if (zval > m_domainRange->m_zmax)
                    {
                        ncnt_up++;
                    }
                }

                // check for all verts to be less or greater than
                // range so that if element spans thin range then
                // it is still included
                if ((ncnt_up == nverts) || (ncnt_low == nverts))
                {
                    returnval = false;
                }
            }
        }

        if (m_domainRange->m_compElmts == 2)
        {
            returnval = false;
            for (unsigned i = 0; i < geom.GetNumEdges(); ++i)
            {
                if (m_domainRange->m_traceIDs.count(geom.GetEid(i)))
                {
                    returnval = true;
                    break;
                }
            }
        }
    }
    return returnval;
}

/* Domain checker for 3D geometries */
bool MeshGraph::CheckRange(Geometry3D &geom)
{
    bool returnval = true;

    if (m_domainRange != LibUtilities::NullDomainRangeShPtr)
    {
        int nverts = geom.GetNumVerts();

        if (m_domainRange->m_doXrange)
        {
            int ncnt_low = 0;
            int ncnt_up  = 0;

            for (int i = 0; i < nverts; ++i)
            {
                NekDouble xval = (*geom.GetVertex(i))[0];
                if (xval < m_domainRange->m_xmin)
                {
                    ncnt_low++;
                }

                if (xval > m_domainRange->m_xmax)
                {
                    ncnt_up++;
                }
            }

            // check for all verts to be less or greater than
            // range so that if element spans thin range then
            // it is still included
            if ((ncnt_up == nverts) || (ncnt_low == nverts))
            {
                returnval = false;
            }
        }

        if (m_domainRange->m_doYrange)
        {
            int ncnt_low = 0;
            int ncnt_up  = 0;
            for (int i = 0; i < nverts; ++i)
            {
                NekDouble yval = (*geom.GetVertex(i))[1];
                if (yval < m_domainRange->m_ymin)
                {
                    ncnt_low++;
                }

                if (yval > m_domainRange->m_ymax)
                {
                    ncnt_up++;
                }
            }

            // check for all verts to be less or greater than
            // range so that if element spans thin range then
            // it is still included
            if ((ncnt_up == nverts) || (ncnt_low == nverts))
            {
                returnval = false;
            }
        }

        if (m_domainRange->m_doZrange)
        {
            int ncnt_low = 0;
            int ncnt_up  = 0;
            for (int i = 0; i < nverts; ++i)
            {
                NekDouble zval = (*geom.GetVertex(i))[2];

                if (zval < m_domainRange->m_zmin)
                {
                    ncnt_low++;
                }

                if (zval > m_domainRange->m_zmax)
                {
                    ncnt_up++;
                }
            }

            // check for all verts to be less or greater than
            // range so that if element spans thin range then
            // it is still included
            if ((ncnt_up == nverts) || (ncnt_low == nverts))
            {
                returnval = false;
            }
        }

        if (m_domainRange->m_checkShape)
        {
            if (geom.GetShapeType() != m_domainRange->m_shapeType)
            {
                returnval = false;
            }
        }

        if (m_domainRange->m_compElmts == 3)
        {
            returnval = false;

            for (unsigned i = 0; i < geom.GetNumFaces(); ++i)
            {
                if (m_domainRange->m_traceIDs.count(geom.GetFid(i)))
                {
                    returnval = true;
                    break;
                }
            }
        }
    }

    return returnval;
}

bool MeshGraph::CheckRange(MeshEntity &e)
{
    bool returnval = true;

    if (m_domainRange != LibUtilities::NullDomainRangeShPtr)
    {
        if (m_domainRange->m_doXrange || m_domainRange->m_doYrange ||
            m_domainRange->m_doZrange)
        {
            WARNINGL2(false, "Not able to use check with coordinates at "
                             "partitioning stage.");
        }

        if (m_domainRange->m_compElmts == m_meshDimension)
        {
            returnval = false;
            for (unsigned i = 0; i < e.list.size(); ++i)
            {
                if (m_domainRange->m_traceIDs.count(e.list[i]))
                {
                    returnval = true;
                    break;
                }
            }
        }
    }
    return returnval;
}

/**
 *
 */
Geometry *MeshGraph::GetCompositeItem(int whichComposite, int whichItem)
{
    Geometry *returnval = nullptr;
    bool error          = false;

    if (whichComposite >= 0 && whichComposite < int(m_meshComposites.size()))
    {
        if (whichItem >= 0 &&
            whichItem < int(m_meshComposites[whichComposite]->m_geomVec.size()))
        {
            returnval = m_meshComposites[whichComposite]->m_geomVec[whichItem];
        }
        else
        {
            error = true;
        }
    }
    else
    {
        error = true;
    }

    if (error)
    {
        std::ostringstream errStream;
        errStream << "Unable to access composite item [" << whichComposite
                  << "][" << whichItem << "].";

        std::string testStr = errStream.str();

        NEKERROR(ErrorUtil::efatal, testStr.c_str());
    }

    return returnval;
}

/**
 *
 */
void MeshGraph::GetCompositeList(const std::string &compositeStr,
                                 CompositeMap &compositeVector) const
{
    // Parse the composites into a list.
    std::vector<unsigned int> seqVector;
    bool parseGood =
        ParseUtils::GenerateSeqVector(compositeStr.c_str(), seqVector);

    ASSERTL0(
        parseGood && !seqVector.empty(),
        (std::string("Unable to read composite index range: ") + compositeStr)
            .c_str());

    std::vector<unsigned int> addedVector; // Vector of those composites already
                                           // added to compositeVector;
    for (auto iter = seqVector.begin(); iter != seqVector.end(); ++iter)
    {
        // Only add a new one if it does not already exist in vector.
        // Can't go back and delete with a vector, so prevent it from
        // being added in the first place.
        if (std::find(addedVector.begin(), addedVector.end(), *iter) ==
            addedVector.end())
        {

            // If the composite listed is not found and we are working
            // on a partitioned mesh, silently ignore it.
            if (m_meshComposites.find(*iter) == m_meshComposites.end() &&
                m_meshPartitioned)
            {
                continue;
            }

            addedVector.push_back(*iter);
            ASSERTL0(m_meshComposites.find(*iter) != m_meshComposites.end(),
                     "Composite not found.");
            CompositeSharedPtr composite = m_meshComposites.find(*iter)->second;

            if (composite)
            {
                compositeVector[*iter] = composite;
            }
            else
            {
                NEKERROR(ErrorUtil::ewarning,
                         ("Undefined composite: " + std::to_string(*iter)));
            }
        }
    }
}

/**
 *
 */
const ExpansionInfoMap &MeshGraph::GetExpansionInfo(const std::string variable)
{
    ExpansionInfoMapShPtr returnval;

    if (m_expansionMapShPtrMap.count(variable))
    {
        returnval = m_expansionMapShPtrMap.find(variable)->second;
    }
    else
    {
        if (m_expansionMapShPtrMap.count("DefaultVar") == 0)
        {
            NEKERROR(
                ErrorUtil::efatal,
                (std::string(
                     "Unable to find expansion vector definition for field: ") +
                 variable)
                    .c_str());
        }
        returnval = m_expansionMapShPtrMap.find("DefaultVar")->second;
        m_expansionMapShPtrMap[variable] = returnval;

        NEKERROR(
            ErrorUtil::ewarning,
            (std::string(
                 "Using Default variable expansion definition for field: ") +
             variable)
                .c_str());
    }

    return *returnval;
}

/**
 *
 */
ExpansionInfoShPtr MeshGraph::GetExpansionInfo(Geometry *geom,
                                               const std::string variable)
{
    ExpansionInfoMapShPtr expansionMap =
        m_expansionMapShPtrMap.find(variable)->second;

    auto iter = expansionMap->find(geom->GetGlobalID());
    ASSERTL1(iter != expansionMap->end(),
             "Could not find expansion " + std::to_string(geom->GetGlobalID()) +
                 " in expansion for variable " + variable);
    return iter->second;
}

/**
 *
 */
void MeshGraph::SetExpansionInfo(
    std::vector<LibUtilities::FieldDefinitionsSharedPtr> &fielddef)
{
    int i, j, k, cnt, id;
    Geometry *geom = nullptr;

    ExpansionInfoMapShPtr expansionMap;

    // Loop over fields and determine unique fields string and
    // declare whole expansion list
    for (i = 0; i < fielddef.size(); ++i)
    {
        for (j = 0; j < fielddef[i]->m_fields.size(); ++j)
        {
            std::string field = fielddef[i]->m_fields[j];
            if (m_expansionMapShPtrMap.count(field) == 0)
            {
                expansionMap                  = SetUpExpansionInfoMap();
                m_expansionMapShPtrMap[field] = expansionMap;

                // check to see if DefaultVar also not set and
                // if so assign it to this expansion
                if (m_expansionMapShPtrMap.count("DefaultVar") == 0)
                {
                    m_expansionMapShPtrMap["DefaultVar"] = expansionMap;
                }
            }
        }
    }

    // loop over all elements find the geometry shared ptr and
    // set up basiskey vector
    for (i = 0; i < fielddef.size(); ++i)
    {
        cnt                                        = 0;
        std::vector<std::string> fields            = fielddef[i]->m_fields;
        std::vector<unsigned int> nmodes           = fielddef[i]->m_numModes;
        std::vector<LibUtilities::BasisType> basis = fielddef[i]->m_basis;
        bool pointDef                              = fielddef[i]->m_pointsDef;
        bool numPointDef = fielddef[i]->m_numPointsDef;

        // Check points and numpoints
        std::vector<unsigned int> npoints            = fielddef[i]->m_numPoints;
        std::vector<LibUtilities::PointsType> points = fielddef[i]->m_points;

        bool UniOrder = fielddef[i]->m_uniOrder;

        for (j = 0; j < fielddef[i]->m_elementIDs.size(); ++j)
        {

            LibUtilities::BasisKeyVector bkeyvec;
            id = fielddef[i]->m_elementIDs[j];

            switch (fielddef[i]->m_shapeType)
            {
                case LibUtilities::eSegment:
                {
                    if (m_segGeoms.count(fielddef[i]->m_elementIDs[j]) == 0)
                    {
                        // skip element likely from parallel read
                        if (!UniOrder)
                        {
                            cnt++;
                            cnt += fielddef[i]->m_numHomogeneousDir;
                        }
                        continue;
                    }
                    geom = m_segGeoms[fielddef[i]->m_elementIDs[j]].get();

                    LibUtilities::PointsKey pkey(
                        nmodes[cnt] + 1, LibUtilities::eGaussLobattoLegendre);

                    if (numPointDef && pointDef)
                    {
                        const LibUtilities::PointsKey pkey1(npoints[cnt],
                                                            points[0]);
                        pkey = pkey1;
                    }
                    else if (!numPointDef && pointDef)
                    {
                        const LibUtilities::PointsKey pkey1(nmodes[cnt] + 1,
                                                            points[0]);
                        pkey = pkey1;
                    }
                    else if (numPointDef && !pointDef)
                    {
                        const LibUtilities::PointsKey pkey1(
                            npoints[cnt], LibUtilities::eGaussLobattoLegendre);
                        pkey = pkey1;
                    }

                    LibUtilities::BasisKey bkey(basis[0], nmodes[cnt], pkey);

                    if (!UniOrder)
                    {
                        cnt++;
                        cnt += fielddef[i]->m_numHomogeneousDir;
                    }
                    bkeyvec.push_back(bkey);
                }
                break;
                case LibUtilities::eTriangle:
                {
                    if (m_triGeoms.count(fielddef[i]->m_elementIDs[j]) == 0)
                    {
                        // skip element likely from parallel read
                        if (!UniOrder)
                        {
                            cnt += 2;
                            cnt += fielddef[i]->m_numHomogeneousDir;
                        }
                        continue;
                    }
                    geom = m_triGeoms[fielddef[i]->m_elementIDs[j]].get();

                    LibUtilities::PointsKey pkey(
                        nmodes[cnt] + 1, LibUtilities::eGaussLobattoLegendre);
                    if (numPointDef && pointDef)
                    {
                        const LibUtilities::PointsKey pkey2(npoints[cnt],
                                                            points[0]);
                        pkey = pkey2;
                    }
                    else if (!numPointDef && pointDef)
                    {
                        const LibUtilities::PointsKey pkey2(nmodes[cnt] + 1,
                                                            points[0]);
                        pkey = pkey2;
                    }
                    else if (numPointDef && !pointDef)
                    {
                        const LibUtilities::PointsKey pkey2(
                            npoints[cnt], LibUtilities::eGaussLobattoLegendre);
                        pkey = pkey2;
                    }
                    LibUtilities::BasisKey bkey(basis[0], nmodes[cnt], pkey);

                    bkeyvec.push_back(bkey);

                    LibUtilities::PointsKey pkey1(
                        nmodes[cnt + 1], LibUtilities::eGaussRadauMAlpha1Beta0);
                    if (numPointDef && pointDef)
                    {
                        const LibUtilities::PointsKey pkey2(npoints[cnt + 1],
                                                            points[1]);
                        pkey1 = pkey2;
                    }
                    else if (!numPointDef && pointDef)
                    {
                        const LibUtilities::PointsKey pkey2(nmodes[cnt + 1],
                                                            points[1]);
                        pkey1 = pkey2;
                    }
                    else if (numPointDef && !pointDef)
                    {
                        const LibUtilities::PointsKey pkey2(
                            npoints[cnt + 1],
                            LibUtilities::eGaussRadauMAlpha1Beta0);
                        pkey1 = pkey2;
                    }
                    LibUtilities::BasisKey bkey1(basis[1], nmodes[cnt + 1],
                                                 pkey1);
                    bkeyvec.push_back(bkey1);

                    if (!UniOrder)
                    {
                        cnt += 2;
                        cnt += fielddef[i]->m_numHomogeneousDir;
                    }
                }
                break;
                case LibUtilities::eQuadrilateral:
                {
                    if (m_quadGeoms.count(fielddef[i]->m_elementIDs[j]) == 0)
                    {
                        // skip element likely from parallel read
                        if (!UniOrder)
                        {
                            cnt += 2;
                            cnt += fielddef[i]->m_numHomogeneousDir;
                        }
                        continue;
                    }

                    geom = m_quadGeoms[fielddef[i]->m_elementIDs[j]].get();

                    for (int b = 0; b < 2; ++b)
                    {
                        LibUtilities::PointsKey pkey(
                            nmodes[cnt + b] + 1,
                            LibUtilities::eGaussLobattoLegendre);

                        if (numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                npoints[cnt + b], points[b]);
                            pkey = pkey2;
                        }
                        else if (!numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                nmodes[cnt + b] + 1, points[b]);
                            pkey = pkey2;
                        }
                        else if (numPointDef && !pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                npoints[cnt + b],
                                LibUtilities::eGaussLobattoLegendre);
                            pkey = pkey2;
                        }
                        LibUtilities::BasisKey bkey(basis[b], nmodes[cnt + b],
                                                    pkey);
                        bkeyvec.push_back(bkey);
                    }

                    if (!UniOrder)
                    {
                        cnt += 2;
                        cnt += fielddef[i]->m_numHomogeneousDir;
                    }
                }
                break;

                case LibUtilities::eTetrahedron:
                {
                    k = fielddef[i]->m_elementIDs[j];

                    // allow for possibility that fielddef is
                    // larger than m_graph which can happen in
                    // parallel runs
                    if (m_tetGeoms.count(k) == 0)
                    {
                        if (!UniOrder)
                        {
                            cnt += 3;
                        }
                        continue;
                    }
                    geom = m_tetGeoms[k].get();

                    {
                        LibUtilities::PointsKey pkey(
                            nmodes[cnt] + 1,
                            LibUtilities::eGaussLobattoLegendre);

                        if (numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(npoints[cnt],
                                                                points[0]);
                            pkey = pkey2;
                        }
                        else if (!numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(nmodes[cnt] + 1,
                                                                points[0]);
                            pkey = pkey2;
                        }
                        else if (numPointDef && !pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                npoints[cnt],
                                LibUtilities::eGaussLobattoLegendre);
                            pkey = pkey2;
                        }

                        LibUtilities::BasisKey bkey(basis[0], nmodes[cnt],
                                                    pkey);

                        bkeyvec.push_back(bkey);
                    }
                    {
                        LibUtilities::PointsKey pkey(
                            nmodes[cnt + 1],
                            LibUtilities::eGaussRadauMAlpha1Beta0);

                        if (numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                npoints[cnt + 1], points[1]);
                            pkey = pkey2;
                        }
                        else if (!numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                nmodes[cnt + 1] + 1, points[1]);
                            pkey = pkey2;
                        }
                        else if (numPointDef && !pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                npoints[cnt + 1],
                                LibUtilities::eGaussRadauMAlpha1Beta0);
                            pkey = pkey2;
                        }

                        LibUtilities::BasisKey bkey(basis[1], nmodes[cnt + 1],
                                                    pkey);

                        bkeyvec.push_back(bkey);
                    }

                    {
                        LibUtilities::PointsKey pkey(
                            nmodes[cnt + 2],
                            LibUtilities::eGaussRadauMAlpha2Beta0);

                        if (numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                npoints[cnt + 2], points[2]);
                            pkey = pkey2;
                        }
                        else if (!numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                nmodes[cnt + 2] + 1, points[2]);
                            pkey = pkey2;
                        }
                        else if (numPointDef && !pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                npoints[cnt + 2],
                                LibUtilities::eGaussRadauMAlpha1Beta0);
                            pkey = pkey2;
                        }

                        LibUtilities::BasisKey bkey(basis[2], nmodes[cnt + 2],
                                                    pkey);

                        bkeyvec.push_back(bkey);
                    }

                    if (!UniOrder)
                    {
                        cnt += 3;
                    }
                }
                break;
                case LibUtilities::ePrism:
                {
                    k = fielddef[i]->m_elementIDs[j];
                    if (m_prismGeoms.count(k) == 0)
                    {
                        if (!UniOrder)
                        {
                            cnt += 3;
                        }
                        continue;
                    }
                    geom = m_prismGeoms[k].get();

                    for (int b = 0; b < 2; ++b)
                    {
                        LibUtilities::PointsKey pkey(
                            nmodes[cnt + b] + 1,
                            LibUtilities::eGaussLobattoLegendre);

                        if (numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                npoints[cnt + b], points[b]);
                            pkey = pkey2;
                        }
                        else if (!numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                nmodes[cnt + b] + 1, points[b]);
                            pkey = pkey2;
                        }
                        else if (numPointDef && !pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                npoints[cnt + b],
                                LibUtilities::eGaussLobattoLegendre);
                            pkey = pkey2;
                        }

                        LibUtilities::BasisKey bkey(basis[b], nmodes[cnt + b],
                                                    pkey);
                        bkeyvec.push_back(bkey);
                    }

                    {
                        LibUtilities::PointsKey pkey(
                            nmodes[cnt + 2],
                            LibUtilities::eGaussRadauMAlpha1Beta0);

                        if (numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                npoints[cnt + 2], points[2]);
                            pkey = pkey2;
                        }
                        else if (!numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                nmodes[cnt + 2] + 1, points[2]);
                            pkey = pkey2;
                        }
                        else if (numPointDef && !pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                npoints[cnt + 2],
                                LibUtilities::eGaussLobattoLegendre);
                            pkey = pkey2;
                        }

                        LibUtilities::BasisKey bkey(basis[2], nmodes[cnt + 2],
                                                    pkey);
                        bkeyvec.push_back(bkey);
                    }

                    if (!UniOrder)
                    {
                        cnt += 3;
                    }
                }
                break;
                case LibUtilities::ePyramid:
                {
                    k = fielddef[i]->m_elementIDs[j];

                    if (m_pyrGeoms.count(k) == 0)
                    {
                        if (!UniOrder)
                        {
                            cnt += 3;
                        }
                        continue;
                    }

                    geom = m_pyrGeoms[k].get();

                    for (int b = 0; b < 2; ++b)
                    {
                        LibUtilities::PointsKey pkey(
                            nmodes[cnt + b] + 1,
                            LibUtilities::eGaussLobattoLegendre);

                        if (numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                npoints[cnt + b], points[b]);
                            pkey = pkey2;
                        }
                        else if (!numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                nmodes[cnt + b] + 1, points[b]);
                            pkey = pkey2;
                        }
                        else if (numPointDef && !pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                npoints[cnt + b],
                                LibUtilities::eGaussLobattoLegendre);
                            pkey = pkey2;
                        }

                        LibUtilities::BasisKey bkey(basis[b], nmodes[cnt + b],
                                                    pkey);
                        bkeyvec.push_back(bkey);
                    }

                    {
                        LibUtilities::PointsKey pkey(
                            nmodes[cnt + 2],
                            LibUtilities::eGaussRadauMAlpha2Beta0);

                        if (numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                npoints[cnt + 2], points[2]);
                            pkey = pkey2;
                        }
                        else if (!numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                nmodes[cnt + 2] + 1, points[2]);
                            pkey = pkey2;
                        }
                        else if (numPointDef && !pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                npoints[cnt + 2],
                                LibUtilities::eGaussLobattoLegendre);
                            pkey = pkey2;
                        }

                        LibUtilities::BasisKey bkey(basis[2], nmodes[cnt + 2],
                                                    pkey);
                        bkeyvec.push_back(bkey);
                    }

                    if (!UniOrder)
                    {
                        cnt += 3;
                    }
                }
                break;
                case LibUtilities::eHexahedron:
                {
                    k = fielddef[i]->m_elementIDs[j];
                    if (m_hexGeoms.count(k) == 0)
                    {
                        if (!UniOrder)
                        {
                            cnt += 3;
                        }
                        continue;
                    }

                    geom = m_hexGeoms[k].get();

                    for (int b = 0; b < 3; ++b)
                    {
                        LibUtilities::PointsKey pkey(
                            nmodes[cnt + b],
                            LibUtilities::eGaussLobattoLegendre);

                        if (numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                npoints[cnt + b], points[b]);
                            pkey = pkey2;
                        }
                        else if (!numPointDef && pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                nmodes[cnt + b] + 1, points[b]);
                            pkey = pkey2;
                        }
                        else if (numPointDef && !pointDef)
                        {
                            const LibUtilities::PointsKey pkey2(
                                npoints[cnt + b],
                                LibUtilities::eGaussLobattoLegendre);
                            pkey = pkey2;
                        }

                        LibUtilities::BasisKey bkey(basis[b], nmodes[cnt + b],
                                                    pkey);
                        bkeyvec.push_back(bkey);
                    }

                    if (!UniOrder)
                    {
                        cnt += 3;
                    }
                }
                break;
                default:
                    geom = nullptr;
                    ASSERTL0(false, "Need to set up for pyramid and prism 3D "
                                    "ExpansionInfo");
                    break;
            }

            for (k = 0; k < fields.size(); ++k)
            {
                expansionMap = m_expansionMapShPtrMap.find(fields[k])->second;
                if ((*expansionMap).find(id) != (*expansionMap).end())
                {
                    (*expansionMap)[id]->m_geomPtr        = geom;
                    (*expansionMap)[id]->m_basisKeyVector = bkeyvec;
                }
            }
        }
    }
}

/**
 *
 */
void MeshGraph::SetExpansionInfo(
    std::vector<LibUtilities::FieldDefinitionsSharedPtr> &fielddef,
    std::vector<std::vector<LibUtilities::PointsType>> &pointstype)
{
    int i, j, k, cnt, id;
    Geometry *geom = nullptr;

    ExpansionInfoMapShPtr expansionMap;

    // Loop over fields and determine unique fields string and
    // declare whole expansion list
    for (i = 0; i < fielddef.size(); ++i)
    {
        for (j = 0; j < fielddef[i]->m_fields.size(); ++j)
        {
            std::string field = fielddef[i]->m_fields[j];
            if (m_expansionMapShPtrMap.count(field) == 0)
            {
                expansionMap                  = SetUpExpansionInfoMap();
                m_expansionMapShPtrMap[field] = expansionMap;

                // check to see if DefaultVar also not set and
                // if so assign it to this expansion
                if (m_expansionMapShPtrMap.count("DefaultVar") == 0)
                {
                    m_expansionMapShPtrMap["DefaultVar"] = expansionMap;
                }
            }
        }
    }

    // loop over all elements find the geometry shared ptr and
    // set up basiskey vector
    for (i = 0; i < fielddef.size(); ++i)
    {
        cnt                                        = 0;
        std::vector<std::string> fields            = fielddef[i]->m_fields;
        std::vector<unsigned int> nmodes           = fielddef[i]->m_numModes;
        std::vector<LibUtilities::BasisType> basis = fielddef[i]->m_basis;
        bool UniOrder                              = fielddef[i]->m_uniOrder;

        for (j = 0; j < fielddef[i]->m_elementIDs.size(); ++j)
        {
            LibUtilities::BasisKeyVector bkeyvec;
            id = fielddef[i]->m_elementIDs[j];

            switch (fielddef[i]->m_shapeType)
            {
                case LibUtilities::eSegment:
                {
                    k = fielddef[i]->m_elementIDs[j];
                    ASSERTL0(m_segGeoms.find(k) != m_segGeoms.end(),
                             "Failed to find geometry with same global id.");
                    geom = m_segGeoms[k].get();

                    const LibUtilities::PointsKey pkey(nmodes[cnt],
                                                       pointstype[i][0]);
                    LibUtilities::BasisKey bkey(basis[0], nmodes[cnt], pkey);
                    if (!UniOrder)
                    {
                        cnt++;
                        cnt += fielddef[i]->m_numHomogeneousDir;
                    }
                    bkeyvec.push_back(bkey);
                }
                break;
                case LibUtilities::eTriangle:
                {
                    k = fielddef[i]->m_elementIDs[j];
                    ASSERTL0(m_triGeoms.find(k) != m_triGeoms.end(),
                             "Failed to find geometry with same global id.");
                    geom = m_triGeoms[k].get();
                    for (int b = 0; b < 2; ++b)
                    {
                        const LibUtilities::PointsKey pkey(nmodes[cnt + b],
                                                           pointstype[i][b]);
                        LibUtilities::BasisKey bkey(basis[b], nmodes[cnt + b],
                                                    pkey);
                        bkeyvec.push_back(bkey);
                    }

                    if (!UniOrder)
                    {
                        cnt += 2;
                        cnt += fielddef[i]->m_numHomogeneousDir;
                    }
                }
                break;
                case LibUtilities::eQuadrilateral:
                {
                    k = fielddef[i]->m_elementIDs[j];
                    ASSERTL0(m_quadGeoms.find(k) != m_quadGeoms.end(),
                             "Failed to find geometry with same global id");
                    geom = m_quadGeoms[k].get();

                    for (int b = 0; b < 2; ++b)
                    {
                        const LibUtilities::PointsKey pkey(nmodes[cnt + b],
                                                           pointstype[i][b]);
                        LibUtilities::BasisKey bkey(basis[b], nmodes[cnt + b],
                                                    pkey);
                        bkeyvec.push_back(bkey);
                    }

                    if (!UniOrder)
                    {
                        cnt += 2;
                        cnt += fielddef[i]->m_numHomogeneousDir;
                    }
                }
                break;
                case LibUtilities::eTetrahedron:
                {
                    k = fielddef[i]->m_elementIDs[j];
                    ASSERTL0(m_tetGeoms.find(k) != m_tetGeoms.end(),
                             "Failed to find geometry with same global id");
                    geom = m_tetGeoms[k].get();

                    for (int b = 0; b < 3; ++b)
                    {
                        const LibUtilities::PointsKey pkey(nmodes[cnt + b],
                                                           pointstype[i][b]);
                        LibUtilities::BasisKey bkey(basis[b], nmodes[cnt + b],
                                                    pkey);
                        bkeyvec.push_back(bkey);
                    }

                    if (!UniOrder)
                    {
                        cnt += 3;
                    }
                }
                break;
                case LibUtilities::ePyramid:
                {
                    k = fielddef[i]->m_elementIDs[j];
                    ASSERTL0(m_pyrGeoms.find(k) != m_pyrGeoms.end(),
                             "Failed to find geometry with same global id");
                    geom = m_pyrGeoms[k].get();

                    for (int b = 0; b < 3; ++b)
                    {
                        const LibUtilities::PointsKey pkey(nmodes[cnt + b],
                                                           pointstype[i][b]);
                        LibUtilities::BasisKey bkey(basis[b], nmodes[cnt + b],
                                                    pkey);
                        bkeyvec.push_back(bkey);
                    }

                    if (!UniOrder)
                    {
                        cnt += 3;
                    }
                }
                break;
                case LibUtilities::ePrism:
                {
                    k = fielddef[i]->m_elementIDs[j];
                    ASSERTL0(m_prismGeoms.find(k) != m_prismGeoms.end(),
                             "Failed to find geometry with same global id");
                    geom = m_prismGeoms[k].get();

                    for (int b = 0; b < 3; ++b)
                    {
                        const LibUtilities::PointsKey pkey(nmodes[cnt + b],
                                                           pointstype[i][b]);
                        LibUtilities::BasisKey bkey(basis[b], nmodes[cnt + b],
                                                    pkey);
                        bkeyvec.push_back(bkey);
                    }

                    if (!UniOrder)
                    {
                        cnt += 3;
                    }
                }
                break;
                case LibUtilities::eHexahedron:
                {
                    k = fielddef[i]->m_elementIDs[j];
                    ASSERTL0(m_hexGeoms.find(k) != m_hexGeoms.end(),
                             "Failed to find geometry with same global id");
                    geom = m_hexGeoms[k].get();

                    for (int b = 0; b < 3; ++b)
                    {
                        const LibUtilities::PointsKey pkey(nmodes[cnt + b],
                                                           pointstype[i][b]);
                        LibUtilities::BasisKey bkey(basis[b], nmodes[cnt + b],
                                                    pkey);
                        bkeyvec.push_back(bkey);
                    }

                    if (!UniOrder)
                    {
                        cnt += 3;
                    }
                }
                break;
                default:
                    ASSERTL0(false, "Need to set up for pyramid and prism 3D "
                                    "ExpansionInfo");
                    break;
            }

            for (k = 0; k < fields.size(); ++k)
            {
                expansionMap = m_expansionMapShPtrMap.find(fields[k])->second;
                if ((*expansionMap).find(id) != (*expansionMap).end())
                {
                    (*expansionMap)[id]->m_geomPtr        = geom;
                    (*expansionMap)[id]->m_basisKeyVector = bkeyvec;
                }
            }
        }
    }
}

/**
 * \brief Reset all points keys to have equispaced points with
 * optional arguemt of \a npoints which redefines how many
 * points are to be used.
 */
void MeshGraph::SetExpansionInfoToEvenlySpacedPoints(int npoints)
{
    // iterate over all defined expansions
    for (auto it = m_expansionMapShPtrMap.begin();
         it != m_expansionMapShPtrMap.end(); ++it)
    {
        for (auto expIt = it->second->begin(); expIt != it->second->end();
             ++expIt)
        {
            for (int i = 0; i < expIt->second->m_basisKeyVector.size(); ++i)
            {
                LibUtilities::BasisKey bkeyold =
                    expIt->second->m_basisKeyVector[i];

                int npts;

                if (npoints) // use input
                {
                    npts = npoints;
                }
                else
                {
                    npts = bkeyold.GetNumModes();
                }
                npts = std::max(npts, 2);

                const LibUtilities::PointsKey pkey(
                    npts, LibUtilities::ePolyEvenlySpaced);
                LibUtilities::BasisKey bkeynew(bkeyold.GetBasisType(),
                                               bkeyold.GetNumModes(), pkey);
                expIt->second->m_basisKeyVector[i] = bkeynew;
            }
        }
    }
}

/**
 * \brief Reset all points keys to have expansion order of \a
 *  nmodes.  we keep the point distribution the same and make
 *  the number of points the same difference from the number
 *  of modes as the original expansion definition
 */
void MeshGraph::SetExpansionInfoToNumModes(int nmodes)
{
    // iterate over all defined expansions
    for (auto it = m_expansionMapShPtrMap.begin();
         it != m_expansionMapShPtrMap.end(); ++it)
    {
        for (auto expIt = it->second->begin(); expIt != it->second->end();
             ++expIt)
        {
            for (int i = 0; i < expIt->second->m_basisKeyVector.size(); ++i)
            {
                LibUtilities::BasisKey bkeyold =
                    expIt->second->m_basisKeyVector[i];

                int npts =
                    nmodes + (bkeyold.GetNumPoints() - bkeyold.GetNumModes());

                const LibUtilities::PointsKey pkey(npts,
                                                   bkeyold.GetPointsType());
                LibUtilities::BasisKey bkeynew(bkeyold.GetBasisType(), nmodes,
                                               pkey);
                expIt->second->m_basisKeyVector[i] = bkeynew;
            }
        }
    }
}

/**
 * \brief Reset all points keys to have expansion order of \a
 *  nmodes.  we keep the point distribution the same and make
 *  the number of points the same difference from the number
 *  of modes as the original expansion definition
 */
void MeshGraph::SetExpansionInfoToPointOrder(int npts)
{
    // iterate over all defined expansions
    for (auto it = m_expansionMapShPtrMap.begin();
         it != m_expansionMapShPtrMap.end(); ++it)
    {
        for (auto expIt = it->second->begin(); expIt != it->second->end();
             ++expIt)
        {
            for (int i = 0; i < expIt->second->m_basisKeyVector.size(); ++i)
            {
                LibUtilities::BasisKey bkeyold =
                    expIt->second->m_basisKeyVector[i];

                const LibUtilities::PointsKey pkey(npts,
                                                   bkeyold.GetPointsType());

                LibUtilities::BasisKey bkeynew(bkeyold.GetBasisType(),
                                               bkeyold.GetNumModes(), pkey);
                expIt->second->m_basisKeyVector[i] = bkeynew;
            }
        }
    }
}

/**
 * For each element of shape given by \a shape in field \a
 * var, replace the current BasisKeyVector describing the
 * expansion in each dimension, with the one provided by \a
 * keys.
 *
 * @TODO: Allow selection of elements through a CompositeVector,
 * as well as by type.
 *
 * @param   shape     The shape of elements to be changed.
 * @param   keys      The new basis vector to apply to those elements.
 */
void MeshGraph::SetBasisKey(LibUtilities::ShapeType shape,
                            LibUtilities::BasisKeyVector &keys, std::string var)
{
    ExpansionInfoMapShPtr expansionMap =
        m_expansionMapShPtrMap.find(var)->second;
    ResetExpansionInfoToBasisKey(expansionMap, shape, keys);
}

void MeshGraph::ResetExpansionInfoToBasisKey(
    ExpansionInfoMapShPtr &expansionMap, LibUtilities::ShapeType shape,
    LibUtilities::BasisKeyVector &keys)
{
    for (auto elemIter = expansionMap->begin(); elemIter != expansionMap->end();
         ++elemIter)
    {
        if ((elemIter->second)->m_geomPtr->GetShapeType() == shape)
        {
            (elemIter->second)->m_basisKeyVector = keys;
        }
    }
}

/**
 *
 */
LibUtilities::BasisKeyVector MeshGraph::DefineBasisKeyFromExpansionType(
    Geometry *in, ExpansionType type, const int nummodes)
{
    LibUtilities::BasisKeyVector returnval;

    LibUtilities::ShapeType shape = in->GetShapeType();

    int quadoffset = 1;
    switch (type)
    {
        case eModified:
        case eModifiedGLLRadau10:
            quadoffset = 1;
            break;
        case eModifiedQuadPlus1:
            quadoffset = 2;
            break;
        case eModifiedQuadPlus2:
            quadoffset = 3;
            break;
        default:
            break;
    }

    switch (type)
    {
        case eModified:
        case eModifiedQuadPlus1:
        case eModifiedQuadPlus2:
        case eModifiedGLLRadau10:
        {
            switch (shape)
            {
                case LibUtilities::eSegment:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes + quadoffset,
                        LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eModified_A,
                                                nummodes, pkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eQuadrilateral:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes + quadoffset,
                        LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eModified_A,
                                                nummodes, pkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eHexahedron:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes + quadoffset,
                        LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eModified_A,
                                                nummodes, pkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eTriangle:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes + quadoffset,
                        LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eModified_A,
                                                nummodes, pkey);
                    returnval.push_back(bkey);

                    const LibUtilities::PointsKey pkey1(
                        nummodes + quadoffset - 1,
                        LibUtilities::eGaussRadauMAlpha1Beta0);
                    LibUtilities::BasisKey bkey1(LibUtilities::eModified_B,
                                                 nummodes, pkey1);

                    returnval.push_back(bkey1);
                }
                break;
                case LibUtilities::eTetrahedron:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes + quadoffset,
                        LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eModified_A,
                                                nummodes, pkey);
                    returnval.push_back(bkey);

                    const LibUtilities::PointsKey pkey1(
                        nummodes + quadoffset - 1,
                        LibUtilities::eGaussRadauMAlpha1Beta0);
                    LibUtilities::BasisKey bkey1(LibUtilities::eModified_B,
                                                 nummodes, pkey1);
                    returnval.push_back(bkey1);

                    if (type == eModifiedGLLRadau10)
                    {
                        const LibUtilities::PointsKey pkey2(
                            nummodes + quadoffset - 1,
                            LibUtilities::eGaussRadauMAlpha1Beta0);
                        LibUtilities::BasisKey bkey2(LibUtilities::eModified_C,
                                                     nummodes, pkey2);
                        returnval.push_back(bkey2);
                    }
                    else
                    {
                        const LibUtilities::PointsKey pkey2(
                            nummodes + quadoffset - 1,
                            LibUtilities::eGaussRadauMAlpha2Beta0);
                        LibUtilities::BasisKey bkey2(LibUtilities::eModified_C,
                                                     nummodes, pkey2);
                        returnval.push_back(bkey2);
                    }
                }
                break;
                case LibUtilities::ePyramid:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes + quadoffset,
                        LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eModified_A,
                                                nummodes, pkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);

                    const LibUtilities::PointsKey pkey1(
                        nummodes + quadoffset - 1,
                        LibUtilities::eGaussRadauMAlpha2Beta0);
                    LibUtilities::BasisKey bkey1(LibUtilities::eModifiedPyr_C,
                                                 nummodes, pkey1);
                    returnval.push_back(bkey1);
                }
                break;
                case LibUtilities::ePrism:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes + quadoffset,
                        LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eModified_A,
                                                nummodes, pkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);

                    const LibUtilities::PointsKey pkey1(
                        nummodes + quadoffset - 1,
                        LibUtilities::eGaussRadauMAlpha1Beta0);
                    LibUtilities::BasisKey bkey1(LibUtilities::eModified_B,
                                                 nummodes, pkey1);
                    returnval.push_back(bkey1);
                }
                break;
                default:
                {
                    ASSERTL0(false,
                             "Expansion not defined in switch for this shape");
                }
                break;
            }
        }
        break;

        case eGLL_Lagrange:
        {
            switch (shape)
            {
                case LibUtilities::eSegment:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes + 1, LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eGLL_Lagrange,
                                                nummodes, pkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eQuadrilateral:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes + 1, LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eGLL_Lagrange,
                                                nummodes, pkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eTriangle: // define with corrects points key
                    // and change to Ortho on construction
                    {
                        const LibUtilities::PointsKey pkey(
                            nummodes + 1, LibUtilities::eGaussLobattoLegendre);
                        LibUtilities::BasisKey bkey(LibUtilities::eGLL_Lagrange,
                                                    nummodes, pkey);
                        returnval.push_back(bkey);

                        const LibUtilities::PointsKey pkey1(
                            nummodes, LibUtilities::eGaussRadauMAlpha1Beta0);
                        LibUtilities::BasisKey bkey1(LibUtilities::eOrtho_B,
                                                     nummodes, pkey1);
                        returnval.push_back(bkey1);
                    }
                    break;
                case LibUtilities::eHexahedron:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes + 1, LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eGLL_Lagrange,
                                                nummodes, pkey);

                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                default:
                {
                    ASSERTL0(false,
                             "Expansion not defined in switch  for this shape");
                }
                break;
            }
        }
        break;

        case eGauss_Lagrange:
        {
            switch (shape)
            {
                case LibUtilities::eSegment:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eGaussGaussLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eGauss_Lagrange,
                                                nummodes, pkey);

                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eQuadrilateral:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eGaussGaussLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eGauss_Lagrange,
                                                nummodes, pkey);

                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eHexahedron:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eGaussGaussLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eGauss_Lagrange,
                                                nummodes, pkey);

                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                default:
                {
                    ASSERTL0(false,
                             "Expansion not defined in switch  for this shape");
                }
                break;
            }
        }
        break;

        case eOrthogonal:
        {
            switch (shape)
            {
                case LibUtilities::eSegment:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes + 1, LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eOrtho_A,
                                                nummodes, pkey);

                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eTriangle:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes + 1, LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eOrtho_A,
                                                nummodes, pkey);

                    returnval.push_back(bkey);

                    const LibUtilities::PointsKey pkey1(
                        nummodes, LibUtilities::eGaussRadauMAlpha1Beta0);
                    LibUtilities::BasisKey bkey1(LibUtilities::eOrtho_B,
                                                 nummodes, pkey1);

                    returnval.push_back(bkey1);
                }
                break;
                case LibUtilities::eQuadrilateral:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes + 1, LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eOrtho_A,
                                                nummodes, pkey);

                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eTetrahedron:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes + 1, LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eOrtho_A,
                                                nummodes, pkey);

                    returnval.push_back(bkey);

                    const LibUtilities::PointsKey pkey1(
                        nummodes, LibUtilities::eGaussRadauMAlpha1Beta0);
                    LibUtilities::BasisKey bkey1(LibUtilities::eOrtho_B,
                                                 nummodes, pkey1);

                    returnval.push_back(bkey1);

                    const LibUtilities::PointsKey pkey2(
                        nummodes, LibUtilities::eGaussRadauMAlpha2Beta0);
                    LibUtilities::BasisKey bkey2(LibUtilities::eOrtho_C,
                                                 nummodes, pkey2);
                }
                break;
                default:
                {
                    ASSERTL0(false,
                             "Expansion not defined in switch  for this shape");
                }
                break;
            }
        }
        break;

        case eGLL_Lagrange_SEM:
        {
            switch (shape)
            {
                case LibUtilities::eSegment:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eGLL_Lagrange,
                                                nummodes, pkey);

                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eQuadrilateral:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eGLL_Lagrange,
                                                nummodes, pkey);

                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eHexahedron:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey(LibUtilities::eGLL_Lagrange,
                                                nummodes, pkey);

                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                default:
                {
                    ASSERTL0(false,
                             "Expansion not defined in switch  for this shape");
                }
                break;
            }
        }
        break;

        case eFourier:
        {
            switch (shape)
            {
                case LibUtilities::eSegment:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eFourierEvenlySpaced);
                    LibUtilities::BasisKey bkey(LibUtilities::eFourier,
                                                nummodes, pkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eQuadrilateral:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eFourierEvenlySpaced);
                    LibUtilities::BasisKey bkey(LibUtilities::eFourier,
                                                nummodes, pkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eHexahedron:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eFourierEvenlySpaced);
                    LibUtilities::BasisKey bkey(LibUtilities::eFourier,
                                                nummodes, pkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                default:
                {
                    ASSERTL0(false,
                             "Expansion not defined in switch  for this shape");
                }
                break;
            }
        }
        break;

        case eFourierSingleMode:
        {
            switch (shape)
            {
                case LibUtilities::eSegment:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey(
                        LibUtilities::eFourierSingleMode, nummodes, pkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eQuadrilateral:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey(
                        LibUtilities::eFourierSingleMode, nummodes, pkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eHexahedron:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey(
                        LibUtilities::eFourierSingleMode, nummodes, pkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                default:
                {
                    ASSERTL0(false,
                             "Expansion not defined in switch  for this shape");
                }
                break;
            }
        }
        break;

        case eFourierHalfModeRe:
        {
            switch (shape)
            {
                case LibUtilities::eSegment:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey(
                        LibUtilities::eFourierHalfModeRe, nummodes, pkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eQuadrilateral:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey(
                        LibUtilities::eFourierHalfModeRe, nummodes, pkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eHexahedron:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey(
                        LibUtilities::eFourierHalfModeRe, nummodes, pkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                default:
                {
                    ASSERTL0(false,
                             "Expansion not defined in switch  for this shape");
                }
                break;
            }
        }
        break;

        case eFourierHalfModeIm:
        {
            switch (shape)
            {
                case LibUtilities::eSegment:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey(
                        LibUtilities::eFourierHalfModeIm, nummodes, pkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eQuadrilateral:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey(
                        LibUtilities::eFourierHalfModeIm, nummodes, pkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eHexahedron:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey(
                        LibUtilities::eFourierHalfModeIm, nummodes, pkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                default:
                {
                    ASSERTL0(false,
                             "Expansion not defined in switch  for this shape");
                }
                break;
            }
        }
        break;

        case eChebyshev:
        {
            switch (shape)
            {
                case LibUtilities::eSegment:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eGaussGaussChebyshev);
                    LibUtilities::BasisKey bkey(LibUtilities::eChebyshev,
                                                nummodes, pkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eQuadrilateral:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eGaussGaussChebyshev);
                    LibUtilities::BasisKey bkey(LibUtilities::eChebyshev,
                                                nummodes, pkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                case LibUtilities::eHexahedron:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eGaussGaussChebyshev);
                    LibUtilities::BasisKey bkey(LibUtilities::eChebyshev,
                                                nummodes, pkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                    returnval.push_back(bkey);
                }
                break;
                default:
                {
                    ASSERTL0(false,
                             "Expansion not defined in switch  for this shape");
                }
                break;
            }
        }
        break;

        case eFourierChebyshev:
        {
            switch (shape)
            {
                case LibUtilities::eQuadrilateral:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eFourierEvenlySpaced);
                    LibUtilities::BasisKey bkey(LibUtilities::eFourier,
                                                nummodes, pkey);
                    returnval.push_back(bkey);

                    const LibUtilities::PointsKey pkey1(
                        nummodes, LibUtilities::eGaussGaussChebyshev);
                    LibUtilities::BasisKey bkey1(LibUtilities::eChebyshev,
                                                 nummodes, pkey1);
                    returnval.push_back(bkey1);
                }
                break;
                default:
                {
                    ASSERTL0(false,
                             "Expansion not defined in switch  for this shape");
                }
                break;
            }
        }
        break;

        case eChebyshevFourier:
        {
            switch (shape)
            {
                case LibUtilities::eQuadrilateral:
                {
                    const LibUtilities::PointsKey pkey1(
                        nummodes, LibUtilities::eGaussGaussChebyshev);
                    LibUtilities::BasisKey bkey1(LibUtilities::eChebyshev,
                                                 nummodes, pkey1);
                    returnval.push_back(bkey1);

                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eFourierEvenlySpaced);
                    LibUtilities::BasisKey bkey(LibUtilities::eFourier,
                                                nummodes, pkey);
                    returnval.push_back(bkey);
                }
                break;
                default:
                {
                    ASSERTL0(false,
                             "Expansion not defined in switch  for this shape");
                }
                break;
            }
        }
        break;

        case eFourierModified:
        {
            switch (shape)
            {
                case LibUtilities::eQuadrilateral:
                {
                    const LibUtilities::PointsKey pkey(
                        nummodes, LibUtilities::eFourierEvenlySpaced);
                    LibUtilities::BasisKey bkey(LibUtilities::eFourier,
                                                nummodes, pkey);
                    returnval.push_back(bkey);

                    const LibUtilities::PointsKey pkey1(
                        nummodes + 1, LibUtilities::eGaussLobattoLegendre);
                    LibUtilities::BasisKey bkey1(LibUtilities::eModified_A,
                                                 nummodes, pkey1);
                    returnval.push_back(bkey1);
                }
                break;
                default:
                {
                    ASSERTL0(false,
                             "Expansion not defined in switch  for this shape");
                }
                break;
            }
        }
        break;

        default:
        {
            ASSERTL0(false, "Expansion type not defined");
        }
        break;
    }

    return returnval;
}

/**
 *
 */
LibUtilities::BasisKeyVector MeshGraph::DefineBasisKeyFromExpansionTypeHomo(
    Geometry *in, ExpansionType type_x, ExpansionType type_y,
    ExpansionType type_z, const int nummodes_x, const int nummodes_y,
    const int nummodes_z)
{
    LibUtilities::BasisKeyVector returnval;

    LibUtilities::ShapeType shape = in->GetShapeType();

    switch (shape)
    {
        case LibUtilities::eSegment:
        {
            ASSERTL0(false, "Homogeneous expansion not defined for this shape");
        }
        break;

        case LibUtilities::eQuadrilateral:
        {
            ASSERTL0(false, "Homogeneous expansion not defined for this shape");
        }
        break;

        case LibUtilities::eHexahedron:
        {
            switch (type_x)
            {
                case eFourier:
                {
                    const LibUtilities::PointsKey pkey1(
                        nummodes_x, LibUtilities::eFourierEvenlySpaced);
                    LibUtilities::BasisKey bkey1(LibUtilities::eFourier,
                                                 nummodes_x, pkey1);
                    returnval.push_back(bkey1);
                }
                break;

                case eFourierSingleMode:
                {
                    const LibUtilities::PointsKey pkey1(
                        nummodes_x, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey1(
                        LibUtilities::eFourierSingleMode, nummodes_x, pkey1);
                    returnval.push_back(bkey1);
                }
                break;

                case eFourierHalfModeRe:
                {
                    const LibUtilities::PointsKey pkey1(
                        nummodes_x, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey1(
                        LibUtilities::eFourierHalfModeRe, nummodes_x, pkey1);
                    returnval.push_back(bkey1);
                }
                break;

                case eFourierHalfModeIm:
                {
                    const LibUtilities::PointsKey pkey1(
                        nummodes_x, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey1(
                        LibUtilities::eFourierHalfModeIm, nummodes_x, pkey1);
                    returnval.push_back(bkey1);
                }
                break;

                case eChebyshev:
                {
                    const LibUtilities::PointsKey pkey1(
                        nummodes_x, LibUtilities::eGaussGaussChebyshev);
                    LibUtilities::BasisKey bkey1(LibUtilities::eChebyshev,
                                                 nummodes_x, pkey1);
                    returnval.push_back(bkey1);
                }
                break;

                default:
                {
                    ASSERTL0(false, "Homogeneous expansion can be of Fourier "
                                    "or Chebyshev type only");
                }
                break;
            }

            switch (type_y)
            {
                case eFourier:
                {
                    const LibUtilities::PointsKey pkey2(
                        nummodes_y, LibUtilities::eFourierEvenlySpaced);
                    LibUtilities::BasisKey bkey2(LibUtilities::eFourier,
                                                 nummodes_y, pkey2);
                    returnval.push_back(bkey2);
                }
                break;

                case eFourierSingleMode:
                {
                    const LibUtilities::PointsKey pkey2(
                        nummodes_y, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey2(
                        LibUtilities::eFourierSingleMode, nummodes_y, pkey2);
                    returnval.push_back(bkey2);
                }
                break;

                case eFourierHalfModeRe:
                {
                    const LibUtilities::PointsKey pkey2(
                        nummodes_y, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey2(
                        LibUtilities::eFourierHalfModeRe, nummodes_y, pkey2);
                    returnval.push_back(bkey2);
                }
                break;

                case eFourierHalfModeIm:
                {
                    const LibUtilities::PointsKey pkey2(
                        nummodes_y, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey2(
                        LibUtilities::eFourierHalfModeIm, nummodes_y, pkey2);
                    returnval.push_back(bkey2);
                }
                break;

                case eChebyshev:
                {
                    const LibUtilities::PointsKey pkey2(
                        nummodes_y, LibUtilities::eGaussGaussChebyshev);
                    LibUtilities::BasisKey bkey2(LibUtilities::eChebyshev,
                                                 nummodes_y, pkey2);
                    returnval.push_back(bkey2);
                }
                break;

                default:
                {
                    ASSERTL0(false, "Homogeneous expansion can be of Fourier "
                                    "or Chebyshev type only");
                }
                break;
            }

            switch (type_z)
            {
                case eFourier:
                {
                    const LibUtilities::PointsKey pkey3(
                        nummodes_z, LibUtilities::eFourierEvenlySpaced);
                    LibUtilities::BasisKey bkey3(LibUtilities::eFourier,
                                                 nummodes_z, pkey3);
                    returnval.push_back(bkey3);
                }
                break;

                case eFourierSingleMode:
                {
                    const LibUtilities::PointsKey pkey3(
                        nummodes_z, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey3(
                        LibUtilities::eFourierSingleMode, nummodes_z, pkey3);
                    returnval.push_back(bkey3);
                }
                break;

                case eFourierHalfModeRe:
                {
                    const LibUtilities::PointsKey pkey3(
                        nummodes_z, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey3(
                        LibUtilities::eFourierHalfModeRe, nummodes_z, pkey3);
                    returnval.push_back(bkey3);
                }
                break;

                case eFourierHalfModeIm:
                {
                    const LibUtilities::PointsKey pkey3(
                        nummodes_z, LibUtilities::eFourierSingleModeSpaced);
                    LibUtilities::BasisKey bkey3(
                        LibUtilities::eFourierHalfModeIm, nummodes_z, pkey3);
                    returnval.push_back(bkey3);
                }
                break;

                case eChebyshev:
                {
                    const LibUtilities::PointsKey pkey3(
                        nummodes_z, LibUtilities::eGaussGaussChebyshev);
                    LibUtilities::BasisKey bkey3(LibUtilities::eChebyshev,
                                                 nummodes_z, pkey3);
                    returnval.push_back(bkey3);
                }
                break;

                default:
                {
                    ASSERTL0(false, "Homogeneous expansion can be of Fourier "
                                    "or Chebyshev type only");
                }
                break;
            }
        }
        break;

        case LibUtilities::eTriangle:
        {
            ASSERTL0(false, "Homogeneous expansion not defined for this shape");
        }
        break;

        case LibUtilities::eTetrahedron:
        {
            ASSERTL0(false, "Homogeneous expansion not defined for this shape");
        }
        break;

        default:
            ASSERTL0(false, "Expansion not defined in switch  for this shape");
            break;
    }

    return returnval;
}

/**
 * Generate a single vector of ExpansionInfo structs mapping global element
 * ID to a corresponding Geometry shared pointer and basis key.
 *
 * ExpansionInfo map ensures elements which appear in multiple composites
 * within the domain are only listed once.
 */
ExpansionInfoMapShPtr MeshGraph::SetUpExpansionInfoMap(void)
{
    ExpansionInfoMapShPtr returnval;
    returnval = MemoryManager<ExpansionInfoMap>::AllocateSharedPtr();

    for (auto &d : m_domain)
    {
        for (auto &compIter : d.second)
        {
            // regular elements first
            for (auto &x : compIter.second->m_geomVec)
            {
                if (x->CalcGeomType() != SpatialDomains::eDeformed)
                {
                    LibUtilities::BasisKeyVector def;
                    ExpansionInfoShPtr expansionElementShPtr =
                        MemoryManager<ExpansionInfo>::AllocateSharedPtr(x, def);
                    int id           = x->GetGlobalID();
                    (*returnval)[id] = expansionElementShPtr;
                }
            }
            // deformed elements
            for (auto &x : compIter.second->m_geomVec)
            {
                if (x->CalcGeomType() == SpatialDomains::eDeformed)
                {
                    LibUtilities::BasisKeyVector def;
                    ExpansionInfoShPtr expansionElementShPtr =
                        MemoryManager<ExpansionInfo>::AllocateSharedPtr(x, def);
                    int id           = x->GetGlobalID();
                    (*returnval)[id] = expansionElementShPtr;
                }
            }
        }
    }

    return returnval;
}

/**
 * @brief Returns a string representation of a composite.
 */
std::string MeshGraph::GetCompositeString(CompositeSharedPtr comp)
{
    if (comp->m_geomVec.size() == 0)
    {
        return "";
    }

    // Create a map that gets around the issue of mapping faces -> F and edges
    // -> E inside the tag.
    std::map<LibUtilities::ShapeType, std::pair<std::string, std::string>>
        compMap;
    compMap[LibUtilities::ePoint]         = std::make_pair("V", "V");
    compMap[LibUtilities::eSegment]       = std::make_pair("S", "E");
    compMap[LibUtilities::eQuadrilateral] = std::make_pair("Q", "F");
    compMap[LibUtilities::eTriangle]      = std::make_pair("T", "F");
    compMap[LibUtilities::eTetrahedron]   = std::make_pair("A", "A");
    compMap[LibUtilities::ePyramid]       = std::make_pair("P", "P");
    compMap[LibUtilities::ePrism]         = std::make_pair("R", "R");
    compMap[LibUtilities::eHexahedron]    = std::make_pair("H", "H");

    std::stringstream s;

    Geometry *firstGeom = comp->m_geomVec[0];
    int shapeDim        = firstGeom->GetShapeDim();
    std::string tag     = (shapeDim < m_meshDimension)
                              ? compMap[firstGeom->GetShapeType()].second
                              : compMap[firstGeom->GetShapeType()].first;

    std::vector<unsigned int> idxList;
    std::transform(comp->m_geomVec.begin(), comp->m_geomVec.end(),
                   std::back_inserter(idxList),
                   [](Geometry *geom) { return geom->GetGlobalID(); });

    s << " " << tag << "[" << ParseUtils::GenerateSeqString(idxList) << "] ";
    return s.str();
}

/**
 * @brief Refine the elements which has at least one vertex inside the
 *        surface region.
 *
 * @param expansionMap    shared pointer for the ExpansionInfoMap.
 * @param region          Object which holds the information provided by the
 *                        user. For example, the radius, coordinates, etc.
 * @param geomVecIter     pointer for the Geometry.
 */
void MeshGraph::PRefinementElmts(ExpansionInfoMapShPtr &expansionMap,
                                 RefRegion *&region, Geometry *geomVecIter)
{
    bool updateExpansion = false;
    Array<OneD, NekDouble> coords(m_spaceDimension, 0.0);

    for (int i = 0; i < geomVecIter->GetNumVerts(); ++i)
    {
        // Get coordinates from the vertex
        geomVecIter->GetVertex(i)->GetCoords(coords);
        updateExpansion = region->v_Contains(coords);

        // Update expansion
        // Change number of modes and number of points (if needed).
        if (updateExpansion)
        {
            // Information of the expansion for a specific element
            auto expInfoID = expansionMap->find(geomVecIter->GetGlobalID());
            if (m_useExpansionType)
            {
                std::vector<unsigned int> nModes = region->GetNumModes();
                (expInfoID->second)->m_basisKeyVector =
                    MeshGraph::DefineBasisKeyFromExpansionType(
                        geomVecIter,
                        (ExpansionType)(expInfoID->second)
                            ->m_basisKeyVector.begin()
                            ->GetBasisType(),
                        nModes[0]);
            }
            else
            {
                int cnt = 0;
                LibUtilities::BasisKeyVector updatedBasisKey;
                std::vector<unsigned int> nModes  = region->GetNumModes();
                std::vector<unsigned int> nPoints = region->GetNumPoints();
                for (auto basis = expInfoID->second->m_basisKeyVector.begin();
                     basis != expInfoID->second->m_basisKeyVector.end();
                     ++basis)
                {
                    // Generate Basis key using information
                    const LibUtilities::PointsKey pkey(nPoints[cnt],
                                                       basis->GetPointsType());
                    updatedBasisKey.push_back(LibUtilities::BasisKey(
                        basis->GetBasisType(), nModes[cnt], pkey));
                    cnt++;
                }
                (expInfoID->second)->m_basisKeyVector = updatedBasisKey;
            }
            updateExpansion = false;
        }
    }
}

/**
 * @brief Set the refinement information. This function selects the
 *        composites and the corresponding surface regions that must
 *        be used to refine the elements.
 *
 * @param expansionMap    shared pointer for the ExpansionInfoMap
 */
void MeshGraph::SetRefinementInfo(ExpansionInfoMapShPtr &expansionMap)
{
    // Loop over the refinement ids
    for (auto pRefinement = m_refComposite.begin();
         pRefinement != m_refComposite.end(); ++pRefinement)
    {
        // For each refinement id, there might be more than one composite,
        // since each refinement id can be related to more than one
        // composite.
        for (auto compVecIter = pRefinement->second.begin();
             compVecIter != pRefinement->second.end(); ++compVecIter)
        {
            for (auto geomVecIter = compVecIter->second->m_geomVec.begin();
                 geomVecIter != compVecIter->second->m_geomVec.end();
                 ++geomVecIter)
            {
                // Loop over the refinements provided by the user storage
                // in the m_refRegion in order to provide the correct
                // refinement region data to PRefinementElmts function.
                for (auto region = m_refRegion.begin();
                     region != m_refRegion.end(); ++region)
                {
                    // Check if the refID are the same in order to refine
                    // the region.
                    if (region->first == pRefinement->first)
                    {
                        // The geomVecIter corresponds the geometry information
                        // of the composite to be refined.
                        PRefinementElmts(expansionMap, region->second,
                                         *geomVecIter);
                    }
                }
            }
        }
    }
}

/**
 * @brief Read refinement information provided by the user in the xml file.
 *        In this function, it reads the reference id, the radius, the
 *        coordinates, the type of the method, number of modes, and number
 *        of quadrature points if necessary.
 */
void MeshGraph::ReadRefinementInfo()
{
    // Find the Refinement tag
    TiXmlElement *refinementTypes = m_session->GetElement("NEKTAR/REFINEMENTS");

    if (refinementTypes)
    {
        TiXmlElement *refinement = refinementTypes->FirstChildElement();
        ASSERTL0(refinement, "Unable to find entries in REFINEMENTS tag "
                             "in file");
        std::string refType = refinement->Value();

        if (refType == "R")
        {
            while (refinement)
            {
                std::vector<NekDouble> coord1Vector, coord2Vector;
                std::vector<unsigned int> nModesVector, nPointsVector;

                // Extract Refinement ID
                const char *idStr = refinement->Attribute("REF");
                ASSERTL0(idStr, "REF was not defined in REFINEMENT section "
                                "of input");

                unsigned id = std::stoul(idStr);

                // Extract Radius
                const char *radiusStr = refinement->Attribute("RADIUS");
                ASSERTL0(radiusStr, "RADIUS was not defined in REFINEMENT "
                                    "section of input");

                NekDouble radius = std::stod(radiusStr);

                // Extract Coordinate 1
                const char *c1Str = refinement->Attribute("COORDINATE1");
                ASSERTL0(c1Str, "COORDINATE1 was not defined in REFINEMENT"
                                "section of input");

                std::string coord1String = c1Str;
                bool valid =
                    ParseUtils::GenerateVector(coord1String, coord1Vector);
                ASSERTL0(valid, "Unable to correctly parse the axes "
                                "values for COORDINATE1");

                ASSERTL0(coord1Vector.size() == m_spaceDimension,
                         "Number of coordinates do not match the space "
                         "dimension for COORDINATE1");

                // Refinement Type
                const char *rType = refinement->Attribute("TYPE");
                ASSERTL0(rType, "TYPE was not defined in REFINEMENT "
                                "section of input");

                // Extract Coordinate 2
                const char *c2Str = refinement->Attribute("COORDINATE2");

                if (strcmp(rType, "STANDARD") == 0)
                {
                    ASSERTL0(c2Str, "COORDINATE2 was not defined in REFINEMENT "
                                    "section of input");

                    std::string coord2String = c2Str;
                    valid =
                        ParseUtils::GenerateVector(coord2String, coord2Vector);
                    ASSERTL0(valid, "Unable to correctly parse the axes "
                                    "values for COORDINATE2");
                    ASSERTL0(coord2Vector.size() == m_spaceDimension,
                             "Number of coordinates do not match the space "
                             "dimension for COORDINATE2");

                    // The STANDARD TYPE approach only accepts meshes that have
                    // the same dimension as the space dimension.
                    ASSERTL0(
                        m_spaceDimension == m_meshDimension,
                        "The mesh dimension must match the space dimension");
                }
                else if (strcmp(rType, "SPHERE") == 0)
                {
                    // COORDINATE2 is not necessary for this TYPE.
                    ASSERTL0(!c2Str, "COORDINATE2 should not be defined in "
                                     "REFINEMENT section of input for the "
                                     "SPHERE TYPE");

                    coord2Vector.clear();
                }
                else
                {
                    NEKERROR(Nektar::ErrorUtil::efatal,
                             "Invalid refinement type");
                }

                // Extract number of modes
                // Check if the expansion was defined individually
                if (m_useExpansionType == false)
                {
                    const char *nModesStr = refinement->Attribute("NUMMODES");
                    ASSERTL0(nModesStr, "NUMMODES was not defined in "
                                        "Refinement section of input");

                    std::string numModesStr = nModesStr;
                    valid =
                        ParseUtils::GenerateVector(numModesStr, nModesVector);
                    ASSERTL0(valid, "Unable to correctly parse the "
                                    "number of modes");

                    // Extract number of points
                    const char *nPointsStr = refinement->Attribute("NUMPOINTS");
                    ASSERTL0(nPointsStr, "NUMPOINTS was not defined in "
                                         "Refinement section of input");

                    std::string numPointsStr = nPointsStr;
                    valid =
                        ParseUtils::GenerateVector(numPointsStr, nPointsVector);
                    ASSERTL0(valid, "Unable to correctly parse the "
                                    "number of modes");
                }
                else // if m_useExpansionType=true
                {
                    const char *nModesStr = refinement->Attribute("NUMMODES");
                    ASSERTL0(nModesStr,
                             "NUMMODES was not defined in Refinement "
                             "section of input");

                    std::string numModesStr = nModesStr;
                    int n_modesRef;
                    if (m_session)
                    {
                        LibUtilities::Equation nummodesEqn(
                            m_session->GetInterpreter(), numModesStr);
                        n_modesRef = (int)nummodesEqn.Evaluate();
                    }
                    else
                    {
                        n_modesRef = std::stoi(numModesStr);
                    }
                    nModesVector.push_back(n_modesRef);
                    nPointsVector.clear(); // No points.
                }

                // Instantiate an object
                if (strcmp(rType, "STANDARD") == 0)
                {
                    switch (m_spaceDimension)
                    {
                        case 3:
                        {
                            // Polymorphism of object
                            // Instantiate RefRegionCylinder object
                            RefRegion *refInfo = new RefRegionCylinder(
                                m_spaceDimension, radius, coord1Vector,
                                coord2Vector, nModesVector, nPointsVector);
                            // Map: refinement ID, refinement region
                            // object
                            m_refRegion[id] = refInfo;
                            break;
                        }
                        case 2:
                        {
                            RefRegion *refInfo = new RefRegionParallelogram(
                                m_spaceDimension, radius, coord1Vector,
                                coord2Vector, nModesVector, nPointsVector);
                            m_refRegion[id] = refInfo;
                            break;
                        }
                        case 1:
                        {
                            RefRegion *refInfo = new RefRegionLine(
                                m_spaceDimension, radius, coord1Vector,
                                coord2Vector, nModesVector, nPointsVector);
                            m_refRegion[id] = refInfo;
                            break;
                        }
                    }
                }
                else
                {
                    RefRegion *refInfo = new RefRegionSphere(
                        m_spaceDimension, radius, coord1Vector, coord2Vector,
                        nModesVector, nPointsVector);
                    m_refRegion[id] = refInfo;
                }

                refinement = refinement->NextSiblingElement("R");
            }
        }
    }
    else
    {
        NEKERROR(Nektar::ErrorUtil::efatal,
                 "Unable to find REFINEMENTS tag in file");
    }
}

void MeshGraph::ReadExpansionInfo()
{
    // Find the Expansions tag
    TiXmlElement *expansionTypes = m_session->GetElement("NEKTAR/EXPANSIONS");
    LibUtilities::SessionReader::GetXMLElementTimeLevel(
        expansionTypes, m_session->GetTimeLevel());

    ASSERTL0(expansionTypes, "Unable to find EXPANSIONS tag in file.");

    if (expansionTypes)
    {
        // Find the Expansion type
        TiXmlElement *expansion = expansionTypes->FirstChildElement();
        ASSERTL0(expansion, "Unable to find entries in EXPANSIONS tag in "
                            "file.");
        std::string expType           = expansion->Value();
        std::vector<std::string> vars = m_session->GetVariables();

        if (expType == "E")
        {
            int i;
            m_expansionMapShPtrMap.clear();
            ExpansionInfoMapShPtr expansionMap;

            /// Expansiontypes will contain composite,
            /// nummodes, and expansiontype (eModified, or
            /// eOrthogonal) Or a full list of data of
            /// basistype, nummodes, pointstype, numpoints;

            /// Expansiontypes may also contain a list of
            /// fields that this expansion relates to. If this
            /// does not exist the variable is set to "DefaultVar".
            /// "DefaultVar" is used as the default for any
            /// variables not explicitly listed in FIELDS.

            // Collect all composites of the domain to control which
            // composites are defined for each variable.
            std::map<int, bool> domainCompList;
            for (auto &d : m_domain)
            {
                for (auto &c : d.second)
                {
                    domainCompList[c.first] = false;
                }
            }
            std::map<std::string, std::map<int, bool>> fieldDomainCompList;

            while (expansion)
            {
                // Extract Composites
                std::string compositeStr = expansion->Attribute("COMPOSITE");
                ASSERTL0(compositeStr.length() > 3,
                         "COMPOSITE must be specified in expansion "
                         "definition");
                int beg = compositeStr.find_first_of("[");
                int end = compositeStr.find_first_of("]");
                std::string compositeListStr =
                    compositeStr.substr(beg + 1, end - beg - 1);

                std::map<int, CompositeSharedPtr> compositeVector;
                GetCompositeList(compositeListStr, compositeVector);

                // Extract Fields if any
                const char *fStr = expansion->Attribute("FIELDS");
                std::vector<std::string> fieldStrings;

                if (fStr) // extract fields.
                {
                    std::string fieldStr = fStr;
                    bool valid = ParseUtils::GenerateVector(fieldStr.c_str(),
                                                            fieldStrings);
                    ASSERTL0(valid, "Unable to correctly parse the field "
                                    "string in ExpansionTypes.");

                    // see if field exists
                    if (m_expansionMapShPtrMap.count(fieldStrings[0]))
                    {
                        expansionMap =
                            m_expansionMapShPtrMap.find(fieldStrings[0])
                                ->second;
                    }
                    else
                    {
                        expansionMap = SetUpExpansionInfoMap();
                    }

                    // make sure all fields in this search point
                    // are asigned to same expansion map
                    for (i = 0; i < fieldStrings.size(); ++i)
                    {
                        if (vars.size() && std::count(vars.begin(), vars.end(),
                                                      fieldStrings[i]) == 0)
                        {
                            ASSERTL0(false, "Variable '" + fieldStrings[i] +
                                                "' defined in EXPANSIONS is not"
                                                " defined in VARIABLES.");
                        }

                        if (m_expansionMapShPtrMap.count(fieldStrings[i]) == 0)
                        {
                            m_expansionMapShPtrMap[fieldStrings[i]] =
                                expansionMap;

                            // set true to the composites where
                            // expansion is defined
                            fieldDomainCompList[fieldStrings[i]] =
                                domainCompList;
                            for (auto c = compositeVector.begin();
                                 c != compositeVector.end(); ++c)
                            {
                                fieldDomainCompList.find(fieldStrings[i])
                                    ->second.find(c->first)
                                    ->second = true;
                            }
                        }
                        else
                        {
                            for (auto c = compositeVector.begin();
                                 c != compositeVector.end(); ++c)
                            {
                                if (fieldDomainCompList.find(fieldStrings[i])
                                        ->second.find(c->first)
                                        ->second == false)
                                {
                                    fieldDomainCompList.find(fieldStrings[i])
                                        ->second.find(c->first)
                                        ->second = true;
                                }
                                else
                                {
                                    ASSERTL0(false,
                                             "Expansion vector for "
                                             "variable '" +
                                                 fieldStrings[i] +
                                                 "' is already setup for "
                                                 "C[" +
                                                 std::to_string(c->first) +
                                                 "].");
                                }
                            }
                            expansionMap =
                                m_expansionMapShPtrMap.find(fieldStrings[i])
                                    ->second;
                        }
                    }
                }
                else // If no FIELDS attribute, DefaultVar is genereted.
                {
                    if (m_expansionMapShPtrMap.count("DefaultVar") == 0)
                    {
                        expansionMap = SetUpExpansionInfoMap();
                        m_expansionMapShPtrMap["DefaultVar"] = expansionMap;

                        fieldDomainCompList["DefaultVar"] = domainCompList;
                        for (auto c = compositeVector.begin();
                             c != compositeVector.end(); ++c)
                        {
                            fieldDomainCompList.find("DefaultVar")
                                ->second.find(c->first)
                                ->second = true;
                        }
                    }
                    else
                    {
                        for (auto c = compositeVector.begin();
                             c != compositeVector.end(); ++c)
                        {
                            if (fieldDomainCompList.find("DefaultVar")
                                    ->second.find(c->first)
                                    ->second == false)
                            {
                                fieldDomainCompList.find("DefaultVar")
                                    ->second.find(c->first)
                                    ->second = true;
                            }
                            else
                            {
                                ASSERTL0(false, "Default expansion already "
                                                "defined for C[" +
                                                    std::to_string(c->first) +
                                                    "].");
                            }
                        }
                        expansionMap =
                            m_expansionMapShPtrMap.find("DefaultVar")->second;
                    }
                }

                /// Mandatory components...optional are to follow later.
                m_useExpansionType           = false;
                ExpansionType expansion_type = eNoExpansionType;
                int num_modes                = 0;

                LibUtilities::BasisKeyVector basiskeyvec;
                const char *tStr = expansion->Attribute("TYPE");

                if (tStr) // use type string to define expansion
                {
                    std::string typeStr       = tStr;
                    const std::string *begStr = kExpansionTypeStr;
                    const std::string *endStr =
                        kExpansionTypeStr + eExpansionTypeSize;
                    const std::string *expStr =
                        std::find(begStr, endStr, typeStr);

                    ASSERTL0(expStr != endStr, "Invalid expansion type.");
                    expansion_type = (ExpansionType)(expStr - begStr);

                    /// \todo solvers break the pattern 'instantiate
                    /// Session -> instantiate MeshGraph' and parse
                    /// command line arguments by themselves; one needs
                    /// to unify command line arguments handling.
                    /// Solvers tend to call MeshGraph::Read statically
                    /// -> m_session is not defined -> no info about
                    /// command line arguments presented
                    /// ASSERTL0(m_session != 0, "One needs to
                    /// instantiate SessionReader first");

                    const char *nStr = expansion->Attribute("NUMMODES");
                    ASSERTL0(nStr, "NUMMODES was not defined in EXPANSION "
                                   "section of input");
                    std::string nummodesStr = nStr;

                    // ASSERTL0(m_session,"Session should be defined to
                    // evaluate nummodes ");
                    if (m_session)
                    {
                        LibUtilities::Equation nummodesEqn(
                            m_session->GetInterpreter(), nummodesStr);
                        num_modes = (int)nummodesEqn.Evaluate();
                    }
                    else
                    {
                        num_modes = std::stoi(nummodesStr);
                    }

                    m_useExpansionType = true;
                }
                else // assume expansion is defined individually
                {
                    // Extract the attributes.
                    const char *bTypeStr = expansion->Attribute("BASISTYPE");
                    ASSERTL0(bTypeStr, "TYPE or BASISTYPE was not defined in "
                                       "EXPANSION section of input");
                    std::string basisTypeStr = bTypeStr;

                    // interpret the basis type string.
                    std::vector<std::string> basisStrings;
                    std::vector<LibUtilities::BasisType> basis;
                    bool valid = ParseUtils::GenerateVector(
                        basisTypeStr.c_str(), basisStrings);
                    ASSERTL0(valid,
                             "Unable to correctly parse the basis types.");
                    for (std::vector<std::string>::size_type i = 0;
                         i < basisStrings.size(); i++)
                    {
                        valid = false;
                        for (unsigned int j = 0;
                             j < LibUtilities::SIZE_BasisType; j++)
                        {
                            if (LibUtilities::BasisTypeMap[j] ==
                                basisStrings[i])
                            {
                                basis.push_back((LibUtilities::BasisType)j);
                                valid = true;
                                break;
                            }
                        }
                        ASSERTL0(valid, std::string("Unable to correctly "
                                                    "parse the basis type: ")
                                            .append(basisStrings[i])
                                            .c_str());
                    }
                    const char *nModesStr = expansion->Attribute("NUMMODES");
                    ASSERTL0(nModesStr, "NUMMODES was not defined in EXPANSION "
                                        "section of input");

                    std::string numModesStr = nModesStr;
                    std::vector<unsigned int> numModes;
                    valid = ParseUtils::GenerateVector(numModesStr.c_str(),
                                                       numModes);
                    ASSERTL0(valid, "Unable to correctly parse the "
                                    "number of modes.");
                    ASSERTL0(numModes.size() == basis.size(),
                             "information for num modes does not match the "
                             "number of basis");

                    const char *pTypeStr = expansion->Attribute("POINTSTYPE");
                    ASSERTL0(pTypeStr, "POINTSTYPE was not defined in "
                                       "EXPANSION section of input");
                    std::string pointsTypeStr = pTypeStr;
                    // interpret the points type string.
                    std::vector<std::string> pointsStrings;
                    std::vector<LibUtilities::PointsType> points;
                    valid = ParseUtils::GenerateVector(pointsTypeStr.c_str(),
                                                       pointsStrings);
                    ASSERTL0(valid,
                             "Unable to correctly parse the points types.");
                    for (std::vector<std::string>::size_type i = 0;
                         i < pointsStrings.size(); i++)
                    {
                        valid = false;
                        for (unsigned int j = 0;
                             j < LibUtilities::SIZE_PointsType; j++)
                        {
                            if (LibUtilities::kPointsTypeStr[j] ==
                                pointsStrings[i])
                            {
                                points.push_back((LibUtilities::PointsType)j);
                                valid = true;
                                break;
                            }
                        }
                        ASSERTL0(valid, std::string("Unable to correctly "
                                                    "parse the points type: ")
                                            .append(pointsStrings[i])
                                            .c_str());
                    }

                    const char *nPointsStr = expansion->Attribute("NUMPOINTS");
                    ASSERTL0(nPointsStr, "NUMPOINTS was not defined in "
                                         "EXPANSION section of input");
                    std::string numPointsStr = nPointsStr;
                    std::vector<unsigned int> numPoints;
                    valid = ParseUtils::GenerateVector(numPointsStr.c_str(),
                                                       numPoints);
                    ASSERTL0(valid, "Unable to correctly parse the "
                                    "number of points.");
                    ASSERTL0(numPoints.size() == numPoints.size(),
                             "information for num points does not match the "
                             "number of basis");

                    for (int i = 0; i < basis.size(); ++i)
                    {
                        // Generate Basis key  using information
                        const LibUtilities::PointsKey pkey(numPoints[i],
                                                           points[i]);
                        basiskeyvec.push_back(LibUtilities::BasisKey(
                            basis[i], numModes[i], pkey));
                    }
                }

                // Extract refinement id if any
                const char *refIDsStr = expansion->Attribute("REFIDS");
                if (refIDsStr)
                {
                    std::vector<NekDouble> refIDsVector;
                    std::string refIDsString = refIDsStr;
                    bool valid =
                        ParseUtils::GenerateVector(refIDsString, refIDsVector);
                    ASSERTL0(valid, "Unable to correctly parse the ids "
                                    "values of input");

                    // Map to link the refinement ids and composites
                    for (auto iter = refIDsVector.begin();
                         iter != refIDsVector.end(); ++iter)
                    {
                        m_refComposite[*iter] = compositeVector;
                    }
                    m_refFlag = true;
                }

                // Now have composite and basiskeys.  Cycle through
                // all composites for the geomShPtrs and set the modes
                // and types for the elements contained in the element
                // list.
                for (auto compVecIter = compositeVector.begin();
                     compVecIter != compositeVector.end(); ++compVecIter)
                {
                    for (auto geomVecIter =
                             compVecIter->second->m_geomVec.begin();
                         geomVecIter != compVecIter->second->m_geomVec.end();
                         ++geomVecIter)
                    {
                        auto x =
                            expansionMap->find((*geomVecIter)->GetGlobalID());
                        ASSERTL0(
                            x != expansionMap->end(),
                            "Expansion " +
                                std::to_string((*geomVecIter)->GetGlobalID()) +
                                " not found!!");
                        if (m_useExpansionType)
                        {
                            (x->second)->m_basisKeyVector =
                                MeshGraph::DefineBasisKeyFromExpansionType(
                                    *geomVecIter, expansion_type, num_modes);
                        }
                        else
                        {
                            ASSERTL0((*geomVecIter)->GetShapeDim() ==
                                         basiskeyvec.size(),
                                     " There is an incompatible expansion "
                                     "dimension with geometry dimension");
                            (x->second)->m_basisKeyVector = basiskeyvec;
                        }
                    }
                }

                expansion = expansion->NextSiblingElement("E");
            }

            // Check if all the domain has been defined for the existing
            // fields excluding DefaultVar. Fill the absent composites
            // of a field if the DefaultVar is defined for that
            // composite
            for (auto f = fieldDomainCompList.begin();
                 f != fieldDomainCompList.end(); ++f)
            {
                if (f->first != "DefaultVar")
                {
                    for (auto c = f->second.begin(); c != f->second.end(); ++c)
                    {
                        if (c->second == false &&
                            fieldDomainCompList.find("DefaultVar")
                                    ->second.find(c->first)
                                    ->second == true)
                        {
                            // Copy DefaultVar into the missing
                            // composite by cycling through the element
                            // list.
                            for (auto geomVecIter =
                                     m_meshComposites.find(c->first)
                                         ->second->m_geomVec.begin();
                                 geomVecIter != m_meshComposites.find(c->first)
                                                    ->second->m_geomVec.end();
                                 ++geomVecIter)
                            {
                                auto xDefaultVar =
                                    m_expansionMapShPtrMap.find("DefaultVar")
                                        ->second->find(
                                            (*geomVecIter)->GetGlobalID());

                                auto xField =
                                    m_expansionMapShPtrMap.find(f->first)
                                        ->second->find(
                                            (*geomVecIter)->GetGlobalID());

                                (xField->second)->m_basisKeyVector =
                                    (xDefaultVar->second)->m_basisKeyVector;
                            }
                            c->second = true;
                            NEKERROR(ErrorUtil::ewarning,
                                     (std::string("Using Default expansion "
                                                  "definition for "
                                                  "field '") +
                                      f->first +
                                      "' in composite "
                                      "C[" +
                                      std::to_string(c->first) + "].")
                                         .c_str());
                        }
                        ASSERTL0(c->second, "There is no expansion defined for "
                                            "variable '" +
                                                f->first + "' in C[" +
                                                std::to_string(c->first) +
                                                "].");
                    }
                }
            }
            // Ensure m_expansionMapShPtrMap has an entry for all
            // variables listed in CONDITIONS/VARIABLES section if
            // DefaultVar is defined.
            for (i = 0; i < vars.size(); ++i)
            {
                if (m_expansionMapShPtrMap.count(vars[i]) == 0)
                {
                    if (m_expansionMapShPtrMap.count("DefaultVar"))
                    {
                        expansionMap =
                            m_expansionMapShPtrMap.find("DefaultVar")->second;
                        m_expansionMapShPtrMap[vars[i]] = expansionMap;

                        NEKERROR(ErrorUtil::ewarning,
                                 (std::string("Using Default expansion "
                                              "definition for field "
                                              "'") +
                                  vars[i] + "'.")
                                     .c_str());
                    }
                    else
                    {
                        ASSERTL0(false, "Variable '" + vars[i] +
                                            "' is missing"
                                            " in FIELDS attribute of EXPANSIONS"
                                            " tag.");
                    }
                }
            }
            // Define "DefaultVar" if not set by user.
            if (m_expansionMapShPtrMap.count("DefaultVar") == 0)
            {
                // Originally assignment was using
                // m_expansionMapShPtrMap["DefaultVar"] =
                // m_expansionMapShPtrMap.begin()->second; but on
                // certain macOS versions, this was causing a seg fault
                // so switched to storing addr first - see #271
                ExpansionInfoMapShPtr firstEntryAddr =
                    m_expansionMapShPtrMap.begin()->second;
                m_expansionMapShPtrMap["DefaultVar"] = firstEntryAddr;
            }

            // If the user defined the refinement section correctly,
            // the refinement secion is going to be uploaded and set.
            if (m_refFlag)
            {
                // Read refinement info
                ReadRefinementInfo();

                // Set refinement info in the given region
                // Verify and set the elements which needs p-refinement
                SetRefinementInfo(expansionMap);
            }
        }
        else if (expType == "H")
        {
            int i;
            m_expansionMapShPtrMap.clear();
            ExpansionInfoMapShPtr expansionMap;

            // Collect all composites of the domain to control which
            // composites are defined for each variable.
            std::map<int, bool> domainCompList;
            for (auto &d : m_domain)
            {
                for (auto &c : d.second)
                {
                    domainCompList[c.first] = false;
                }
            }
            std::map<std::string, std::map<int, bool>> fieldDomainCompList;

            while (expansion)
            {
                // Extract Composites
                std::string compositeStr = expansion->Attribute("COMPOSITE");
                ASSERTL0(compositeStr.length() > 3,
                         "COMPOSITE must be specified in expansion "
                         "definition");
                int beg = compositeStr.find_first_of("[");
                int end = compositeStr.find_first_of("]");
                std::string compositeListStr =
                    compositeStr.substr(beg + 1, end - beg - 1);

                std::map<int, CompositeSharedPtr> compositeVector;
                GetCompositeList(compositeListStr, compositeVector);

                // Extract Fields if any
                const char *fStr = expansion->Attribute("FIELDS");
                std::vector<std::string> fieldStrings;

                if (fStr) // extract fields.
                {
                    std::string fieldStr = fStr;
                    bool valid = ParseUtils::GenerateVector(fieldStr.c_str(),
                                                            fieldStrings);
                    ASSERTL0(valid, "Unable to correctly parse the field "
                                    "string in ExpansionTypes.");

                    // see if field exists
                    if (m_expansionMapShPtrMap.count(fieldStrings[0]))
                    {
                        expansionMap =
                            m_expansionMapShPtrMap.find(fieldStrings[0])
                                ->second;
                    }
                    else
                    {
                        expansionMap = SetUpExpansionInfoMap();
                    }

                    // make sure all fields in this search point
                    // are asigned to same expansion map
                    for (i = 0; i < fieldStrings.size(); ++i)
                    {
                        if (vars.size() && std::count(vars.begin(), vars.end(),
                                                      fieldStrings[i]) == 0)
                        {
                            ASSERTL0(false, "Variable '" + fieldStrings[i] +
                                                "' defined in EXPANSIONS is not"
                                                " defined in VARIABLES.");
                        }

                        if (m_expansionMapShPtrMap.count(fieldStrings[i]) == 0)
                        {
                            m_expansionMapShPtrMap[fieldStrings[i]] =
                                expansionMap;

                            // set true to the composites where
                            // expansion is defined
                            fieldDomainCompList[fieldStrings[i]] =
                                domainCompList;
                            for (auto c = compositeVector.begin();
                                 c != compositeVector.end(); ++c)
                            {
                                fieldDomainCompList.find(fieldStrings[i])
                                    ->second.find(c->first)
                                    ->second = true;
                            }
                        }
                        else
                        {
                            for (auto c = compositeVector.begin();
                                 c != compositeVector.end(); ++c)
                            {
                                if (fieldDomainCompList.find(fieldStrings[i])
                                        ->second.find(c->first)
                                        ->second == false)
                                {
                                    fieldDomainCompList.find(fieldStrings[i])
                                        ->second.find(c->first)
                                        ->second = true;
                                }
                                else
                                {
                                    ASSERTL0(false,
                                             "Expansion vector for "
                                             "variable '" +
                                                 fieldStrings[i] +
                                                 "' is already setup for "
                                                 "C[" +
                                                 std::to_string(c->first) +
                                                 "].");
                                }
                            }
                            expansionMap =
                                m_expansionMapShPtrMap.find(fieldStrings[i])
                                    ->second;
                        }
                    }
                }
                else // If no FIELDS attribute, DefaultVar is genereted.
                {
                    if (m_expansionMapShPtrMap.count("DefaultVar") == 0)
                    {
                        expansionMap = SetUpExpansionInfoMap();
                        m_expansionMapShPtrMap["DefaultVar"] = expansionMap;

                        fieldDomainCompList["DefaultVar"] = domainCompList;
                        for (auto c = compositeVector.begin();
                             c != compositeVector.end(); ++c)
                        {
                            fieldDomainCompList.find("DefaultVar")
                                ->second.find(c->first)
                                ->second = true;
                        }
                    }
                    else
                    {
                        for (auto c = compositeVector.begin();
                             c != compositeVector.end(); ++c)
                        {
                            if (fieldDomainCompList.find("DefaultVar")
                                    ->second.find(c->first)
                                    ->second == false)
                            {
                                fieldDomainCompList.find("DefaultVar")
                                    ->second.find(c->first)
                                    ->second = true;
                            }
                            else
                            {
                                ASSERTL0(false, "Default expansion already "
                                                "defined for C[" +
                                                    std::to_string(c->first) +
                                                    "].");
                            }
                        }
                        expansionMap =
                            m_expansionMapShPtrMap.find("DefaultVar")->second;
                    }
                }

                /// Mandatory components...optional are to follow later.
                ExpansionType expansion_type_x = eNoExpansionType;
                ExpansionType expansion_type_y = eNoExpansionType;
                ExpansionType expansion_type_z = eNoExpansionType;
                int num_modes_x                = 0;
                int num_modes_y                = 0;
                int num_modes_z                = 0;

                LibUtilities::BasisKeyVector basiskeyvec;

                const char *tStr_x = expansion->Attribute("TYPE-X");

                if (tStr_x) // use type string to define expansion
                {
                    std::string typeStr       = tStr_x;
                    const std::string *begStr = kExpansionTypeStr;
                    const std::string *endStr =
                        kExpansionTypeStr + eExpansionTypeSize;
                    const std::string *expStr =
                        std::find(begStr, endStr, typeStr);

                    ASSERTL0(expStr != endStr, "Invalid expansion type.");
                    expansion_type_x = (ExpansionType)(expStr - begStr);

                    const char *nStr = expansion->Attribute("NUMMODES-X");
                    ASSERTL0(nStr, "NUMMODES-X was not defined in EXPANSION "
                                   "section of input");
                    std::string nummodesStr = nStr;

                    // ASSERTL0(m_session,"Session should be defined to
                    // evaluate nummodes ");

                    if (m_session)
                    {
                        LibUtilities::Equation nummodesEqn(
                            m_session->GetInterpreter(), nummodesStr);
                        num_modes_x = (int)nummodesEqn.Evaluate();
                    }
                    else
                    {
                        num_modes_x = std::stoi(nummodesStr);
                    }
                }

                const char *tStr_y = expansion->Attribute("TYPE-Y");

                if (tStr_y) // use type string to define expansion
                {
                    std::string typeStr       = tStr_y;
                    const std::string *begStr = kExpansionTypeStr;
                    const std::string *endStr =
                        kExpansionTypeStr + eExpansionTypeSize;
                    const std::string *expStr =
                        std::find(begStr, endStr, typeStr);

                    ASSERTL0(expStr != endStr, "Invalid expansion type.");
                    expansion_type_y = (ExpansionType)(expStr - begStr);

                    const char *nStr = expansion->Attribute("NUMMODES-Y");
                    ASSERTL0(nStr, "NUMMODES-Y was not defined in EXPANSION "
                                   "section of input");
                    std::string nummodesStr = nStr;

                    // ASSERTL0(m_session,"Session should be defined to
                    // evaluate nummodes ");
                    if (m_session)
                    {
                        LibUtilities::Equation nummodesEqn(
                            m_session->GetInterpreter(), nummodesStr);
                        num_modes_y = (int)nummodesEqn.Evaluate();
                    }
                    else
                    {
                        num_modes_y = std::stoi(nummodesStr);
                    }
                }

                const char *tStr_z = expansion->Attribute("TYPE-Z");

                if (tStr_z) // use type string to define expansion
                {
                    std::string typeStr       = tStr_z;
                    const std::string *begStr = kExpansionTypeStr;
                    const std::string *endStr =
                        kExpansionTypeStr + eExpansionTypeSize;
                    const std::string *expStr =
                        std::find(begStr, endStr, typeStr);

                    ASSERTL0(expStr != endStr, "Invalid expansion type.");
                    expansion_type_z = (ExpansionType)(expStr - begStr);

                    const char *nStr = expansion->Attribute("NUMMODES-Z");
                    ASSERTL0(nStr, "NUMMODES-Z was not defined in EXPANSION "
                                   "section of input");
                    std::string nummodesStr = nStr;

                    // ASSERTL0(m_session,"Session should be defined to
                    // evaluate nummodes ");
                    if (m_session)
                    {
                        LibUtilities::Equation nummodesEqn(
                            m_session->GetInterpreter(), nummodesStr);
                        num_modes_z = (int)nummodesEqn.Evaluate();
                    }
                    else
                    {
                        num_modes_z = std::stoi(nummodesStr);
                    }
                }

                for (auto compVecIter = compositeVector.begin();
                     compVecIter != compositeVector.end(); ++compVecIter)
                {
                    for (auto geomVecIter =
                             compVecIter->second->m_geomVec.begin();
                         geomVecIter != compVecIter->second->m_geomVec.end();
                         ++geomVecIter)
                    {
                        for (auto expVecIter = expansionMap->begin();
                             expVecIter != expansionMap->end(); ++expVecIter)
                        {

                            (expVecIter->second)->m_basisKeyVector =
                                DefineBasisKeyFromExpansionTypeHomo(
                                    *geomVecIter, expansion_type_x,
                                    expansion_type_y, expansion_type_z,
                                    num_modes_x, num_modes_y, num_modes_z);
                        }
                    }
                }

                expansion = expansion->NextSiblingElement("H");
            }

            // Check if all the domain has been defined for the existing
            // fields excluding DefaultVar. Fill the absent composites
            // of a field if the DefaultVar is defined for that
            // composite
            for (auto f = fieldDomainCompList.begin();
                 f != fieldDomainCompList.end(); ++f)
            {
                if (f->first != "DefaultVar")
                {
                    for (auto c = f->second.begin(); c != f->second.end(); ++c)
                    {
                        if (c->second == false &&
                            fieldDomainCompList.find("DefaultVar")
                                    ->second.find(c->first)
                                    ->second == true)
                        {
                            // Copy DefaultVar into the missing
                            // composite by cycling through the element
                            // list.
                            for (auto geomVecIter =
                                     m_meshComposites.find(c->first)
                                         ->second->m_geomVec.begin();
                                 geomVecIter != m_meshComposites.find(c->first)
                                                    ->second->m_geomVec.end();
                                 ++geomVecIter)
                            {
                                auto xDefaultVar =
                                    m_expansionMapShPtrMap.find("DefaultVar")
                                        ->second->find(
                                            (*geomVecIter)->GetGlobalID());

                                auto xField =
                                    m_expansionMapShPtrMap.find(f->first)
                                        ->second->find(
                                            (*geomVecIter)->GetGlobalID());

                                (xField->second)->m_basisKeyVector =
                                    (xDefaultVar->second)->m_basisKeyVector;
                            }
                            c->second = true;
                            NEKERROR(ErrorUtil::ewarning,
                                     (std::string("Using Default expansion "
                                                  "definition for "
                                                  "field '") +
                                      f->first +
                                      "' in composite "
                                      "C[" +
                                      std::to_string(c->first) + "].")
                                         .c_str());
                        }
                        ASSERTL0(c->second, "There is no expansion defined for "
                                            "variable '" +
                                                f->first + "' in C[" +
                                                std::to_string(c->first) +
                                                "].");
                    }
                }
            }
            // Ensure m_expansionMapShPtrMap has an entry for all
            // variables listed in CONDITIONS/VARIABLES section if
            // DefaultVar is defined.
            for (i = 0; i < vars.size(); ++i)
            {
                if (m_expansionMapShPtrMap.count(vars[i]) == 0)
                {
                    if (m_expansionMapShPtrMap.count("DefaultVar"))
                    {
                        expansionMap =
                            m_expansionMapShPtrMap.find("DefaultVar")->second;
                        m_expansionMapShPtrMap[vars[i]] = expansionMap;

                        NEKERROR(ErrorUtil::ewarning,
                                 (std::string("Using Default expansion "
                                              "definition for field "
                                              "'") +
                                  vars[i] + "'.")
                                     .c_str());
                    }
                    else
                    {
                        ASSERTL0(false, "Variable '" + vars[i] +
                                            "' is missing"
                                            " in FIELDS attribute of EXPANSIONS"
                                            " tag.");
                    }
                }
            }
            // Define "DefaultVar" if not set by user.
            if (m_expansionMapShPtrMap.count("DefaultVar") == 0)
            {
                // Originally assignment was using
                // m_expansionMapShPtrMap["DefaultVar"] =
                // m_expansionMapShPtrMap.begin()->second; but on
                // certain macOS versions, This was causing a seg fault
                // so switched to storing addr first - see #271
                ExpansionInfoMapShPtr firstEntryAddr =
                    m_expansionMapShPtrMap.begin()->second;
                m_expansionMapShPtrMap["DefaultVar"] = firstEntryAddr;
            }
        }
        else if (expType == "ELEMENTS") // Reading a file with the
                                        // expansion definition
        {
            std::vector<LibUtilities::FieldDefinitionsSharedPtr> fielddefs;

            // This has to use the XML reader since we are treating the
            // already parsed XML as a standard FLD file.
            std::shared_ptr<LibUtilities::FieldIOXml> f =
                std::make_shared<LibUtilities::FieldIOXml>(m_session->GetComm(),
                                                           false);
            f->ImportFieldDefs(
                LibUtilities::XmlDataSource::create(m_session->GetDocument()),
                fielddefs, true);
            std::cout << "    Number of elements: " << fielddefs.size()
                      << std::endl;
            SetExpansionInfo(fielddefs);
        }
        else if (expType == "F")
        {
            ASSERTL0(expansion->Attribute("FILE"),
                     "Attribute FILE expected for type F expansion");
            std::string filenameStr = expansion->Attribute("FILE");
            ASSERTL0(!filenameStr.empty(),
                     "A filename must be specified for the FILE "
                     "attribute of expansion");

            std::vector<LibUtilities::FieldDefinitionsSharedPtr> fielddefs;
            LibUtilities::FieldIOSharedPtr f =
                LibUtilities::FieldIO::CreateForFile(m_session, filenameStr);
            f->Import(filenameStr, fielddefs);
            SetExpansionInfo(fielddefs);
        }
        else
        {
            ASSERTL0(false, "Expansion type not defined");
        }
    }
}

GeometryLinkSharedPtr MeshGraph::GetElementsFromEdge(Geometry1D *edge)
{
    // Search tris and quads
    // Need to iterate through vectors because there may be multiple
    // occurrences.

    GeometryLinkSharedPtr ret =
        GeometryLinkSharedPtr(new std::vector<std::pair<Geometry *, int>>);

    TriGeom *triGeomPtr;
    QuadGeom *quadGeomPtr;

    for (auto &d : m_domain)
    {
        for (auto &compIter : d.second)
        {
            for (auto &geomIter : compIter.second->m_geomVec)
            {
                triGeomPtr  = static_cast<TriGeom *>(geomIter);
                quadGeomPtr = static_cast<QuadGeom *>(geomIter);

                if (triGeomPtr || quadGeomPtr)
                {
                    if (triGeomPtr)
                    {
                        for (int i = 0; i < triGeomPtr->GetNumEdges(); i++)
                        {
                            if (triGeomPtr->GetEdge(i)->GetGlobalID() ==
                                edge->GetGlobalID())
                            {
                                ret->push_back(std::make_pair(triGeomPtr, i));
                                break;
                            }
                        }
                    }
                    else if (quadGeomPtr)
                    {
                        for (int i = 0; i < quadGeomPtr->GetNumEdges(); i++)
                        {
                            if (quadGeomPtr->GetEdge(i)->GetGlobalID() ==
                                edge->GetGlobalID())
                            {
                                ret->push_back(std::make_pair(quadGeomPtr, i));
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    return ret;
}

GeometryLinkSharedPtr MeshGraph::GetElementsFromFace(Geometry2D *face)
{
    auto it = m_faceToElMap.find(face->GetGlobalID());

    ASSERTL0(it != m_faceToElMap.end(), "Unable to find corresponding face!");

    return it->second;
}

/**
 * @brief Given a 3D geometry object #element, populate the face to
 * element map #m_faceToElMap which maps faces to their corresponding
 * element(s).
 *
 * @param element  Element to process.
 * @param kNfaces  Number of faces of #element. Should be removed and
 * put into Geometry3D as a virtual member function.
 */
void MeshGraph::PopulateFaceToElMap(Geometry3D *element, int kNfaces)
{
    // Set up face -> element map
    for (int i = 0; i < kNfaces; ++i)
    {
        int faceId = element->GetFace(i)->GetGlobalID();

        // Search map to see if face already exists.
        auto it = m_faceToElMap.find(faceId);

        if (it == m_faceToElMap.end())
        {
            GeometryLinkSharedPtr tmp = GeometryLinkSharedPtr(
                new std::vector<std::pair<Geometry *, int>>);
            tmp->push_back(std::make_pair(element, i));
            m_faceToElMap[faceId] = tmp;
        }
        else
        {
            it->second->push_back(std::make_pair(element, i));
        }
    }
}

/**
 * @brief Create mesh entities for this graph.
 *
 * This function will create a map of all mesh entities of the current
 * graph, which can then be used within the mesh partitioner to
 * construct an appropriate partitioning.
 */
std::map<int, MeshEntity> MeshGraph::CreateMeshEntities()
{
    std::map<int, MeshEntity> elements;
    switch (m_meshDimension)
    {
        case 1:
        {
            for (auto &i : m_segGeoms)
            {
                MeshEntity e;
                e.id = e.origId = i.first;
                e.list.push_back(i.second->GetVertex(0)->GetGlobalID());
                e.list.push_back(i.second->GetVertex(1)->GetGlobalID());
                e.ghost        = false;
                elements[e.id] = e;
            }
        }
        break;
        case 2:
        {
            for (auto &i : m_triGeoms)
            {
                MeshEntity e;
                e.id = e.origId = i.first;
                e.list.push_back(i.second->GetEdge(0)->GetGlobalID());
                e.list.push_back(i.second->GetEdge(1)->GetGlobalID());
                e.list.push_back(i.second->GetEdge(2)->GetGlobalID());
                e.ghost        = false;
                elements[e.id] = e;
            }
            for (auto &i : m_quadGeoms)
            {
                MeshEntity e;
                e.id = e.origId = i.first;
                e.list.push_back(i.second->GetEdge(0)->GetGlobalID());
                e.list.push_back(i.second->GetEdge(1)->GetGlobalID());
                e.list.push_back(i.second->GetEdge(2)->GetGlobalID());
                e.list.push_back(i.second->GetEdge(3)->GetGlobalID());
                e.ghost        = false;
                elements[e.id] = e;
            }
        }
        break;
        case 3:
        {
            for (auto &i : m_tetGeoms)
            {
                MeshEntity e;
                e.id = e.origId = i.first;
                e.list.push_back(i.second->GetFace(0)->GetGlobalID());
                e.list.push_back(i.second->GetFace(1)->GetGlobalID());
                e.list.push_back(i.second->GetFace(2)->GetGlobalID());
                e.list.push_back(i.second->GetFace(3)->GetGlobalID());
                e.ghost        = false;
                elements[e.id] = e;
            }
            for (auto &i : m_pyrGeoms)
            {
                MeshEntity e;
                e.id = e.origId = i.first;
                e.list.push_back(i.second->GetFace(0)->GetGlobalID());
                e.list.push_back(i.second->GetFace(1)->GetGlobalID());
                e.list.push_back(i.second->GetFace(2)->GetGlobalID());
                e.list.push_back(i.second->GetFace(3)->GetGlobalID());
                e.list.push_back(i.second->GetFace(4)->GetGlobalID());
                e.ghost        = false;
                elements[e.id] = e;
            }
            for (auto &i : m_prismGeoms)
            {
                MeshEntity e;
                e.id = e.origId = i.first;
                e.list.push_back(i.second->GetFace(0)->GetGlobalID());
                e.list.push_back(i.second->GetFace(1)->GetGlobalID());
                e.list.push_back(i.second->GetFace(2)->GetGlobalID());
                e.list.push_back(i.second->GetFace(3)->GetGlobalID());
                e.list.push_back(i.second->GetFace(4)->GetGlobalID());
                e.ghost        = false;
                elements[e.id] = e;
            }
            for (auto &i : m_hexGeoms)
            {
                MeshEntity e;
                e.id = e.origId = i.first;
                e.list.push_back(i.second->GetFace(0)->GetGlobalID());
                e.list.push_back(i.second->GetFace(1)->GetGlobalID());
                e.list.push_back(i.second->GetFace(2)->GetGlobalID());
                e.list.push_back(i.second->GetFace(3)->GetGlobalID());
                e.list.push_back(i.second->GetFace(4)->GetGlobalID());
                e.list.push_back(i.second->GetFace(5)->GetGlobalID());
                e.ghost        = false;
                elements[e.id] = e;
            }
        }
        break;
    }

    return elements;
}

CompositeDescriptor MeshGraph::CreateCompositeDescriptor()
{
    CompositeDescriptor ret;

    for (auto &comp : m_meshComposites)
    {
        std::pair<LibUtilities::ShapeType, std::vector<int>> tmp;
        tmp.first = comp.second->m_geomVec[0]->GetShapeType();

        tmp.second.resize(comp.second->m_geomVec.size());
        for (size_t i = 0; i < tmp.second.size(); ++i)
        {
            tmp.second[i] = comp.second->m_geomVec[i]->GetGlobalID();
        }

        ret[comp.first] = tmp;
    }

    return ret;
}

void MeshGraph::SetDomainRange(LibUtilities::DomainRangeShPtr rng)
{
    m_domainRange = rng;
}

void MeshGraph::SetDomainRange(NekDouble xmin, NekDouble xmax, NekDouble ymin,
                               NekDouble ymax, NekDouble zmin, NekDouble zmax)
{
    m_domainRange->m_checkShape = false;

    if (m_domainRange == LibUtilities::NullDomainRangeShPtr)
    {
        m_domainRange =
            MemoryManager<LibUtilities::DomainRange>::AllocateSharedPtr();
        m_domainRange->m_doXrange = true;
    }

    m_domainRange->m_xmin = xmin;
    m_domainRange->m_xmax = xmax;

    if (ymin == NekConstants::kNekUnsetDouble)
    {
        m_domainRange->m_doYrange = false;
    }
    else
    {
        m_domainRange->m_doYrange = true;
        m_domainRange->m_ymin     = ymin;
        m_domainRange->m_ymax     = ymax;
    }

    if (zmin == NekConstants::kNekUnsetDouble)
    {
        m_domainRange->m_doZrange = false;
    }
    else
    {
        m_domainRange->m_doZrange = true;
        m_domainRange->m_zmin     = zmin;
        m_domainRange->m_zmax     = zmax;
    }
}

void MeshGraph::Clear()
{
    m_pointGeoms.clear();
    m_curvedEdges.clear();
    m_curvedFaces.clear();
    m_segGeoms.clear();
    m_triGeoms.clear();
    m_quadGeoms.clear();
    m_tetGeoms.clear();
    m_pyrGeoms.clear();
    m_prismGeoms.clear();
    m_hexGeoms.clear();
    m_meshComposites.clear();
    m_compositesLabels.clear();
    m_domain.clear();
    m_expansionMapShPtrMap.clear();
    m_faceToElMap.clear();
}

} // namespace Nektar::SpatialDomains
