////////////////////////////////////////////////////////////////////////////////
//
//  File: Zones.cpp
//
//  For more information, please see: http://www.nektar.info/
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  License for the specific language governing rights and limitations under
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
//  Description: Zones used in the non-conformal interfaces
//               and ALE implementations
//
////////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/ParseUtils.h>
#include <SpatialDomains/MeshGraph.h>
#include <SpatialDomains/Movement/Zones.h>
#include <tinyxml.h>

namespace Nektar::SpatialDomains
{

ZoneBase::ZoneBase(MovementType type, int indx, int domainID,
                   CompositeMap domain, int coordDim)
    : m_type(type), m_id(indx), m_domainID(domainID), m_domain(domain),
      m_coordDim(coordDim)
{
    for (auto &comp : domain)
    {
        for (auto &geom : comp.second->m_geomVec)
        {
            m_elements.emplace_back(geom);

            // Fill constituent elements (i.e. faces and edges)
            switch (geom->GetShapeDim())
            {
                case 3:
                    for (int i = 0; i < geom->GetNumFaces(); ++i)
                    {
                        m_constituentElements[2].insert(geom->GetFace(i));
                    }
                    /* fall through */
                case 2:
                    for (int i = 0; i < geom->GetNumEdges(); ++i)
                    {
                        m_constituentElements[1].insert(geom->GetEdge(i));
                    }
                    /* fall through */
                case 1:
                    m_constituentElements[0].insert(geom);
                    /* fall through */
                default:
                    break;
            }
        }
    }

    // The seenVerts/Edges/Faces keeps track of geometry so duplicates aren't
    // emplaced in to the storage vector
    std::unordered_set<int> seenVerts, seenEdges, seenFaces;

    // Fill verts/edges/faces vector storage that are to be moved each timestep
    for (auto &comp : m_domain)
    {
        for (auto &geom : comp.second->m_geomVec)
        {
            for (int i = 0; i < geom->GetNumVerts(); ++i)
            {
                PointGeom *vert = geom->GetVertex(i);

                if (seenVerts.find(vert->GetGlobalID()) != seenVerts.end())
                {
                    continue;
                }

                seenVerts.insert(vert->GetGlobalID());
                m_verts.emplace_back(vert);
            }

            for (int i = 0; i < geom->GetNumEdges(); ++i)
            {
                SegGeom *edge = static_cast<SegGeom *>(geom->GetEdge(i));

                if (seenEdges.find(edge->GetGlobalID()) != seenEdges.end())
                {
                    continue;
                }

                seenEdges.insert(edge->GetGlobalID());

                Curve *curve = edge->GetCurve();
                if (!curve)
                {
                    continue;
                }

                m_curves.emplace_back(curve);
            }

            for (int i = 0; i < geom->GetNumFaces(); ++i)
            {
                Geometry2D *face = static_cast<Geometry2D *>(geom->GetFace(i));

                if (seenFaces.find(face->GetGlobalID()) != seenFaces.end())
                {
                    continue;
                }

                seenFaces.insert(face->GetGlobalID());

                Curve *curve = face->GetCurve();
                if (!curve)
                {
                    continue;
                }

                m_curves.emplace_back(curve);
            }
        }
    }

    // Copy points so we know original positions.
    for (auto &pt : m_verts)
    {
        m_origVerts.emplace_back(*pt);
    }

    for (auto &curve : m_curves)
    {
        for (auto &pt : curve->m_points)
        {
            m_origVerts.emplace_back(*pt);
        }
    }
}

ZoneRotate::ZoneRotate(int id, int domainID, const CompositeMap &domain,
                       const int coordDim, const NekPoint<NekDouble> &origin,
                       const DNekVec &axis,
                       const LibUtilities::EquationSharedPtr &angularVelEqn,
                       const NekDouble rampTime, const NekDouble sector,
                       const Array<OneD, NekDouble> &base)
    : ZoneBase(MovementType::eRotate, id, domainID, domain, coordDim),
      m_origin(origin), m_axis(axis), m_angularVelEqn(angularVelEqn),
      m_rampTime(rampTime), m_sector(sector), m_base(base)
{
    // Construct rotation matrix
    m_W(0, 1) = -m_axis[2];
    m_W(0, 2) = m_axis[1];
    m_W(1, 0) = m_axis[2];
    m_W(1, 2) = -m_axis[0];
    m_W(2, 0) = -m_axis[1];
    m_W(2, 1) = m_axis[0];

    m_W2 = m_W * m_W;
}

void ZoneBase::ClearBoundingBoxes()
{
    // Clear bboxes (these will be regenerated next time GetBoundingBox is
    // called)
    for (auto &el : m_elements)
    {
        el->ClearBoundingBox();

        int nfaces = el->GetNumFaces();
        for (int i = 0; i < nfaces; ++i)
        {
            el->GetFace(i)->ClearBoundingBox();
        }

        int nedges = el->GetNumEdges();
        for (int i = 0; i < nedges; ++i)
        {
            el->GetEdge(i)->ClearBoundingBox();
        }
    }
}

NekDouble ZoneRotate::GetAngularVel(const NekDouble &time) const
{
    if (time < m_rampTime)
    {
        return m_angularVelEqn->Evaluate(0, 0, 0, time) * time / m_rampTime;
    }
    else
    {
        return m_angularVelEqn->Evaluate(0, 0, 0, time);
    }
}

// Calculate angle rotated at given time
NekDouble ZoneRotate::GetAngle(const NekDouble &time)
{
    // This works for a linear ramp
    if (time < m_rampTime)
    {
        m_angle = GetAngularVel(time) * (time) / 2;
    }
    else
    {
        m_angle = GetAngularVel(m_rampTime) * m_rampTime / 2 +
                  GetAngularVel(time) * (time - m_rampTime);
    }
    return m_angle;
}

// Calculate new location of points using Rodrigues formula
bool ZoneRotate::v_Move(NekDouble time)
{
    NekDouble angle = GetAngle(time);

    if (angle == 0.0)
    {
        return false;
    }

    // Identity matrix
    DNekMat rot(3, 3, 0.0);
    rot(0, 0) = 1.0;
    rot(1, 1) = 1.0;
    rot(2, 2) = 1.0;

    // Rodrigues' rotation formula in matrix form
    rot = rot + sin(angle) * m_W + (1 - cos(angle)) * m_W2;

    int cnt = 0;
    for (auto &vert : m_verts)
    {
        NekPoint<NekDouble> pnt = m_origVerts[cnt] - m_origin;
        DNekVec pntVec          = {pnt[0], pnt[1], pnt[2]};

        DNekVec newLoc = rot * pntVec;

        vert->UpdatePosition(newLoc(0) + m_origin[0], newLoc(1) + m_origin[1],
                             newLoc(2) + m_origin[2]);
        cnt++;
    }

    for (auto &curve : m_curves)
    {
        for (auto &vert : curve->m_points)
        {
            NekPoint<NekDouble> pnt = m_origVerts[cnt] - m_origin;
            DNekVec pntVec          = {pnt[0], pnt[1], pnt[2]};

            DNekVec newLoc = rot * pntVec;

            vert->UpdatePosition(newLoc(0) + m_origin[0],
                                 newLoc(1) + m_origin[1],
                                 newLoc(2) + m_origin[2]);
            cnt++;
        }
    }

    ClearBoundingBoxes();

    return true;
}

void ZoneRotate::Rotate(Array<OneD, NekDouble> &gloCoord,
                        const NekDouble &angle)
{
    // Identity matrix
    DNekMat rot(3, 3, 0.0);
    rot(0, 0) = 1.0;
    rot(1, 1) = 1.0;
    rot(2, 2) = 1.0;

    // Rodrigues' rotation formula in matrix form
    rot = rot + sin(angle) * m_W + (1 - cos(angle)) * m_W2;

    DNekVec pntVec = {gloCoord[0] - m_origin[0], gloCoord[1] - m_origin[1],
                      gloCoord[2] - m_origin[2]};

    DNekVec newLoc = rot * pntVec;

    gloCoord[0] = newLoc(0) + m_origin[0];
    gloCoord[1] = newLoc(1) + m_origin[1];
    gloCoord[2] = newLoc(2) + m_origin[2];
}

ZoneTranslate::ZoneTranslate(
    int id, int domainID, const CompositeMap &domain, const int coordDim,
    const Array<OneD, LibUtilities::EquationSharedPtr> &velocityEqns,
    const Array<OneD, LibUtilities::EquationSharedPtr> &displacementEqns)
    : ZoneBase(MovementType::eTranslate, id, domainID, domain, coordDim),
      m_velocityEqns(velocityEqns), m_displacementEqns(displacementEqns)
{
    int NumVerts = m_origVerts.size();
    // Bounding length
    m_ZoneLength = Array<OneD, NekDouble>(3);
    // Bounding box
    m_ZoneBox = Array<OneD, NekDouble>(6);
    // NekDouble minx, miny, minz, maxx, maxy, maxz;
    Array<OneD, NekDouble> min(3), max(3);
    Array<OneD, NekDouble> x(3);
    // Initialise max and min
    for (int j = 0; j < 3; ++j)
    {
        min[j] = std::numeric_limits<double>::max();
        max[j] = -std::numeric_limits<double>::max();
    }
    // Loop over all original vertexes to get the box and length
    for (int i = 0; i < NumVerts; ++i)
    {
        PointGeom p = m_origVerts[i];
        p.GetCoords(x[0], x[1], x[2]);
        for (int j = 0; j < 3; ++j)
        {
            min[j] = (x[j] < min[j] ? x[j] : min[j]);
            max[j] = (x[j] > max[j] ? x[j] : max[j]);
        }
    }
    for (int j = 0; j < 3; ++j)
    {
        m_ZoneBox[j]     = min[j];
        m_ZoneBox[j + 3] = max[j];
        m_ZoneLength[j]  = max[j] - min[j];
    }
}

std::vector<NekDouble> ZoneTranslate::GetVel(NekDouble &time) const
{
    std::vector<NekDouble> vel(m_coordDim);
    for (int i = 0; i < m_coordDim; ++i)
    {
        vel[i] = m_velocityEqns[i]->Evaluate(0, 0, 0, time);
    }

    return vel;
}

std::vector<NekDouble> ZoneTranslate::GetDisp(NekDouble &time)
{
    m_disp = std::vector<NekDouble>(m_coordDim);
    for (int i = 0; i < m_coordDim; ++i)
    {
        m_disp[i] = m_displacementEqns[i]->Evaluate(0, 0, 0, time);
    }
    return m_disp;
}

bool ZoneTranslate::v_Move(NekDouble time)
{
    auto disp = GetDisp(time);

    int cnt = 0;
    for (auto &vert : m_verts)
    {
        Array<OneD, NekDouble> newLoc(3, 0.0);
        auto pnt = m_origVerts[cnt];

        for (int i = 0; i < m_coordDim; ++i)
        {
            newLoc[i] = pnt(i) + disp[i];
        }

        vert->UpdatePosition(newLoc[0], newLoc[1], newLoc[2]);
        cnt++;
    }

    for (auto &curve : m_curves)
    {
        for (auto &vert : curve->m_points)
        {
            Array<OneD, NekDouble> newLoc(3, 0.0);
            auto pnt = m_origVerts[cnt];

            for (int i = 0; i < m_coordDim; ++i)
            {
                newLoc[i] = pnt(i) + disp[i];
            }

            vert->UpdatePosition(newLoc[0], newLoc[1], newLoc[2]);
            cnt++;
        }
    }

    ClearBoundingBoxes();

    return true;
}

bool ZoneFixed::v_Move([[maybe_unused]] NekDouble time)
{
    return false;
}

std::vector<NekDouble> ZoneFixed::v_GetDisp() const
{
    std::vector<NekDouble> disp(m_coordDim);
    for (int i = 0; i < m_coordDim; ++i)
    {
        disp[i] = 0.0;
    }

    return disp;
}
NekDouble ZoneFixed::v_GetAngle() const
{
    return 0.0;
}

} // namespace Nektar::SpatialDomains
