////////////////////////////////////////////////////////////////////////////////
//
//  File: QuadGeom.cpp
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
//
////////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Foundations/Interp.h>
#include <SpatialDomains/QuadGeom.h>

#include <SpatialDomains/Curve.hpp>
#include <SpatialDomains/GeomFactors.h>
#include <SpatialDomains/SegGeom.h>
#include <SpatialDomains/XmapFactory.hpp>
#include <StdRegions/StdQuadExp.h>

namespace Nektar::SpatialDomains
{

XmapFactory<StdRegions::StdQuadExp, 2> &GetStdQuadFactory()
{
    static XmapFactory<StdRegions::StdQuadExp, 2> factory;
    return factory;
}

QuadGeom::QuadGeom()
{
    m_shapeType = LibUtilities::eQuadrilateral;
}

QuadGeom::QuadGeom(const int id, std::array<SegGeom *, kNedges> edges,
                   Curve *curve)
    : Geometry2D(edges[0]->GetVertex(0)->GetCoordim(), curve)
{
    int j;

    m_shapeType = LibUtilities::eQuadrilateral;
    m_globalID  = id;

    /// Copy the edge pointers
    m_edges = edges;

    for (j = 0; j < kNedges; ++j)
    {
        m_eorient[j] =
            SegGeom::GetEdgeOrientation(*edges[j], *edges[(j + 1) % kNedges]);
        m_verts[j] =
            edges[j]->GetVertex(m_eorient[j] == StdRegions::eForwards ? 0 : 1);
    }

    for (j = 2; j < kNedges; ++j)
    {
        m_eorient[j] = m_eorient[j] == StdRegions::eBackwards
                           ? StdRegions::eForwards
                           : StdRegions::eBackwards;
    }

    m_coordim = edges[0]->GetVertex(0)->GetCoordim();
    ASSERTL0(m_coordim > 1, "Cannot call function with dim == 1");
}

QuadGeom::QuadGeom(const QuadGeom &in) : Geometry2D(in)
{
    // From Geometry
    m_shapeType = in.m_shapeType;
    m_globalID  = in.m_globalID;

    // From QuadGeom
    m_verts = in.m_verts;
    m_edges = in.m_edges;
    for (int i = 0; i < kNedges; i++)
    {
        m_eorient[i] = in.m_eorient[i];
    }
}

int QuadGeom::v_AllLeftCheck(const Array<OneD, const NekDouble> &gloCoord)
{
    int nc = 1, d0 = m_manifold[0], d1 = m_manifold[1];
    if (0 == m_edgeNormal.size())
    {
        m_edgeNormal = Array<OneD, Array<OneD, NekDouble>>(m_verts.size());
        Array<OneD, Array<OneD, NekDouble>> x(2);
        x[0] = Array<OneD, NekDouble>(3);
        x[1] = Array<OneD, NekDouble>(3);
        m_verts[0]->GetCoords(x[0]);
        int i0 = 1, i1 = 0, direction = 1;
        for (size_t i = 0; i < m_verts.size(); ++i)
        {
            i0 ^= 1;
            i1 ^= 1;
            m_verts[(i + 1) % m_verts.size()]->GetCoords(x[i1]);
            if (m_edges[i]->GetXmap()->GetBasis(0)->GetNumModes() > 2)
            {
                continue;
            }
            m_edgeNormal[i]    = Array<OneD, NekDouble>(2);
            m_edgeNormal[i][0] = x[i0][d1] - x[i1][d1];
            m_edgeNormal[i][1] = x[i1][d0] - x[i0][d0];
        }
        if (m_coordim == 3)
        {
            for (size_t i = 0; i < m_verts.size(); ++i)
            {
                if (m_edgeNormal[i].size() == 2)
                {
                    m_verts[i]->GetCoords(x[0]);
                    m_verts[(i + 2) % m_verts.size()]->GetCoords(x[1]);
                    if (m_edgeNormal[i][0] * (x[1][d0] - x[0][d0]) <
                        m_edgeNormal[i][1] * (x[0][d1] - x[1][d1]))
                    {
                        direction = -1;
                    }
                    break;
                }
            }
        }
        if (direction == -1)
        {
            for (size_t i = 0; i < m_verts.size(); ++i)
            {
                if (m_edgeNormal[i].size() == 2)
                {
                    m_edgeNormal[i][0] = -m_edgeNormal[i][0];
                    m_edgeNormal[i][1] = -m_edgeNormal[i][1];
                }
            }
        }
    }

    Array<OneD, NekDouble> vertex(3);
    for (size_t i = 0; i < m_verts.size(); ++i)
    {
        int i1 = (i + 1) % m_verts.size();
        if (m_verts[i]->GetGlobalID() < m_verts[i1]->GetGlobalID())
        {
            m_verts[i]->GetCoords(vertex);
        }
        else
        {
            m_verts[i1]->GetCoords(vertex);
        }
        if (m_edgeNormal[i].size() == 0)
        {
            nc = 0; // not sure
            continue;
        }
        if (m_edgeNormal[i][0] * (gloCoord[d0] - vertex[d0]) <
            m_edgeNormal[i][1] * (vertex[d1] - gloCoord[d1]))
        {
            return -1; // outside
        }
    }
    // 3D manifold needs to check the distance
    if (m_coordim == 3)
    {
        nc = 0;
    }
    // nc: 1 (side element), 0 (maybe inside), -1 (outside)
    return nc;
}

void QuadGeom::SetUpXmap()
{
    int order0 = std::max(m_edges[0]->GetXmap()->GetBasis(0)->GetNumModes(),
                          m_edges[2]->GetXmap()->GetBasis(0)->GetNumModes());
    int order1 = std::max(m_edges[1]->GetXmap()->GetBasis(0)->GetNumModes(),
                          m_edges[3]->GetXmap()->GetBasis(0)->GetNumModes());

    std::array<LibUtilities::BasisKey, 2> basis = {
        LibUtilities::BasisKey(
            LibUtilities::eModified_A, order0,
            LibUtilities::PointsKey(order0 + 1,
                                    LibUtilities::eGaussLobattoLegendre)),
        LibUtilities::BasisKey(
            LibUtilities::eModified_A, order1,
            LibUtilities::PointsKey(order1 + 1,
                                    LibUtilities::eGaussLobattoLegendre))};

    m_xmap = GetStdQuadFactory().CreateInstance(basis);
}

NekDouble QuadGeom::v_GetCoord(const int i,
                               const Array<OneD, const NekDouble> &Lcoord)
{
    ASSERTL1(m_state == ePtsFilled, "Geometry is not in physical space");

    Array<OneD, NekDouble> tmp(m_xmap->GetTotPoints());
    m_xmap->BwdTrans(m_coeffs[i], tmp);

    return m_xmap->PhysEvaluate(Lcoord, tmp);
}

StdRegions::Orientation QuadGeom::GetFaceOrientation(const QuadGeom &face1,
                                                     const QuadGeom &face2,
                                                     bool doRot, int dir,
                                                     NekDouble angle,
                                                     NekDouble tol)
{
    return GetFaceOrientation(face1.m_verts, face2.m_verts, doRot, dir, angle,
                              tol);
}

/**
 * Calculate the orientation of face2 to face1 (note this is
 * not face1 to face2!).
 */
StdRegions::Orientation QuadGeom::GetFaceOrientation(
    std::array<PointGeom *, 4> face1, std::array<PointGeom *, 4> face2,
    bool doRot, int dir, NekDouble angle, NekDouble tol)
{
    int i, j, vmap[4] = {-1, -1, -1, -1};

    if (doRot)
    {
        PointGeom rotPt;

        for (i = 0; i < 4; ++i)
        {
            rotPt.Rotate((*face1[i]), dir, angle);
            for (j = 0; j < 4; ++j)
            {
                if (rotPt.dist(*face2[j]) < tol)
                {
                    vmap[j] = i;
                    break;
                }
            }
        }
    }
    else
    {

        NekDouble x, y, z, x1, y1, z1, cx = 0.0, cy = 0.0, cz = 0.0;

        // For periodic faces, we calculate the vector between the centre
        // points of the two faces. (For connected faces this will be
        // zero). We can then use this to determine alignment later in the
        // algorithm.
        for (i = 0; i < 4; ++i)
        {
            cx += (*face2[i])(0) - (*face1[i])(0);
            cy += (*face2[i])(1) - (*face1[i])(1);
            cz += (*face2[i])(2) - (*face1[i])(2);
        }
        cx /= 4;
        cy /= 4;
        cz /= 4;

        // Now construct a mapping which takes us from the vertices of one
        // face to the other. That is, vertex j of face2 corresponds to
        // vertex vmap[j] of face1.
        for (i = 0; i < 4; ++i)
        {
            x = (*face1[i])(0);
            y = (*face1[i])(1);
            z = (*face1[i])(2);
            for (j = 0; j < 4; ++j)
            {
                x1 = (*face2[j])(0) - cx;
                y1 = (*face2[j])(1) - cy;
                z1 = (*face2[j])(2) - cz;
                if (sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y) +
                         (z1 - z) * (z1 - z)) < 1e-8)
                {
                    vmap[j] = i;
                    break;
                }
            }
        }
    }

    // Use the mapping to determine the eight alignment options between
    // faces.
    if (vmap[1] == (vmap[0] + 1) % 4)
    {
        switch (vmap[0])
        {
            case 0:
                return StdRegions::eDir1FwdDir1_Dir2FwdDir2;
                break;
            case 1:
                return StdRegions::eDir1BwdDir2_Dir2FwdDir1;
                break;
            case 2:
                return StdRegions::eDir1BwdDir1_Dir2BwdDir2;
                break;
            case 3:
                return StdRegions::eDir1FwdDir2_Dir2BwdDir1;
                break;
        }
    }
    else
    {
        switch (vmap[0])
        {
            case 0:
                return StdRegions::eDir1FwdDir2_Dir2FwdDir1;
                break;
            case 1:
                return StdRegions::eDir1BwdDir1_Dir2FwdDir2;
                break;
            case 2:
                return StdRegions::eDir1BwdDir2_Dir2BwdDir1;
                break;
            case 3:
                return StdRegions::eDir1FwdDir1_Dir2BwdDir2;
                break;
        }
    }

    ASSERTL0(false, "unable to determine face orientation");
    return StdRegions::eDir1FwdDir1_Dir2FwdDir2;
}

/**
 * Set up GeoFac for this geometry using Coord quadrature distribution
 */
GeomType QuadGeom::v_CalcGeomType()
{
    if (!m_setupState)
    {
        QuadGeom::v_Setup();
    }
    QuadGeom::v_FillGeom();

    GeomType Gtype = eRegular;

    // We will first check whether we have a regular or deformed
    // geometry. We will define regular as those cases where the
    // Jacobian and the metric terms of the derivative are constants
    // (i.e. not coordinate dependent)

    // Check to see if expansions are linear
    // If not linear => deformed geometry
    if ((m_xmap->GetBasisNumModes(0) != 2) ||
        (m_xmap->GetBasisNumModes(1) != 2))
    {
        Gtype = eDeformed;
    }

    // For linear expansions, the mapping from standard to local
    // element is given by the relation:
    // x_i = 0.25 * [ ( x_i^A + x_i^B + x_i^C + x_i^D)       +
    //                (-x_i^A + x_i^B + x_i^C - x_i^D)*xi_1  +
    //                (-x_i^A - x_i^B + x_i^C + x_i^D)*xi_2  +
    //                ( x_i^A - x_i^B + x_i^C - x_i^D)*xi_1*xi_2 ]
    //
    // The jacobian of the transformation and the metric terms
    // dxi_i/dx_j, involve only terms of the form dx_i/dxi_j (both
    // for coordim == 2 or 3). Inspecting the formula above, it can
    // be appreciated that the derivatives dx_i/dxi_j will be
    // constant, if the coefficient of the non-linear term is zero.
    //
    // That is why for regular geometry, we require
    //
    //     x_i^A - x_i^B + x_i^C - x_i^D = 0
    //
    // or equivalently
    //
    //     x_i^A - x_i^B = x_i^D - x_i^C
    //
    // This corresponds to quadrilaterals which are paralellograms.
    m_manifold    = Array<OneD, int>(m_coordim);
    m_manifold[0] = 0;
    m_manifold[1] = 1;
    if (m_coordim == 3)
    {
        PointGeom e01, e21, norm;
        e01.Sub(*m_verts[0], *m_verts[1]);
        e21.Sub(*m_verts[3], *m_verts[1]);
        norm.Mult(e01, e21);
        int tmpi   = 0;
        double tmp = std::fabs(norm[0]);
        if (tmp < fabs(norm[1]))
        {
            tmp  = fabs(norm[1]);
            tmpi = 1;
        }
        if (tmp < fabs(norm[2]))
        {
            tmpi = 2;
        }
        m_manifold[0] = (tmpi + 1) % 3;
        m_manifold[1] = (tmpi + 2) % 3;
        m_manifold[2] = (tmpi + 3) % 3;
    }

    if (Gtype == eRegular)
    {
        Array<OneD, Array<OneD, NekDouble>> verts(m_verts.size());
        for (int i = 0; i < m_verts.size(); ++i)
        {
            verts[i] = Array<OneD, NekDouble>(3);
            m_verts[i]->GetCoords(verts[i]);
        }
        // a00 + a01 xi1 + a02 xi2 + a03 xi1 xi2
        // a10 + a11 xi1 + a12 xi2 + a03 xi1 xi2
        m_isoParameter = Array<OneD, Array<OneD, NekDouble>>(2);
        for (int i = 0; i < 2; i++)
        {
            unsigned int d    = m_manifold[i];
            m_isoParameter[i] = Array<OneD, NekDouble>(4, 0.);
            // Karniadakis, Sherwin 2005, Appendix D
            NekDouble A          = verts[0][d];
            NekDouble B          = verts[1][d];
            NekDouble D          = verts[2][d];
            NekDouble C          = verts[3][d];
            m_isoParameter[i][0] = 0.25 * (A + B + C + D);  // 1
            m_isoParameter[i][1] = 0.25 * (-A + B - C + D); // xi1
            m_isoParameter[i][2] = 0.25 * (-A - B + C + D); // xi2
            m_isoParameter[i][3] = 0.25 * (A - B - C + D);  // xi1*xi2
            NekDouble tmp =
                fabs(m_isoParameter[i][1]) + fabs(m_isoParameter[i][2]);
            if (fabs(m_isoParameter[i][3]) > tmp * NekConstants::kNekZeroTol)
            {
                Gtype = eDeformed;
            }
        }
    }

    if (Gtype == eRegular)
    {
        v_CalculateInverseIsoParam();
    }
    else if (m_straightEdge)
    {
        PreSolveStraightEdge();
    }

    return Gtype;
}

GeomFactorsUniquePtr QuadGeom::v_GenGeomFactors(
    LibUtilities::PointsKeyVector &keyTgt)
{
    GeomType Gtype = CalcGeomType();

    return ObjPoolManager<GeomFactors>::AllocateUniquePtr(
        Gtype, m_coordim, m_xmap, m_coeffs, keyTgt);
}

/**
 * Note verts and edges are listed according to anticlockwise
 * convention but points in _coeffs have to be in array format from
 * left to right.
 */
void QuadGeom::v_FillGeom()
{
    // check to see if geometry structure is already filled
    if (m_state != ePtsFilled)
    {
        int i, j, k;
        int nEdgeCoeffs;

        if (m_curve)
        {
            int npts     = m_curve->m_points.size();
            int nEdgePts = (int)sqrt(static_cast<NekDouble>(npts));
            Array<OneD, NekDouble> tmp(npts);
            Array<OneD, NekDouble> tmp2(m_xmap->GetTotPoints());
            LibUtilities::PointsKey curveKey(nEdgePts, m_curve->m_ptype);

            // Sanity checks:
            // - Curved faces should have square number of points;
            // - Each edge should have sqrt(npts) points.
            ASSERTL0(nEdgePts * nEdgePts == npts,
                     "NUMPOINTS should be a square number in"
                     " quadrilteral " +
                         std::to_string(m_globalID));

            for (i = 0; i < kNedges; ++i)
            {
                ASSERTL0(m_edges[i]->GetXmap()->GetNcoeffs() == nEdgePts,
                         "Number of edge points does not correspond to "
                         "number of face points in quadrilateral " +
                             std::to_string(m_globalID));
            }

            for (i = 0; i < m_coordim; ++i)
            {
                for (j = 0; j < npts; ++j)
                {
                    tmp[j] = (m_curve->m_points[j]->GetPtr())[i];
                }

                // Interpolate m_curve points to GLL points
                LibUtilities::Interp2D(curveKey, curveKey, tmp,
                                       m_xmap->GetBasis(0)->GetPointsKey(),
                                       m_xmap->GetBasis(1)->GetPointsKey(),
                                       tmp2);

                // Forwards transform to get coefficient space.
                m_xmap->FwdTrans(tmp2, m_coeffs[i]);
            }
        }

        // Now fill in edges.
        Array<OneD, unsigned int> mapArray;
        Array<OneD, int> signArray;

        for (i = 0; i < kNedges; i++)
        {
            m_edges[i]->FillGeom();
            m_xmap->GetTraceToElementMap(i, mapArray, signArray, m_eorient[i]);

            nEdgeCoeffs = m_edges[i]->GetXmap()->GetNcoeffs();

            for (j = 0; j < m_coordim; j++)
            {
                for (k = 0; k < nEdgeCoeffs; k++)
                {
                    m_coeffs[j][mapArray[k]] =
                        signArray[k] * (m_edges[i]->GetCoeffs(j))[k];
                }
            }
        }

        m_state = ePtsFilled;
    }
}

void QuadGeom::PreSolveStraightEdge()
{
    int i0, i1, j1, j2;
    if (fabs(m_isoParameter[0][3]) >= fabs(m_isoParameter[1][3]))
    {
        i0 = 0;
        i1 = 1;
    }
    else
    {
        i1 = 0;
        i0 = 1;
        m_straightEdge |= 2;
    }
    NekDouble gamma = m_isoParameter[i1][3] / m_isoParameter[i0][3];
    std::vector<NekDouble> c(3);
    for (int i = 0; i < 3; ++i)
    {
        c[i] = m_isoParameter[i1][i] - gamma * m_isoParameter[i0][i];
    }
    if (fabs(c[1]) >= fabs(c[2]))
    {
        j1 = 1;
        j2 = 2;
    }
    else
    {
        j1 = 2;
        j2 = 1;
        m_straightEdge |= 4;
    }
    NekDouble beta = c[j2] / c[j1];
    if (i0 == 1)
    {
        Vmath::Vcopy(4, m_isoParameter[1], 1, m_isoParameter[0], 1);
    }
    if (j1 == 2)
    {
        NekDouble temp        = m_isoParameter[0][j1];
        m_isoParameter[0][j1] = m_isoParameter[0][j2];
        m_isoParameter[0][j2] = temp;
    }
    m_isoParameter[0][2] -= m_isoParameter[0][1] * beta;
    m_isoParameter[1][0] = c[0];
    m_isoParameter[1][1] = 1. / c[j1];
    m_isoParameter[1][2] = beta;
    m_isoParameter[1][3] = gamma;
}

int QuadGeom::v_GetDir(const int i, [[maybe_unused]] const int j) const
{
    return i % 2;
}

void QuadGeom::v_Reset(CurveMap &curvedEdges, CurveMap &curvedFaces)
{
    Geometry::v_Reset(curvedEdges, curvedFaces);
    CurveMap::iterator it = curvedFaces.find(m_globalID);

    if (it != curvedFaces.end())
    {
        m_curve = it->second.get();
    }

    for (int i = 0; i < 4; ++i)
    {
        m_edges[i]->Reset(curvedEdges, curvedFaces);
    }

    SetUpXmap();
    SetUpCoeffs(m_xmap->GetNcoeffs());
}

void QuadGeom::v_Setup()
{
    if (!m_setupState)
    {
        for (int i = 0; i < 4; ++i)
        {
            m_edges[i]->Setup();
        }
        SetUpXmap();
        SetUpCoeffs(m_xmap->GetNcoeffs());

        // Check to see if expansions are linear
        m_straightEdge = 1;
        if ((m_xmap->GetBasisNumModes(0) != 2) ||
            (m_xmap->GetBasisNumModes(1) != 2))
        {
            m_straightEdge = 0;
        }

        m_setupState = true;
    }
}

} // namespace Nektar::SpatialDomains
