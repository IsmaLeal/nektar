///////////////////////////////////////////////////////////////////////////////
//
// File: FilterDOMeasures.cpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description: SolverUtils Filter writing the asymmetry measures of the
//              sudden-expansion campaign from the expansion itself.
//
// Every OutputFrequency steps, from the mean velocity (ubar, vbar), the S
// DO modes (u_i, v_i) and the particle coefficients Y_pi (S = 0 for a DNS):
//
//   <f, g>     = sum_q w_q J_q (f_u g_u + f_v g_v)_q over the quadrature
//                points, the solver's m_physWeights; exact for products of
//                two fields of the expansion.
//   phi        = antisym(u_fp) / ||antisym(u_fp)||, antisym(u) = (u - R u)/2,
//                R the mirror y -> -y (u even, v odd); u_fp read from the
//                FUNCTION named by the FixedPoint parameter; R is a
//                permutation of the quadrature points (symmetric mesh).
//   A0 = <ubar, phi>, A_i = <u_i, phi>, a_i = <ubar, u_i>, G_ij = <u_i, u_j>,
//   <ubar, ubar>; over the elements with x >= 0: <1,1>, <1,y>, <ubar,1>,
//   <ubar,y>, <u_i,1>, <u_i,y>.
//   centreline y = 0, x >= 0: v of the mean and of every mode at the edge
//   quadrature points; c = int vbar^2 dx, b_i = int vbar v_i dx,
//   C_ij = int v_i v_j dx (1D Gauss-Lobatto rule of each edge, exact);
//   v(SENSOR_X, 0) from the polynomial of the edge holding it.
//   walls y = +-WALL_Y: du/dn (n into the fluid) of the mean and of every
//   mode at the edge points, the derivative of the element polynomial on
//   its edge; p at the same points.
//   per particle p: A_p = A0 + sum_i Y_pi A_i; M1_p = a_1 + Y_p1;
//   M_p = sgn(v_p(SENSOR_X, 0)) sqrt(c + 2 b.Y_p + Y_p C Y_p);
//   y_b,p = (<1,y> - <ubar,y> - sum_i Y_pi <u_i,y>) /
//           (<1,1> - <ubar,1> - sum_i Y_pi <u_i,1>);
//   Delta L_p = L_up - L_low, L the downstream end of the main bubble of
//   that wall: walking from the step downstream, the first stretch with
//   du/dn > 0 (the corner eddy) is skipped, the first stretch with
//   du/dn < 0 is the bubble and L is the root of the edge polynomial where
//   it ends; no negative stretch gives the x of the outlet.
//
// Output: one HDF5 file, one row per output, flushed after each row; the
// centreline and wall values go every LineOutputFrequency steps.
//
///////////////////////////////////////////////////////////////////////////////

#include <IncNavierStokesSolver/EquationSystems/DOVelocityCorrectionScheme.h>
#include <IncNavierStokesSolver/Filters/FilterDOMeasures.h>
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/BasicUtils/VmathArray.hpp>
#include <LibUtilities/LinearAlgebra/Blas.hpp>
#include <SpatialDomains/GeomFactors.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace Nektar::SolverUtils
{

std::string FilterDOMeasures::className =
    GetFilterFactory().RegisterCreatorFunction("DOMeasures",
                                               FilterDOMeasures::create);

// the sudden-expansion geometry the measures are defined on
static const NekDouble AXIS_Y      = 0.0;   // the centreline
static const NekDouble WALL_Y      = 1.5;   // the expansion walls, y = +-WALL_Y
static const NekDouble EXPANSION_X = 0.0;   // the step; x >= EXPANSION_X is the expansion
static const NekDouble SENSOR_X    = 2.0;   // where the sign of v is read for M
static const NekDouble GEOM_TOL    = 1e-8;  // coordinate tolerance for the lines above
static const NekDouble MIRROR_Q    = 1e-6;  // cell size of the mirror lookup
static const NekDouble MIRROR_TOL  = 1e-7;  // largest accepted point -> mirror distance

FilterDOMeasures::FilterDOMeasures(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const std::shared_ptr<EquationSystem> &pEquation,
    const ParamMap &pParams)
    : Filter(pSession, pEquation)
{
    m_outName = Filter::SetupOutput(".h5", pParams);

    auto it = pParams.find("OutputFrequency");
    if (it != pParams.end())
    {
        LibUtilities::Equation equ(m_session->GetInterpreter(), it->second);
        m_outputFrequency = round(equ.Evaluate());
    }
    ASSERTL0(m_outputFrequency > 0, "FilterDOMeasures: OutputFrequency must "
                                    "be positive.");

    m_lineFrequency = m_outputFrequency;
    it              = pParams.find("LineOutputFrequency");
    if (it != pParams.end())
    {
        LibUtilities::Equation equ(m_session->GetInterpreter(), it->second);
        m_lineFrequency = round(equ.Evaluate());
    }
    ASSERTL0(m_lineFrequency > 0 && m_lineFrequency % m_outputFrequency == 0,
             "FilterDOMeasures: LineOutputFrequency must be a positive "
             "multiple of OutputFrequency.");

    it = pParams.find("FixedPoint");
    if (it != pParams.end())
    {
        m_fixedPointName = it->second;
    }
}

bool FilterDOMeasures::v_IsTimeDependent()
{
    return true;
}

void FilterDOMeasures::v_Initialise(
    const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
    const NekDouble &time)
{
    auto dom = std::dynamic_pointer_cast<Nektar::DOVelocityCorrectionScheme>(
        m_equ.lock());
    ASSERTL0(dom, "FilterDOMeasures requires "
                  "SolverType=DOVelocityCorrectionScheme.");

    const auto &velocity = dom->GetVelocityIdx();
    ASSERTL0(velocity.size() == 2, "FilterDOMeasures: 2D only.");
    m_nModes     = dom->GetNumDOModes();
    m_nParticles = dom->GetNumDOParticles();
    m_comm       = pFields[0]->GetComm()->GetRowComm();
    m_nPhys      = pFields[0]->GetTotPoints();
    m_index      = 0;

    const auto &varNames = m_session->GetVariables();
    for (size_t j = 0; j < varNames.size(); ++j)
    {
        if (varNames[j] == "p")
        {
            m_pIdx = (int)j;
        }
    }
    ASSERTL0(m_pIdx >= 0, "FilterDOMeasures: no pressure variable p.");

    // one quadrature rule for every element and both directions: the edge
    // values of every line then share one set of points and one polynomial
    auto exp0  = pFields[0]->GetExp(0);
    auto pkey0 = exp0->GetBasis(0)->GetPointsKey();
    for (int e = 0; e < pFields[0]->GetExpSize(); ++e)
    {
        auto exp = pFields[0]->GetExp(e);
        ASSERTL0(exp->DetShapeType() == LibUtilities::eQuadrilateral,
                 "FilterDOMeasures: quadrilateral elements only.");
        ASSERTL0(exp->GetBasis(0)->GetPointsKey() == pkey0 &&
                     exp->GetBasis(1)->GetPointsKey() == pkey0,
                 "FilterDOMeasures: every element needs the same "
                 "quadrature in both directions.");
        ASSERTL0(exp->GetGeomFactors()->GetGtype() ==
                     SpatialDomains::eRegular,
                 "FilterDOMeasures: straight-sided elements only.");
    }
    m_nptsEdge = pkey0.GetNumPoints();
    {
        Array<OneD, NekDouble> lo(1, (NekDouble)m_nptsEdge),
            hi(1, (NekDouble)m_nptsEdge);
        m_comm->AllReduce(lo, LibUtilities::ReduceMin);
        m_comm->AllReduce(hi, LibUtilities::ReduceMax);
        ASSERTL0(lo[0] == hi[0], "FilterDOMeasures: the quadrature differs "
                                 "between ranks.");
    }
    m_seg = MemoryManager<StdRegions::StdSegExp>::AllocateSharedPtr(
        exp0->GetBasis(0)->GetBasisKey());

    // quadrature weights w_q J_q, as the solver uses them
    m_w = dom->GetPhysWeights();
    ASSERTL0((int)m_w.size() == m_nPhys, "FilterDOMeasures: weights size.");

    Array<OneD, NekDouble> x(m_nPhys), y(m_nPhys), z(m_nPhys, 0.0);
    pFields[0]->GetCoords(x, y, z);

    // the elements with x >= EXPANSION_X (the step is a mesh line)
    m_wPlus  = Array<OneD, NekDouble>(m_nPhys, 0.0);
    m_wPlusY = Array<OneD, NekDouble>(m_nPhys, 0.0);
    for (int e = 0; e < pFields[0]->GetExpSize(); ++e)
    {
        const int off = pFields[0]->GetPhys_Offset(e);
        const int nq  = pFields[0]->GetTotPoints(e);
        bool inside = true, outside = true;
        for (int q = 0; q < nq; ++q)
        {
            inside  = inside && x[off + q] >= EXPANSION_X - GEOM_TOL;
            outside = outside && x[off + q] <= EXPANSION_X + GEOM_TOL;
        }
        ASSERTL0(inside || outside, "FilterDOMeasures: an element crosses "
                                    "the step x = EXPANSION_X.");
        if (inside)
        {
            for (int q = 0; q < nq; ++q)
            {
                m_wPlus[off + q]  = m_w[off + q];
                m_wPlusY[off + q] = m_w[off + q] * y[off + q];
            }
        }
    }
    {
        Array<OneD, NekDouble> s(2);
        s[0] = Vmath::Vsum(m_nPhys, m_wPlus.data(), 1);
        s[1] = Vmath::Vsum(m_nPhys, m_wPlusY.data(), 1);
        m_comm->AllReduce(s, LibUtilities::ReduceSum);
        m_areaPlus = s[0];
        m_yPlus    = s[1];
    }

    FindLines(pFields, x, y);
    BuildPhi(pFields, x, y);

    // scratch
    const int S = m_nModes;
    m_physU = Array<OneD, NekDouble>(m_nPhys);
    m_physV = Array<OneD, NekDouble>(m_nPhys);
    m_wu    = Array<OneD, NekDouble>(m_nPhys);
    m_wv    = Array<OneD, NekDouble>(m_nPhys);
    m_wui   = Array<OneD, NekDouble>(m_nPhys);
    m_wvi   = Array<OneD, NekDouble>(m_nPhys);
    int nqMax = 0, ncMax = 0;
    for (int e = 0; e < pFields[0]->GetExpSize(); ++e)
    {
        nqMax = std::max(nqMax, pFields[0]->GetTotPoints(e));
        ncMax = std::max(ncMax, pFields[m_pIdx]->GetNcoeffs(e));
    }
    m_elmtBuf  = Array<OneD, NekDouble>(std::max(nqMax, 1));
    m_elmtBuf2 = Array<OneD, NekDouble>(std::max({nqMax, ncMax, 1}));
    m_edgeBuf.resize(S + 1);
    for (int f = 0; f <= S; ++f)
    {
        m_edgeBuf[f] = Array<OneD, NekDouble>(m_nptsEdge);
    }
    if (m_comm->GetRank() == 0)
    {
        m_tau = Array<OneD, NekDouble>(
            std::max(1, m_nParticles * 2 * m_nWallPts));
        OpenOutput();
        std::cout << "FilterDOMeasures: S = " << S << ", Np = " << m_nParticles
                  << ", " << m_nAxisEdges << " centreline edges (x >= "
                  << EXPANSION_X << "), " << m_nWallEdges << " + "
                  << m_nWallEdges << " wall edges (y = +-" << WALL_Y << "), "
                  << m_nptsEdge << " points per edge, sensor at x = "
                  << SENSOR_X << "\n"
                  << "FilterDOMeasures: ||antisym(u_fp)|| = "
                  << std::setprecision(10) << m_antisymNorm
                  << ", largest point-to-mirror distance "
                  << std::setprecision(3) << m_mirrorMismatch
                  << "; writing " << m_outName << " every "
                  << m_outputFrequency << " steps, lines every "
                  << m_lineFrequency << std::endl;
    }

    Compute(pFields, 0, time);
}

void FilterDOMeasures::v_Update(
    const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
    const NekDouble &time)
{
    ++m_index;
    if (m_index % m_outputFrequency != 0)
    {
        return;
    }
    Compute(pFields, (int)m_index, time);
}

void FilterDOMeasures::v_Finalise(
    [[maybe_unused]] const Array<OneD, const MultiRegions::ExpListSharedPtr>
        &pFields,
    [[maybe_unused]] const NekDouble &time)
{
    if (m_comm->GetRank() == 0 && m_file)
    {
        m_sets.clear();
        m_file.reset();
    }
}

/**
 * The values of elmtVals (one element, its quadrature points) on the edge
 * e.edge, in the element's own order of the edge points.
 */
void FilterDOMeasures::ExtractEdge(
    const LocalRegions::ExpansionSharedPtr &exp, const LineEdge &e,
    const Array<OneD, const NekDouble> &elmtVals, Array<OneD, NekDouble> &out)
{
    exp->GetTracePhysVals(e.edge, exp, elmtVals, out);
}

/**
 * Finds the centreline edges with x >= EXPANSION_X and the wall edges of
 * this rank, their global positions along each line, and the edge holding
 * the sensor.
 */
void FilterDOMeasures::FindLines(
    const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
    const Array<OneD, const NekDouble> &x, const Array<OneD, const NekDouble> &y)
{
    const int n = m_nptsEdge;
    Array<OneD, NekDouble> xe(n), ye(n);

    for (int el = 0; el < pFields[0]->GetExpSize(); ++el)
    {
        auto exp      = pFields[0]->GetExp(el);
        const int off = pFields[0]->GetPhys_Offset(el);
        const int nq  = pFields[0]->GetTotPoints(el);
        Array<OneD, NekDouble> xel(nq, const_cast<NekDouble *>(x.data() + off),
                                   eArrayWrapper);
        Array<OneD, NekDouble> yel(nq, const_cast<NekDouble *>(y.data() + off),
                                   eArrayWrapper);
        const NekDouble yCentroid = Vmath::Vsum(nq, yel.data(), 1) / nq;

        for (int j = 0; j < exp->GetNtraces(); ++j)
        {
            exp->GetTracePhysVals(j, exp, xel, xe);
            exp->GetTracePhysVals(j, exp, yel, ye);

            bool onAxis = true, onLower = true, onUpper = true;
            for (int k = 0; k < n; ++k)
            {
                onAxis  = onAxis && std::abs(ye[k] - AXIS_Y) < GEOM_TOL &&
                         xe[k] >= EXPANSION_X - GEOM_TOL;
                onLower = onLower && std::abs(ye[k] + WALL_Y) < GEOM_TOL;
                onUpper = onUpper && std::abs(ye[k] - WALL_Y) < GEOM_TOL;
            }
            // a centreline edge belongs to two elements: the one above it
            // keeps it, so no rank counts it twice
            onAxis = onAxis && yCentroid > AXIS_Y;
            if (!onAxis && !onLower && !onUpper)
            {
                continue;
            }

            LineEdge le;
            le.elmt = el;
            le.edge = j;
            le.gpos = -1;
            le.npts = n;
            le.sign = onUpper ? -1.0 : 1.0;
            le.x    = Array<OneD, NekDouble>(n);
            le.w    = Array<OneD, NekDouble>(n);
            le.sorted = Array<OneD, int>(n);
            Vmath::Vcopy(n, xe, 1, le.x, 1);
            std::iota(le.sorted.data(), le.sorted.data() + n, 0);
            std::sort(le.sorted.data(), le.sorted.data() + n,
                      [&](int a, int b) { return le.x[a] < le.x[b]; });
            // a straight horizontal edge: Jacobian = half its length
            const NekDouble jac = 0.5 * std::abs(le.x[n - 1] - le.x[0]);
            ASSERTL0(jac > GEOM_TOL, "FilterDOMeasures: degenerate edge.");
            const auto &w1d = exp->GetBasis(j % 2)->GetW();
            Vmath::Smul(n, jac, w1d, 1, le.w, 1);

            if (onAxis)
            {
                m_axis.push_back(le);
            }
            else
            {
                m_wall[onLower ? 0 : 1].push_back(le);
            }
        }
    }

    AssignPositions(m_axis, m_nAxisEdges, m_xAxis);
    int nLower = 0, nUpper = 0;
    AssignPositions(m_wall[0], nLower, m_xWall[0]);
    AssignPositions(m_wall[1], nUpper, m_xWall[1]);
    ASSERTL0(m_nAxisEdges > 0, "FilterDOMeasures: no centreline edge with "
                               "x >= EXPANSION_X found on y = AXIS_Y.");
    ASSERTL0(nLower > 0 && nLower == nUpper,
             "FilterDOMeasures: the two walls y = +-WALL_Y must hold the "
             "same positive number of edges (found " +
                 std::to_string(nLower) + " and " + std::to_string(nUpper) +
                 ").");
    m_nWallEdges = nLower;
    m_nAxisPts   = m_nAxisEdges * n;
    m_nWallPts   = m_nWallEdges * n;

    // the centreline edge holding the sensor: min x <= SENSOR_X < max x, so
    // a vertex at SENSOR_X belongs to the edge downstream of it only
    int nSensor = 0;
    for (size_t i = 0; i < m_axis.size(); ++i)
    {
        const auto &e      = m_axis[i];
        const NekDouble x0 = e.x[0], x1 = e.x[n - 1];
        if (std::min(x0, x1) <= SENSOR_X && SENSOR_X < std::max(x0, x1))
        {
            m_sensorEdge = (int)i;
            m_sensorXi   = -1.0 + 2.0 * (SENSOR_X - x0) / (x1 - x0);
            ++nSensor;
        }
    }
    {
        Array<OneD, NekDouble> cnt(1, (NekDouble)nSensor);
        m_comm->AllReduce(cnt, LibUtilities::ReduceSum);
        ASSERTL0(cnt[0] == 1.0, "FilterDOMeasures: the sensor x = SENSOR_X "
                                "must lie on exactly one centreline edge.");
    }
}

/**
 * Orders the edges of one line by x over all ranks: gpos of each local edge,
 * the global number of edges, and the sorted x of all the line's points.
 */
void FilterDOMeasures::AssignPositions(std::vector<LineEdge> &edges,
                                       int &nEdgesGlobal,
                                       Array<OneD, NekDouble> &xGlobal)
{
    const int n     = m_nptsEdge;
    const int nR    = m_comm->GetSize();
    const int nLoc  = (int)edges.size();
    std::vector<NekDouble> midLoc(nLoc);
    for (int i = 0; i < nLoc; ++i)
    {
        midLoc[i] = 0.5 * (edges[i].x[0] + edges[i].x[n - 1]);
    }

    // the midpoints of every edge of the line: each rank writes its own at
    // its offset and the sum over ranks assembles them. A rank may hold no
    // edge of this line, which AllGatherv does not allow (it sends the whole
    // container, never zero elements).
    int offset   = 0;
    nEdgesGlobal = nLoc;
    if (nR > 1)
    {
        Array<OneD, int> cntLoc(1, nLoc), counts(nR);
        m_comm->AllGather(cntLoc, counts);
        nEdgesGlobal = 0;
        for (int r = 0; r < nR; ++r)
        {
            if (r == m_comm->GetRank())
            {
                offset = nEdgesGlobal;
            }
            nEdgesGlobal += counts[r];
        }
    }
    Array<OneD, NekDouble> mid(std::max(nEdgesGlobal, 1), 0.0);
    for (int i = 0; i < nLoc; ++i)
    {
        mid[offset + i] = midLoc[i];
    }
    if (nR > 1)
    {
        m_comm->AllReduce(mid, LibUtilities::ReduceSum);
    }

    std::vector<NekDouble> sortedMid(mid.data(), mid.data() + nEdgesGlobal);
    std::sort(sortedMid.begin(), sortedMid.end());
    for (int g = 1; g < nEdgesGlobal; ++g)
    {
        ASSERTL0(sortedMid[g] - sortedMid[g - 1] > GEOM_TOL,
                 "FilterDOMeasures: two edges of one line share a midpoint.");
    }
    for (int i = 0; i < nLoc; ++i)
    {
        edges[i].gpos = (int)(std::lower_bound(sortedMid.begin(),
                                               sortedMid.end(), midLoc[i]) -
                              sortedMid.begin());
    }

    xGlobal = Array<OneD, NekDouble>(std::max(nEdgesGlobal * n, 1), 0.0);
    for (int i = 0; i < nLoc; ++i)
    {
        for (int k = 0; k < n; ++k)
        {
            xGlobal[edges[i].gpos * n + k] = edges[i].x[edges[i].sorted[k]];
        }
    }
    m_comm->AllReduce(xGlobal, LibUtilities::ReduceSum);
}

/**
 * phi = antisym(u_fp) / ||antisym(u_fp)|| at the quadrature points. u_fp
 * comes from the FixedPoint FUNCTION; its mirror image is read off the
 * quadrature point at (x, -y), found by coordinates over all ranks.
 */
void FilterDOMeasures::BuildPhi(
    const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
    const Array<OneD, const NekDouble> &x, const Array<OneD, const NekDouble> &y)
{
    auto equ = m_equ.lock();
    auto dom = std::dynamic_pointer_cast<Nektar::DOVelocityCorrectionScheme>(equ);
    const auto &velocity = dom->GetVelocityIdx();
    const auto &varNames = m_session->GetVariables();
    const std::string vu = varNames[velocity[0]], vv = varNames[velocity[1]];
    ASSERTL0(m_session->DefinesFunction(m_fixedPointName, vu) &&
                 m_session->DefinesFunction(m_fixedPointName, vv),
             "FilterDOMeasures: FUNCTION " + m_fixedPointName +
                 " must define " + vu + " and " + vv + " (the fixed point).");
    auto fn = equ->GetFunction(m_fixedPointName, pFields[velocity[0]]);
    Array<OneD, NekDouble> ufp(m_nPhys), vfp(m_nPhys);
    fn->Evaluate(vu, ufp);
    fn->Evaluate(vv, vfp);

    // every rank's points and fixed-point values
    const int nR = m_comm->GetSize();
    int nAll     = m_nPhys;
    Array<OneD, const NekDouble> gx, gy, gu, gv;
    if (nR == 1)
    {
        gx = x;
        gy = y;
        gu = ufp;
        gv = vfp;
    }
    else
    {
        Array<OneD, int> cntLoc(1, m_nPhys), counts(nR), offsets(nR);
        m_comm->AllGather(cntLoc, counts);
        nAll = 0;
        for (int r = 0; r < nR; ++r)
        {
            offsets[r] = nAll;
            nAll += counts[r];
        }
        Array<OneD, NekDouble> tx(nAll), ty(nAll), tu(nAll), tv(nAll);
        Array<OneD, NekDouble> xs(m_nPhys), ys(m_nPhys);
        Vmath::Vcopy(m_nPhys, x, 1, xs, 1);
        Vmath::Vcopy(m_nPhys, y, 1, ys, 1);
        m_comm->AllGatherv(xs, tx, counts, offsets);
        m_comm->AllGatherv(ys, ty, counts, offsets);
        m_comm->AllGatherv(ufp, tu, counts, offsets);
        m_comm->AllGatherv(vfp, tv, counts, offsets);
        gx = tx;
        gy = ty;
        gu = tu;
        gv = tv;
    }

    // lookup by quantised coordinates; points shared by elements repeat
    // with equal values, the first one kept is as good as any
    typedef std::pair<long long, long long> Key;
    std::map<Key, int> cell;
    for (int q = 0; q < nAll; ++q)
    {
        cell.emplace(Key(std::llround(gx[q] / MIRROR_Q),
                         std::llround(gy[q] / MIRROR_Q)),
                     q);
    }
    Array<OneD, NekDouble> au(m_nPhys), av(m_nPhys);
    NekDouble worst = 0.0;
    for (int q = 0; q < m_nPhys; ++q)
    {
        const NekDouble xm = x[q], ym = -y[q];
        const long long ix = std::llround(xm / MIRROR_Q),
                        iy = std::llround(ym / MIRROR_Q);
        int best        = -1;
        NekDouble dBest = 1e300;
        for (long long dx = -1; dx <= 1; ++dx)
        {
            for (long long dy = -1; dy <= 1; ++dy)
            {
                auto it = cell.find(Key(ix + dx, iy + dy));
                if (it == cell.end())
                {
                    continue;
                }
                const NekDouble d = std::hypot(gx[it->second] - xm,
                                               gy[it->second] - ym);
                if (d < dBest)
                {
                    dBest = d;
                    best  = it->second;
                }
            }
        }
        ASSERTL0(best >= 0 && dBest < MIRROR_TOL,
                 "FilterDOMeasures: no mirror image for the quadrature "
                 "point (" + std::to_string(x[q]) + ", " +
                     std::to_string(y[q]) + "): the mesh is not symmetric "
                     "in y.");
        worst = std::max(worst, dBest);
        au[q] = 0.5 * (ufp[q] - gu[best]);
        av[q] = 0.5 * (vfp[q] + gv[best]);
    }
    {
        Array<OneD, NekDouble> r(1, worst);
        m_comm->AllReduce(r, LibUtilities::ReduceMax);
        m_mirrorMismatch = r[0];
    }

    NekDouble norm2 = 0.0;
    {
        Array<OneD, NekDouble> t(m_nPhys);
        Vmath::Vmul(m_nPhys, m_w, 1, au, 1, t, 1);
        norm2 = Vmath::Dot(m_nPhys, t.data(), au.data());
        Vmath::Vmul(m_nPhys, m_w, 1, av, 1, t, 1);
        norm2 += Vmath::Dot(m_nPhys, t.data(), av.data());
        Array<OneD, NekDouble> r(1, norm2);
        m_comm->AllReduce(r, LibUtilities::ReduceSum);
        norm2 = r[0];
    }
    ASSERTL0(norm2 > 0.0, "FilterDOMeasures: the fixed point is symmetric, "
                          "phi is undefined.");
    m_antisymNorm = std::sqrt(norm2);
    m_phiU  = Array<OneD, NekDouble>(m_nPhys);
    m_phiV  = Array<OneD, NekDouble>(m_nPhys);
    m_wPhiU = Array<OneD, NekDouble>(m_nPhys);
    m_wPhiV = Array<OneD, NekDouble>(m_nPhys);
    Vmath::Smul(m_nPhys, 1.0 / m_antisymNorm, au, 1, m_phiU, 1);
    Vmath::Smul(m_nPhys, 1.0 / m_antisymNorm, av, 1, m_phiV, 1);
    Vmath::Vmul(m_nPhys, m_w, 1, m_phiU, 1, m_wPhiU, 1);
    Vmath::Vmul(m_nPhys, m_w, 1, m_phiV, 1, m_wPhiV, 1);
}

template <class T>
void FilterDOMeasures::AddSet(const std::string &name,
                              const std::vector<hsize_t> &dims)
{
    namespace H5 = LibUtilities::H5;
    OutSet s;
    s.dims  = dims;
    s.space = std::make_shared<H5::DataSpace>(dims);
    s.set   = m_file->CreateDataSet(name, H5::PredefinedDataType::Native<T>(),
                                    s.space);
    m_sets[name] = s;
}

template <class T>
void FilterDOMeasures::PutRow(const std::string &name, hsize_t row,
                              const std::vector<T> &data)
{
    auto &s = m_sets.at(name);
    std::vector<hsize_t> start(s.dims.size(), 0), count(s.dims);
    start[0] = row;
    count[0] = 1;
    hsize_t n = 1;
    for (size_t d = 1; d < s.dims.size(); ++d)
    {
        n *= s.dims[d];
    }
    ASSERTL0(data.size() == n, "FilterDOMeasures: row size of " + name);
    ASSERTL0(row < s.dims[0], "FilterDOMeasures: row overflow of " + name);
    s.space->SelectRange(start, count);
    s.set->Write(data, s.space);
}

/**
 * Creates the file and every dataset, sized by the number of outputs the
 * run will make. Rank 0 only.
 */
void FilterDOMeasures::OpenOutput()
{
    namespace H5 = LibUtilities::H5;
    auto equ     = m_equ.lock();
    auto dom = std::dynamic_pointer_cast<Nektar::DOVelocityCorrectionScheme>(equ);
    const hsize_t S = m_nModes, Np = m_nParticles;
    const int steps = equ->GetSteps();
    m_nRows     = steps / m_outputFrequency + 1;
    m_nLineRows = steps / m_lineFrequency + 1;
    m_row = m_lineRow = 0;

    m_file   = H5::File::Create(m_outName, H5F_ACC_TRUNC);
    auto hdr = m_file->CreateGroup("header");
    hdr->SetAttribute("format", std::string("DO_MEASURES_V1"));
    hdr->SetAttribute("n_modes", m_nModes);
    hdr->SetAttribute("n_particles", m_nParticles);
    hdr->SetAttribute("dt", dom->GetTimeStep());
    hdr->SetAttribute("output_frequency", (int)m_outputFrequency);
    hdr->SetAttribute("line_output_frequency", (int)m_lineFrequency);
    hdr->SetAttribute("antisym_norm", m_antisymNorm);
    hdr->SetAttribute("mirror_mismatch", m_mirrorMismatch);
    hdr->SetAttribute("area_plus", m_areaPlus);
    hdr->SetAttribute("int_y_plus", m_yPlus);
    hdr->SetAttribute("n_axis_edges", m_nAxisEdges);
    hdr->SetAttribute("n_wall_edges", m_nWallEdges);
    hdr->SetAttribute("npts_edge", m_nptsEdge);
    hdr->SetAttribute("axis_y", AXIS_Y);
    hdr->SetAttribute("wall_y", WALL_Y);
    hdr->SetAttribute("expansion_x", EXPANSION_X);
    hdr->SetAttribute("sensor_x", SENSOR_X);

    const hsize_t T = m_nRows, TL = m_nLineRows;
    const hsize_t nA = m_nAxisPts, nW = m_nWallPts;

    AddSet<double>("t", {T});
    AddSet<int>("step", {T});
    AddSet<double>("A0", {T});
    AddSet<double>("mean_norm2", {T});
    AddSet<double>("int_ubar", {T});
    AddSet<double>("int_y_ubar", {T});
    AddSet<double>("c", {T});
    AddSet<double>("v2_mean", {T});
    AddSet<float>("A_p", {T, Np});
    AddSet<float>("M_p", {T, Np});
    AddSet<float>("yb_p", {T, Np});
    AddSet<float>("dL_p", {T, Np});
    if (S > 0)
    {
        AddSet<double>("Ai", {T, S});
        AddSet<double>("a", {T, S});
        AddSet<double>("G", {T, S, S});
        AddSet<double>("int_ui", {T, S});
        AddSet<double>("int_y_ui", {T, S});
        AddSet<double>("b", {T, S});
        AddSet<double>("C", {T, S, S});
        AddSet<double>("v2_modes", {T, S});
        AddSet<float>("M1_p", {T, Np});
        AddSet<float>("Y1_p", {T, Np});
    }
    AddSet<int>("line_step", {TL});
    AddSet<double>("line_t", {TL});
    AddSet<double>("v_axis", {TL, S + 1, nA});
    AddSet<double>("dudn_wall", {TL, S + 1, 2, nW});
    AddSet<double>("p_wall", {TL, S + 1, 2, nW});
    AddSet<double>("x_axis", {nA});
    AddSet<double>("x_wall", {2, nW});

    // rows not yet written read as step -1
    {
        std::vector<int> none(1, -1);
        for (hsize_t r = 0; r < T; ++r)
        {
            PutRow<int>("step", r, none);
        }
        for (hsize_t r = 0; r < TL; ++r)
        {
            PutRow<int>("line_step", r, none);
        }
    }
    m_sets.at("x_axis").set->Write(
        std::vector<double>(m_xAxis.data(), m_xAxis.data() + nA));
    {
        std::vector<double> xw(2 * nW);
        for (int w = 0; w < 2; ++w)
        {
            std::copy(m_xWall[w].data(), m_xWall[w].data() + nW,
                      xw.begin() + w * nW);
        }
        m_sets.at("x_wall").set->Write(xw);
    }
}

/**
 * L of one wall from du/dn of one particle at the wall's points (sorted by
 * x, nEdges edges of m_nptsEdge points; xw their x): walking downstream
 * from the step, skip the first stretch with du/dn >= 0 (the corner eddy),
 * enter the first stretch with du/dn < 0 (the main bubble) and return the x
 * where it ends, the root of the edge polynomial between the last negative
 * point and the next non-negative one. No negative stretch, or a bubble
 * reaching the outlet, gives the x of the last point.
 */
NekDouble FilterDOMeasures::Reattachment(const NekDouble *tau,
                                         const NekDouble *xw,
                                         int nEdges) const
{
    const int n = m_nptsEdge, N = nEdges * n;
    int k = 0;
    while (k < N && tau[k] >= 0.0)
    {
        ++k;
    }
    if (k == N)
    {
        return xw[N - 1];
    }
    while (k < N && tau[k] < 0.0)
    {
        ++k;
    }
    if (k == N)
    {
        return xw[N - 1];
    }
    // tau[k-1] < 0 <= tau[k]
    if (tau[k] == 0.0 || k % n == 0)
    {
        // a root on a point, or a sign change across the vertex shared by
        // two edges (the derivative of the expansion jumps there)
        return xw[k];
    }
    const int g        = k / n;
    const NekDouble x0 = xw[g * n], x1 = xw[g * n + n - 1];
    Array<OneD, NekDouble> vals(n, const_cast<NekDouble *>(tau + g * n),
                                eArrayWrapper);
    Array<OneD, NekDouble> xi(1);
    NekDouble a = -1.0 + 2.0 * (xw[k - 1] - x0) / (x1 - x0);
    NekDouble b = -1.0 + 2.0 * (xw[k] - x0) / (x1 - x0);
    for (int it = 0; it < 100 && b - a > 1e-13; ++it)
    {
        xi[0]             = 0.5 * (a + b);
        const NekDouble f = m_seg->PhysEvaluate(xi, vals);
        if (f < 0.0)
        {
            a = xi[0];
        }
        else
        {
            b = xi[0];
        }
    }
    return x0 + 0.5 * (0.5 * (a + b) + 1.0) * (x1 - x0);
}

/**
 * Everything of one output: the inner products, the lines, the particles,
 * and the row of the file.
 */
void FilterDOMeasures::Compute(
    const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
    int step, NekDouble time)
{
    auto dom = std::dynamic_pointer_cast<Nektar::DOVelocityCorrectionScheme>(
        m_equ.lock());
    const auto &velocity = dom->GetVelocityIdx();
    const int S = m_nModes, Np = m_nParticles, nPhys = m_nPhys;
    const int n = m_nptsEdge, nA = m_nAxisPts, nW = m_nWallPts;
    const bool lines = (step % m_lineFrequency == 0);
    const auto &modePhys = dom->GetDOModePhys();
    const auto &pCoeffs  = dom->GetDOModePCoeffs();
    const int nPC        = pFields[m_pIdx]->GetNcoeffs();
    const bool modeP     = (S > 0 && (int)pCoeffs.size() == S * nPC);

    // the mean velocity at the quadrature points
    const NekDouble *ub, *vb;
    {
        auto fu = pFields[velocity[0]], fv = pFields[velocity[1]];
        if (fu->GetPhysState())
        {
            ub = fu->GetPhys().data();
        }
        else
        {
            fu->BwdTrans(fu->GetCoeffs(), m_physU);
            ub = m_physU.data();
        }
        if (fv->GetPhysState())
        {
            vb = fv->GetPhys().data();
        }
        else
        {
            fv->BwdTrans(fv->GetCoeffs(), m_physV);
            vb = m_physV.data();
        }
    }
    auto U = [&](int i, int c) -> const NekDouble * {
        return modePhys.data() + (i * 2 + c) * nPhys;
    };
    auto wrap = [](const NekDouble *p, int size) {
        return Array<OneD, NekDouble>(size, const_cast<NekDouble *>(p),
                                      eArrayWrapper);
    };

    // ---- inner products over the domain, one packed reduction
    // A0 | Ai(S) | a(S) | mean_norm2 | G(S*S) | int_ubar | int_y_ubar |
    // int_ui(S) | int_y_ui(S)
    const int iA0 = 0, iAi = 1, ia = 1 + S, iNorm = 1 + 2 * S,
              iG = 2 + 2 * S, iIu = 2 + 2 * S + S * S, iIyu = iIu + 1,
              iIui = iIu + 2, iIyui = iIui + S, nPk = iIyui + S;
    Array<OneD, NekDouble> pk(nPk, 0.0);
    Vmath::Vmul(nPhys, m_w.data(), 1, ub, 1, m_wu.data(), 1);
    Vmath::Vmul(nPhys, m_w.data(), 1, vb, 1, m_wv.data(), 1);
    pk[iA0] = Vmath::Dot(nPhys, m_wu.data(), m_phiU.data()) +
              Vmath::Dot(nPhys, m_wv.data(), m_phiV.data());
    pk[iNorm] = Vmath::Dot(nPhys, m_wu.data(), ub) +
                Vmath::Dot(nPhys, m_wv.data(), vb);
    pk[iIu]  = Vmath::Dot(nPhys, m_wPlus.data(), ub);
    pk[iIyu] = Vmath::Dot(nPhys, m_wPlusY.data(), ub);
    for (int i = 0; i < S; ++i)
    {
        pk[iAi + i] = Vmath::Dot(nPhys, m_wPhiU.data(), U(i, 0)) +
                      Vmath::Dot(nPhys, m_wPhiV.data(), U(i, 1));
        pk[ia + i] = Vmath::Dot(nPhys, m_wu.data(), U(i, 0)) +
                     Vmath::Dot(nPhys, m_wv.data(), U(i, 1));
        pk[iIui + i]  = Vmath::Dot(nPhys, m_wPlus.data(), U(i, 0));
        pk[iIyui + i] = Vmath::Dot(nPhys, m_wPlusY.data(), U(i, 0));
        Vmath::Vmul(nPhys, m_w.data(), 1, U(i, 0), 1, m_wui.data(), 1);
        Vmath::Vmul(nPhys, m_w.data(), 1, U(i, 1), 1, m_wvi.data(), 1);
        for (int j = i; j < S; ++j)
        {
            const NekDouble g = Vmath::Dot(nPhys, m_wui.data(), U(j, 0)) +
                                Vmath::Dot(nPhys, m_wvi.data(), U(j, 1));
            pk[iG + i * S + j] = pk[iG + j * S + i] = g;
        }
    }
    m_comm->AllReduce(pk, LibUtilities::ReduceSum);

    // ---- centreline: c | b(S) | C(S*S) | v2(S+1); the line values
    const int ic = 0, ib = 1, iC = 1 + S, iv2 = 1 + S + S * S,
              nAx = iv2 + S + 1;
    Array<OneD, NekDouble> ax(nAx, 0.0);
    Array<OneD, NekDouble> vAxis(std::max((S + 1) * nA, 1), 0.0);
    for (size_t ie = 0; ie < m_axis.size(); ++ie)
    {
        const auto &e = m_axis[ie];
        auto exp      = pFields[0]->GetExp(e.elmt);
        const int off = pFields[0]->GetPhys_Offset(e.elmt);
        const int nq  = pFields[0]->GetTotPoints(e.elmt);
        for (int f = 0; f <= S; ++f)
        {
            const NekDouble *src = (f == 0) ? vb + off : U(f - 1, 1) + off;
            ExtractEdge(exp, e, wrap(src, nq), m_edgeBuf[f]);
            for (int k = 0; k < n; ++k)
            {
                vAxis[f * nA + e.gpos * n + k] = m_edgeBuf[f][e.sorted[k]];
            }
        }
        for (int k = 0; k < n; ++k)
        {
            const NekDouble w = e.w[k], v0 = m_edgeBuf[0][k];
            ax[ic] += w * v0 * v0;
            for (int i = 0; i < S; ++i)
            {
                const NekDouble vi = m_edgeBuf[1 + i][k];
                ax[ib + i] += w * v0 * vi;
                for (int j = i; j < S; ++j)
                {
                    ax[iC + i * S + j] += w * vi * m_edgeBuf[1 + j][k];
                }
            }
        }
        if ((int)ie == m_sensorEdge)
        {
            Array<OneD, NekDouble> xi(1, m_sensorXi);
            for (int f = 0; f <= S; ++f)
            {
                ax[iv2 + f] = m_seg->PhysEvaluate(xi, m_edgeBuf[f]);
            }
        }
    }
    for (int i = 0; i < S; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            ax[iC + i * S + j] = ax[iC + j * S + i];
        }
    }
    m_comm->AllReduce(ax, LibUtilities::ReduceSum);
    if (lines)
    {
        m_comm->AllReduce(vAxis, LibUtilities::ReduceSum);
    }

    // ---- walls: du/dn and p of the mean and of every mode at the points
    // layout f * (2 nW) + wall * nW + gpos * n + k
    const int nWW = 2 * nW;
    Array<OneD, NekDouble> dudn((S + 1) * nWW, 0.0), pw((S + 1) * nWW, 0.0);
    for (int w = 0; w < 2; ++w)
    {
        for (const auto &e : m_wall[w])
        {
            auto exp       = pFields[velocity[0]]->GetExp(e.elmt);
            auto pexp      = pFields[m_pIdx]->GetExp(e.elmt);
            const int off  = pFields[0]->GetPhys_Offset(e.elmt);
            const int nq   = pFields[0]->GetTotPoints(e.elmt);
            const int coff = pFields[m_pIdx]->GetCoeff_Offset(e.elmt);
            const int nc   = pFields[m_pIdx]->GetNcoeffs(e.elmt);
            for (int f = 0; f <= S; ++f)
            {
                const NekDouble *src = (f == 0) ? ub + off : U(f - 1, 0) + off;
                exp->PhysDeriv(1, wrap(src, nq), m_elmtBuf);
                ExtractEdge(exp, e, m_elmtBuf, m_edgeBuf[f]);
                for (int k = 0; k < n; ++k)
                {
                    dudn[f * nWW + w * nW + e.gpos * n + k] =
                        e.sign * m_edgeBuf[f][e.sorted[k]];
                }
            }
            if (lines)
            {
                for (int f = 0; f <= S; ++f)
                {
                    if (f > 0 && !modeP)
                    {
                        break;
                    }
                    const NekDouble *cf =
                        (f == 0) ? pFields[m_pIdx]->GetCoeffs().data() + coff
                                 : pCoeffs.data() + (f - 1) * nPC + coff;
                    pexp->BwdTrans(wrap(cf, nc), m_elmtBuf2);
                    ExtractEdge(pexp, e, m_elmtBuf2, m_edgeBuf[f]);
                    for (int k = 0; k < n; ++k)
                    {
                        pw[f * nWW + w * nW + e.gpos * n + k] =
                            m_edgeBuf[f][e.sorted[k]];
                    }
                }
            }
        }
    }
    m_comm->AllReduce(dudn, LibUtilities::ReduceSum);
    if (lines)
    {
        m_comm->AllReduce(pw, LibUtilities::ReduceSum);
    }

    // ---- the particles (Y gathered on every rank; the rest on rank 0)
    Array<OneD, NekDouble> Y(std::max(1, Np * S), 0.0);
    dom->GatherYi(Y);
    if (m_comm->GetRank() != 0)
    {
        return;
    }

    // du/dn of every particle at every wall point: tau = Y D + mean
    if (S > 0)
    {
        Blas::Cdgemm(Np, nWW, S, 1.0, Y.data(), S, dudn.data() + nWW, nWW,
                     0.0, m_tau.data(), nWW);
    }
    else
    {
        Vmath::Zero(Np * nWW, m_tau.data(), 1);
    }
    for (int p = 0; p < Np; ++p)
    {
        Vmath::Vadd(nWW, dudn.data(), 1, m_tau.data() + p * nWW, 1,
                    m_tau.data() + p * nWW, 1);
    }

    std::vector<float> Ap(Np), Mp(Np), ybp(Np), dLp(Np), M1p(Np), Y1p(Np);
    for (int p = 0; p < Np; ++p)
    {
        const NekDouble *Yp = Y.data() + p * S;
        NekDouble A = pk[iA0], s = ax[iv2], q = ax[ic];
        NekDouble num = m_yPlus - pk[iIyu], den = m_areaPlus - pk[iIu];
        for (int i = 0; i < S; ++i)
        {
            A += Yp[i] * pk[iAi + i];
            s += Yp[i] * ax[iv2 + 1 + i];
            q += 2.0 * Yp[i] * ax[ib + i];
            num -= Yp[i] * pk[iIyui + i];
            den -= Yp[i] * pk[iIui + i];
            for (int j = 0; j < S; ++j)
            {
                q += Yp[i] * ax[iC + i * S + j] * Yp[j];
            }
        }
        Ap[p]  = (float)A;
        Mp[p]  = (float)((s < 0.0 ? -1.0 : 1.0) * std::sqrt(std::max(q, 0.0)));
        ybp[p] = (float)(num / den);
        const NekDouble *tp = m_tau.data() + p * nWW;
        dLp[p] = (float)(Reattachment(tp + nW, m_xWall[1].data(), m_nWallEdges) -
                         Reattachment(tp, m_xWall[0].data(), m_nWallEdges));
        if (S > 0)
        {
            M1p[p] = (float)(pk[ia] + Yp[0]);
            Y1p[p] = (float)Yp[0];
        }
    }

    // ---- the row
    if (m_row >= m_nRows)
    {
        if (m_row == m_nRows)
        {
            std::cerr << "FilterDOMeasures: more outputs than rows "
                         "allocated; the rest are dropped." << std::endl;
        }
        ++m_row;
        return;
    }
    auto vec = [](const NekDouble *p, int size) {
        return std::vector<double>(p, p + size);
    };
    PutRow<double>("t", m_row, {time});
    PutRow<int>("step", m_row, {step});
    PutRow<double>("A0", m_row, {pk[iA0]});
    PutRow<double>("mean_norm2", m_row, {pk[iNorm]});
    PutRow<double>("int_ubar", m_row, {pk[iIu]});
    PutRow<double>("int_y_ubar", m_row, {pk[iIyu]});
    PutRow<double>("c", m_row, {ax[ic]});
    PutRow<double>("v2_mean", m_row, {ax[iv2]});
    PutRow<float>("A_p", m_row, Ap);
    PutRow<float>("M_p", m_row, Mp);
    PutRow<float>("yb_p", m_row, ybp);
    PutRow<float>("dL_p", m_row, dLp);
    if (S > 0)
    {
        PutRow<double>("Ai", m_row, vec(pk.data() + iAi, S));
        PutRow<double>("a", m_row, vec(pk.data() + ia, S));
        PutRow<double>("G", m_row, vec(pk.data() + iG, S * S));
        PutRow<double>("int_ui", m_row, vec(pk.data() + iIui, S));
        PutRow<double>("int_y_ui", m_row, vec(pk.data() + iIyui, S));
        PutRow<double>("b", m_row, vec(ax.data() + ib, S));
        PutRow<double>("C", m_row, vec(ax.data() + iC, S * S));
        PutRow<double>("v2_modes", m_row, vec(ax.data() + iv2 + 1, S));
        PutRow<float>("M1_p", m_row, M1p);
        PutRow<float>("Y1_p", m_row, Y1p);
    }
    ++m_row;
    if (lines && m_lineRow < m_nLineRows)
    {
        PutRow<int>("line_step", m_lineRow, {step});
        PutRow<double>("line_t", m_lineRow, {time});
        PutRow<double>("v_axis", m_lineRow, vec(vAxis.data(), (S + 1) * nA));
        PutRow<double>("dudn_wall", m_lineRow, vec(dudn.data(), (S + 1) * nWW));
        PutRow<double>("p_wall", m_lineRow, vec(pw.data(), (S + 1) * nWW));
        ++m_lineRow;
    }
    H5Fflush(m_file->GetId(), H5F_SCOPE_GLOBAL);
}

} // namespace Nektar::SolverUtils
