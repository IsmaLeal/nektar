///////////////////////////////////////////////////////////////////////////////
//
// File: FilterDOMeasures.h
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
// Description: SolverUtils Filter that writes, every OutputFrequency steps,
//              the asymmetry measures of the sudden-expansion campaign
//              computed on the expansion with the solver's own quadrature:
//              the inner products with the reference direction phi, the
//              centreline and wall lines, and per particle A, M, M1, y_b,
//              Delta L and Y_1. Runs on DOVelocityCorrectionScheme, with
//              DOModes = 0 (a DNS) included.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_INCNAVIERSTOKES_FILTERS_FILTERDOMEASURES_H
#define NEKTAR_INCNAVIERSTOKES_FILTERS_FILTERDOMEASURES_H

#include <LibUtilities/BasicUtils/H5.h>
#include <SolverUtils/Filters/Filter.h>
#include <StdRegions/StdSegExp.h>

#include <map>
#include <string>
#include <vector>

namespace Nektar::SolverUtils
{

class FilterDOMeasures : public Filter
{
public:
    static FilterSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const std::shared_ptr<EquationSystem> &pEquation,
        const ParamMap &pParams)
    {
        return MemoryManager<FilterDOMeasures>::AllocateSharedPtr(
            pSession, pEquation, pParams);
    }

    static std::string className;

    FilterDOMeasures(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const std::shared_ptr<EquationSystem> &pEquation,
        const ParamMap &pParams);

    ~FilterDOMeasures() override = default;

protected:
    void v_Initialise(
        const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
        const NekDouble &time) override;
    void v_Update(
        const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
        const NekDouble &time) override;
    void v_Finalise(
        const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
        const NekDouble &time) override;
    bool v_IsTimeDependent() override;

private:
    /// one edge of a mesh line (the centreline or a wall) held by this rank
    struct LineEdge
    {
        int elmt;                     ///< local element index
        int edge;                     ///< local edge index in the element
        int gpos;                     ///< position along the line, all ranks
        int npts;                     ///< quadrature points on the edge
        NekDouble sign;               ///< walls: +1 lower, -1 upper (normal
                                      ///< into the fluid); centreline: 1
        Array<OneD, int> sorted;      ///< sorted[k] = raw index of the k-th
                                      ///< point by increasing x
        Array<OneD, NekDouble> x;     ///< x of the points, raw element order
        Array<OneD, NekDouble> w;     ///< 1D quadrature weight times the edge
                                      ///< Jacobian, raw element order
    };

    /// a dataset of the output file and its dataspace, for row writes
    struct OutSet
    {
        LibUtilities::H5::DataSetSharedPtr   set;
        LibUtilities::H5::DataSpaceSharedPtr space;
        std::vector<hsize_t>                 dims;
    };

    void FindLines(
        const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
        const Array<OneD, const NekDouble> &x,
        const Array<OneD, const NekDouble> &y);
    void AssignPositions(std::vector<LineEdge> &edges, int &nEdgesGlobal,
                         Array<OneD, NekDouble> &xGlobal);
    void BuildPhi(
        const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
        const Array<OneD, const NekDouble> &x,
        const Array<OneD, const NekDouble> &y);
    void OpenOutput();
    void Compute(
        const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
        int step, NekDouble time);
    void ExtractEdge(const LocalRegions::ExpansionSharedPtr &exp,
                     const LineEdge &e,
                     const Array<OneD, const NekDouble> &elmtVals,
                     Array<OneD, NekDouble> &out);
    NekDouble Reattachment(const NekDouble *tau, const NekDouble *xw,
                           int nEdges) const;
    template <class T>
    void AddSet(const std::string &name, const std::vector<hsize_t> &dims);
    template <class T>
    void PutRow(const std::string &name, hsize_t row,
                const std::vector<T> &data);

    // parameters
    std::string  m_outName;
    unsigned int m_outputFrequency = 1;
    unsigned int m_lineFrequency   = 1;
    std::string  m_fixedPointName  = "FixedPoint";

    // sizes
    int m_nModes     = 0;   ///< S
    int m_nParticles = 0;   ///< Np
    int m_nPhys      = 0;   ///< quadrature points of this rank
    int m_pIdx       = -1;  ///< index of the pressure field
    int m_nptsEdge   = 0;   ///< quadrature points per edge (uniform)
    unsigned int m_index = 0;
    hsize_t m_row = 0, m_lineRow = 0, m_nRows = 0, m_nLineRows = 0;

    LibUtilities::CommSharedPtr m_comm;

    // ---- quadrature weights and the x >= 0 part of the domain
    Array<OneD, NekDouble> m_w;        ///< w_q J_q (the solver's m_physWeights)
    Array<OneD, NekDouble> m_wPlus;    ///< m_w on the elements with x >= 0, 0 elsewhere
    Array<OneD, NekDouble> m_wPlusY;   ///< m_wPlus times y
    NekDouble m_areaPlus = 0.0;        ///< <1, 1>_+
    NekDouble m_yPlus    = 0.0;        ///< <1, y>_+

    // ---- the reference direction phi and its weighted copy
    Array<OneD, NekDouble> m_phiU, m_phiV, m_wPhiU, m_wPhiV;
    NekDouble m_antisymNorm   = 0.0;   ///< ||antisym(u_fp)||
    NekDouble m_mirrorMismatch = 0.0;  ///< largest distance point -> mirror point

    // ---- the lines
    std::vector<LineEdge> m_axis;      ///< centreline edges with x >= 0
    std::vector<LineEdge> m_wall[2];   ///< [0] lower wall, [1] upper wall
    int m_nAxisEdges = 0, m_nWallEdges = 0;   ///< over all ranks
    int m_nAxisPts = 0, m_nWallPts = 0;       ///< edges times points per edge
    Array<OneD, NekDouble> m_xAxis;           ///< x of the centreline points, sorted
    Array<OneD, NekDouble> m_xWall[2];        ///< x of the wall points, sorted
    int m_sensorEdge = -1;                    ///< local centreline edge holding SENSOR_X
    NekDouble m_sensorXi = 0.0;               ///< its local coordinate there
    StdRegions::StdSegExpSharedPtr m_seg;     ///< the edge polynomial, for evaluation

    // ---- scratch
    Array<OneD, NekDouble> m_physU, m_physV;             ///< BwdTrans fallbacks
    Array<OneD, NekDouble> m_wu, m_wv, m_wui, m_wvi;      ///< weighted fields
    Array<OneD, NekDouble> m_elmtBuf, m_elmtBuf2;        ///< one element
    std::vector<Array<OneD, NekDouble>> m_edgeBuf;       ///< (S+1) edges
    Array<OneD, NekDouble> m_tau;                        ///< Np x 2 x nWallPts

    // ---- output
    LibUtilities::H5::FileSharedPtr m_file;
    std::map<std::string, OutSet>   m_sets;
};

} // namespace Nektar::SolverUtils

#endif // NEKTAR_INCNAVIERSTOKES_FILTERS_FILTERDOMEASURES_H
