///////////////////////////////////////////////////////////////////////////////
//
// File: LinearAdvectionDiffusionReaction.cpp
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
// Description: LinearAdvectionDiffusionReaction operator implementations
//
///////////////////////////////////////////////////////////////////////////////

#include <Collections/Collection.h>
#include <Collections/IProduct.h>
#include <Collections/MatrixFreeBase.h>
#include <Collections/Operator.h>
#include <MatrixFreeOps/Operator.hpp>

using namespace std;

namespace Nektar::Collections
{

using LibUtilities::eHexahedron;
using LibUtilities::ePrism;
using LibUtilities::ePyramid;
using LibUtilities::eQuadrilateral;
using LibUtilities::eSegment;
using LibUtilities::eTetrahedron;
using LibUtilities::eTriangle;

/**
 * @brief LinearAdvectionDiffusionReaction help class to calculate the size of
 * the collection that is given as an input and as an output to the
 * LinearAdvectionDiffusionReaction Operator. The size evaluation takes into
 * account that the evaluation of the LinearAdvectionDiffusionReaction operator
 * takes input from the coeff space and gives the output in the coeff space too.
 */
class LinearAdvectionDiffusionReaction_Helper : virtual public Operator
{
protected:
    LinearAdvectionDiffusionReaction_Helper()
    {
        // expect input to be number of elements by the number of coefficients
        m_inputSize      = m_numElmt * m_stdExp->GetNcoeffs();
        m_inputSizeOther = m_numElmt * m_stdExp->GetTotPoints();

        // expect output to be number of elements by the number of coefficients
        // computation is from coeff space to coeff space
        m_outputSize      = m_inputSize;
        m_outputSizeOther = m_inputSizeOther;
    }
};

/**
 * @brief LinearAdvectionDiffusionReaction operator using LocalRegions
 * implementation.
 */
class LinearAdvectionDiffusionReaction_NoCollection final
    : virtual public Operator,
      LinearAdvectionDiffusionReaction_Helper
{
public:
    OPERATOR_CREATE(LinearAdvectionDiffusionReaction_NoCollection)

    ~LinearAdvectionDiffusionReaction_NoCollection() final = default;

    void operator()(const Array<OneD, const NekDouble> &entry0,
                    Array<OneD, NekDouble> &entry1,
                    [[maybe_unused]] Array<OneD, NekDouble> &entry2,
                    [[maybe_unused]] Array<OneD, NekDouble> &entry3,
                    [[maybe_unused]] Array<OneD, NekDouble> &wsp) final
    {
        unsigned int nmodes = m_expList[0]->GetNcoeffs();
        unsigned int nphys  = m_expList[0]->GetTotPoints();
        Array<OneD, NekDouble> tmp;

        for (int n = 0; n < m_numElmt; ++n)
        {
            // Restrict varcoeffs to size of element
            StdRegions::VarCoeffMap varcoeffs = StdRegions::NullVarCoeffMap;
            if (m_varcoeffs.size())
            {
                varcoeffs =
                    StdRegions::RestrictCoeffMap(m_varcoeffs, n * nphys, nphys);
            }

            StdRegions::StdMatrixKey mkey(
                StdRegions::eLinearAdvectionDiffusionReaction,
                (m_expList)[n]->DetShapeType(), *(m_expList)[n], m_factors,
                varcoeffs);
            m_expList[n]->GeneralMatrixOp(entry0 + n * nmodes,
                                          tmp = entry1 + n * nmodes, mkey);
        }
    }

    void operator()([[maybe_unused]] int dir,
                    [[maybe_unused]] const Array<OneD, const NekDouble> &input,
                    [[maybe_unused]] Array<OneD, NekDouble> &output,
                    [[maybe_unused]] Array<OneD, NekDouble> &wsp) final
    {
        NEKERROR(ErrorUtil::efatal, "Not valid for this operator.");
    }

    void UpdateFactors(StdRegions::FactorMap factors) override
    {
        m_factors = factors;
    }

    /**
     * @brief Check whether necessary advection velocities are supplied
     * and copy into local member for varcoeff map.
     *
     * @param varcoeffs Map of variable coefficients
     */
    void UpdateVarcoeffs(StdRegions::VarCoeffMap &varcoeffs) override
    {
        m_varcoeffs = varcoeffs;
    }

protected:
    int m_dim;
    int m_coordim;
    vector<StdRegions::StdExpansionSharedPtr> m_expList;
    StdRegions::FactorMap m_factors;
    StdRegions::VarCoeffMap m_varcoeffs;

private:
    LinearAdvectionDiffusionReaction_NoCollection(
        vector<StdRegions::StdExpansionSharedPtr> pCollExp,
        CoalescedGeomDataSharedPtr pGeomData, StdRegions::FactorMap factors)
        : Operator(pCollExp, pGeomData, factors),
          LinearAdvectionDiffusionReaction_Helper()
    {
        m_expList = pCollExp;
        m_dim     = pCollExp[0]->GetNumBases();
        m_coordim = pCollExp[0]->GetCoordim();

        m_factors   = factors;
        m_varcoeffs = StdRegions::NullVarCoeffMap;
    }
};

/// Factory initialisation for the LinearAdvectionDiffusionReaction_NoCollection
/// operators
OperatorKey LinearAdvectionDiffusionReaction_NoCollection::m_typeArr[] = {
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eSegment, eLinearAdvectionDiffusionReaction, eNoCollection,
                    false),
        LinearAdvectionDiffusionReaction_NoCollection::create,
        "LinearAdvectionDiffusionReaction_NoCollection_Seg"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eTriangle, eLinearAdvectionDiffusionReaction, eNoCollection,
                    false),
        LinearAdvectionDiffusionReaction_NoCollection::create,
        "LinearAdvectionDiffusionReaction_NoCollection_Tri"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eTriangle, eLinearAdvectionDiffusionReaction, eNoCollection,
                    true),
        LinearAdvectionDiffusionReaction_NoCollection::create,
        "LinearAdvectionDiffusionReaction_NoCollection_NodalTri"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eQuadrilateral, eLinearAdvectionDiffusionReaction,
                    eNoCollection, false),
        LinearAdvectionDiffusionReaction_NoCollection::create,
        "LinearAdvectionDiffusionReaction_NoCollection_Quad"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eTetrahedron, eLinearAdvectionDiffusionReaction,
                    eNoCollection, false),
        LinearAdvectionDiffusionReaction_NoCollection::create,
        "LinearAdvectionDiffusionReaction_NoCollection_Tet"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eTetrahedron, eLinearAdvectionDiffusionReaction,
                    eNoCollection, true),
        LinearAdvectionDiffusionReaction_NoCollection::create,
        "LinearAdvectionDiffusionReaction_NoCollection_NodalTet"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(ePyramid, eLinearAdvectionDiffusionReaction, eNoCollection,
                    false),
        LinearAdvectionDiffusionReaction_NoCollection::create,
        "LinearAdvectionDiffusionReaction_NoCollection_Pyr"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(ePrism, eLinearAdvectionDiffusionReaction, eNoCollection,
                    false),
        LinearAdvectionDiffusionReaction_NoCollection::create,
        "LinearAdvectionDiffusionReaction_NoCollection_Prism"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(ePrism, eLinearAdvectionDiffusionReaction, eNoCollection,
                    true),
        LinearAdvectionDiffusionReaction_NoCollection::create,
        "LinearAdvectionDiffusionReaction_NoCollection_NodalPrism"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eHexahedron, eLinearAdvectionDiffusionReaction,
                    eNoCollection, false),
        LinearAdvectionDiffusionReaction_NoCollection::create,
        "LinearAdvectionDiffusionReaction_NoCollection_Hex")};

/**
 * @brief LinearAdvectionDiffusionReaction operator using LocalRegions
 * implementation.
 */
class LinearAdvectionDiffusionReaction_IterPerExp final
    : virtual public Operator,
      LinearAdvectionDiffusionReaction_Helper
{
public:
    OPERATOR_CREATE(LinearAdvectionDiffusionReaction_IterPerExp)

    ~LinearAdvectionDiffusionReaction_IterPerExp() final = default;

    void operator()(const Array<OneD, const NekDouble> &input,
                    Array<OneD, NekDouble> &output,
                    [[maybe_unused]] Array<OneD, NekDouble> &output1,
                    [[maybe_unused]] Array<OneD, NekDouble> &output2,
                    Array<OneD, NekDouble> &wsp) final
    {
        const int nCoeffs = m_stdExp->GetNcoeffs();
        const int nPhys   = m_stdExp->GetTotPoints();

        ASSERTL1(input.size() >= m_numElmt * nCoeffs,
                 "input array size is insufficient");
        ASSERTL1(output.size() >= m_numElmt * nCoeffs,
                 "output array size is insufficient");

        Array<OneD, NekDouble> tmpphys, t1;
        Array<OneD, Array<OneD, NekDouble>> dtmp(3);
        Array<OneD, Array<OneD, NekDouble>> tmp(3);

        tmpphys = wsp;
        for (int i = 1; i < m_coordim + 1; ++i)
        {
            dtmp[i - 1] = wsp + i * nPhys;
            tmp[i - 1]  = wsp + (i + m_coordim) * nPhys;
        }

        for (int i = 0; i < m_numElmt; ++i)
        {
            // Std u
            m_stdExp->BwdTrans(input + i * nCoeffs, tmpphys);

            // Std \nabla u
            m_stdExp->PhysDeriv(tmpphys, dtmp[0], dtmp[1], dtmp[2]);

            // Transform Std \nabla u -> Local \nabla u
            // tmp[0] = dxi1/dx du/dxi1 + dxi2/dx du/dxi2 = du / dx
            // tmp[1] = dxi1/dy du/dxi1 + dxi2/dy du/dxi2 = du / dy
            if (m_isDeformed)
            {
                for (int j = 0; j < m_coordim; ++j)
                {
                    Vmath::Vmul(nPhys,
                                m_derivFac[j * m_dim].origin() + i * nPhys, 1,
                                &dtmp[0][0], 1, &tmp[j][0], 1);

                    for (int k = 1; k < m_dim; ++k)
                    {
                        Vmath::Vvtvp(
                            nPhys,
                            m_derivFac[j * m_dim + k].origin() + i * nPhys, 1,
                            &dtmp[k][0], 1, &tmp[j][0], 1, &tmp[j][0], 1);
                    }
                }
            }
            else
            {
                for (int j = 0; j < m_coordim; ++j)
                {
                    Vmath::Smul(nPhys, m_derivFac[j * m_dim][i], &dtmp[0][0], 1,
                                &tmp[j][0], 1);

                    for (int k = 1; k < m_dim; ++k)
                    {
                        Vmath::Svtvp(nPhys, m_derivFac[j * m_dim + k][i],
                                     &dtmp[k][0], 1, &tmp[j][0], 1, &tmp[j][0],
                                     1);
                    }
                }
            }

            /// Mass term
            // tmpphys *= \lambda = \lambda u
            Vmath::Smul(nPhys, -m_lambda, tmpphys, 1, tmpphys, 1);

            /// Add advection term
            // tmpphys += V[0] * du / dx + V[1] * du / dy
            for (int j = 0; j < m_coordim; ++j)
            {
                Vmath::Vvtvp(nPhys, m_advVel[j] + i * nPhys, 1, tmp[j], 1,
                             tmpphys, 1, tmpphys, 1);
            }

            // Multiply by Jacobian for inner product
            if (m_isDeformed)
            {
                Vmath::Vmul(nPhys, m_jac + i * nPhys, 1, tmpphys, 1, tmpphys,
                            1);
            }
            else
            {
                Vmath::Smul(nPhys, m_jac[i], tmpphys, 1, tmpphys, 1);
            }

            // Compute inner product wrt base
            m_stdExp->IProductWRTBase(tmpphys, t1 = output + i * nCoeffs);

            /// Determine Laplacian term
            // add derivFactors for IProdWRTDerivBase below
            // dtmp[0] = dxi1/dx du/dx + dxi1/dy du/dy
            // dtmp[1] = dxi2/dx du/dx + dxi2/dy du/dy
            if (m_isDeformed)
            {
                for (int j = 0; j < m_dim; ++j)
                {
                    Vmath::Vmul(nPhys, m_derivFac[j].origin() + i * nPhys, 1,
                                &tmp[0][0], 1, &dtmp[j][0], 1);

                    for (int k = 1; k < m_coordim; ++k)
                    {
                        Vmath::Vvtvp(
                            nPhys,
                            m_derivFac[j + k * m_dim].origin() + i * nPhys, 1,
                            &tmp[k][0], 1, &dtmp[j][0], 1, &dtmp[j][0], 1);
                    }
                }

                // calculate Iproduct WRT Std Deriv
                for (int j = 0; j < m_dim; ++j)
                {

                    // multiply by Jacobian
                    Vmath::Vmul(nPhys, m_jac + i * nPhys, 1, dtmp[j], 1,
                                dtmp[j], 1);

                    m_stdExp->IProductWRTDerivBase(j, dtmp[j], tmp[0]);
                    Vmath::Vadd(nCoeffs, tmp[0], 1, output + i * nCoeffs, 1,
                                t1 = output + i * nCoeffs, 1);
                }
            }
            else
            {
                for (int j = 0; j < m_dim; ++j)
                {
                    Vmath::Smul(nPhys, m_derivFac[j][i], &tmp[0][0], 1,
                                &dtmp[j][0], 1);

                    for (int k = 1; k < m_coordim; ++k)
                    {
                        Vmath::Svtvp(nPhys, m_derivFac[j + k * m_dim][i],
                                     &tmp[k][0], 1, &dtmp[j][0], 1, &dtmp[j][0],
                                     1);
                    }
                }

                // calculate Iproduct WRT Std Deriv
                for (int j = 0; j < m_dim; ++j)
                {
                    // multiply by Jacobian for integration
                    Vmath::Smul(nPhys, m_jac[i], dtmp[j], 1, dtmp[j], 1);

                    m_stdExp->IProductWRTDerivBase(j, dtmp[j], tmp[0]);
                    Vmath::Vadd(nCoeffs, tmp[0], 1, output + i * nCoeffs, 1,
                                t1 = output + i * nCoeffs, 1);
                }
            }
        }
    }

    void operator()([[maybe_unused]] int dir,
                    [[maybe_unused]] const Array<OneD, const NekDouble> &input,
                    [[maybe_unused]] Array<OneD, NekDouble> &output,
                    [[maybe_unused]] Array<OneD, NekDouble> &wsp) final
    {
        NEKERROR(ErrorUtil::efatal, "Not valid for this operator.");
    }

    /**
     * @brief Check the validity of supplied constant factors.
     *
     * @param factors Map of factors
     */
    void UpdateFactors(StdRegions::FactorMap factors) override
    {
        m_factors = factors;

        // Check Lambda constant of LinearAdvectionDiffusionReaction operator
        auto x = factors.find(StdRegions::eFactorLambda);
        ASSERTL1(
            x != factors.end(),
            "Constant factor not defined: " +
                std::string(
                    StdRegions::ConstFactorTypeMap[StdRegions::eFactorLambda]));
        m_lambda = x->second;
    }

    /**
     * @brief Check whether necessary advection velocities are supplied
     * and copy into local array for operator.
     *
     * @param varcoeffs Map of variable coefficients
     */
    void UpdateVarcoeffs(StdRegions::VarCoeffMap &varcoeffs) override
    {
        // Check whether any varcoeffs are provided
        if (varcoeffs.empty())
        {
            return;
        }

        // Check advection velocities count
        int ndir = 0;
        for (auto &x : advVelTypes)
        {
            if (varcoeffs.count(x))
            {
                ndir++;
            }
        }
        ASSERTL0(ndir, "Must define at least one advection velocity");
        ASSERTL1(ndir <= m_coordim,
                 "Number of constants is larger than coordinate dimensions");

        // Copy new varcoeffs
        m_varcoeffs = varcoeffs;

        // Hold advection velocity reference
        // separate copy required for operator
        for (int i = 0; i < m_coordim; i++)
        {
            m_advVel[i] = m_varcoeffs.find(advVelTypes[i])->second.GetValue();
        }
    }

protected:
    Array<TwoD, const NekDouble> m_derivFac;
    Array<OneD, const NekDouble> m_jac;
    int m_dim;
    int m_coordim;
    StdRegions::FactorMap m_factors;
    StdRegions::VarCoeffMap m_varcoeffs;
    NekDouble m_lambda;
    Array<OneD, Array<OneD, NekDouble>> m_advVel;
    const StdRegions::VarCoeffType advVelTypes[3] = {StdRegions::eVarCoeffVelX,
                                                     StdRegions::eVarCoeffVelY,
                                                     StdRegions::eVarCoeffVelZ};

private:
    LinearAdvectionDiffusionReaction_IterPerExp(
        vector<StdRegions::StdExpansionSharedPtr> pCollExp,
        CoalescedGeomDataSharedPtr pGeomData, StdRegions::FactorMap factors)
        : Operator(pCollExp, pGeomData, factors),
          LinearAdvectionDiffusionReaction_Helper()
    {
        m_dim     = pCollExp[0]->GetShapeDimension();
        m_coordim = pCollExp[0]->GetCoordim();
        int nqtot = m_stdExp->GetTotPoints();

        m_derivFac = pGeomData->GetDerivFactors(pCollExp);
        m_jac      = pGeomData->GetJac(pCollExp);
        m_wspSize  = (2 * m_coordim + 1) * nqtot;

        m_lambda    = 1.0;
        m_advVel    = Array<OneD, Array<OneD, NekDouble>>(m_coordim);
        m_factors   = StdRegions::NullFactorMap;
        m_varcoeffs = StdRegions::NullVarCoeffMap;
        this->UpdateFactors(factors);
    }
};

/// Factory initialisation for the
/// LinearAdvectionDiffusionReaction_IterPerExp
OperatorKey LinearAdvectionDiffusionReaction_IterPerExp::m_typeArr[] = {
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eSegment, eLinearAdvectionDiffusionReaction, eIterPerExp,
                    false),
        LinearAdvectionDiffusionReaction_IterPerExp::create,
        "LinearAdvectionDiffusionReaction_IterPerExp_Seg"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eTriangle, eLinearAdvectionDiffusionReaction, eIterPerExp,
                    false),
        LinearAdvectionDiffusionReaction_IterPerExp::create,
        "LinearAdvectionDiffusionReaction_IterPerExp_Tri"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eTriangle, eLinearAdvectionDiffusionReaction, eIterPerExp,
                    true),
        LinearAdvectionDiffusionReaction_IterPerExp::create,
        "LinearAdvectionDiffusionReaction_IterPerExp_NodalTri"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eQuadrilateral, eLinearAdvectionDiffusionReaction,
                    eIterPerExp, false),
        LinearAdvectionDiffusionReaction_IterPerExp::create,
        "LinearAdvectionDiffusionReaction_IterPerExp_Quad"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eTetrahedron, eLinearAdvectionDiffusionReaction,
                    eIterPerExp, false),
        LinearAdvectionDiffusionReaction_IterPerExp::create,
        "LinearAdvectionDiffusionReaction_IterPerExp_Tet"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eTetrahedron, eLinearAdvectionDiffusionReaction,
                    eIterPerExp, true),
        LinearAdvectionDiffusionReaction_IterPerExp::create,
        "LinearAdvectionDiffusionReaction_IterPerExp_NodalTet"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(ePyramid, eLinearAdvectionDiffusionReaction, eIterPerExp,
                    false),
        LinearAdvectionDiffusionReaction_IterPerExp::create,
        "LinearAdvectionDiffusionReaction_IterPerExp_Pyr"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(ePrism, eLinearAdvectionDiffusionReaction, eIterPerExp,
                    false),
        LinearAdvectionDiffusionReaction_IterPerExp::create,
        "LinearAdvectionDiffusionReaction_IterPerExp_Prism"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(ePrism, eLinearAdvectionDiffusionReaction, eIterPerExp,
                    true),
        LinearAdvectionDiffusionReaction_IterPerExp::create,
        "LinearAdvectionDiffusionReaction_IterPerExp_NodalPrism"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eHexahedron, eLinearAdvectionDiffusionReaction, eIterPerExp,
                    false),
        LinearAdvectionDiffusionReaction_IterPerExp::create,
        "LinearAdvectionDiffusionReaction_IterPerExp_Hex")};

/*
 * @brief LinearAdvectionDiffusionReaction operator using matrix free
 * operators.
 */
class LinearAdvectionDiffusionReaction_MatrixFree final
    : virtual public Operator,
      MatrixFreeBase,
      LinearAdvectionDiffusionReaction_Helper
{
public:
    OPERATOR_CREATE(LinearAdvectionDiffusionReaction_MatrixFree)

    ~LinearAdvectionDiffusionReaction_MatrixFree() final = default;

    void operator()(const Array<OneD, const NekDouble> &input,
                    Array<OneD, NekDouble> &output0,
                    [[maybe_unused]] Array<OneD, NekDouble> &output1,
                    [[maybe_unused]] Array<OneD, NekDouble> &output2,
                    [[maybe_unused]] Array<OneD, NekDouble> &wsp) final
    {
        (*m_oper)(input, output0);
    }

    void operator()([[maybe_unused]] int dir,
                    [[maybe_unused]] const Array<OneD, const NekDouble> &input,
                    [[maybe_unused]] Array<OneD, NekDouble> &output,
                    [[maybe_unused]] Array<OneD, NekDouble> &wsp) final
    {
        NEKERROR(ErrorUtil::efatal, "Not valid for this operator.");
    }

    /**
     *
     */
    void UpdateFactors(StdRegions::FactorMap factors) override
    {
        m_factors = factors;

        // Set lambda for this call
        auto x = factors.find(StdRegions::eFactorLambda);
        ASSERTL1(
            x != factors.end(),
            "Constant factor not defined: " +
                std::string(
                    StdRegions::ConstFactorTypeMap[StdRegions::eFactorLambda]));
        m_oper->SetLambda(-1 * x->second);
    }

    /**
     * @brief Check whether necessary advection velocities are supplied,
     * interleave and copy into vectorised containers.
     *
     * @param varcoeffs Map of variable coefficients
     */
    void UpdateVarcoeffs(StdRegions::VarCoeffMap &varcoeffs) override
    {
        // Check varcoeffs empty?
        if (varcoeffs.empty())
        {
            return;
        }

        // Check whether varcoeffs need update (copied from
        // GlobalMatrixKey.cpp) This is essential to only do the update +
        // interleaving after each time step instead of doing it every time
        // we apply the operator.
        bool update = false;
        if (m_varcoeffs.size() < varcoeffs.size())
        {
            update = true;
        }
        else if (m_varcoeffs.size() > varcoeffs.size())
        {
            update = true;
        }
        else
        {
            StdRegions::VarCoeffMap::const_iterator x, y;
            for (x = m_varcoeffs.begin(), y = varcoeffs.begin();
                 x != m_varcoeffs.end(); ++x, ++y)
            {
                if (x->second.GetHash() < y->second.GetHash())
                {
                    update = true;
                    break;
                }
                if (x->second.GetHash() > y->second.GetHash())
                {
                    update = true;
                    break;
                }
            }
        }
        // return if no update required
        if (!update)
        {
            return;
        }
        // else copy
        m_varcoeffs = varcoeffs;

        // Check advection velocities count
        int ndir = 0;
        for (auto &x : advVelTypes)
        {
            if (m_varcoeffs.count(x))
            {
                ndir++;
            }
        }
        ASSERTL0(ndir, "Must define at least one advection velocity");
        ASSERTL1(ndir >= m_coordim,
                 "Number of coordingates is larger than number of constants "
                 "provided (ndir = " +
                     std::to_string(ndir) +
                     ", m_coordim = " + std::to_string(m_coordim) + ")");

        // Extract advection velocities, interleave and pass to
        // libMatrixFree Multiply by -1/lambda for combined (Mass +
        // Advection) IProduct operation in MatrixFree/AdvectionKernel
        auto lambda = m_factors.find(StdRegions::eFactorLambda);
        Array<OneD, Array<OneD, NekDouble>> advVel(m_coordim);
        for (int i = 0; i < m_coordim; i++)
        {
            advVel[i] = Array<OneD, NekDouble>(
                m_varcoeffs.find(advVelTypes[i])->second.GetValue());
            Vmath::Smul(advVel[i].size(), -1 / lambda->second, advVel[i], 1,
                        advVel[i], 1);
        }

        // Interleave Advection Velocities
        using namespace tinysimd;
        using vec_t = simd<NekDouble>;
        typedef std::vector<vec_t, tinysimd::allocator<vec_t>> VecVec_t;

        // Arguments of GetDerivFactorsInterLeave function
        int nElmt = m_nElmtPad;

        ASSERTL1(nElmt % vec_t::width == 0,
                 "Number of elements not divisible by vector "
                 "width, padding not yet implemented.");

        int nBlocks = nElmt / vec_t::width;

        unsigned int n_vel = m_coordim;
        alignas(vec_t::alignment) NekDouble vec[vec_t::width];

        VecVec_t newAdvVel;
        int nq        = m_nqtot / m_numElmt;
        int totalsize = m_nqtot; // nq * n_vel;

        // The advection velocity varies at every quad point
        // It is independent of DEFORMED
        newAdvVel.resize(nBlocks * n_vel * nq);
        auto *advVel_ptr = &newAdvVel[0];
        for (int e = 0; e < nBlocks; ++e)
        {
            for (int q = 0; q < nq; q++)
            {
                for (int dir = 0; dir < n_vel; ++dir, ++advVel_ptr)
                {
                    for (int j = 0; j < vec_t::width; ++j)
                    {
                        if ((vec_t::width * e + j) * nq + q < totalsize)
                        {
                            vec[j] =
                                advVel[dir][(vec_t::width * e + j) * nq + q];
                        }
                        else
                        {
                            vec[j] = 0.0;
                        }
                    }
                    (*advVel_ptr).load(&vec[0]);
                }
            }
        }

        m_oper->SetAdvectionVelocities(
            MemoryManager<VecVec_t>::AllocateSharedPtr(newAdvVel));
    }

private:
    std::shared_ptr<MatrixFree::LinearAdvectionDiffusionReaction> m_oper;
    unsigned int m_nmtot;
    unsigned int m_nqtot;
    int m_coordim;
    StdRegions::FactorMap m_factors;
    StdRegions::VarCoeffMap m_varcoeffs;
    const StdRegions::VarCoeffType advVelTypes[3] = {StdRegions::eVarCoeffVelX,
                                                     StdRegions::eVarCoeffVelY,
                                                     StdRegions::eVarCoeffVelZ};

    LinearAdvectionDiffusionReaction_MatrixFree(
        vector<StdRegions::StdExpansionSharedPtr> pCollExp,
        CoalescedGeomDataSharedPtr pGeomData, StdRegions::FactorMap factors)
        : Operator(pCollExp, pGeomData, factors),
          MatrixFreeBase(pCollExp[0]->GetStdExp()->GetNcoeffs(),
                         pCollExp[0]->GetStdExp()->GetNcoeffs(),
                         pCollExp.size()),
          LinearAdvectionDiffusionReaction_Helper()
    {
        m_nmtot = m_numElmt * pCollExp[0]->GetStdExp()->GetNcoeffs();

        const auto dim = pCollExp[0]->GetStdExp()->GetShapeDimension();
        m_coordim      = dim;

        // Basis vector.
        std::vector<LibUtilities::BasisSharedPtr> basis(dim);
        for (auto i = 0; i < dim; ++i)
        {
            basis[i] = pCollExp[0]->GetBasis(i);
        }

        // Get shape type
        auto shapeType = pCollExp[0]->GetStdExp()->DetShapeType();

        // Generate operator string and create operator.
        std::string op_string = "LinearAdvectionDiffusionReaction";
        op_string += MatrixFree::GetOpstring(shapeType, m_isDeformed);
        auto oper = MatrixFree::GetOperatorFactory().CreateInstance(
            op_string, basis, pCollExp.size());

        // Get N quadpoints with padding
        m_nqtot = m_numElmt * pCollExp[0]->GetStdExp()->GetTotPoints();

        // set up required copies for operations
        oper->SetUpBdata(basis);
        oper->SetUpDBdata(basis);
        oper->SetUpD(basis);
        oper->SetUpZW(basis);

        // Set Jacobian
        oper->SetJac(pGeomData->GetJacInterLeave(pCollExp, m_nElmtPad));

        // Store derivative factor
        oper->SetDF(pGeomData->GetDerivFactorsInterLeave(pCollExp, m_nElmtPad));

        m_oper = std::dynamic_pointer_cast<
            MatrixFree::LinearAdvectionDiffusionReaction>(oper);
        ASSERTL0(m_oper, "Failed to cast pointer.");

        // Set factors
        m_factors   = StdRegions::NullFactorMap;
        m_varcoeffs = StdRegions::NullVarCoeffMap;
        this->UpdateFactors(factors);
    }
};

/// Factory initialisation for the
/// LinearAdvectionDiffusionReaction_MatrixFree operators
OperatorKey LinearAdvectionDiffusionReaction_MatrixFree::m_typeArr[] = {
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eQuadrilateral, eLinearAdvectionDiffusionReaction,
                    eMatrixFree, false),
        LinearAdvectionDiffusionReaction_MatrixFree::create,
        "LinearAdvectionDiffusionReaction_MatrixFree_Quad"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eTriangle, eLinearAdvectionDiffusionReaction, eMatrixFree,
                    false),
        LinearAdvectionDiffusionReaction_MatrixFree::create,
        "LinearAdvectionDiffusionReaction_MatrixFree_Tri"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eHexahedron, eLinearAdvectionDiffusionReaction, eMatrixFree,
                    false),
        LinearAdvectionDiffusionReaction_MatrixFree::create,
        "LinearAdvectionDiffusionReaction_MatrixFree_Hex"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(ePrism, eLinearAdvectionDiffusionReaction, eMatrixFree,
                    false),
        LinearAdvectionDiffusionReaction_MatrixFree::create,
        "LinearAdvectionDiffusionReaction_MatrixFree_Prism"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(ePyramid, eLinearAdvectionDiffusionReaction, eMatrixFree,
                    false),
        LinearAdvectionDiffusionReaction_MatrixFree::create,
        "LinearAdvectionDiffusionReaction_MatrixFree_Pyr"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eTetrahedron, eLinearAdvectionDiffusionReaction,
                    eMatrixFree, false),
        LinearAdvectionDiffusionReaction_MatrixFree::create,
        "LinearAdvectionDiffusionReaction_MatrixFree_Tet"),
};

} // namespace Nektar::Collections
