///////////////////////////////////////////////////////////////////////////////
//
// File: NekSys.h
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// License for the specific language governing rights and limitations under
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
// Description: NekSys header
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_UTILITIES_LINEAR_ALGEBRA_NEK_NekSys_H
#define NEKTAR_LIB_UTILITIES_LINEAR_ALGEBRA_NEK_NekSys_H

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/LinearAlgebra/NekMatrix.hpp>
#include <iomanip>

namespace Nektar::LibUtilities
{

// =====================================================================
// ==== DEFINITION OF THE CLASS  NekSysOperators
// ==== Defines operators needed by iterative solver of nonlinear or
// ==== linear system
// =====================================================================
class NekSysOperators
{
public:
    typedef const Array<OneD, const NekDouble> InArrayType;
    typedef Array<OneD, NekDouble> OutArrayType;

    typedef std::function<void(InArrayType &, OutArrayType &, const bool &)>
        FunctorType1;
    typedef std::function<void(InArrayType &, OutArrayType &, OutArrayType &,
                               const bool &)>
        FunctorType2;
    typedef Array<OneD, FunctorType1> FunctorType1Array;
    typedef Array<OneD, FunctorType2> FunctorType2Array;
    static const int nfunctor1 = 4;
    static const int nfunctor2 = 1;

    NekSysOperators(void) : m_functors1(nfunctor1), m_functors2(nfunctor2)
    {
    }

    NekSysOperators(const NekSysOperators &in)
        : m_functors1(nfunctor1), m_functors2(nfunctor2)
    {
        for (int i = 0; i < nfunctor1; ++i)
        {
            m_functors1[i] = in.m_functors1[i];
        }
        for (int i = 0; i < nfunctor2; ++i)
        {
            m_functors2[i] = in.m_functors2[i];
        }
    }

    NekSysOperators &operator=(const NekSysOperators &in)
    {
        for (int i = 0; i < nfunctor1; ++i)
        {
            m_functors1[i] = in.m_functors1[i];
        }
        for (int i = 0; i < nfunctor2; ++i)
        {
            m_functors2[i] = in.m_functors2[i];
        }

        return *this;
    }

    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineNekSysResEval(FuncPointerT func, ObjectPointerT obj)
    {
        m_functors1[0] =
            std::bind(func, obj, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3);
    }
    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineNekSysLhsEval(FuncPointerT func, ObjectPointerT obj)
    {
        m_functors1[1] =
            std::bind(func, obj, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3);
    }
    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineNekSysPrecon(FuncPointerT func, ObjectPointerT obj)
    {
        m_functors1[2] =
            std::bind(func, obj, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3);
    }
    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineAssembleLoc(FuncPointerT func, ObjectPointerT obj)
    {
        m_functors1[3] =
            std::bind(func, obj, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3);
    }
    template <typename FuncPointerT, typename ObjectPointerT>
    void DefineNekSysFixPointIte(FuncPointerT func, ObjectPointerT obj)
    {
        m_functors2[0] =
            std::bind(func, obj, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4);
    }

    // check if this operator is defined
    bool CheckNekSysFixPointIte()
    {
        return m_functors2[0] ? true : false;
    }

    inline void DoNekSysResEval(InArrayType &inarray, OutArrayType &outarray,
                                const bool &flag = false) const
    {
        ASSERTL1(m_functors1[0], "DoNekSysResEval should be defined");
        m_functors1[0](inarray, outarray, flag);
    }

    inline void DoNekSysLhsEval(InArrayType &inarray, OutArrayType &outarray,
                                const bool &flag = false) const
    {
        ASSERTL1(m_functors1[1], "DoNekSysLhsEval should be defined");
        m_functors1[1](inarray, outarray, flag);
    }

    inline void DoNekSysPrecon(InArrayType &inarray, OutArrayType &outarray,
                               const bool &flag = false) const
    {
        if (m_functors1[2])
        {
            m_functors1[2](inarray, outarray, flag);
        }
        else
        {
            Vmath::Vcopy(outarray.size(), inarray, 1, outarray, 1);
        }
    }

    inline void DoAssembleLoc(InArrayType &xn, OutArrayType &xn1,
                              const bool &flag = false) const
    {
        ASSERTL1(m_functors1[3], "DoAssembleLoc should be defined");
        m_functors1[3](xn, xn1, flag);
    }

    // xn contains with initial guess and overwritten by solution on exit
    // xn1 is filled by final residual for convergence check
    inline void DoNekSysFixPointIte(InArrayType &rhs, OutArrayType &xn,
                                    OutArrayType &xn1,
                                    const bool &flag = false) const
    {
        ASSERTL1(m_functors2[0], "DoNekSysFixPointIte should be defined");
        m_functors2[0](rhs, xn, xn1, flag);
    }

protected:
    /* Defines operators
        DoNekSysResEval   :
            evaluations the residual of the Nonlinear/Linear system
            ie. the residual b-Ax and N(x) for linear and
            nonlinear systems, respectively
            May not be used for linear system.
        DoNekSysLhsEval   :
            evaluations the LHS of the Nonlinear/Linear system (Ax),
            where A is the matrix and x is solution vector.
            For linear system A is the coefficient matrix;
            For nonlinear system A is the coefficient matrix in
            each nonlinear iterations, for example A is the
            Jacobian matrix for Newton method;
        DoNekSysPrecon      :
            Preconditioning operator of the system.
        DoAssembleLoc       :
            Operator to assemble array and scater it back to the lcoal storage
            if flag is true will zero Dirichlet conditions
        DoNekSysFixPointIte :
    */
    FunctorType1Array m_functors1;

    /* Defines one operator
        DoNekSysFixPointIte  :
            Operator to calculate RHS of fixed point iterations
            (x^{n+1}=M^{-1}(b-N*x^{n}), with M+N=A).
    */
    FunctorType2Array m_functors2;
};

class NekSysKey
{
public:
    NekSysKey()  = default;
    ~NekSysKey() = default;

    // NekLinSysIter
    int m_NekLinSysMaxIterations   = 5000;
    NekDouble m_NekLinSysTolerance = NekConstants::kNekIterativeTol;

    // NekNonlinSysIter
    int m_NekNonlinSysMaxIterations            = 100;
    NekDouble m_NekNonLinSysTolerance          = NekConstants::kNekIterativeTol;
    NekDouble m_NonlinIterTolRelativeL2        = 1.0E-06;
    std::string m_LinSysIterSolverTypeInNonlin = "GMRES";

    // GMRES
    int m_LinSysMaxStorage        = 100;
    int m_KrylovMaxHessMatBand    = 101;
    bool m_NekLinSysLeftPrecon    = false;
    bool m_NekLinSysRightPrecon   = true;
    bool m_GMRESCentralDifference = false;
};

class NekSys;

typedef std::shared_ptr<NekSys> NekSysSharedPtr;

class NekSys : public std::enable_shared_from_this<NekSys>
{
public:
    /// Support creation through MemoryManager.
    friend class MemoryManager<NekSys>;

    LIB_UTILITIES_EXPORT static NekSysSharedPtr CreateInstance(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const LibUtilities::CommSharedPtr &vRowComm, const int nDimen,
        const NekSysKey &pKey)
    {
        NekSysSharedPtr p = MemoryManager<NekSys>::AllocateSharedPtr(
            pSession, vRowComm, nDimen, pKey);
        return p;
    }

    LIB_UTILITIES_EXPORT NekSys(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const LibUtilities::CommSharedPtr &vRowComm, const int nDimen,
        const NekSysKey &pKey);
    LIB_UTILITIES_EXPORT virtual ~NekSys() = default;

    LIB_UTILITIES_EXPORT void InitObject()
    {
        v_InitObject();
    }

    LIB_UTILITIES_EXPORT int SolveSystem(
        const int nGlobal, const Array<OneD, const NekDouble> &pInput,
        Array<OneD, NekDouble> &pOutput, const int nDir = 0)
    {
        return v_SolveSystem(nGlobal, pInput, pOutput, nDir);
    }

    LIB_UTILITIES_EXPORT inline const NekSysOperators &GetSysOperators()
    {
        return m_operator;
    }

    LIB_UTILITIES_EXPORT inline void SetSysOperators(const NekSysOperators &in)
    {
        v_SetSysOperators(in);
    }

    LIB_UTILITIES_EXPORT void SetFlagWarnings(bool in)
    {
        m_FlagWarnings = in;
    }

    LIB_UTILITIES_EXPORT void SetRhsMagnitude(const NekDouble mag)
    {
        m_rhs_magnitude = mag;
    }

protected:
    LibUtilities::CommSharedPtr m_rowComm;
    bool m_converged;
    bool m_root;
    bool m_verbose;
    bool m_FlagWarnings;
    int m_SysDimen;

    NekSysOperators m_operator;

    NekDouble m_rhs_magnitude = NekConstants::kNekUnsetDouble;

    virtual void v_InitObject()
    {
    }

    virtual void v_SetSysOperators(const NekSysOperators &in)
    {
        m_operator = in;
    }

    virtual int v_SolveSystem(
        [[maybe_unused]] const int nGlobal,
        [[maybe_unused]] const Array<OneD, const NekDouble> &pInput,
        [[maybe_unused]] Array<OneD, NekDouble> &pOutput,
        [[maybe_unused]] const int nDir)
    {
        ASSERTL0(false, "LinSysIterSolver NOT CORRECT.");
        return 0;
    }
};

} // namespace Nektar::LibUtilities
#endif
