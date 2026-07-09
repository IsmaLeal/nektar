///////////////////////////////////////////////////////////////////////////////
//
// File: GlobalLinSysIterativeStaticCond.h
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
// Description: GlobalLinSysIterativeStaticCond header
//
///////////////////////////////////////////////////////////////////////////////
#ifndef NEKTAR_LIB_MULTIREGIONS_GLOBALLINSYSITERATIVESTATICCOND_H
#define NEKTAR_LIB_MULTIREGIONS_GLOBALLINSYSITERATIVESTATICCOND_H

#include <LibUtilities/LinearAlgebra/SparseMatrixFwd.hpp>
#include <MultiRegions/GlobalLinSysIterative.h>
#include <MultiRegions/GlobalLinSysStaticCond.h>
#include <MultiRegions/GlobalMatrix.h>

namespace Nektar::MultiRegions
{
// Forward declarations
class ExpList;
class GlobalLinSysIterativeStaticCond;

typedef std::shared_ptr<GlobalLinSysIterativeStaticCond>
    GlobalLinSysIterativeStaticCondSharedPtr;

enum LocalMatrixStorageStrategy
{
    eNoStrategy,
    eContiguous,
    eNonContiguous,
    eSparse
};

const char *const LocalMatrixStorageStrategyMap[] = {
    "Contiguous", "Non-contiguous", "Sparse"};

/// A global linear system.
class GlobalLinSysIterativeStaticCond : virtual public GlobalLinSysIterative,
                                        virtual public GlobalLinSysStaticCond
{
public:
    typedef NekSparseDiagBlkMatrix<StorageSmvBsr<NekDouble>>
        DNekSmvBsrDiagBlkMat;
    typedef std::shared_ptr<DNekSmvBsrDiagBlkMat> DNekSmvBsrDiagBlkMatSharedPtr;

    /// Creates an instance of this class
    static GlobalLinSysSharedPtr create(
        const GlobalLinSysKey &pLinSysKey,
        const std::weak_ptr<ExpList> &pExpList,
        const std::shared_ptr<AssemblyMap> &pLocToGloMap)
    {
        GlobalLinSysSharedPtr p =
            MemoryManager<GlobalLinSysIterativeStaticCond>::AllocateSharedPtr(
                pLinSysKey, pExpList, pLocToGloMap);
        p->InitObject();
        return p;
    }

    /// Name of class
    static std::string className;
    static std::string className2;

    /// Constructor for full direct matrix solve.
    MULTI_REGIONS_EXPORT GlobalLinSysIterativeStaticCond(
        const GlobalLinSysKey &mkey, const std::weak_ptr<ExpList> &pExpList,
        const std::shared_ptr<AssemblyMap> &locToGloMap);

    /// Constructor for full direct matrix solve.
    MULTI_REGIONS_EXPORT GlobalLinSysIterativeStaticCond(
        const GlobalLinSysKey &mkey, const std::weak_ptr<ExpList> &pExpList,
        const DNekScalBlkMatSharedPtr pSchurCompl,
        const DNekScalBlkMatSharedPtr pBinvD, const DNekScalBlkMatSharedPtr pC,
        const DNekScalBlkMatSharedPtr pInvD,
        const std::shared_ptr<AssemblyMap> &locToGloMap,
        const PreconditionerSharedPtr pPrecon);

    ~GlobalLinSysIterativeStaticCond() override;

    /// Batched analogue of Solve for nCol right-hand sides of this system:
    /// per-column static condensation and back-substitution around a single
    /// batched conjugate-gradient solve (one AllReduce per iteration for
    /// all columns). Returns false without touching the outputs when the
    /// configuration is unsupported (multi-level static condensation,
    /// local-space or non-CG solver, successive-RHS projection), in which
    /// case the caller must fall back to per-column Solve calls. Only
    /// called explicitly (DO solver); no existing path reaches it.
    MULTI_REGIONS_EXPORT bool SolveBatched(
        const int nCol, const Array<OneD, const Array<OneD, NekDouble>> &pLocInputs,
        Array<OneD, Array<OneD, NekDouble>> &pLocOutputs,
        const AssemblyMapSharedPtr &pLocToGloMap);

protected:
    void v_InitObject() override;

    /// Assemble the Schur complement matrix.
    void v_AssembleSchurComplement(
        const std::shared_ptr<AssemblyMap> locToGloMap) override;

    /// Perform a Shur-complement matrix multiply operation.
    void v_DoMatrixMultiply(const Array<OneD, NekDouble> &pInput,
                            Array<OneD, NekDouble> &pOutput) override;

    void v_UniqueMap() override;

    DNekScalBlkMatSharedPtr v_GetStaticCondBlock(unsigned int n) override;
    GlobalLinSysStaticCondSharedPtr v_Recurse(
        const GlobalLinSysKey &mkey, const std::weak_ptr<ExpList> &pExpList,
        const DNekScalBlkMatSharedPtr pSchurCompl,
        const DNekScalBlkMatSharedPtr pBinvD, const DNekScalBlkMatSharedPtr pC,
        const DNekScalBlkMatSharedPtr pInvD,
        const std::shared_ptr<AssemblyMap> &locToGloMap) override;

    void v_PreSolve(int scLevel, Array<OneD, NekDouble> &F_bnd) override;
    void v_BasisFwdTransform(Array<OneD, NekDouble> &pInOut) override;
    void v_CoeffsBwdTransform(Array<OneD, NekDouble> &pInOut) override;
    void v_CoeffsFwdTransform(const Array<OneD, NekDouble> &pInput,
                              Array<OneD, NekDouble> &pOutput) override;

private:
    /// Dense storage for block Schur complement matrix
    std::vector<double> m_storage;
    /// Vector of pointers to local matrix data
    std::vector<const double *> m_denseBlocks;
    /// Ranks of local matrices
    Array<OneD, unsigned int> m_rows;
    /// Scaling factors for local matrices
    Array<OneD, NekDouble> m_scale;
    /// Sparse representation of Schur complement matrix at this level
    DNekSmvBsrDiagBlkMatSharedPtr m_sparseSchurCompl;
    /// Utility strings
    static std::string storagedef;
    static std::string storagelookupIds[];

    /// Prepares local representation of Schur complement
    /// stored as a sparse block-diagonal matrix.
    void PrepareLocalSchurComplement();

    /// Solve the matrix system
    void v_SolveLinearSystem(const int pNumRows,
                             const Array<OneD, const NekDouble> &pInput,
                             Array<OneD, NekDouble> &pOutput,
                             const AssemblyMapSharedPtr &locToGloMap,
                             const int pNumDir) override;
};
} // namespace Nektar::MultiRegions

#endif
