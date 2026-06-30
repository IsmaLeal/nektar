///////////////////////////////////////////////////////////////////////////////
//
// File: DOReducedCGEigenBasis.h
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
// Description: Reduced homogeneous CG Laplacian eigenbasis helper for DOVelocityCorrectionScheme.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERS_DOREDCGEIGENBASIS_H
#define NEKTAR_SOLVERS_DOREDCGEIGENBASIS_H

#include <MultiRegions/ContField.h>

#include <memory>
#include <vector>

namespace Nektar
{
class DOReducedCGEigenBasis
{
public:
    struct Eigenpair
    {
        NekDouble lambda = 0.0;
        NekDouble residual = 0.0;
        std::vector<NekDouble> reduced;
    };

    explicit DOReducedCGEigenBasis(
        const MultiRegions::ContFieldSharedPtr &field);
    ~DOReducedCGEigenBasis();

    int GetNumHomCoeffs() const;
    std::vector<Eigenpair> ComputeSmallest(int nev);
    void ExportToLocalAndPhys(const std::vector<NekDouble> &reduced,
                              Array<OneD, NekDouble> &local,
                              Array<OneD, NekDouble> &phys);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace Nektar

#endif
