///////////////////////////////////////////////////////////////////////////////
//
// File: MovingFrameFar.h
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
// Description: Abstract base class for Extrapolate.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERS_MOVINGFRAMEFAR_H
#define NEKTAR_SOLVERS_MOVINGFRAMEFAR_H

#include <IncNavierStokesSolver/BoundaryConditions/IncBaseCondition.h>
#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <MultiRegions/ExpList.h>

namespace Nektar
{

class MovingFrameFar : public IncBaseCondition
{
public:
    friend class MemoryManager<MovingFrameFar>;

    static IncBaseConditionSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr pSession,
        Array<OneD, MultiRegions::ExpListSharedPtr> pFields,
        Array<OneD, SpatialDomains::BoundaryConditionShPtr> cond,
        Array<OneD, MultiRegions::ExpListSharedPtr> exp, int nbnd, int spacedim,
        int bnddim)
    {
        IncBaseConditionSharedPtr p =
            MemoryManager<MovingFrameFar>::AllocateSharedPtr(
                pSession, pFields, cond, exp, nbnd, spacedim, bnddim);
        p->Initialise(pSession);
        return p;
    }

    static std::string className;

protected:
    std::vector<LibUtilities::EquationSharedPtr> m_definedVels;

    MovingFrameFar(const LibUtilities::SessionReaderSharedPtr pSession,
                   Array<OneD, MultiRegions::ExpListSharedPtr> pFields,
                   Array<OneD, SpatialDomains::BoundaryConditionShPtr> cond,
                   Array<OneD, MultiRegions::ExpListSharedPtr> exp, int nbnd,
                   int spacedim, int bnddim);

    ~MovingFrameFar() override = default;

    void v_Initialise(
        const LibUtilities::SessionReaderSharedPtr &pSession) override;

    void v_Update(const Array<OneD, const Array<OneD, NekDouble>> &fields,
                  const Array<OneD, const Array<OneD, NekDouble>> &Adv,
                  std::map<std::string, NekDouble> &params) override;
};

} // namespace Nektar

#endif
