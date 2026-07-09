///////////////////////////////////////////////////////////////////////////////
//
// File: FilterDOArchive.h
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
// Description: SolverUtils Filter that writes DO_ARCHIVE_V2 snapshots
//              (mean fields + DO mode coeffs/phys + Yi particles) at a
//              user-configured cadence, decoupled from the Checkpoint
//              filter.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_INCNAVIERSTOKES_FILTERS_FILTERDOARCHIVE_H
#define NEKTAR_INCNAVIERSTOKES_FILTERS_FILTERDOARCHIVE_H

#include <LibUtilities/BasicUtils/H5.h>
#include <SolverUtils/Filters/Filter.h>

#include <fstream>

namespace Nektar::SolverUtils
{

class FilterDOArchive : public Filter
{
public:
    static FilterSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const std::shared_ptr<EquationSystem> &pEquation,
        const ParamMap &pParams)
    {
        return MemoryManager<FilterDOArchive>::AllocateSharedPtr(
            pSession, pEquation, pParams);
    }

    static std::string className;

    FilterDOArchive(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const std::shared_ptr<EquationSystem> &pEquation,
        const ParamMap &pParams);

    ~FilterDOArchive() override = default;

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
    void WriteSnapshot(
        const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
        int step, NekDouble time);

    enum class Fmt { Text, Hdf5, Field };

    std::ofstream                  m_outFile;       // Text mode
    LibUtilities::H5::FileSharedPtr m_h5File;       // Hdf5 mode
    LibUtilities::FieldIOSharedPtr  m_fld;          // Field mode (MPI-aware)
    std::string                    m_outName;
    std::string                    m_outBase;       // Field mode: prefix for per-snapshot files
    Fmt                            m_fmt             = Fmt::Text;
    unsigned int                   m_outputFrequency = 1;
    unsigned int                   m_index           = 0;
    int                            m_snapIdx         = 0;
};

} // namespace Nektar::SolverUtils

#endif // NEKTAR_INCNAVIERSTOKES_FILTERS_FILTERDOARCHIVE_H
