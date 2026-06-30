///////////////////////////////////////////////////////////////////////////////
//
// File: FilterDOArchive.h
//
// Description: SolverUtils Filter that writes DO_ARCHIVE_V2 snapshots
//              (mean fields + DO mode coeffs/phys + Yi particles) at a
//              user-configured cadence. Replaces the cadence-mining
//              static-state code that was previously inside DOVelocityCorrectionScheme.cpp.
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

#endif
