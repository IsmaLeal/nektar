///////////////////////////////////////////////////////////////////////////////
//
// File: FilterDOArchive.cpp
//
// Description: SolverUtils Filter that writes DO_ARCHIVE_V2 snapshots
//              consumed by py_utils/load_chk.py. The cadence and output
//              filename come from the filter's own XML parameters
//              (OutputFrequency, OutputFile), so the archive output is
//              now decoupled from the Checkpoint filter.
//
///////////////////////////////////////////////////////////////////////////////

#include <IncNavierStokesSolver/Filters/FilterDOArchive.h>
#include <IncNavierStokesSolver/EquationSystems/DOVelocityCorrectionScheme.h>
#include <MultiRegions/ContField.h>

#include <iomanip>
#include <sstream>

namespace Nektar::SolverUtils
{

std::string FilterDOArchive::className =
    GetFilterFactory().RegisterCreatorFunction(
        "DOArchive", FilterDOArchive::create);

FilterDOArchive::FilterDOArchive(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const std::shared_ptr<EquationSystem> &pEquation,
    const ParamMap &pParams)
    : Filter(pSession, pEquation)
{
    // OutputFile (uses Filter::SetupOutput for backup-on-overwrite)
    std::string ext = ".do_archive";
    m_outName       = Filter::SetupOutput(ext, pParams);

    // OutputFrequency (default 1)
    auto it = pParams.find("OutputFrequency");
    if (it == pParams.end())
    {
        m_outputFrequency = 1;
    }
    else
    {
        LibUtilities::Equation equ(m_session->GetInterpreter(), it->second);
        m_outputFrequency = round(equ.Evaluate());
    }

    // Format selector:
    //   Text  (default) -- legacy plain-text DO_ARCHIVE_V2; single-rank only
    //   Hdf5             -- single-file DO_ARCHIVE_H5_V1; single-rank only
    //   Field            -- per-snapshot Nektar FieldIO .fld files (HDF5 under
    //                      MPI); fully MPI-aware. Output is a directory of
    //                      files named <basename>.do_<step>.fld.
    auto fit = pParams.find("Format");
    if (fit != pParams.end())
    {
        if (fit->second == "Hdf5" || fit->second == "HDF5")
        {
            m_fmt = Fmt::Hdf5;
            m_outName += ".h5";
        }
        else if (fit->second == "Field" || fit->second == "FieldIO")
        {
            m_fmt = Fmt::Field;
            // strip ".do_archive" so snapshots become
            // <basename>.do_<step>.fld.
            const std::string suff(".do_archive");
            m_outBase = (m_outName.size() >= suff.size() &&
                         m_outName.compare(m_outName.size() - suff.size(),
                                           suff.size(), suff) == 0)
                            ? m_outName.substr(
                                0, m_outName.size() - suff.size())
                            : m_outName;
        }
    }
}

void FilterDOArchive::v_Initialise(
    const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
    const NekDouble &time)
{
    m_index   = 0;
    auto comm = m_session->GetComm();

    auto dom = std::dynamic_pointer_cast<
        Nektar::DOVelocityCorrectionScheme>(m_equ.lock());
    ASSERTL0(dom, "FilterDOArchive requires "
             "SolverType=DOVelocityCorrectionScheme.");

    // MPI > 1: only Format=Field is MPI-aware. Suppress for Text/Hdf5 with
    // a clear warning so the simulation runs uninterrupted.
    if (comm->GetSize() > 1 && m_fmt != Fmt::Field)
    {
        if (comm->GetRank() == 0)
        {
            std::cerr << "[DOVelocityCorrectionScheme][FilterDOArchive]"
                         " WARNING: MPI ranks="
                      << comm->GetSize()
                      << "; DOArchive Format="
                      << (m_fmt == Fmt::Hdf5 ? "Hdf5" : "Text")
                      << " output is suppressed (single-rank only). Use "
                         "Format=Field for MPI-aware archives.\n";
        }
        m_outputFrequency = 0;
        return;
    }

    if (m_fmt == Fmt::Field)
    {
        // Field mode: collective FieldIO writer. Every rank participates.
        m_fld = LibUtilities::FieldIO::CreateDefault(m_session);
    }

    if (comm->GetRank() == 0)
    {
        const int nVel  = dom->GetVelocityIdx().size();
        const int nPhys = pFields[0]->GetTotPoints();
        if (m_fmt == Fmt::Hdf5)
        {
            namespace H5 = LibUtilities::H5;
            m_h5File           = H5::File::Create(m_outName, H5F_ACC_TRUNC);
            auto hdr           = m_h5File->CreateGroup("header");
            hdr->SetAttribute("format_version",
                              std::string("DO_ARCHIVE_H5_V1"));
            hdr->SetAttribute("n_modes",        dom->GetNumDOModes());
            hdr->SetAttribute("n_particles",    dom->GetNumDOParticles());
            hdr->SetAttribute("n_vel",          nVel);
            hdr->SetAttribute("n_phys",         nPhys);
            hdr->SetAttribute("dt",             dom->GetTimeStep());
            const std::vector<std::string> vnames = m_session->GetVariables();
            hdr->SetAttribute("var_names", vnames);
        }
        else if (m_fmt == Fmt::Text)
        {
            m_outFile.open(m_outName, std::ios::out | std::ios::trunc);
            ASSERTL0(m_outFile.is_open(),
                     "FilterDOArchive: unable to open '" + m_outName + "'");
            m_outFile << std::setprecision(17);
            m_outFile << "DO_ARCHIVE_V2\n"
                      << "N_MODES "     << dom->GetNumDOModes()       << "\n"
                      << "N_PARTICLES " << dom->GetNumDOParticles()   << "\n"
                      << "N_VEL "       << nVel                       << "\n"
                      << "N_PTS "       << nPhys                      << "\n"
                      << "DT "          << dom->GetTimeStep()         << "\n"
                      << "END_HEADER\n";
        }
        // Field mode: nothing rank-0-only to do at init; per-snapshot files.
    }

    // initial snapshot at t=0
    WriteSnapshot(pFields, 0, time);
}

void FilterDOArchive::v_Update(
    const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
    const NekDouble &time)
{
    ++m_index;
    if (m_outputFrequency == 0)                  return;
    if ((m_index % m_outputFrequency) != 0)      return;
    WriteSnapshot(pFields, static_cast<int>(m_index), time);
}

void FilterDOArchive::v_Finalise(
    [[maybe_unused]] const Array<OneD, const MultiRegions::ExpListSharedPtr>
                              &pFields,
    [[maybe_unused]] const NekDouble &time)
{
    if (m_session->GetComm()->GetRank() == 0)
    {
        if (m_outFile.is_open()) m_outFile.close();
        m_h5File.reset();   // closes the HDF5 file in Hdf5 mode
    }
}

bool FilterDOArchive::v_IsTimeDependent()
{
    return true;
}

void FilterDOArchive::WriteSnapshot(
    const Array<OneD, const MultiRegions::ExpListSharedPtr> &pFields,
    int step, NekDouble time)
{
    auto dom = std::dynamic_pointer_cast<
        Nektar::DOVelocityCorrectionScheme>(m_equ.lock());
    ASSERTL0(dom, "FilterDOArchive requires "
             "SolverType=DOVelocityCorrectionScheme.");

    const auto &velocity     = dom->GetVelocityIdx();
    const int   nVel         = velocity.size();
    const int   nPhys        = pFields[0]->GetTotPoints();
    const int   nDOModes     = dom->GetNumDOModes();
    const int   nDOParticles = dom->GetNumDOParticles();
    const auto &doModePhys   = dom->GetDOModePhys();
    const auto &doModeCoeffs = dom->GetDOModeCoeffs();
    const auto &Yi           = dom->GetYi();
    const auto &varNames     = m_session->GetVariables();
    const int   nCo          = pFields[velocity[0]]->GetNcoeffs();

    // Field mode: collective; all ranks participate in FieldIO::Write.
    if (m_fmt == Fmt::Field)
    {
        // Build FieldDef list once from the first ExpList's element layout.
        std::vector<LibUtilities::FieldDefinitionsSharedPtr> FieldDef =
            pFields[0]->GetFieldDefinitions();
        std::vector<std::vector<NekDouble>> FieldData(FieldDef.size());

        // Mean fields: u, v, [w], p -- one per session variable.
        for (size_t j = 0; j < varNames.size(); ++j)
        {
            for (size_t i = 0; i < FieldDef.size(); ++i)
            {
                FieldDef[i]->m_fields.push_back(varNames[j]);
                pFields[0]->AppendFieldData(FieldDef[i], FieldData[i],
                                            pFields[j]->UpdateCoeffs());
            }
        }

        // Mode fields: mode_<i>_<varname>, packaged from m_DOModeCoeffs.
        // Each rank's slice is per-element-major (same layout as Nektar's
        // ExpList m_coeffs); AppendFieldData reads per-element blocks and the
        // collective FieldIO::Write below assembles them globally via element
        // IDs. Shared-DOF inconsistencies at the FP-noise level may exist but
        // are bounded -- empirically <2 in mode_u/v after one POD-init step.
        for (int m = 0; m < nDOModes; ++m)
        {
            for (int c = 0; c < nVel; ++c)
            {
                const std::string fieldName =
                    "mode_" + std::to_string(m) + "_" + varNames[velocity[c]];
                Array<OneD, NekDouble> coeffSlice(
                    nCo,
                    const_cast<NekDouble *>(doModeCoeffs.data() +
                                            (m * nVel + c) * nCo),
                    eArrayWrapper);
                for (size_t i = 0; i < FieldDef.size(); ++i)
                {
                    FieldDef[i]->m_fields.push_back(fieldName);
                    pFields[velocity[c]]->AppendFieldData(
                        FieldDef[i], FieldData[i], coeffSlice);
                }
            }
        }

        // Per-snapshot file name.
        std::ostringstream fn;
        fn << m_outBase << ".do_" << std::setw(6) << std::setfill('0')
           << step << ".fld";

        // Metadata: scalars + Yi as hex of doubles (~12 KB for typical sizes).
        LibUtilities::FieldMetaDataMap meta;
        meta["DOVelocityCorrectionScheme_step"]        = std::to_string(step);
        meta["DOVelocityCorrectionScheme_time"]        = std::to_string(time);
        meta["DOVelocityCorrectionScheme_n_modes"] =
            std::to_string(nDOModes);
        meta["DOVelocityCorrectionScheme_n_particles"] =
            std::to_string(nDOParticles);
        meta["DOVelocityCorrectionScheme_n_vel"] =
            std::to_string(nVel);
        meta["DOVelocityCorrectionScheme_dt"] =
            std::to_string(dom->GetTimeStep());
        // Yi as hex (rank 0 has the same Yi as everyone -- replicated).
        {
            const int nFlat = nDOParticles * nDOModes;
            std::ostringstream hex;
            hex << std::hex << std::setfill('0');
            const unsigned char *bytes =
                reinterpret_cast<const unsigned char *>(Yi.data());
            for (int b = 0; b < nFlat * (int)sizeof(NekDouble); ++b)
            {
                hex << std::setw(2) << static_cast<unsigned>(bytes[b]);
            }
            meta["DOVelocityCorrectionScheme_Yi_hex"] = hex.str();
        }
        // Pressure mode coefficients as hex for lossless restart.
        {
            const auto &pCoeffs = dom->GetDOModePCoeffs();
            const int nFlatP = pCoeffs.size();
            std::ostringstream hexP;
            hexP << std::hex << std::setfill('0');
            const unsigned char *bytesP =
                reinterpret_cast<const unsigned char *>(pCoeffs.data());
            for (int b = 0; b < nFlatP * (int)sizeof(NekDouble); ++b)
            {
                hexP << std::setw(2) << static_cast<unsigned>(bytesP[b]);
            }
            meta["DOVelocityCorrectionScheme_PCoeffs_hex"] = hexP.str();
        }

        // Stochastic forcing state (only when forcing is active).
        const int nForcingChannels = dom->GetNumForcingChannels();
        if (nForcingChannels > 0)
        {
            const auto &eta    = dom->GetForcingEta();
            const int nFlatEta = nDOParticles * nForcingChannels;
            std::ostringstream hexEta;
            hexEta << std::hex << std::setfill('0');
            const unsigned char *bytesEta =
                reinterpret_cast<const unsigned char *>(eta.data());
            for (int b = 0; b < nFlatEta * (int)sizeof(NekDouble); ++b)
                hexEta << std::setw(2) << static_cast<unsigned>(bytesEta[b]);
            meta["DOVelocityCorrectionScheme_ForcingEta_hex"] = hexEta.str();

            std::ostringstream rngSs;
            dom->SerializeForcingRng(rngSs);
            meta["DOVelocityCorrectionScheme_ForcingRng"] = rngSs.str();
        }

        m_fld->Write(fn.str(), FieldDef, FieldData, meta);
        return;
    }

    if (m_session->GetComm()->GetRank() != 0) return;

    if (m_fmt == Fmt::Hdf5)
    {
        namespace H5 = LibUtilities::H5;
        H5::GroupSharedPtr g =
            m_h5File->CreateGroup("snap_" + std::to_string(m_snapIdx));
        g->SetAttribute("step", step);
        g->SetAttribute("time", static_cast<double>(time));
        for (size_t i = 0; i < varNames.size(); ++i)
        {
            const auto &c = pFields[i]->GetCoeffs();
            std::vector<double> v(c.data(),
                                  c.data() + pFields[i]->GetNcoeffs());
            g->CreateWriteDataSet("mean_" + varNames[i], v);
        }
        const int nMP = nDOModes * nVel * nPhys;
        g->CreateWriteDataSet(
            "mode_phys",
            std::vector<double>(doModePhys.data(), doModePhys.data() + nMP));
        const int nMC = nDOModes * nVel * nCo;
        g->CreateWriteDataSet(
            "mode_coeffs",
            std::vector<double>(doModeCoeffs.data(),
                                doModeCoeffs.data() + nMC));
        const int nFlat = nDOParticles * nDOModes;
        g->CreateWriteDataSet(
            "Yi", std::vector<double>(Yi.data(), Yi.data() + nFlat));
        ++m_snapIdx;
        return;
    }

    // ---- Text snapshot (legacy DO_ARCHIVE_V2 format) ----
    // OU state is not serialized in Text format (legacy limitation).
    m_outFile << "SNAPSHOT " << step << " " << time << "\n"
              << "RNG 0\n";
    m_outFile << "MEAN_FIELDS_BEGIN\n";
    for (size_t i = 0; i < varNames.size(); ++i)
    {
        const int   nc     = pFields[i]->GetNcoeffs();
        const auto &coeffs = pFields[i]->GetCoeffs();
        m_outFile << "FIELD " << varNames[i] << " " << nc << "\n";
        for (int k = 0; k < nc; ++k)
        {
            m_outFile << coeffs[k] << "\n";
        }
    }
    m_outFile << "MEAN_FIELDS_END\n";

    m_outFile << "MODE_FIELDS_BEGIN " << nDOModes << " " << nVel << " "
              << nPhys << "\n";
    const int nTot = nDOModes * nVel * nPhys;
    for (int k = 0; k < nTot; ++k)
    {
        m_outFile << doModePhys[k] << "\n";
    }
    m_outFile << "MODE_FIELDS_END\n";

    m_outFile << "MODE_COEFFS_BEGIN " << nDOModes << " " << nVel << "\n";
    for (int m = 0; m < nDOModes; ++m)
    {
        for (int c = 0; c < nVel; ++c)
        {
            m_outFile << "MODE_COEFF " << m << " " << c << " " << nCo << "\n";
            const NekDouble *p = doModeCoeffs.data() + (m * nVel + c) * nCo;
            for (int k = 0; k < nCo; ++k)
            {
                m_outFile << p[k] << "\n";
            }
        }
    }
    m_outFile << "MODE_COEFFS_END\n";

    m_outFile << "YI_BEGIN " << nDOParticles << " " << nDOModes << "\n";
    const int nFlat = nDOParticles * nDOModes;
    for (int k = 0; k < nFlat; ++k)
    {
        m_outFile << Yi[k] << "\n";
    }
    m_outFile << "YI_END\n";

    // OU amplitudes not serialized in Text format; write placeholder zeros.
    m_outFile << "OU_BEGIN " << nDOParticles << " " << nDOModes << "\n";
    for (int k = 0; k < nFlat; ++k)
    {
        m_outFile << "0\n";
    }
    m_outFile << "OU_END\n";

    m_outFile << "END_SNAPSHOT\n";
}

} // namespace Nektar::SolverUtils
