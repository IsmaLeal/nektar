///////////////////////////////////////////////////////////////////////////////
//
// File: DOPODInitialiser.cpp
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
// Description: POD-based DO mode initialiser implementation.
//
///////////////////////////////////////////////////////////////////////////////

#include <IncNavierStokesSolver/EquationSystems/DOPODInitialiser.h>

#include <LibUtilities/BasicUtils/FieldIO.h>
#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/LinearAlgebra/Blas.hpp>
#include <LibUtilities/LinearAlgebra/Lapack.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <regex>
#include <stdexcept>

namespace Nektar
{
// ============================================================================
// Glob expansion: support a single '*' in the basename portion of `pattern`.
// ============================================================================
std::vector<std::string> DOPODInitialiser::ExpandGlob(const std::string &pattern)
{
    fs::path p(pattern);
    fs::path dir = p.parent_path();
    if (dir.empty())
    {
        dir = fs::current_path();
    }
    if (!fs::exists(dir) || !fs::is_directory(dir))
    {
        return {};
    }

    const std::string fname = p.filename().string();
    const auto star = fname.find('*');

    std::vector<std::string> matches;
    if (star == std::string::npos)
    {
        // Exact match (no wildcard)
        if (fs::exists(p))
        {
            matches.push_back(fs::absolute(p).string());
        }
        return matches;
    }

    const std::string prefix = fname.substr(0, star);
    const std::string suffix = fname.substr(star + 1);

    for (const auto &entry : fs::directory_iterator(dir))
    {
        if (!entry.is_regular_file() && !entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        if (name.size() < prefix.size() + suffix.size()) continue;
        if (name.compare(0, prefix.size(), prefix) != 0) continue;
        if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
            continue;
        matches.push_back(fs::absolute(entry.path()).string());
    }

    // Sort: prefer numeric sort by trailing _<digits> immediately before suffix.
    static const std::regex tail_re("_(\\d+)\\.[A-Za-z0-9]+$");
    auto extract_idx = [](const std::string &full) -> long long {
        std::smatch m;
        if (std::regex_search(full, m, tail_re))
        {
            try { return std::stoll(m[1].str()); } catch (...) { return -1; }
        }
        return -1;
    };
    bool allNumeric = true;
    for (const auto &s : matches)
    {
        if (extract_idx(s) < 0) { allNumeric = false; break; }
    }
    if (allNumeric)
    {
        std::sort(matches.begin(), matches.end(),
                  [&](const std::string &a, const std::string &b) {
                      return extract_idx(a) < extract_idx(b);
                  });
    }
    else
    {
        std::sort(matches.begin(), matches.end());
    }
    return matches;
}

// ============================================================================
// Impl
// ============================================================================
class DOPODInitialiser::Impl
{
public:
    Impl(const LibUtilities::SessionReaderSharedPtr      &session,
         const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
         const Array<OneD, int>                            &velocity,
         const Config                                      &cfg)
        : m_session(session), m_fields(fields), m_velocity(velocity), m_cfg(cfg)
    {
        ASSERTL0(m_cfg.numModes > 0,
                 "DOPODInitialiser: numModes must be > 0.");
        ASSERTL0(!m_cfg.snapshotFiles.empty(),
                 "DOPODInitialiser: snapshotFiles is empty.");
        ASSERTL0((int)m_cfg.snapshotFiles.size() >= m_cfg.numModes,
                 "DOPODInitialiser: number of snapshots must be >= numModes.");

        m_nVel    = (int)m_velocity.size();
        m_nCoeffs = m_fields[m_velocity[0]]->GetNcoeffs();
        m_nPhys   = m_fields[m_velocity[0]]->GetTotPoints();

        for (int c = 1; c < m_nVel; ++c)
        {
            ASSERTL0(m_fields[m_velocity[c]]->GetNcoeffs() == m_nCoeffs &&
                         m_fields[m_velocity[c]]->GetTotPoints() == m_nPhys,
                     "DOPODInitialiser: velocity components must share the "
                     "same expansion sizes.");
        }
    }

    void Compute()
    {
        const int K = (int)m_cfg.snapshotFiles.size();
        const int S = m_cfg.numModes;

        if (m_cfg.verbose)
        {
            std::cout << "[DOVelocityCorrectionScheme][POD] " << K << " snapshots, S=" << S
                      << ", nVel=" << m_nVel << ", nCoeffs=" << m_nCoeffs
                      << ", nPhys=" << m_nPhys << "\n";
        }

        // 1) Load all snapshot velocity coeffs.
        m_snapCoeffs.assign(K, std::vector<std::vector<NekDouble>>(m_nVel));
        for (int k = 0; k < K; ++k)
        {
            LoadSnapshot(m_cfg.snapshotFiles[k], m_snapCoeffs[k]);
            if (m_cfg.verbose)
            {
                std::cout << "[DOVelocityCorrectionScheme][POD] loaded "
                          << m_cfg.snapshotFiles[k] << "\n";
            }
        }

        // 2) Compute mean and subtract from each snapshot (in coeff space).
        std::vector<std::vector<NekDouble>> mean(m_nVel,
                                                 std::vector<NekDouble>(m_nCoeffs, 0.0));
        switch (m_cfg.meanType)
        {
            case MeanType::TimeMean:
            {
                const NekDouble inv = 1.0 / static_cast<NekDouble>(K);
                for (int k = 0; k < K; ++k)
                    for (int c = 0; c < m_nVel; ++c)
                        for (int j = 0; j < m_nCoeffs; ++j)
                            mean[c][j] += inv * m_snapCoeffs[k][c][j];
                break;
            }
            case MeanType::FirstSnapshot:
            {
                for (int c = 0; c < m_nVel; ++c)
                    mean[c] = m_snapCoeffs[0][c];
                break;
            }
            case MeanType::ProvidedMeanField:
            {
                ASSERTL0(!m_cfg.meanFile.empty(),
                         "DOPODInitialiser: PODMeanType=ProvidedMeanField "
                         "requires PODMeanFile.");
                std::vector<std::vector<NekDouble>> tmp(m_nVel);
                LoadSnapshot(m_cfg.meanFile, tmp);
                for (int c = 0; c < m_nVel; ++c) mean[c] = std::move(tmp[c]);
                break;
            }
        }
        for (int k = 0; k < K; ++k)
            for (int c = 0; c < m_nVel; ++c)
                for (int j = 0; j < m_nCoeffs; ++j)
                    m_snapCoeffs[k][c][j] -= mean[c][j];
        m_mean = mean;  // stash for ExportMean()

        // 3) Build the K x K correlation matrix C[i,j] = <snap_i, snap_j>_M.
        //    snapMat[c] packs m_snapCoeffs[i][c] contiguously (K x nCoeffs
        //    row-major = nCoeffs x K Fortran column-major, lda=nCoeffs).
        //    For each j: compute mb = M*snap_j[c] once, then update the
        //    upper triangle C[0..j, j] with one Dgemv('T') call.
        std::vector<std::vector<NekDouble>> snapMat(
            m_nVel, std::vector<NekDouble>((size_t)K * m_nCoeffs));
        for (int c = 0; c < m_nVel; ++c)
            for (int i = 0; i < K; ++i)
                std::copy(m_snapCoeffs[i][c].begin(), m_snapCoeffs[i][c].end(),
                          snapMat[c].data() + (size_t)i * m_nCoeffs);

        std::vector<NekDouble> C(K * K, 0.0);
        Array<OneD, NekDouble> ip(m_nCoeffs);
        Array<OneD, NekDouble> physBuf(m_nPhys);
        std::vector<NekDouble> mb(m_nCoeffs);
        for (int j = 0; j < K; ++j)
        {
            for (int c = 0; c < m_nVel; ++c)
            {
                // BwdTrans(snap_j.coeffs[c]) -> phys, then IProductWRTBase -> M*coeffs
                Array<OneD, NekDouble> coeffArr(m_nCoeffs,
                                                m_snapCoeffs[j][c].data(),
                                                eArrayWrapper);
                m_fields[m_velocity[c]]->BwdTrans(coeffArr, physBuf);
                m_fields[m_velocity[c]]->IProductWRTBase(physBuf, ip);
                std::memcpy(mb.data(), ip.data(),
                            sizeof(NekDouble) * m_nCoeffs);
                Blas::Dgemv('T', m_nCoeffs, j + 1, 1.0,
                            snapMat[c].data(), m_nCoeffs,
                            mb.data(), 1,
                            1.0, C.data() + j, K);
            }
        }
        // MPI: C entries are partial sums; one bulk AllReduce on K² values.
        m_fields[m_velocity[0]]->GetComm()->GetRowComm()->AllReduce(
            C, LibUtilities::ReduceSum);
        // Symmetrize (lower from upper).
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < i; ++j)
                C[i * K + j] = C[j * K + i];

        // 4) Eigendecomposition via Dspev (packed symmetric, ascending).
        //    Packed-upper format: ap[i + j*(j+1)/2] = C(i, j) for i <= j.
        std::vector<NekDouble> ap((size_t)K * (K + 1) / 2);
        for (int j = 0; j < K; ++j)
            for (int i = 0; i <= j; ++i)
                ap[i + (size_t)j * (j + 1) / 2] = C[i * K + j];
        std::vector<NekDouble> w(K);
        std::vector<NekDouble> z((size_t)K * K);
        std::vector<NekDouble> work(3 * K);
        int info = 0;
        Lapack::Dspev('V', 'U', K, ap.data(), w.data(), z.data(), K,
                      work.data(), info);
        ASSERTL0(info == 0, "DOPODInitialiser: Dspev failed.");

        // 5) Pick top S eigenpairs (descending). Discard near-zero eigenvalues.
        //    Dspev returns ascending eigenvalues, so descending index = K-1-k.
        const NekDouble lambda_max = std::max(w[K - 1], (NekDouble)1.0e-300);
        const NekDouble lambda_floor = 1.0e-12 * lambda_max;
        m_sigmas.clear();
        m_eigVecs.assign(S, std::vector<NekDouble>(K, 0.0));
        m_modeCoeffs.assign(S, std::vector<std::vector<NekDouble>>(
                                   m_nVel, std::vector<NekDouble>(m_nCoeffs, 0.0)));
        m_modePhys.assign(S, std::vector<std::vector<NekDouble>>(
                                 m_nVel, std::vector<NekDouble>(m_nPhys, 0.0)));
        m_K = K;

        m_totalEnergy = 0.0;
        for (int k = 0; k < K; ++k)
            m_totalEnergy += std::max(w[k], (NekDouble)0.0);

        m_capturedEnergy = 0.0;
        for (int k = 0; k < S; ++k)
        {
            const int idx = K - 1 - k;
            const NekDouble lambda = w[idx];
            ASSERTL0(lambda > lambda_floor,
                     "DOPODInitialiser: requested more POD modes than the "
                     "snapshot ensemble can support (rank deficient). "
                     "Reduce DOModes or supply more linearly-independent "
                     "snapshots past the spinup window.");
            const NekDouble sigma = std::sqrt(lambda);
            m_sigmas.push_back(sigma);
            m_capturedEnergy += lambda;

            // Cache eigenvector v_{:,k} (= v_{p,k} = z[idx*K + p]) for caller-
            // side Y projection (Y_{p,i} = sigma_i * v_{p,i}).
            for (int p = 0; p < K; ++p)
                m_eigVecs[k][p] = z[(size_t)idx * K + p];

            // Synthesise mode k:  phi_k = (1/sigma) Sum_i z[i, idx] * snap_i
            // (mass-orthonormality of phi_k follows from
            //   <phi_k, phi_l>_M
            //     = (1/sqrt(lambda_k lambda_l)) v_k^T C v_l
            //     = (1/sqrt(lambda_k lambda_l)) lambda_l delta_{kl}
            //     = delta_{kl}.)
            const NekDouble inv_sigma = 1.0 / sigma;
            for (int c = 0; c < m_nVel; ++c)
            {
                for (int i = 0; i < K; ++i)
                {
                    const NekDouble a = inv_sigma * m_eigVecs[k][i];
                    if (a == 0.0) continue;
                    NekDouble *dst = m_modeCoeffs[k][c].data();
                    const NekDouble *src = m_snapCoeffs[i][c].data();
                    for (int q = 0; q < m_nCoeffs; ++q) dst[q] += a * src[q];
                }
                // BwdTrans into phys; eArrayWrapper is REQUIRED — without it
                // Nektar's Array<OneD,T>(n, ptr, ...) defaults to eArrayCopy
                // and BwdTrans writes into a temporary that never reaches our
                // std::vector storage. Do not change without re-reading
                // SharedArray.hpp::Array constructors.
                Array<OneD, NekDouble> coeffArr(m_nCoeffs,
                                                m_modeCoeffs[k][c].data(),
                                                eArrayWrapper);
                Array<OneD, NekDouble> physArr(m_nPhys,
                                               m_modePhys[k][c].data(),
                                               eArrayWrapper);
                m_fields[m_velocity[c]]->BwdTrans(coeffArr, physArr);
            }
        }

        // 6) Self-check: max |<phi_i, phi_j>_M - delta_{ij}| should be O(eps).
        //    Catches any future bug in the synthesis or BwdTrans wrapping.
        //    AllReduce the (S+1)*S/2 inner products in one bulk call.
        const int nIp = (S * (S + 1)) / 2;
        std::vector<NekDouble> innerVec(nIp, 0.0);
        Array<OneD, NekDouble> physTmp(m_nPhys), ipTmp(m_nCoeffs);
        int idx = 0;
        for (int k = 0; k < S; ++k)
            for (int l = k; l < S; ++l, ++idx)
            {
                NekDouble inner = 0.0;
                for (int c = 0; c < m_nVel; ++c)
                {
                    Array<OneD, NekDouble> phi_l(m_nPhys,
                                                 const_cast<NekDouble *>(
                                                     m_modePhys[l][c].data()),
                                                 eArrayWrapper);
                    m_fields[m_velocity[c]]->IProductWRTBase(phi_l, ipTmp);
                    for (int q = 0; q < m_nCoeffs; ++q)
                        inner += m_modeCoeffs[k][c][q] * ipTmp[q];
                }
                innerVec[idx] = inner;
            }
        m_fields[m_velocity[0]]->GetComm()->GetRowComm()->AllReduce(
            innerVec, LibUtilities::ReduceSum);
        NekDouble maxOrthoErr = 0.0;
        idx = 0;
        for (int k = 0; k < S; ++k)
            for (int l = k; l < S; ++l, ++idx)
            {
                const NekDouble target = (k == l) ? 1.0 : 0.0;
                maxOrthoErr = std::max(maxOrthoErr,
                                       std::abs(innerVec[idx] - target));
            }
        if (maxOrthoErr > 1.0e-6)
        {
            std::cout << "[DOVelocityCorrectionScheme][POD] WARNING mass-ortho "
                      << "self-check max-err = " << maxOrthoErr
                      << " > 1e-6. Downstream ReOrthonormalise will MGS-correct "
                      << "this; expected for rank-deficient ensembles (e.g. ICs "
                      << "with only a few random scalars and many requested DOModes).\n";
        }
        else if (m_cfg.verbose)
        {
            std::cout << "[DOVelocityCorrectionScheme][POD] mass-ortho self-check max-err = "
                      << maxOrthoErr << "\n";
        }
        ASSERTL0(maxOrthoErr < 1.0e-3,
                 "DOPODInitialiser: synthesised modes failed mass-orthonormality "
                 "self-check (max-err > 1e-3). This is large enough that downstream "
                 "MGS may not recover; check for duplicate snapshots, severe "
                 "polynomial-order mismatch with the current expansion, or a "
                 "build-environment problem with Lapack::Dspev.");

        if (m_cfg.verbose)
        {
            std::cout << "[DOVelocityCorrectionScheme][POD] sigma=";
            for (auto s : m_sigmas) std::cout << " " << s;
            std::cout << "  energy="
                      << (m_totalEnergy > 0
                              ? m_capturedEnergy / m_totalEnergy
                              : 0.0)
                      << "\n";
        }

    }

    void ExportMean(Array<OneD, NekDouble> &meanCoeffs) const
    {
        ASSERTL0(!m_mean.empty(),
                 "DOPODInitialiser: Compute() must run before ExportMean.");
        for (int c = 0; c < m_nVel; ++c)
        {
            Vmath::Vcopy(m_nCoeffs, m_mean[c].data(), 1,
                         meanCoeffs.data() + c * m_nCoeffs, 1);
        }
    }

    void ExportToDOMode(Array<OneD, NekDouble> &modePhys,
                        Array<OneD, NekDouble> &modeCoeffs) const
    {
        const int S = m_cfg.numModes;
        ASSERTL0((int)m_modeCoeffs.size() == S,
                 "DOPODInitialiser: Compute() not run before Export.");
        for (int i = 0; i < S; ++i)
        {
            for (int c = 0; c < m_nVel; ++c)
            {
                const int cOff = (i * m_nVel + c) * m_nCoeffs;
                const int pOff = (i * m_nVel + c) * m_nPhys;
                Vmath::Vcopy(m_nCoeffs, m_modeCoeffs[i][c].data(), 1,
                             modeCoeffs.data() + cOff, 1);
                Vmath::Vcopy(m_nPhys, m_modePhys[i][c].data(), 1,
                             modePhys.data() + pOff, 1);
            }
        }
    }

    const std::vector<NekDouble> &SingularValues() const { return m_sigmas; }

    const std::vector<std::vector<NekDouble>> &EigenVectors() const
    {
        return m_eigVecs;
    }

    int NumSnapshots() const { return m_K; }

    NekDouble EnergyFraction() const
    {
        return m_totalEnergy > 0 ? m_capturedEnergy / m_totalEnergy : 0.0;
    }

    /// Project (snap_p - mean) onto the supplied post-reorth modes via the
    /// velocity mass inner product. Writes Yi[p*S + k] for p in [0, Kproj)
    /// and k in [0, S); leaves Yi unchanged for p in [Kproj, nParticles).
    /// m_snapCoeffs must be alive (Compute() was called and snapshots were
    /// not freed). Inner products are AllReduced across MPI ranks.
    void RecomputeYiByProjection(
        const Array<OneD, NekDouble> &modePhys,
        const Array<OneD, NekDouble> &modeCoeffs,
        Array<OneD, NekDouble>       &Yi,
        int                           nParticles)
    {
        const int S = m_cfg.numModes;
        const int K = (int)m_cfg.snapshotFiles.size();
        ASSERTL0(S > 0 && K > 0 && !m_snapCoeffs.empty(),
                 "DOPODInitialiser::RecomputeYiByProjection: "
                 "Compute() must be called first.");
        const int Kproj = std::min(K, nParticles);

        // Precompute ip[k][c] = IProductWRTBase(mode_k_c_phys) once per (k,c).
        // Use an owned scratch buffer — modePhys is aliased into live solver
        // state and must not be wrapped with eArrayWrapper here.
        Array<OneD, NekDouble> physScratch(m_nPhys);
        std::vector<std::vector<Array<OneD, NekDouble>>> ipKC(
            S, std::vector<Array<OneD, NekDouble>>(m_nVel));
        for (int k = 0; k < S; ++k)
            for (int c = 0; c < m_nVel; ++c)
            {
                ipKC[k][c] = Array<OneD, NekDouble>(m_nCoeffs, 0.0);
                Vmath::Vcopy(m_nPhys,
                             modePhys.data() + (k * m_nVel + c) * m_nPhys, 1,
                             physScratch.data(), 1);
                m_fields[m_velocity[c]]->IProductWRTBase(physScratch, ipKC[k][c]);
            }
        (void)modeCoeffs;

        std::vector<NekDouble> Ylocal(Kproj * S, 0.0);
        for (int p = 0; p < Kproj; ++p)
            for (int k = 0; k < S; ++k)
            {
                NekDouble s = 0.0;
                for (int c = 0; c < m_nVel; ++c)
                    s += Vmath::Dot(m_nCoeffs,
                                    m_snapCoeffs[p][c].data(), 1,
                                    ipKC[k][c].data(), 1);
                Ylocal[p * S + k] = s;
            }
        m_fields[m_velocity[0]]->GetComm()->GetRowComm()->AllReduce(
            Ylocal, LibUtilities::ReduceSum);
        for (int p = 0; p < Kproj; ++p)
            for (int k = 0; k < S; ++k)
                Yi[p * S + k] = Ylocal[p * S + k];
    }

private:
    /// Load one .chk/.fld file's velocity components (u/v[/w]) into per-component
    /// coeff vectors. Mirrors the third EquationSystem::ImportFld signature
    /// (FieldIO::CreateForFile + Import + ExtractDataToCoeffs), but emits raw
    /// std::vector<NekDouble> per component.
    void LoadSnapshot(const std::string &filename,
                      std::vector<std::vector<NekDouble>> &outCoeffs)
    {
        std::vector<LibUtilities::FieldDefinitionsSharedPtr> FieldDef;
        std::vector<std::vector<NekDouble>> FieldData;
        LibUtilities::FieldIOSharedPtr fld =
            LibUtilities::FieldIO::CreateForFile(m_session, filename);
        fld->Import(filename, FieldDef, FieldData);

        // Polynomial-order sanity check (warning only): if the snapshot's
        // m_numModes differs from the current expansion, ExtractDataToCoeffs
        // will silently interpolate. POD remains valid but the user probably
        // wants to know.
        if (!m_orderWarningEmitted && !FieldDef.empty() &&
            !FieldDef[0]->m_numModes.empty())
        {
            const unsigned snapOrder = FieldDef[0]->m_numModes[0];
            // m_fields[0]->GetExp(0)->GetBasisNumModes(0) gives the current
            // first-element first-direction polynomial order.
            const unsigned curOrder =
                m_fields[m_velocity[0]]->GetExp(0)->GetBasisNumModes(0);
            if (snapOrder != curOrder)
            {
                std::cout << "[DOVelocityCorrectionScheme][POD] WARNING: snapshot " << filename
                          << " uses polynomial order " << snapOrder
                          << " but current expansion is " << curOrder
                          << "; ExtractDataToCoeffs will interpolate.\n";
                m_orderWarningEmitted = true;  // emit once, not per-snapshot
            }
        }

        outCoeffs.assign(m_nVel, std::vector<NekDouble>(m_nCoeffs, 0.0));
        for (int c = 0; c < m_nVel; ++c)
        {
            std::string varName = m_session->GetVariable(m_velocity[c]);
            Array<OneD, NekDouble> coeffArr(m_nCoeffs, 0.0);
            bool found = false;
            for (size_t i = 0; i < FieldDef.size(); ++i)
            {
                const auto &fields = FieldDef[i]->m_fields;
                if (std::find(fields.begin(), fields.end(), varName) ==
                    fields.end())
                {
                    continue;
                }
                m_fields[m_velocity[c]]->ExtractDataToCoeffs(
                    FieldDef[i], FieldData[i], varName, coeffArr);
                found = true;
            }
            ASSERTL0(found,
                     "DOPODInitialiser: variable '" + varName +
                         "' not found in snapshot " + filename);
            std::memcpy(outCoeffs[c].data(), coeffArr.data(),
                        sizeof(NekDouble) * m_nCoeffs);
        }
    }

    LibUtilities::SessionReaderSharedPtr               m_session;
    Array<OneD, MultiRegions::ExpListSharedPtr>        m_fields;
    Array<OneD, int>                                   m_velocity;
    Config                                             m_cfg;

    int  m_nVel                 = 0;
    int  m_nCoeffs              = 0;
    int  m_nPhys                = 0;
    bool m_orderWarningEmitted  = false;

    /// Per-snapshot coefficient buffers: [snapshot][component][coeff].
    std::vector<std::vector<std::vector<NekDouble>>> m_snapCoeffs;

    /// Output POD modes: [mode][component][coeff/phys].
    std::vector<std::vector<std::vector<NekDouble>>> m_modeCoeffs;
    std::vector<std::vector<std::vector<NekDouble>>> m_modePhys;

    /// Mean field used by Compute() (per cfg.meanType): [component][coeff].
    std::vector<std::vector<NekDouble>> m_mean;

    std::vector<NekDouble>              m_sigmas;
    /// m_eigVecs[k][p] = v_{p,k}, the K-vector eigenvector for mode k.
    std::vector<std::vector<NekDouble>> m_eigVecs;
    int                                 m_K              = 0;
    NekDouble                           m_totalEnergy    = 0.0;
    NekDouble                           m_capturedEnergy = 0.0;
};

// ============================================================================
// Public surface
// ============================================================================
DOPODInitialiser::DOPODInitialiser(
    const LibUtilities::SessionReaderSharedPtr      &session,
    const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    const Array<OneD, int>                            &velocity,
    const Config                                      &cfg)
    : m_impl(std::make_unique<Impl>(session, fields, velocity, cfg))
{
}

DOPODInitialiser::~DOPODInitialiser() = default;

void DOPODInitialiser::Compute()
{
    m_impl->Compute();
}

void DOPODInitialiser::ExportToDOMode(Array<OneD, NekDouble> &modePhys,
                                       Array<OneD, NekDouble> &modeCoeffs) const
{
    m_impl->ExportToDOMode(modePhys, modeCoeffs);
}

void DOPODInitialiser::ExportMean(Array<OneD, NekDouble> &meanCoeffs) const
{
    m_impl->ExportMean(meanCoeffs);
}

const std::vector<NekDouble> &DOPODInitialiser::SingularValues() const
{
    return m_impl->SingularValues();
}

const std::vector<std::vector<NekDouble>> &DOPODInitialiser::EigenVectors() const
{
    return m_impl->EigenVectors();
}

int DOPODInitialiser::NumSnapshots() const
{
    return m_impl->NumSnapshots();
}

NekDouble DOPODInitialiser::EnergyFraction() const
{
    return m_impl->EnergyFraction();
}

void DOPODInitialiser::RecomputeYiByProjection(
    const Array<OneD, NekDouble> &modePhys,
    const Array<OneD, NekDouble> &modeCoeffs,
    Array<OneD, NekDouble>       &Yi,
    int                           nParticles)
{
    m_impl->RecomputeYiByProjection(modePhys, modeCoeffs, Yi, nParticles);
}

} // namespace Nektar
