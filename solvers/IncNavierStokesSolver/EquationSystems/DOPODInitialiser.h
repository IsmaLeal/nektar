///////////////////////////////////////////////////////////////////////////////
//
// File: DOPODInitialiser.h
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
// Description: POD-based DO mode initialiser. Reads a set of velocity
//              snapshots (.chk/.fld), subtracts a chosen mean, computes the
//              method-of-snapshots POD in the velocity mass inner product,
//              and exports the leading S modes into the DOVelocityCorrectionScheme packed layout.
//              The downstream DOVelocityCorrectionScheme::ReOrthonormalise() handles the
//              homogeneous-BC bracket, divergence-free projection, MGS, and
//              final mass-orthonormalisation.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERS_DOPODINITIALISER_H
#define NEKTAR_SOLVERS_DOPODINITIALISER_H

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ExpList.h>

#include <memory>
#include <string>
#include <vector>

namespace Nektar
{
class DOPODInitialiser
{
public:
    enum class MeanType
    {
        TimeMean,           ///< average of all snapshots (default)
        FirstSnapshot,      ///< use the first snapshot as the base flow
        ProvidedMeanField   ///< load mean from cfg.meanFile
    };

    struct Config
    {
        std::vector<std::string> snapshotFiles;     ///< resolved (sorted) snapshot paths
        std::string              meanFile;          ///< only if meanType==ProvidedMeanField
        MeanType                 meanType = MeanType::TimeMean;
        int                      numModes = 0;      ///< S
        bool                     verbose  = false;
    };

    DOPODInitialiser(
        const LibUtilities::SessionReaderSharedPtr      &session,
        const Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
        const Array<OneD, int>                            &velocity,
        const Config                                      &cfg);

    ~DOPODInitialiser();

    /// Load snapshots, subtract mean, compute POD by method-of-snapshots.
    void Compute();

    /// Pack the leading S POD modes into the DOVelocityCorrectionScheme layout
    ///   modePhys  [(i*nVel + c)*nPhys   .. +nPhys )
    ///   modeCoeffs[(i*nVel + c)*nCoeffs .. +nCoeffs)
    /// Mirrors the layout produced by InitialiseModesFromEllipticEigenbasis.
    void ExportToDOMode(Array<OneD, NekDouble> &modePhys,
                        Array<OneD, NekDouble> &modeCoeffs) const;

    /// Pack the mean field computed during Compute() (per cfg.meanType) into
    /// a per-component coeff layout: meanCoeffs[c*nCoeffs + j] for c in [0,nVel).
    void ExportMean(Array<OneD, NekDouble> &meanCoeffs) const;

    /// Singular values sigma_k = sqrt(lambda_k), descending.
    const std::vector<NekDouble> &SingularValues() const;

    /// Method-of-snapshots eigenvectors for the leading S modes:
    ///   eigVecs[k][p] = v_{p,k}  (snapshot p's coefficient on POD mode k).
    /// Shape: S rows of K entries (K = number of snapshots used).
    /// The natural Y init is Y_{p,i} = sigma_i * v_{p,i} (snapshot projection).
    const std::vector<std::vector<NekDouble>> &EigenVectors() const;

    /// Number of snapshots K (i.e. column-count of the eigenvector matrix).
    int NumSnapshots() const;

    /// Energy fraction Sigma_{k=0..S-1} sigma_k^2 / Sigma_{k=0..K-1} sigma_k^2.
    NekDouble EnergyFraction() const;

    /// Re-projects each snapshot onto the supplied post-ReOrthonormalise basis
    /// to rebuild the stochastic coefficients consistently with the modes that
    /// were ACTUALLY stored after reorth (the Stokes-projection step inside
    /// reorth changes the subspace and is non-deterministic under MPI, so the
    /// natural Y init Y_{p,k}=sigma_k v_{p,k} no longer matches the stored
    /// modes). Reads snapshots from disk, subtracts the same mean used by
    /// Compute(), and writes
    ///   Yi[p*nVel*nModes + k] = <(snap_p - mean), mode_k>_M
    /// for p in [0, K) and k in [0, S). For p in [K, nParticles) Yi is left
    /// unchanged (filled by the caller via iid Gaussian).
    /// `modePhys` is the post-reorth mode buffer with the DOVelocityCorrectionScheme packed layout
    /// modePhys[(i*nVel + c)*nPhys .. +nPhys). nParticles is m_nDOParticles
    /// from the caller; only the first K particles are projected here.
    void RecomputeYiByProjection(
        const Array<OneD, NekDouble> &modePhys,
        const Array<OneD, NekDouble> &modeCoeffs,
        Array<OneD, NekDouble>       &Yi,
        int                          nParticles);

    /// Helper: expand a glob pattern (path with one '*' in the basename) into
    /// a sorted list of absolute file paths. Sorts numerically when filenames
    /// match <stem>_<digits>.<ext>; otherwise lexicographically. Returns an
    /// empty vector if no files match.
    static std::vector<std::string> ExpandGlob(const std::string &pattern);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace Nektar

#endif // NEKTAR_SOLVERS_DOPODINITIALISER_H
