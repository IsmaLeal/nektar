///////////////////////////////////////////////////////////////////////////////
//
// File: TimeIntegrationSchemeSDC.h
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
// Description: Header file of time integration scheme SDC base class
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_UTILITIES_TIME_INTEGRATION_TIME_INTEGRATION_SCHEME_SDC
#define NEKTAR_LIB_UTILITIES_TIME_INTEGRATION_TIME_INTEGRATION_SCHEME_SDC

#define LUE LIB_UTILITIES_EXPORT

#include <string>

#include <LibUtilities/BasicUtils/VmathArray.hpp>
#include <LibUtilities/Foundations/Foundations.hpp>
#include <LibUtilities/Foundations/ManagerAccess.h>
#include <LibUtilities/Foundations/Points.h>
#include <LibUtilities/Polylib/Polylib.h>
#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>

namespace Nektar::LibUtilities
{

///////////////////////////////////////////////////////////////////////////////
/// Class for spectral deferred correction integration.
class TimeIntegrationSchemeSDC : public TimeIntegrationScheme
{
public:
    TimeIntegrationSchemeSDC(std::string variant, size_t order,
                             std::vector<NekDouble> freeParams)
        : TimeIntegrationScheme(variant, order, freeParams),
          m_name("SpectralDeferredCorrection")
    {
        ASSERTL0(freeParams.size() == 2,
                 "SDC Time integration scheme invalid number "
                 "of free parameters, expected two "
                 "<theta, number of quadrature>, received " +
                     std::to_string(freeParams.size()));

        ASSERTL0(0.0 <= freeParams[0] && freeParams[0] <= 1.0,
                 "Spectral Deferred Correction Time integration "
                 "scheme bad parameter numbers (0.0 - 1.0): " +
                     std::to_string(freeParams[0]));

        // Set scheme properties
        m_variant    = variant;
        m_order      = order;
        m_freeParams = freeParams;
        m_theta      = m_freeParams[0];
        m_nQuadPts   = m_freeParams[1];

        if (m_variant == "Equidistant")
        {
            ASSERTL0(1 <= m_nQuadPts,
                     m_variant +
                         " quadrature require quadrature "
                         "numbers (>=1" +
                         "): " + std::to_string(freeParams[1]));

            m_first_quadrature = (m_nQuadPts == 1) ? false : true;
            m_last_quadrature  = (m_nQuadPts == 1) ? false : true;
            m_ordermin         = 1;
            m_ordermax         = m_nQuadPts;
            m_pointsKey        = LibUtilities::PointsKey(
                m_nQuadPts, LibUtilities::ePolyEvenlySpaced);
        }
        else if (m_variant == "GaussLobattoLegendre")
        {
            ASSERTL0(2 <= m_nQuadPts,
                     m_variant +
                         " quadrature require quadrature "
                         "numbers (>=2" +
                         "): " + std::to_string(freeParams[1]));

            m_first_quadrature = true;
            m_last_quadrature  = true;
            m_ordermin         = 1;
            m_ordermax         = 2 * m_nQuadPts - 2;
            m_pointsKey        = LibUtilities::PointsKey(
                m_nQuadPts, LibUtilities::eGaussLobattoLegendre);
        }
        else if (m_variant == "GaussRadauLegendre")
        {
            ASSERTL0(2 <= m_nQuadPts,
                     m_variant +
                         " quadrature require quadrature "
                         "numbers (>=1" +
                         "): " + std::to_string(freeParams[1]));

            m_first_quadrature = false;
            m_last_quadrature  = true;
            m_ordermin         = 1;
            m_ordermax         = 2 * m_nQuadPts - 1;
            m_pointsKey        = LibUtilities::PointsKey(
                m_nQuadPts, LibUtilities::eGaussRadauPLegendre);
        }
        else if (m_variant == "GaussGaussLegendre")
        {
            ASSERTL0(1 <= m_nQuadPts,
                     m_variant +
                         " quadrature require quadrature "
                         "numbers (>=1" +
                         "): " + std::to_string(freeParams[1]));

            m_first_quadrature = false;
            m_last_quadrature  = false;
            m_ordermin         = 1;
            m_ordermax         = 2 * m_nQuadPts;
            m_pointsKey        = LibUtilities::PointsKey(
                m_nQuadPts, LibUtilities::eGaussGaussLegendre);
        }
        else
        {
            ASSERTL0(false, "unknow variant (quadrature) type");
        }

        ASSERTL0(m_ordermin <= m_order,
                 "Spectral Deferred Correction Time integration "
                 "scheme bad order numbers (>=" +
                     std::to_string(m_ordermin) +
                     "): " + std::to_string(m_order));

        ASSERTL0(m_ordermax >= m_order,
                 "Spectral Deferred Correction Time integration "
                 "scheme bad order numbers (<=" +
                     std::to_string(m_ordermax) +
                     "): " + std::to_string(m_order));

        // Add one extra quadrature points for i.c., if necessary
        if (!m_first_quadrature)
        {
            m_nQuadPts = m_nQuadPts + 1;
        }

        // Get quadrature points and rescale to [0, 1]
        m_tau         = SingleArray(m_nQuadPts, 0.0);
        size_t offset = m_first_quadrature ? 0 : 1;
        for (size_t i = offset; i < m_nQuadPts; i++)
        {
            NekDouble tau = PointsManager()[m_pointsKey]->GetZ()[i - offset];
            m_tau[i]      = (tau + 1.0) / 2.0;
        }
    }

    /// Destructor
    ~TimeIntegrationSchemeSDC() override
    {
    }

    static TimeIntegrationSchemeSharedPtr create(
        std::string variant, size_t order, std::vector<NekDouble> freeParams)
    {
        TimeIntegrationSchemeSharedPtr p =
            MemoryManager<TimeIntegrationSchemeSDC>::AllocateSharedPtr(
                variant, order, freeParams);

        return p;
    }

    static std::string className;

    LUE void SetPFASST(bool pfasst)
    {
        m_PFASST = pfasst;
    }

    LUE void SetTime(double time)
    {
        m_time = time;
    }

    LUE size_t GetMaxOrder() const
    {
        return m_ordermax;
    }

    LUE bool HasFirstQuadrature() const
    {
        return m_first_quadrature;
    }

    LUE bool HasLastQuadrature() const
    {
        return m_last_quadrature;
    }

    LUE size_t GetQuadPtsNumber() const
    {
        return m_nQuadPts;
    }

    LUE size_t GetNpoints() const
    {
        ASSERTL1(m_uniformVarSizes,
                 "GetNpoints is only valid for uniform-size states.");
        return m_npoints;
    }

    LUE size_t GetNvars() const
    {
        return m_nvars;
    }

    LUE const Array<OneD, const size_t> &GetVarSizes() const
    {
        return m_varSizes;
    }

    LUE size_t GetVarSize(const size_t var) const
    {
        return m_uniformVarSizes ? m_npoints : m_varSizes[var];
    }

    LUE const PointsKey &GetPointsKey() const
    {
        return m_pointsKey;
    }

    LUE const DoubleArray &GetFirstQuadratureSolutionVector() const
    {
        return m_Y[0];
    }

    LUE DoubleArray &UpdateFirstQuadratureSolutionVector()
    {
        return m_Y[0];
    }

    LUE const DoubleArray &GetLastQuadratureSolutionVector() const
    {
        return m_last_quadrature ? m_Y[m_nQuadPts - 1] : m_Y_f;
    }

    LUE DoubleArray &UpdateLastQuadratureSolutionVector()
    {
        return m_last_quadrature ? m_Y[m_nQuadPts - 1] : m_Y_f;
    }

    LUE const TripleArray &GetResidualVector() const
    {
        return m_F;
    }

    LUE TripleArray &UpdateResidualVector()
    {
        return m_F;
    }

    LUE const TripleArray &GetIntegratedResidualVector() const
    {
        return m_QFint;
    }

    LUE TripleArray &UpdateIntegratedResidualVector()
    {
        return m_QFint;
    }

    LUE const TripleArray &GetFAScorrectionVector() const
    {
        return m_FAScorr;
    }

    LUE TripleArray &UpdateFAScorrectionVector()
    {
        return m_FAScorr;
    }

    LUE void ResidualEval(const NekDouble &delta_t, const size_t n)
    {
        v_ResidualEval(delta_t, n);
    }

    LUE void ResidualEval(const NekDouble &delta_t)
    {
        v_ResidualEval(delta_t);
    }

    LUE void ComputeInitialGuess(const NekDouble &delta_t)
    {
        v_ComputeInitialGuess(delta_t);
    }

    LUE void SDCIterationLoop(const NekDouble &delta_t)
    {
        v_SDCIterationLoop(delta_t);
    }

    LUE void UpdateFirstQuadrature(void);
    LUE void UpdateLastQuadrature(void);
    LUE void AddFASCorrectionToSFint(void);
    LUE void UpdateIntegratedResidualSFint(const NekDouble &delta_t);
    LUE void UpdateIntegratedResidualQFint(const NekDouble &delta_t);

protected:
    LUE std::string v_GetName() const override;
    LUE std::string v_GetVariant() const override;
    LUE size_t v_GetOrder() const override;
    LUE std::vector<NekDouble> v_GetFreeParams() const override;
    LUE TimeIntegrationSchemeType v_GetIntegrationSchemeType() const override;
    LUE NekDouble v_GetTimeStability() const override;
    LUE size_t v_GetNumIntegrationPhases() const override;

    /**
     * \brief Gets the solution vector of the ODE
     */
    LUE const TripleArray &v_GetSolutionVector() const override;
    LUE TripleArray &v_UpdateSolutionVector() override;

    /**
     * \brief Sets the solution vector of the ODE
     */
    LUE void v_SetSolutionVector(const size_t Offset,
                                 const DoubleArray &y) override;

    // The worker methods from the base class that are virtual
    LUE void v_InitializeScheme(
        const NekDouble deltaT, ConstDoubleArray &y_0, const NekDouble time,
        const TimeIntegrationSchemeOperators &op) override;

    LUE ConstDoubleArray &v_TimeIntegrate(const size_t timestep,
                                          const NekDouble delta_t) override;

    LUE virtual void v_ResidualEval([[maybe_unused]] const NekDouble &delta_t,
                                    [[maybe_unused]] const size_t n)
    {
        ASSERTL0(false, "Specific version of spectral deferred correction "
                        "not implemented");
    }

    LUE virtual void v_ResidualEval([[maybe_unused]] const NekDouble &delta_t)
    {
        ASSERTL0(false, "Specific version of spectral deferred correction "
                        "not implemented");
    }

    LUE virtual void v_ComputeInitialGuess(
        [[maybe_unused]] const NekDouble &delta_t)
    {
        ASSERTL0(false, "Specific version of spectral deferred correction "
                        "not implemented");
    }

    LUE virtual void v_SDCIterationLoop(
        [[maybe_unused]] const NekDouble &delta_t)
    {
        ASSERTL0(false, "Specific version of spectral deferred correction "
                        "not implemented");
    }

    LUE void v_print(std::ostream &os) const override;
    LUE void v_printFull(std::ostream &os) const override;

    // Variables common to all schemes
    NekDouble m_time;
    std::string m_name;
    std::string m_variant;
    std::vector<NekDouble> m_freeParams;
    TimeIntegrationSchemeType m_schemeType{eNoTimeIntegrationSchemeType};
    TimeIntegrationSchemeOperators m_op;

    // Storage of states and associated timesteps
    PointsKey m_pointsKey; /// Object containing quadrature data
    SingleArray m_tau;     /// Array containing the quadrature points
    DoubleArray m_Y_f;     /// Array containing the last stage values
    TripleArray m_Y;       /// Array containing the stage values
    TripleArray m_F;       /// Array containing the stage derivatives
    TripleArray m_FAScorr; /// Array containing the FAS correction term
    TripleArray m_SFint;   /// Array containing the integrated residual term
    TripleArray m_QFint;   /// Array containing the integrated residual term
    SingleArray m_QMat;    /// Array containing the integration matrix
    SingleArray m_interp;  /// Array containing the interpolation coefficients

    // SDC parameter
    NekDouble m_theta{1.0}; /// SDC parameter
    size_t m_ordermin{0};   /// Minimum order of the integration scheme
    size_t m_ordermax{0};   /// Maximum order of the integration scheme
    size_t m_order{0};      /// Order of the integration scheme
    size_t m_nQuadPts{0};   /// Number of quadrature points
    size_t m_nvars{0};      /// Number of variables in the integration scheme
    size_t m_npoints{0};    /// Uniform number of points in the fast path
    Array<OneD, size_t> m_varSizes; /// Per-variable numbers of points
    bool m_uniformVarSizes{false};
    bool m_first_quadrature{true};
    bool m_last_quadrature{true};
    bool m_initialized{false};
    bool m_PFASST{false};

    inline Array<OneD, size_t> BuildVarSizes(ConstDoubleArray &y) const
    {
        Array<OneD, size_t> varSizes(y.size(), size_t(0));
        for (size_t i = 0; i < y.size(); ++i)
        {
            varSizes[i] = y[i].size();
        }

        return varSizes;
    }

    inline bool IsUniform(const Array<OneD, size_t> &varSizes) const
    {
        if (varSizes.size() == 0)
        {
            return true;
        }

        const size_t npoints = varSizes[0];
        for (size_t i = 1; i < varSizes.size(); ++i)
        {
            if (varSizes[i] != npoints)
            {
                return false;
            }
        }

        return true;
    }

    inline void SetVarSizes(const Array<OneD, size_t> &varSizes)
    {
        m_varSizes        = varSizes;
        m_uniformVarSizes = IsUniform(varSizes);
        m_npoints         = (m_uniformVarSizes && varSizes.size() > 0)
                                ? varSizes[0]
                                : 0;
    }

    inline void CheckVarSizes(const DoubleArray &y) const
    {
        ASSERTL1(y.size() == m_varSizes.size(),
                 "Non-matching number of variables.");

        for (size_t i = 0; i < y.size(); ++i)
        {
            ASSERTL1(y[i].size() == m_varSizes[i],
                     "Non-matching number of coefficients.");
        }
    }

    inline bool HasMatchingVarSizes(const Array<OneD, size_t> &varSizes) const
    {
        if (m_varSizes.size() != varSizes.size())
        {
            return false;
        }

        for (size_t i = 0; i < varSizes.size(); ++i)
        {
            if (m_varSizes[i] != varSizes[i])
            {
                return false;
            }
        }

        return true;
    }

    inline bool CanReuseStorage(ConstDoubleArray &y) const
    {
        return m_initialized && m_nvars == y.size() &&
               HasMatchingVarSizes(BuildVarSizes(y));
    }

}; // end class TimeIntegrationSchemeSDC

} // namespace Nektar::LibUtilities

#endif
