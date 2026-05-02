//////////////////////////////////////////////////////////////////////////////polationMethod
//
// File: TimeIntegrationSchemeGEM.h
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
// Description: Header file of time integration scheme GEM base class
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_UTILITIES_TIME_INTEGRATION_TIME_INTEGRATION_SCHEME_GEM
#define NEKTAR_LIB_UTILITIES_TIME_INTEGRATION_TIME_INTEGRATION_SCHEME_GEM

#define LUE LIB_UTILITIES_EXPORT

#include <string>

#include <LibUtilities/BasicUtils/VmathArray.hpp>
#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>

namespace Nektar::LibUtilities
{

///////////////////////////////////////////////////////////////////////////////
/// Class for spectral deferred correction integration.
class TimeIntegrationSchemeGEM : public TimeIntegrationScheme
{
public:
    TimeIntegrationSchemeGEM(std::string variant, size_t order,
                             std::vector<NekDouble> freeParams)
        : TimeIntegrationScheme(variant, order, freeParams),
          m_name("ExtrapolationMethod")
    {
        ASSERTL0(variant == "" || variant == "ExplicitEuler" ||
                     variant == "ImplicitEuler" || variant == "IMEXEuler" ||
                     variant == "ExplicitMidpoint" ||
                     variant == "ImplicitMidpoint",
                 "Extrapolation Time integration "
                 "scheme bad variant (ExplicitEuler, ImplicitEuler, "
                 "ExplicitMidpoint)")

        if (variant == "IMEXEuler")
        {
            std::cerr << "WARNING: IMEX Euler extrapolation method has been "
                         "implemented but its use is not recommended as the "
                         "approach is affected by order-reduction problems."
                      << std::endl;
        }

        if (variant == "" || variant == "ExplicitEuler" ||
            variant == "ImplicitEuler" || variant == "IMEXEuler")
        {
            ASSERTL0(order >= 1, "Extrapolation Time integration "
                                 "scheme bad order numbers (>=1): " +
                                     std::to_string(order));
        }
        else if (variant == "ExplicitMidpoint" || variant == "ImplicitMidpoint")
        {
            ASSERTL0(order >= 2, "Extrapolation Time integration "
                                 "scheme bad order numbers (>=2): " +
                                     std::to_string(order));

            ASSERTL0(order % 2 == 0,
                     "Extrapolation Time integration "
                     "scheme bad order numbers (even number): " +
                         std::to_string(order));
        }

        m_variant    = variant;
        m_order      = order;
        m_freeParams = freeParams;
        if (variant == "ExplicitEuler" || variant == "ExplicitMidpoint")
        {
            m_schemeType = eExplicit;
        }
        else if (variant == "ImplicitEuler" || variant == "ImplicitMidpoint")
        {
            m_schemeType = eImplicit;
        }
        else if (variant == "IMEXEuler")
        {
            m_schemeType = eIMEX;
        }
    }

    /// Destructor
    ~TimeIntegrationSchemeGEM() override
    {
    }

    static TimeIntegrationSchemeSharedPtr create(
        std::string variant, size_t order, std::vector<NekDouble> freeParams)
    {
        TimeIntegrationSchemeSharedPtr p =
            MemoryManager<TimeIntegrationSchemeGEM>::AllocateSharedPtr(
                variant, order, freeParams);

        return p;
    }

    static std::string className;

protected:
    LUE std::string v_GetName() const override;
    LUE std::string v_GetVariant() const override;
    LUE size_t v_GetOrder() const override;
    LUE std::vector<NekDouble> v_GetFreeParams() const override;
    LUE TimeIntegrationSchemeType v_GetIntegrationSchemeType() const override;
    LUE NekDouble v_GetTimeStability() const override;
    LUE size_t v_GetNumIntegrationPhases() const override;
    LUE const TripleArray &v_GetSolutionVector() const override;
    LUE TripleArray &v_UpdateSolutionVector() override;
    LUE void v_SetSolutionVector(const size_t Offset,
                                 const DoubleArray &y) override;
    LUE void v_InitializeScheme(
        const NekDouble deltaT, ConstDoubleArray &y_0, const NekDouble time,
        const TimeIntegrationSchemeOperators &op) override;
    LUE ConstDoubleArray &v_TimeIntegrate(const size_t timestep,
                                          const NekDouble delta_t) override;
    LUE void v_print(std::ostream &os) const override;
    LUE void v_printFull(std::ostream &os) const override;

    // Variables common to all schemes.
    NekDouble m_time;
    std::string m_name;
    std::string m_variant;
    std::vector<NekDouble> m_freeParams;
    TimeIntegrationSchemeType m_schemeType{eNoTimeIntegrationSchemeType};
    TimeIntegrationSchemeOperators m_op;

    // Storage of previous states and associated timesteps.
    TripleArray m_Y;  /// Array containing the stage values
    TripleArray m_T;  /// Array containing the solution values
    TripleArray m_T0; /// Array containing the solution values
    DoubleArray m_F;  /// Array corresponding to the stage Derivatives
    DoubleArray m_F0; /// Array corresponding to the stage Derivatives

    // GEM parameter
    size_t m_order{0};   // Order of the integration scheme
    size_t m_nvars{0};   // Number of variables in the integration scheme.
    size_t m_npoints{0}; // Uniform number of points in the fast path.
    Array<OneD, size_t> m_varSizes; // Per-variable number of points.
    bool m_uniformVarSizes{false};
    bool m_initialized{false};

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

    inline size_t GetVarSize(const size_t var) const
    {
        return m_uniformVarSizes ? m_npoints : m_varSizes[var];
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

}; // end class TimeIntegrationSchemeGEM

LUE std::ostream &operator<<(std::ostream &os,
                             const TimeIntegrationSchemeGEM &rhs);

LUE std::ostream &operator<<(std::ostream &os,
                             const TimeIntegrationSchemeGEMSharedPtr &rhs);

} // namespace Nektar::LibUtilities

#endif
