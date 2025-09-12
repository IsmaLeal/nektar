///////////////////////////////////////////////////////////////////////////////
//
// File: SchemeInitializer.cpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// License for the specific language governing rights and limitations under
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
// Description: This file isused to add each of the Time Integration Schemes
//              to the NekFactory.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/TimeIntegration/AdamsBashforthTimeIntegrationSchemes.h>
#include <LibUtilities/TimeIntegration/AdamsMoultonTimeIntegrationSchemes.h>
#include <LibUtilities/TimeIntegration/BDFImplicitTimeIntegrationSchemes.h>
#include <LibUtilities/TimeIntegration/CNABTimeIntegrationScheme.h>
#include <LibUtilities/TimeIntegration/DIRKTimeIntegrationSchemes.h>
#include <LibUtilities/TimeIntegration/EulerExponentialTimeIntegrationSchemes.h>
#include <LibUtilities/TimeIntegration/EulerTimeIntegrationSchemes.h>
#include <LibUtilities/TimeIntegration/IMEXTimeIntegrationSchemes.h>
#include <LibUtilities/TimeIntegration/IMEXdirkTimeIntegrationSchemes.h>
#include <LibUtilities/TimeIntegration/MCNABTimeIntegrationScheme.h>
#include <LibUtilities/TimeIntegration/NoSchemeTimeIntegrationScheme.h>
#include <LibUtilities/TimeIntegration/RungeKuttaTimeIntegrationSchemes.h>

#include <LibUtilities/TimeIntegration/TimeIntegrationSchemeFIT.h>

#include <LibUtilities/TimeIntegration/ExplicitTimeIntegrationSchemeSDC.h>
#include <LibUtilities/TimeIntegration/IMEXTimeIntegrationSchemeSDC.h>
#include <LibUtilities/TimeIntegration/ImplicitTimeIntegrationSchemeSDC.h>

#include <LibUtilities/TimeIntegration/TimeIntegrationSchemeGEM.h>

#include <LibUtilities/BasicUtils/SessionReader.h>

namespace Nektar::LibUtilities
{

// Register all the schemes with the Time Integration Scheme Factory...
//
#define FACTORYREGISTER(scheme)                                                \
    std::string scheme##TimeIntegrationScheme::className =                     \
        GetTimeIntegrationSchemeFactory().RegisterCreatorFunction(             \
            #scheme, scheme##TimeIntegrationScheme::create)

// AdamsBashforthTimeIntegrationSchemes.h
FACTORYREGISTER(AdamsBashforth);

// AdamsMoultonTimeIntegrationSchemes.h
FACTORYREGISTER(AdamsMoulton);

// BDFImplicitTimeIntegrationSchemes.h
FACTORYREGISTER(BDFImplicit);

// EulerTimeIntegrationSchemes.h
FACTORYREGISTER(Euler);

// EulerExponentialTimeIntegrationSchemes.h
FACTORYREGISTER(EulerExponential);

// TimeIntegrationSchemesFIT.h
std::string FractionalInTimeIntegrationScheme::className =
    GetTimeIntegrationSchemeFactory().RegisterCreatorFunction(
        "FractionalInTime", FractionalInTimeIntegrationScheme::create);

// CNABTimeIntegrationScheme.h
FACTORYREGISTER(CNAB);

// DIRKTimeIntegrationSchemes.h
FACTORYREGISTER(DIRK);

// IMEXdirkTimeIntegrationSchemes.h
FACTORYREGISTER(IMEXdirk);

// IMEXTimeIntegrationSchemes.h
FACTORYREGISTER(IMEX);

// MCNABTimeIntegrationScheme.h
FACTORYREGISTER(MCNAB);

// RungeKuttaTimeIntegrationSchemes.h
FACTORYREGISTER(RungeKutta);

// Do nothing
FACTORYREGISTER(NoScheme);

// TimeIntegrationSchemesSDC.h
std::string ExplicitTimeIntegrationSchemeSDC::className =
    GetTimeIntegrationSchemeFactory().RegisterCreatorFunction(
        "ExplicitSDC", ExplicitTimeIntegrationSchemeSDC::create);
std::string ImplicitTimeIntegrationSchemeSDC::className =
    GetTimeIntegrationSchemeFactory().RegisterCreatorFunction(
        "ImplicitSDC", ImplicitTimeIntegrationSchemeSDC::create);
std::string IMEXTimeIntegrationSchemeSDC::className =
    GetTimeIntegrationSchemeFactory().RegisterCreatorFunction(
        "IMEXSDC", IMEXTimeIntegrationSchemeSDC::create);

// TimeIntegrationSchemesGEM.h
std::string TimeIntegrationSchemeGEM::className =
    GetTimeIntegrationSchemeFactory().RegisterCreatorFunction(
        "ExtrapolationMethod", TimeIntegrationSchemeGEM::create);

} // namespace Nektar::LibUtilities
