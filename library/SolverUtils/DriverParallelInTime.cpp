///////////////////////////////////////////////////////////////////////////////
//
// File DriverParallelInTime.cpp
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
// Description: Driver class for the parallel-in-time solver
//
///////////////////////////////////////////////////////////////////////////////

#include <iomanip>

#include <LibUtilities/BasicUtils/Timer.h>
#include <LocalRegions/HexExp.h>
#include <LocalRegions/PrismExp.h>
#include <LocalRegions/PyrExp.h>
#include <LocalRegions/QuadExp.h>
#include <LocalRegions/SegExp.h>
#include <LocalRegions/TetExp.h>
#include <LocalRegions/TriExp.h>
#include <SolverUtils/DriverParallelInTime.h>
#include <SpatialDomains/MeshGraphIO.h>
#include <boost/format.hpp>

namespace Nektar::SolverUtils
{

/**
 *
 */
DriverParallelInTime::DriverParallelInTime(
    const LibUtilities::SessionReaderSharedPtr pSession,
    const SpatialDomains::MeshGraphSharedPtr pGraph)
    : Driver(pSession, pGraph)
{
}

/**
 *
 */
void DriverParallelInTime::v_InitObject(std::ostream &out)
{
    try
    {
        // Retrieve the type of evolution operator to use
        m_EvolutionOperator =
            m_session->GetSolverInfoAsEnum<EvolutionOperatorType>(
                "EvolutionOperator");

        m_nTimeLevel = 2; // Only two time levels currently implemented.

        m_equ = Array<OneD, EquationSystemSharedPtr>(m_nTimeLevel);

        // Set the AdvectiveType tag and create EquationSystem objects.
        switch (m_EvolutionOperator)
        {
            case eNonlinear:
                SetParallelInTimeEquationSystem("Convective");
                break;
            case eDirect:
                SetParallelInTimeEquationSystem("Linearised");
                break;
            case eAdjoint:
                SetParallelInTimeEquationSystem("Adjoint");
                break;
            case eSkewSymmetric:
                SetParallelInTimeEquationSystem("SkewSymmetric");
                break;
            default:
                ASSERTL0(false, "Unrecognised evolution operator.");
        }

        // Set pointers.
        m_EqSys = Array<OneD, std::shared_ptr<UnsteadySystem>>(m_nTimeLevel);
        for (size_t timeLevel = 0; timeLevel < m_nTimeLevel; timeLevel++)
        {
            m_EqSys[timeLevel] =
                std::dynamic_pointer_cast<SolverUtils::UnsteadySystem>(
                    m_equ[timeLevel]);
        }

        // Set time communication parameters.
        m_numChunks = m_comm->GetTimeComm()->GetSize();
        m_chunkRank = m_comm->GetTimeComm()->GetRank();
    }
    catch (int e)
    {
        ASSERTL0(e == -1, "No such class class defined.");
        out << "An error occurred during driver initialisation." << std::endl;
    }
}

/**
 *
 */
void DriverParallelInTime::v_Execute([[maybe_unused]] std::ostream &out)
{
}

/**
 * Set the ParallelInTime (coarse solver) session file
 */
void DriverParallelInTime::SetParallelInTimeEquationSystem(
    std::string AdvectiveType)
{
    // Retrieve the equation system to solve.
    ASSERTL0(m_session->DefinesSolverInfo("EqType"),
             "EqType SolverInfo tag must be defined.");
    std::string vEquation = m_session->DefinesSolverInfo("SolverType")
                                ? m_session->GetSolverInfo("SolverType")
                                : m_session->GetSolverInfo("EqType");

    // Check such a module exists for this equation.
    ASSERTL0(GetEquationSystemFactory().ModuleExists(vEquation),
             "EquationSystem '" + vEquation +
                 "' is not defined.\n"
                 "Ensure equation name is correct and module is compiled.\n");

    // Set fine parallel-in-time solver.
    ASSERTL0(m_session->DefinesCmdLineArgument("npt"),
             "Number of parallel-in-time processes npt must be specified.");
    m_session->SetTag("AdvectiveType", AdvectiveType);
    m_session->SetTag("ParallelInTimeSolver", "TimeLevel0");
    m_equ[0] = GetEquationSystemFactory().CreateInstance(vEquation, m_session,
                                                         m_graph);

    // Define argument for the coarse parallel-in-time solver.
    int npx = m_session->DefinesCmdLineArgument("npx")
                  ? m_session->GetCmdLineArgument<int>("npx")
                  : 1;
    int npy = m_session->DefinesCmdLineArgument("npy")
                  ? m_session->GetCmdLineArgument<int>("npy")
                  : 1;
    int npz = m_session->DefinesCmdLineArgument("npz")
                  ? m_session->GetCmdLineArgument<int>("npz")
                  : 1;
    int nsz = m_session->DefinesCmdLineArgument("nsz")
                  ? m_session->GetCmdLineArgument<int>("nsz")
                  : 1;
    int npt = m_session->GetCmdLineArgument<int>("npt");

    // Convert into string.
    std::string npx_string    = std::to_string(npx);
    std::string npy_string    = std::to_string(npy);
    std::string npz_string    = std::to_string(npz);
    std::string nsz_string    = std::to_string(nsz);
    std::string npt_string    = std::to_string(npt);
    std::string driver_string = "Driver=" + m_session->GetSolverInfo("Driver");

    // use-opt-file
    bool useOptFile         = m_session->DefinesCmdLineArgument("use-opt-file");
    std::string optfilename = useOptFile ? m_session->GetFilenames()[0] : "";

    char *argv[] = {const_cast<char *>("Solver"), // this is just a place holder
                    const_cast<char *>("--solverinfo"),
                    const_cast<char *>(driver_string.c_str()),
                    const_cast<char *>("--npx"),
                    const_cast<char *>(npx_string.c_str()),
                    const_cast<char *>("--npy"),
                    const_cast<char *>(npy_string.c_str()),
                    const_cast<char *>("--npz"),
                    const_cast<char *>(npz_string.c_str()),
                    const_cast<char *>("--nsz"),
                    const_cast<char *>(nsz_string.c_str()),
                    const_cast<char *>("--npt"),
                    const_cast<char *>(npt_string.c_str()),
                    const_cast<char *>("-f"),
                    const_cast<char *>("--use-opt-file"),
                    const_cast<char *>(optfilename.c_str()),
                    nullptr};

    size_t argc = useOptFile ? 16 : 14;

    // Get list of session file names.
    std::vector<std::string> sessionFileNames;
    for (auto &filename : m_session->GetFilenames())
    {
        // Remove optfile name, if necessary.
        if (filename.substr(filename.find_last_of(".") + 1) != "opt")
        {
            sessionFileNames.push_back(filename);
        }
    }

    // Set session for coarse solver.
    for (size_t timeLevel = 1; timeLevel < m_nTimeLevel; timeLevel++)
    {
        auto session = LibUtilities::SessionReader::CreateInstance(
            argc, argv, sessionFileNames, m_session->GetComm(), timeLevel);

        // Set graph for coarse solver.
        auto graph = SpatialDomains::MeshGraphIO::Read(
            session, LibUtilities::NullDomainRangeShPtr, true, m_graph);

        // Set BndRegionOrdering (necessary for DG with periodic BC) FIXME
        graph->SetBndRegionOrdering(m_graph->GetBndRegionOrdering());

        // Set CompositeOrdering (necessary for DG with periodic BC) FIXME
        graph->SetCompositeOrdering(m_graph->GetCompositeOrdering());

        // Retrieve the equation system to solve.
        ASSERTL0(session->DefinesSolverInfo("EqType"),
                 "EqType SolverInfo tag must be defined.");
        auto vEquation = session->DefinesSolverInfo("SolverType")
                             ? session->GetSolverInfo("SolverType")
                             : session->GetSolverInfo("EqType");

        // Check such a module exists for this equation.
        ASSERTL0(
            GetEquationSystemFactory().ModuleExists(vEquation),
            "EquationSystem '" + vEquation +
                "' is not defined.\n"
                "Ensure equation name is correct and module is compiled.\n");

        // Set coarse parallel-in-time solver.
        session->SetTag("AdvectiveType", AdvectiveType);
        session->SetTag("ParallelInTimeSolver",
                        "TimeLevel" + std::to_string(timeLevel));
        m_equ[timeLevel] = GetEquationSystemFactory().CreateInstance(
            vEquation, session, graph);
    }
}

/**
 *
 */
void DriverParallelInTime::GetParametersFromSession(void)
{
    // Parallel-in-Time iteration parameters.
    m_tolerPIT      = m_session->DefinesParameter("PITToler")
                          ? m_session->GetParameter("PITToler")
                          : 1.0E-16;
    m_iterMaxPIT    = m_session->DefinesParameter("PITIterMax")
                          ? m_session->GetParameter("PITIterMax")
                          : m_numChunks;
    m_numWindowsPIT = m_session->DefinesParameter("NumWindows")
                          ? m_session->GetParameter("NumWindows")
                          : 1;

    // Other parameters.
    m_exactSolution = m_session->DefinesParameter("ExactSolution")
                          ? m_session->GetParameter("ExactSolution")
                          : 0;
}

/**
 *
 */
void DriverParallelInTime::InitialiseEqSystem(bool turnoff_output)
{
    m_nVar = m_EqSys[0]->GetNvariables();

    // Initialize fine solver.
    if (turnoff_output)
    {
        for (size_t timeLevel = 0; timeLevel < m_nTimeLevel; timeLevel++)
        {
            m_EqSys[timeLevel]->SetInfoSteps(0);
            m_EqSys[timeLevel]->SetCheckpointSteps(0);
        }
    }
    m_EqSys[0]->DoInitialise(true);

    // Initialize coarse solver(s).
    for (size_t timeLevel = 1; timeLevel < m_nTimeLevel; timeLevel++)
    {
        m_EqSys[timeLevel]->DoInitialise(false);
    }

    // Initialize time stepping parameters.
    m_timestep = Array<OneD, NekDouble>(m_nTimeLevel);
    m_nsteps   = Array<OneD, size_t>(m_nTimeLevel);
    m_npts     = Array<OneD, size_t>(m_nTimeLevel);
    for (size_t timeLevel = 0; timeLevel < m_nTimeLevel; timeLevel++)
    {
        m_timestep[timeLevel] =
            m_EqSys[timeLevel]->EquationSystem::GetTimeStep();
        m_nsteps[timeLevel] = m_EqSys[timeLevel]->EquationSystem::GetSteps();
        m_npts[timeLevel]   = m_EqSys[timeLevel]->EquationSystem::GetNpoints();
    }

    // Initialize errors.
    m_exactsoln = Array<OneD, Array<OneD, NekDouble>>(m_nVar);
    for (size_t i = 0; i < m_nVar; ++i)
    {
        m_exactsoln[i] = Array<OneD, NekDouble>(m_npts[0], 0.0);
    }
    m_vL2Errors   = Array<OneD, NekDouble>(m_nVar, 0.0);
    m_vLinfErrors = Array<OneD, NekDouble>(m_nVar, 0.0);
}

/**
 *
 */
void DriverParallelInTime::PrintSolverInfo(std::ostream &out)
{
    if (m_chunkRank == 0 && m_comm->GetSpaceComm()->GetRank() == 0)
    {
        for (size_t timeLevel = 0; timeLevel < m_nTimeLevel; timeLevel++)
        {
            std::cout << std::string(71, '=') << std::endl << std::flush;
            std::cout << "=========================== TIME LEVEL " +
                             std::to_string(timeLevel) +
                             " INFO "
                             "========================="
                      << std::endl
                      << std::flush;

            m_EqSys[timeLevel]->PrintSummary(out);

            std::cout << std::endl << std::flush;
        }
    }
}

/**
 *
 */
void DriverParallelInTime::PrintHeader(const std::string &title, const char c)
{
    if (m_chunkRank == m_numChunks - 1 &&
        m_comm->GetSpaceComm()->GetRank() == 0)
    {
        std::cout << std::endl;
        std::cout << std::string(43, c) << std::endl << std::flush;
        std::cout << title << std::endl << std::flush;
        std::cout << std::string(43, c) << std::endl << std::flush;
    }
}

/**
 *
 */
void DriverParallelInTime::RecvFromPreviousProc(
    Array<OneD, Array<OneD, NekDouble>> &array, int &convergence)
{
    if (m_chunkRank > 0)
    {
        if (!convergence)
        {
            m_comm->GetTimeComm()->Recv(m_chunkRank - 1, convergence);
            for (size_t i = 0; i < m_nVar; ++i)
            {
                m_comm->GetTimeComm()->Recv(m_chunkRank - 1, array[i]);
            }
        }
    }
}

/**
 *
 */
void DriverParallelInTime::RecvFromPreviousProc(Array<OneD, NekDouble> &array)
{
    if (m_chunkRank > 0)
    {
        m_comm->GetTimeComm()->Recv(m_chunkRank - 1, array);
    }
}

/**
 *
 */
void DriverParallelInTime::SendToNextProc(
    Array<OneD, Array<OneD, NekDouble>> &array, int &convergence)
{
    if (m_chunkRank < m_numChunks - 1)
    {
        m_comm->GetTimeComm()->Send(m_chunkRank + 1, convergence);
        for (size_t i = 0; i < m_nVar; ++i)
        {
            m_comm->GetTimeComm()->Send(m_chunkRank + 1, array[i]);
        }
    }
}

/**
 *
 */
void DriverParallelInTime::SendToNextProc(Array<OneD, NekDouble> &array)
{
    if (m_chunkRank < m_numChunks - 1)
    {
        m_comm->GetTimeComm()->Send(m_chunkRank + 1, array);
    }
}

/**
 *
 */
void DriverParallelInTime::CopySolutionVector(
    const Array<OneD, const Array<OneD, NekDouble>> &in,
    Array<OneD, Array<OneD, NekDouble>> &out)
{
    for (size_t i = 0; i < m_nVar; ++i)
    {
        Vmath::Vcopy(in[i].size(), in[i], 1, out[i], 1);
    }
}

/**
 *
 */
void DriverParallelInTime::CopyFromPhysField(
    const size_t timeLevel, Array<OneD, Array<OneD, NekDouble>> &out)
{
    for (size_t i = 0; i < m_nVar; ++i)
    {
        m_EqSys[timeLevel]->CopyFromPhysField(i, out[i]);
    }
}

/**
 *
 */
void DriverParallelInTime::CopyToPhysField(
    const size_t timeLevel, const Array<OneD, const Array<OneD, NekDouble>> &in)
{
    for (size_t i = 0; i < m_nVar; ++i)
    {
        m_EqSys[timeLevel]->CopyToPhysField(i, in[i]);
        m_EqSys[timeLevel]->UpdateFields()[i]->SetPhysState(true);
    }
}

/**
 *
 */
void DriverParallelInTime::UpdateFieldCoeffs(
    const size_t timeLevel, const Array<OneD, const Array<OneD, NekDouble>> &in)
{
    for (size_t i = 0; i < m_nVar; ++i)
    {
        if (in != NullNekDoubleArrayOfArray)
        {
            m_EqSys[timeLevel]->CopyToPhysField(i, in[i]);
        }
        m_EqSys[timeLevel]->UpdateFields()[i]->FwdTrans(
            m_EqSys[timeLevel]->UpdateFields()[i]->GetPhys(),
            m_EqSys[timeLevel]->UpdateFields()[i]->UpdateCoeffs());
    }
}

/**
 *
 */
void DriverParallelInTime::EvaluateExactSolution(const size_t timeLevel,
                                                 const NekDouble &time)
{
    for (size_t i = 0; i < m_nVar; ++i)
    {
        m_EqSys[timeLevel]->EvaluateExactSolution(i, m_exactsoln[i], time);
    }
}

/**
 *
 */
void DriverParallelInTime::SolutionConvergenceMonitoring(const size_t timeLevel,
                                                         const size_t iter)
{
    PrintHeader((boost::format("ITERATION %1%") % iter).str(), '-');
    UpdateErrorNorm(timeLevel, true);
    PrintErrorNorm(timeLevel, true);
}

/**
 *
 */
void DriverParallelInTime::SolutionConvergenceSummary(const size_t timeLevel)
{
    UpdateErrorNorm(timeLevel, false);
    PrintErrorNorm(timeLevel, false);
}

/**
 *
 */
void DriverParallelInTime::UpdateErrorNorm(const size_t timeLevel,
                                           const bool normalized)
{
    for (size_t i = 0; i < m_nVar; ++i)
    {
        m_vL2Errors[i] =
            m_EqSys[timeLevel]->L2Error(i, m_exactsoln[i], normalized);
        m_vLinfErrors[i] = m_EqSys[timeLevel]->LinfError(i, m_exactsoln[i]);
    }
}

/**
 *
 */
void DriverParallelInTime::PrintErrorNorm(const size_t timeLevel,
                                          const bool normalized)
{
    if (m_chunkRank == m_numChunks - 1 &&
        m_comm->GetSpaceComm()->GetRank() == 0)
    {
        for (size_t i = 0; i < m_nVar; ++i)
        {
            if (normalized)
            {
                std::cout << "L2 error (variable "
                          << m_EqSys[timeLevel]->GetVariable(i)
                          << ") : " << m_vL2Errors[i] << std::endl
                          << std::flush;
                std::cout << "Linf error (variable "
                          << m_EqSys[timeLevel]->GetVariable(i)
                          << ") : " << m_vLinfErrors[i] << std::endl
                          << std::flush;
            }
            else
            {
                std::cout << "L 2 error (variable "
                          << m_EqSys[timeLevel]->GetVariable(i)
                          << ") : " << m_vL2Errors[i] << std::endl
                          << std::flush;
                std::cout << "L inf error (variable "
                          << m_EqSys[timeLevel]->GetVariable(i)
                          << ") : " << m_vLinfErrors[i] << std::endl
                          << std::flush;
            }
        }
    }
}

/**
 *
 */
NekDouble DriverParallelInTime::vL2ErrorMax(void)
{
    NekDouble L2Error = 0.0;
    for (size_t i = 0; i < m_nVar; ++i)
    {
        L2Error = std::max(L2Error, m_vL2Errors[i]);
    }
    return L2Error;
}

/**
 *
 */
NekDouble DriverParallelInTime::EstimateCommunicationTime(
    Array<OneD, Array<OneD, NekDouble>> &buffer1,
    Array<OneD, Array<OneD, NekDouble>> &buffer2)
{
    if (m_numChunks == 1)
    {
        return 0.0;
    }
    else
    {
        // Average communication time over niter iteration.
        size_t niter = 20;
        Nektar::LibUtilities::Timer timer;
        for (size_t n = 0; n <= niter; n++)
        {
            if (n == 1)
            {
                timer.Start(); // Ignore the first iteration
            }

            if (m_chunkRank == 0)
            {
                for (size_t i = 0; i < buffer1.size(); ++i)
                {
                    m_comm->GetTimeComm()->Send(m_numChunks - 1, buffer1[i]);
                }
            }

            if (m_chunkRank == m_numChunks - 1)
            {
                for (size_t i = 0; i < buffer2.size(); ++i)
                {
                    m_comm->GetTimeComm()->Recv(0, buffer2[i]);
                }
            }
        }
        timer.Stop();
        return timer.Elapsed().count() / niter;
    }
}

/**
 *
 */
void DriverParallelInTime::Interpolate(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &infield,
    const Array<OneD, MultiRegions::ExpListSharedPtr> &outfield,
    const Array<OneD, Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray)
{
    if (infield.size() != outfield.size())
    {
        NEKERROR(ErrorUtil::efatal, "not the same number of variables")
    }

    for (size_t n = 0; n < infield.size(); ++n)
    {
        // Interpolation from infield -> outfield assuming that infield and
        // outfield are the same explists, but at potentially different
        // polynomial orders.
        if (infield[n]->GetExpSize() != outfield[n]->GetExpSize())
        {
            NEKERROR(ErrorUtil::efatal, "not the same mesh")
        }

        // Assign input/output array.
        auto inphys  = (inarray == NullNekDoubleArrayOfArray)
                           ? infield[n]->UpdatePhys()
                           : inarray[n];
        auto outphys = (outarray == NullNekDoubleArrayOfArray)
                           ? outfield[n]->UpdatePhys()
                           : outarray[n];

        // If same polynomial orders, simply copy solution.
        if (infield[n]->GetTotPoints() == outfield[n]->GetTotPoints())
        {
            Vmath::Vcopy(infield[n]->GetTotPoints(), inphys, 1, outphys, 1);
        }
        // If different polynomial orders, interpolate solution.
        else
        {
            for (size_t i = 0; i < infield[n]->GetExpSize(); ++i)
            {
                // Get the elements.
                auto inElmt  = infield[n]->GetExp(i);
                auto outElmt = outfield[n]->GetExp(i);

                // Get the offset of elements in the storage arrays.
                size_t inoffset  = infield[n]->GetPhys_Offset(i);
                size_t outoffset = outfield[n]->GetPhys_Offset(i);

                // Transform solution from physical to coefficient space.
                Array<OneD, NekDouble> incoeff(inElmt->GetNcoeffs());
                inElmt->FwdTrans(inphys + inoffset, incoeff);

                // Interpolate elements.
                StdRegions::StdExpansionSharedPtr expPtr;
                if (inElmt->DetShapeType() == LibUtilities::Seg)
                {
                    expPtr = std::make_shared<StdRegions::StdSegExp>(
                        LibUtilities::BasisKey(
                            inElmt->GetBasis(0)->GetBasisType(),
                            inElmt->GetBasis(0)->GetNumModes(),
                            outElmt->GetBasis(0)->GetPointsKey()));
                }
                else if (inElmt->DetShapeType() == LibUtilities::Quad)
                {
                    expPtr = std::make_shared<StdRegions::StdQuadExp>(
                        LibUtilities::BasisKey(
                            inElmt->GetBasis(0)->GetBasisType(),
                            inElmt->GetBasis(0)->GetNumModes(),
                            outElmt->GetBasis(0)->GetPointsKey()),
                        LibUtilities::BasisKey(
                            inElmt->GetBasis(1)->GetBasisType(),
                            inElmt->GetBasis(1)->GetNumModes(),
                            outElmt->GetBasis(1)->GetPointsKey()));
                }
                else if (inElmt->DetShapeType() == LibUtilities::Tri)
                {
                    expPtr = std::make_shared<StdRegions::StdTriExp>(
                        LibUtilities::BasisKey(
                            inElmt->GetBasis(0)->GetBasisType(),
                            inElmt->GetBasis(0)->GetNumModes(),
                            outElmt->GetBasis(0)->GetPointsKey()),
                        LibUtilities::BasisKey(
                            inElmt->GetBasis(1)->GetBasisType(),
                            inElmt->GetBasis(1)->GetNumModes(),
                            outElmt->GetBasis(1)->GetPointsKey()));
                }
                else if (inElmt->DetShapeType() == LibUtilities::Hex)
                {
                    expPtr = std::make_shared<StdRegions::StdHexExp>(
                        LibUtilities::BasisKey(
                            inElmt->GetBasis(0)->GetBasisType(),
                            inElmt->GetBasis(0)->GetNumModes(),
                            outElmt->GetBasis(0)->GetPointsKey()),
                        LibUtilities::BasisKey(
                            inElmt->GetBasis(1)->GetBasisType(),
                            inElmt->GetBasis(1)->GetNumModes(),
                            outElmt->GetBasis(1)->GetPointsKey()),
                        LibUtilities::BasisKey(
                            inElmt->GetBasis(2)->GetBasisType(),
                            inElmt->GetBasis(2)->GetNumModes(),
                            outElmt->GetBasis(2)->GetPointsKey()));
                }
                else if (inElmt->DetShapeType() == LibUtilities::Prism)
                {
                    expPtr = std::make_shared<StdRegions::StdPrismExp>(
                        LibUtilities::BasisKey(
                            inElmt->GetBasis(0)->GetBasisType(),
                            inElmt->GetBasis(0)->GetNumModes(),
                            outElmt->GetBasis(0)->GetPointsKey()),
                        LibUtilities::BasisKey(
                            inElmt->GetBasis(1)->GetBasisType(),
                            inElmt->GetBasis(1)->GetNumModes(),
                            outElmt->GetBasis(1)->GetPointsKey()),
                        LibUtilities::BasisKey(
                            inElmt->GetBasis(2)->GetBasisType(),
                            inElmt->GetBasis(2)->GetNumModes(),
                            outElmt->GetBasis(2)->GetPointsKey()));
                }
                else if (inElmt->DetShapeType() == LibUtilities::Pyr)
                {
                    expPtr = std::make_shared<StdRegions::StdPyrExp>(
                        LibUtilities::BasisKey(
                            inElmt->GetBasis(0)->GetBasisType(),
                            inElmt->GetBasis(0)->GetNumModes(),
                            outElmt->GetBasis(0)->GetPointsKey()),
                        LibUtilities::BasisKey(
                            inElmt->GetBasis(1)->GetBasisType(),
                            inElmt->GetBasis(1)->GetNumModes(),
                            outElmt->GetBasis(1)->GetPointsKey()),
                        LibUtilities::BasisKey(
                            inElmt->GetBasis(2)->GetBasisType(),
                            inElmt->GetBasis(2)->GetNumModes(),
                            outElmt->GetBasis(2)->GetPointsKey()));
                }
                else if (inElmt->DetShapeType() == LibUtilities::Tet)
                {
                    expPtr = std::make_shared<StdRegions::StdTetExp>(
                        LibUtilities::BasisKey(
                            inElmt->GetBasis(0)->GetBasisType(),
                            inElmt->GetBasis(0)->GetNumModes(),
                            outElmt->GetBasis(0)->GetPointsKey()),
                        LibUtilities::BasisKey(
                            inElmt->GetBasis(1)->GetBasisType(),
                            inElmt->GetBasis(1)->GetNumModes(),
                            outElmt->GetBasis(1)->GetPointsKey()),
                        LibUtilities::BasisKey(
                            inElmt->GetBasis(2)->GetBasisType(),
                            inElmt->GetBasis(2)->GetNumModes(),
                            outElmt->GetBasis(2)->GetPointsKey()));
                }

                // Transform solution from coefficient to physical space.
                Array<OneD, NekDouble> tmp = outphys + outoffset;
                expPtr->BwdTrans(incoeff, tmp);
            }
        }
    }
}

} // namespace Nektar::SolverUtils
