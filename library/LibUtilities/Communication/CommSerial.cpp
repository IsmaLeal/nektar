///////////////////////////////////////////////////////////////////////////////
//
// File: CommSerial.cpp
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
// Description: Serial (= no) communication implementation
//
///////////////////////////////////////////////////////////////////////////////

#ifdef NEKTAR_USING_PETSC
#include "petscsys.h"
#endif

#include <LibUtilities/Communication/CommSerial.h>

namespace Nektar::LibUtilities
{
std::string CommSerial::className = GetCommFactory().RegisterCreatorFunction(
    "Serial", CommSerial::create, "Single-process serial communication.");

CommSerial::CommSerial(int argc, char *argv[]) : Comm(argc, argv)
{
#ifdef NEKTAR_USING_PETSC
    PetscInitializeNoArguments();
#endif
    m_size = 1;
    m_type = "Serial";
}

CommSerial::~CommSerial() = default;

/**
 *
 */
void CommSerial::v_Finalise()
{
#ifdef NEKTAR_USING_PETSC
    PetscFinalize();
#endif
}

/**
 *
 */
int CommSerial::v_GetRank()
{
    return 0;
}

/**
 *
 */
bool CommSerial::v_TreatAsRankZero()
{
    return true;
}

/**
 *
 */
bool CommSerial::v_IsSerial()
{
    return true;
}

/**
 *
 */
std::tuple<int, int, int> CommSerial::v_GetVersion()
{
    return std::make_tuple(0, 0, 0);
}

/**
 *
 */
void CommSerial::v_Block()
{
}

/**
 *
 */
NekDouble CommSerial::v_Wtime()
{
    return 0;
}

/**
 *
 */
void CommSerial::v_Send([[maybe_unused]] const void *buf,
                        [[maybe_unused]] int count,
                        [[maybe_unused]] CommDataType dt,
                        [[maybe_unused]] int dest)
{
}

/**
 *
 */
void CommSerial::v_Recv([[maybe_unused]] void *buf, [[maybe_unused]] int count,
                        [[maybe_unused]] CommDataType dt,
                        [[maybe_unused]] int source)
{
}

/**
 *
 */
void CommSerial::v_SendRecv(
    [[maybe_unused]] const void *sendbuf, [[maybe_unused]] int sendcount,
    [[maybe_unused]] CommDataType sendtype, [[maybe_unused]] int dest,
    [[maybe_unused]] void *recvbuf, [[maybe_unused]] int recvcount,
    [[maybe_unused]] CommDataType recvtype, [[maybe_unused]] int source)
{
}

/**
 *
 */
void CommSerial::v_AllReduce([[maybe_unused]] void *buf,
                             [[maybe_unused]] int count,
                             [[maybe_unused]] CommDataType dt,
                             [[maybe_unused]] enum ReduceOperator pOp)
{
}

/**
 *
 */
void CommSerial::v_AlltoAll([[maybe_unused]] const void *sendbuf,
                            [[maybe_unused]] int sendcount,
                            [[maybe_unused]] CommDataType sendtype,
                            [[maybe_unused]] void *recvbuf,
                            [[maybe_unused]] int recvcount,
                            [[maybe_unused]] CommDataType recvtype)
{
}

/**
 *
 */
void CommSerial::v_AlltoAllv([[maybe_unused]] const void *sendbuf,
                             [[maybe_unused]] const int *sendcounts,
                             [[maybe_unused]] const int *senddispls,
                             [[maybe_unused]] CommDataType sendtype,
                             [[maybe_unused]] void *recvbuf,
                             [[maybe_unused]] const int *recvcounts,
                             [[maybe_unused]] const int *recvdispls,
                             [[maybe_unused]] CommDataType recvtype)
{
}

/**
 *
 */
void CommSerial::v_AllGather([[maybe_unused]] const void *sendbuf,
                             [[maybe_unused]] int sendcount,
                             [[maybe_unused]] CommDataType sendtype,
                             [[maybe_unused]] void *recvbuf,
                             [[maybe_unused]] int recvcount,
                             [[maybe_unused]] CommDataType recvtype)
{
}

/**
 *
 */
void CommSerial::v_AllGatherv([[maybe_unused]] const void *sendbuf,
                              [[maybe_unused]] int sendcount,
                              [[maybe_unused]] CommDataType sendtype,
                              [[maybe_unused]] void *recvbuf,
                              [[maybe_unused]] const int *recvcounts,
                              [[maybe_unused]] const int *recvdispls,
                              [[maybe_unused]] CommDataType recvtype)
{
}

/**
 *
 */
void CommSerial::v_AllGatherv([[maybe_unused]] void *recvbuf,
                              [[maybe_unused]] const int *recvcounts,
                              [[maybe_unused]] const int *recvdispls,
                              [[maybe_unused]] CommDataType recvtype)
{
}

/**
 *
 */
void CommSerial::v_Bcast([[maybe_unused]] void *buffer,
                         [[maybe_unused]] int count,
                         [[maybe_unused]] CommDataType dt,
                         [[maybe_unused]] int root)
{
}

/**
 *
 */
void CommSerial::v_Gather(const void *sendbuf, int sendcount,
                          CommDataType sendtype, void *recvbuf,
                          [[maybe_unused]] int recvcount,
                          [[maybe_unused]] CommDataType recvtype,
                          [[maybe_unused]] int root)
{
    std::memcpy(recvbuf, sendbuf, sendcount * CommDataTypeGetSize(sendtype));
}

/**
 *
 */
void CommSerial::v_Scatter(const void *sendbuf, int sendcount,
                           CommDataType sendtype, void *recvbuf,
                           [[maybe_unused]] int recvcount,
                           [[maybe_unused]] CommDataType recvtype,
                           [[maybe_unused]] int root)
{
    std::memcpy(recvbuf, sendbuf, sendcount * CommDataTypeGetSize(sendtype));
}

/**
 *
 */
void CommSerial::v_DistGraphCreateAdjacent(
    [[maybe_unused]] int indegree, [[maybe_unused]] const int *sources,
    [[maybe_unused]] const int *sourceweights, [[maybe_unused]] int reorder)
{
}

/**
 *
 */
void CommSerial::v_NeighborAlltoAllv([[maybe_unused]] const void *sendbuf,
                                     [[maybe_unused]] const int *sendcounts,
                                     [[maybe_unused]] const int *senddispls,
                                     [[maybe_unused]] CommDataType sendtype,
                                     [[maybe_unused]] void *recvbuf,
                                     [[maybe_unused]] const int *recvcounts,
                                     [[maybe_unused]] const int *recvdispls,
                                     [[maybe_unused]] CommDataType recvtype)
{
}

/**
 *
 */
void CommSerial::v_Irsend([[maybe_unused]] const void *buf,
                          [[maybe_unused]] int count,
                          [[maybe_unused]] CommDataType dt,
                          [[maybe_unused]] int dest,
                          [[maybe_unused]] CommRequestSharedPtr request,
                          [[maybe_unused]] int loc)
{
}

/**
 *
 */
void CommSerial::v_Isend([[maybe_unused]] const void *buf,
                         [[maybe_unused]] int count,
                         [[maybe_unused]] CommDataType dt,
                         [[maybe_unused]] int dest,
                         [[maybe_unused]] CommRequestSharedPtr request,
                         [[maybe_unused]] int loc)
{
}

/**
 *
 */
void CommSerial::v_SendInit([[maybe_unused]] const void *buf,
                            [[maybe_unused]] int count,
                            [[maybe_unused]] CommDataType dt,
                            [[maybe_unused]] int dest,
                            [[maybe_unused]] CommRequestSharedPtr request,
                            [[maybe_unused]] int loc)
{
}

/**
 *
 */
void CommSerial::v_Irecv([[maybe_unused]] void *buf, [[maybe_unused]] int count,
                         [[maybe_unused]] CommDataType dt,
                         [[maybe_unused]] int source,
                         [[maybe_unused]] CommRequestSharedPtr request,
                         [[maybe_unused]] int loc)
{
}

/**
 *
 */
void CommSerial::v_RecvInit([[maybe_unused]] void *buf,
                            [[maybe_unused]] int count,
                            [[maybe_unused]] CommDataType dt,
                            [[maybe_unused]] int source,
                            [[maybe_unused]] CommRequestSharedPtr request,
                            [[maybe_unused]] int loc)
{
}

/**
 *
 */
void CommSerial::v_StartAll([[maybe_unused]] CommRequestSharedPtr request)
{
}

/**
 *
 */
void CommSerial::v_WaitAll([[maybe_unused]] CommRequestSharedPtr request)
{
}

/**
 *
 */
CommRequestSharedPtr CommSerial::v_CreateRequest([[maybe_unused]] int num)
{
    return std::shared_ptr<CommRequest>(new CommRequest);
}

/**
 *
 */
void CommSerial::v_SplitComm([[maybe_unused]] int pRows,
                             [[maybe_unused]] int pColumns,
                             [[maybe_unused]] int pTime)
{
    ASSERTL0(false, "Cannot split a serial process.");
}

/**
 *
 */
CommSharedPtr CommSerial::v_CommCreateIf(int flag)
{
    if (flag == 0)
    {
        // flag == 0 => get back MPI_COMM_NULL, return a null ptr instead.
        return std::shared_ptr<Comm>();
    }
    else
    {
        // Return a real communicator
        return shared_from_this();
    }
}

} // namespace Nektar::LibUtilities
