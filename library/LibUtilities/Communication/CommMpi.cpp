///////////////////////////////////////////////////////////////////////////////
//
// File: CommMpi.cpp
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
// Description: MPI communication implementation
//
///////////////////////////////////////////////////////////////////////////////

#ifdef NEKTAR_USING_PETSC
#include "petscsys.h"
#endif

#include <LibUtilities/Communication/CommMpi.h>

namespace Nektar::LibUtilities
{

std::string CommMpi::className = GetCommFactory().RegisterCreatorFunction(
    "ParallelMPI", CommMpi::create, "Parallel communication using MPI.");

/**
 *
 */
CommMpi::CommMpi(int narg, char *arg[]) : Comm(narg, arg)
{
    int init = 0;
    MPI_Initialized(&init);

    if (!init)
    {
        int thread_support = 0;
        if (MPI_Init_thread(&narg, &arg, MPI_THREAD_MULTIPLE,
                            &thread_support) != MPI_SUCCESS)
        {
            NEKERROR(
                ErrorUtil::ewarning,
                "Initializing MPI using MPI_Init, if scotch version > 6 and is "
                "compiled with multi-threading, it might cause deadlocks.")
            ASSERTL0(MPI_Init(&narg, &arg) == MPI_SUCCESS,
                     "Failed to initialise MPI");
        }
        // store bool to indicate that Nektar++ is in charge of finalizing MPI.
        m_controls_mpi = true;
    }
    else
    {
        // Another code is in charge of finalizing MPI and this is not the
        // responsiblity of Nektar++
        m_controls_mpi = false;
    }

    m_comm = MPI_COMM_WORLD;
    MPI_Comm_size(m_comm, &m_size);
    MPI_Comm_rank(m_comm, &m_rank);

#ifdef NEKTAR_USING_PETSC
    PetscInitializeNoArguments();
#endif

    m_type = "Parallel MPI";
}

/**
 *
 */
CommMpi::CommMpi(MPI_Comm pComm) : Comm()
{
    m_comm = pComm;
    MPI_Comm_size(m_comm, &m_size);
    MPI_Comm_rank(m_comm, &m_rank);

    m_type = "Parallel MPI";
}

/**
 *
 */
CommMpi::~CommMpi()
{
    int flag;
    MPI_Finalized(&flag);
    if (!flag && m_comm != MPI_COMM_WORLD)
    {
        MPI_Comm_free(&m_comm);
    }
}

/**
 *
 */
MPI_Comm CommMpi::GetComm()
{
    return m_comm;
}

/**
 *
 */
void CommMpi::v_Finalise()
{
#ifdef NEKTAR_USING_PETSC
    PetscFinalize();
#endif
    int flag;
    MPI_Finalized(&flag);
    if ((!flag) && m_controls_mpi)
    {
        MPI_Finalize();
    }
}

/**
 *
 */
int CommMpi::v_GetRank()
{
    return m_rank;
}

/**
 *
 */
bool CommMpi::v_TreatAsRankZero()
{
    return m_rank == 0;
}

/**
 *
 */
bool CommMpi::v_IsSerial()
{
    return m_size == 1;
}

/**
 *
 */
std::tuple<int, int, int> CommMpi::v_GetVersion()
{
    int version, subversion;
    int retval = MPI_Get_version(&version, &subversion);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing GetVersion.");

    return std::make_tuple(version, subversion, 0);
}

/**
 *
 */
void CommMpi::v_Block()
{
    MPI_Barrier(m_comm);
}

/**
 *
 */
double CommMpi::v_Wtime()
{
    return MPI_Wtime();
}

/**
 *
 */
void CommMpi::v_Send(const void *buf, int count, CommDataType dt, int dest)
{
    if (MPISYNC)
    {
        MPI_Ssend(buf, count, dt, dest, 0, m_comm);
    }
    else
    {
        MPI_Send(buf, count, dt, dest, 0, m_comm);
    }
}

/**
 *
 */
void CommMpi::v_Recv(void *buf, int count, CommDataType dt, int source)
{
    MPI_Recv(buf, count, dt, source, 0, m_comm, MPI_STATUS_IGNORE);
}

/**
 *
 */
void CommMpi::v_SendRecv(const void *sendbuf, int sendcount,
                         CommDataType sendtype, int dest, void *recvbuf,
                         int recvcount, CommDataType recvtype, int source)
{
    MPI_Status status;
    int retval = MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, 0, recvbuf,
                              recvcount, recvtype, source, 0, m_comm, &status);

    ASSERTL0(retval == MPI_SUCCESS,
             "MPI error performing send-receive of data.");
}

/**
 *
 */
void CommMpi::v_AllReduce(void *buf, int count, CommDataType dt,
                          enum ReduceOperator pOp)
{
    if (GetSize() == 1)
    {
        return;
    }

    MPI_Op vOp;
    switch (pOp)
    {
        case ReduceMax:
            vOp = MPI_MAX;
            break;
        case ReduceMin:
            vOp = MPI_MIN;
            break;
        case ReduceSum:
        default:
            vOp = MPI_SUM;
            break;
    }
    int retval = MPI_Allreduce(MPI_IN_PLACE, buf, count, dt, vOp, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing All-reduce.");
}

/**
 *
 */
void CommMpi::v_AlltoAll(const void *sendbuf, int sendcount,
                         CommDataType sendtype, void *recvbuf, int recvcount,
                         CommDataType recvtype)
{
    int retval = MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                              recvtype, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing All-to-All.");
}

/**
 *
 */
void CommMpi::v_AlltoAllv(const void *sendbuf, const int *sendcounts,
                          const int *senddispls, CommDataType sendtype,
                          void *recvbuf, const int *recvcounts,
                          const int *recvdispls, CommDataType recvtype)
{
    int retval =
        MPI_Alltoallv(sendbuf, sendcounts, senddispls, sendtype, recvbuf,
                      recvcounts, recvdispls, recvtype, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing All-to-All-v.");
}

/**
 *
 */
void CommMpi::v_AllGather(const void *sendbuf, int sendcount,
                          CommDataType sendtype, void *recvbuf, int recvcount,
                          CommDataType recvtype)
{
    int retval = MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                               recvtype, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing Allgather.");
}

/**
 *
 */
void CommMpi::v_AllGatherv(const void *sendbuf, int sendcount,
                           CommDataType sendtype, void *recvbuf,
                           const int *recvcounts, const int *recvdispls,
                           CommDataType recvtype)
{
    int retval = MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf,
                                recvcounts, recvdispls, recvtype, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing Allgatherv.");
}

/**
 *
 */
void CommMpi::v_AllGatherv(void *recvbuf, const int *recvcounts,
                           const int *recvdispls, CommDataType recvtype)
{
    int retval = MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvbuf,
                                recvcounts, recvdispls, recvtype, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing Allgatherv.");
}

/**
 *
 */
void CommMpi::v_Bcast(void *buffer, int count, CommDataType dt, int root)
{
    int retval = MPI_Bcast(buffer, count, dt, root, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing Bcast-v.");
}

/**
 *
 */
void CommMpi::v_Gather(const void *sendbuf, int sendcount,
                       CommDataType sendtype, void *recvbuf, int recvcount,
                       CommDataType recvtype, int root)
{
    int retval = MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                            recvtype, root, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing Gather.");
}

/**
 *
 */
void CommMpi::v_Scatter(const void *sendbuf, int sendcount,
                        CommDataType sendtype, void *recvbuf, int recvcount,
                        CommDataType recvtype, int root)
{
    int retval = MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                             recvtype, root, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing Scatter.");
}

/**
 *
 */
void CommMpi::v_DistGraphCreateAdjacent(
    [[maybe_unused]] int indegree, [[maybe_unused]] const int *sources,
    [[maybe_unused]] const int *sourceweights, [[maybe_unused]] int reorder)
{
#if MPI_VERSION < 3
    ASSERTL0(false, "MPI_Dist_graph_create_adjacent is not supported in your "
                    "installed MPI version.");
#else
    int retval = MPI_Dist_graph_create_adjacent(
        m_comm, indegree, sources, sourceweights, indegree, sources,
        sourceweights, MPI_INFO_NULL, reorder, &m_comm);

    ASSERTL0(retval == MPI_SUCCESS,
             "MPI error performing Dist_graph_create_adjacent.")
#endif
}

/**
 *
 */
void CommMpi::v_NeighborAlltoAllv([[maybe_unused]] const void *sendbuf,
                                  [[maybe_unused]] const int *sendcounts,
                                  [[maybe_unused]] const int *senddispls,
                                  [[maybe_unused]] CommDataType sendtype,
                                  [[maybe_unused]] void *recvbuf,
                                  [[maybe_unused]] const int *recvcounts,
                                  [[maybe_unused]] const int *recvdispls,
                                  [[maybe_unused]] CommDataType recvtype)
{
#if MPI_VERSION < 3
    ASSERTL0(false, "MPI_Neighbor_alltoallv is not supported in your "
                    "installed MPI version.");
#else
    int retval = MPI_Neighbor_alltoallv(sendbuf, sendcounts, senddispls,
                                        sendtype, recvbuf, recvcounts,
                                        recvdispls, recvtype, m_comm);

    ASSERTL0(retval == MPI_SUCCESS, "MPI error performing NeighborAllToAllV.");
#endif
}

/**
 *
 */
void CommMpi::v_Irsend(const void *buf, int count, CommDataType dt, int dest,
                       CommRequestSharedPtr request, int loc)
{
    CommRequestMpiSharedPtr req =
        std::static_pointer_cast<CommRequestMpi>(request);
    MPI_Irsend(buf, count, dt, dest, 0, m_comm, req->GetRequest(loc));
}

/**
 *
 */
void CommMpi::v_Isend(const void *buf, int count, CommDataType dt, int dest,
                      CommRequestSharedPtr request, int loc)
{
    CommRequestMpiSharedPtr req =
        std::static_pointer_cast<CommRequestMpi>(request);
    MPI_Isend(buf, count, dt, dest, 0, m_comm, req->GetRequest(loc));
}

/**
 *
 */
void CommMpi::v_SendInit(const void *buf, int count, CommDataType dt, int dest,
                         CommRequestSharedPtr request, int loc)
{
    CommRequestMpiSharedPtr req =
        std::static_pointer_cast<CommRequestMpi>(request);
    MPI_Send_init(buf, count, dt, dest, 0, m_comm, req->GetRequest(loc));
}

/**
 *
 */
void CommMpi::v_Irecv(void *buf, int count, CommDataType dt, int source,
                      CommRequestSharedPtr request, int loc)
{
    CommRequestMpiSharedPtr req =
        std::static_pointer_cast<CommRequestMpi>(request);
    MPI_Irecv(buf, count, dt, source, 0, m_comm, req->GetRequest(loc));
}

/**
 *
 */
void CommMpi::v_RecvInit(void *buf, int count, CommDataType dt, int source,
                         CommRequestSharedPtr request, int loc)
{
    CommRequestMpiSharedPtr req =
        std::static_pointer_cast<CommRequestMpi>(request);
    MPI_Recv_init(buf, count, dt, source, 0, m_comm, req->GetRequest(loc));
}

/**
 *
 */
void CommMpi::v_StartAll(CommRequestSharedPtr request)
{
    CommRequestMpiSharedPtr req =
        std::static_pointer_cast<CommRequestMpi>(request);
    if (req->GetNumRequest() != 0)
    {
        MPI_Startall(req->GetNumRequest(), req->GetRequest(0));
    }
}

/**
 *
 */
void CommMpi::v_WaitAll(CommRequestSharedPtr request)
{
    CommRequestMpiSharedPtr req =
        std::static_pointer_cast<CommRequestMpi>(request);
    if (req->GetNumRequest() != 0)
    {
        MPI_Waitall(req->GetNumRequest(), req->GetRequest(0),
                    MPI_STATUSES_IGNORE);
    }
}

/**
 *
 */
CommRequestSharedPtr CommMpi::v_CreateRequest(int num)
{
    return std::shared_ptr<CommRequest>(new CommRequestMpi(num));
}

/**
 * Processes are considered as a grid of size pRows*pColumns. Comm
 * objects are created corresponding to the rows and columns of this
 * grid. The row and column to which this process belongs is stored in
 * #m_commRow and #m_commColumn.
 */
void CommMpi::v_SplitComm(int pRows, int pColumns, int pTime)
{
    if (pTime == 0)
    {
        ASSERTL0(pRows * pColumns == m_size,
                 "Rows/Columns do not match comm size.");

        MPI_Comm newComm;

        // Compute row and column in grid.
        int myCol = m_rank % pColumns;
        int myRow = (m_rank - myCol) / pColumns;

        // Split Comm into rows - all processes with same myRow are put in
        // the same communicator. The rank within this communicator is the
        // column index.
        MPI_Comm_split(m_comm, myRow, myCol, &newComm);
        m_commRow = std::shared_ptr<Comm>(new CommMpi(newComm));

        // Split Comm into columns - all processes with same myCol are put
        // in the same communicator. The rank within this communicator is
        // the row index.
        MPI_Comm_split(m_comm, myCol, myRow, &newComm);
        m_commColumn = std::shared_ptr<Comm>(new CommMpi(newComm));
    }
    else
    {
        ASSERTL0(pRows * pColumns * pTime == m_size,
                 "Rows/Columns/Time do not match comm size.");

        MPI_Comm newComm;

        // Compute row and column in grid.
        int mySpace = m_rank % (pRows * pColumns);
        int myTime  = (m_rank - mySpace) / (pRows * pColumns);
        int myCol   = mySpace % pColumns;
        int myRow   = (mySpace - myCol) / pColumns;

        // Split Comm - all processes with same mySpace are put in
        // the same communicator. The rank within this communicator is the
        // time index.
        MPI_Comm_split(m_comm, mySpace, myTime, &newComm);
        m_commTime = std::shared_ptr<Comm>(new CommMpi(newComm));

        // Split Comm - all processes with same myTime are put in
        // the same communicator. The rank within this communicator is the
        // spatial index.
        MPI_Comm_split(m_comm, myTime, mySpace, &newComm);
        m_commSpace = std::shared_ptr<Comm>(new CommMpi(newComm));

        // Split Comm into rows - all processes with same myRow are put in
        // the same communicator. The rank within this communicator is the
        // column index.
        MPI_Comm_split(m_comm, myRow + pRows * pColumns * myTime, myCol,
                       &newComm);
        m_commRow = std::shared_ptr<Comm>(new CommMpi(newComm));

        // Split Comm into columns - all processes with same myCol are put
        // in the same communicator. The rank within this communicator is
        // the row index.
        MPI_Comm_split(m_comm, myCol + pRows * pColumns * myTime, myRow,
                       &newComm);
        m_commColumn = std::shared_ptr<Comm>(new CommMpi(newComm));
    }
}

/**
 * Create a new communicator if the flag is non-zero.
 */
CommSharedPtr CommMpi::v_CommCreateIf(int flag)
{
    MPI_Comm newComm;
    // color == MPI_UNDEF => not in the new communicator
    // key == 0 on all => use rank to order them. OpenMPI, at least,
    // implies this is faster than ordering them ourselves.
    MPI_Comm_split(m_comm, flag ? flag : MPI_UNDEFINED, 0, &newComm);

    if (flag == 0)
    {
        // flag == 0 => get back MPI_COMM_NULL, return a null ptr instead.
        return std::shared_ptr<Comm>();
    }
    else
    {
        // Return a real communicator
        return std::shared_ptr<Comm>(new CommMpi(newComm));
    }
}

/**
 *
 */
std::pair<CommSharedPtr, CommSharedPtr> CommMpi::v_SplitCommNode()
{
    std::pair<CommSharedPtr, CommSharedPtr> ret;

#if MPI_VERSION < 3
    ASSERTL0(false, "Not implemented for non-MPI-3 versions.");
#else
    // Create an intra-node communicator.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, m_rank,
                        MPI_INFO_NULL, &nodeComm);

    // For rank 0 of the intra-node communicator, split the main
    // communicator. Everyone else will get a null communicator.
    ret.first  = std::shared_ptr<Comm>(new CommMpi(nodeComm));
    ret.second = CommMpi::v_CommCreateIf(ret.first->GetRank() == 0);
    if (ret.first->GetRank() == 0)
    {
        ret.second->SplitComm(1, ret.second->GetSize());
    }
#endif

    return ret;
}

} // namespace Nektar::LibUtilities
