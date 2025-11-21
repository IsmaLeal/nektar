///////////////////////////////////////////////////////////////////////////////
//
// File: ALEHelper.cpp
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
// Description: Helper class for ALE process
//
///////////////////////////////////////////////////////////////////////////////

#include "ALEHelper.h"
#include <LibUtilities/BasicUtils/Timer.h>
#include <StdRegions/StdQuadExp.h>

namespace Nektar::SolverUtils
{

void ALEHelper::v_ALEInitObject(
    [[maybe_unused]] int spaceDim,
    [[maybe_unused]] Array<OneD, MultiRegions::ExpListSharedPtr> &fields)
{
}

void ALEHelper::InitObject(
    [[maybe_unused]] int spaceDim,
    [[maybe_unused]] Array<OneD, MultiRegions::ExpListSharedPtr> &fields)
{
    // Create ALE objects for each interface zone
    if (fields[0]->GetGraph() !=
        nullptr) // homogeneous graphs are missing the graph data
    {
        for (auto &zone : fields[0]->GetGraph()->GetMovement()->GetZones())
        {
            switch (zone.second->GetMovementType())
            {
                case SpatialDomains::MovementType::eFixed:
                    m_ALEs.emplace_back(ALEFixedShPtr(
                        MemoryManager<ALEFixed>::AllocateSharedPtr(
                            zone.second)));
                    break;
                case SpatialDomains::MovementType::eTranslate:
                    m_ALEs.emplace_back(ALETranslateShPtr(
                        MemoryManager<ALETranslate>::AllocateSharedPtr(
                            zone.second)));
                    m_ALESolver = true;
                    break;
                case SpatialDomains::MovementType::eRotate:
                    m_ALEs.emplace_back(ALERotateShPtr(
                        MemoryManager<ALERotate>::AllocateSharedPtr(
                            zone.second)));
                    m_ALESolver = true;
                    break;
                case SpatialDomains::MovementType::eNone:
                    WARNINGL0(false,
                              "Zone cannot have movement type of 'None'.");
                default:
                    break;
            }
        }
    }

    // Update grid velocity
    v_UpdateGridVelocity(0);

    // Updates trace grid velocity
    for (int i = 0; i < m_gridVelocityTrace.size(); ++i)
    {
        m_fieldsALE[0]->ExtractTracePhys(m_gridVelocity[i],
                                         m_gridVelocityTrace[i], true);
    }
}

void ALEHelper::v_UpdateGridVelocity(const NekDouble &time)
{

    // Reset grid velocity to 0
    for (int i = 0; i < m_spaceDim; ++i)
    {
        std::fill(m_gridVelocity[i].begin(), m_gridVelocity[i].end(), 0.0);
    }
    if (m_ALESolver)
    {
        // Now update for each movement zone, adding the grid velocities
        for (auto &ALE : m_ALEs)
        {
            ALE->UpdateGridVel(time, m_fieldsALE, m_gridVelocity);
        }
    }
}

void ALEHelper::v_ALEPreMultiplyMass(
    Array<OneD, Array<OneD, NekDouble>> &fields)
{
    const int nm = m_fieldsALE[0]->GetNcoeffs();
    MultiRegions::GlobalMatrixKey mkey(StdRegions::eMass);

    // Premultiply each field by the mass matrix
    for (int i = 0; i < m_fieldsALE.size(); ++i)
    {
        fields[i] = Array<OneD, NekDouble>(nm);
        m_fieldsALE[i]->GeneralMatrixOp(mkey, m_fieldsALE[i]->GetCoeffs(),
                                        fields[i]);
    }
}

/**
* @brief Update m_fields with u^n by multiplying by inverse mass
   matrix. That's then used in e.g. checkpoint output and L^2 error
   calculation.
*/
void ALEHelper::ALEDoElmtInvMass(
    [[maybe_unused]] Array<OneD, Array<OneD, NekDouble>> &traceNormals,
    Array<OneD, Array<OneD, NekDouble>> &fields,
    [[maybe_unused]] NekDouble time)
{
    // @TODO: Look at geometric factor and junk only what is needed
    // @TODO: Look at collections and see if they offer a speed up
    for (int i = 0; i < m_fieldsALE.size(); ++i)
    {
        m_fieldsALE[i]->MultiplyByElmtInvMass(
            fields[i],
            m_fieldsALE[i]->UpdateCoeffs()); // @TODO: Potentially matrix free?
        m_fieldsALE[i]->BwdTrans(m_fieldsALE[i]->GetCoeffs(),
                                 m_fieldsALE[i]->UpdatePhys());
    }
}

void ALEHelper::ALEDoElmtInvMassBwdTrans(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray)
{
    const int nc   = m_fieldsALE[0]->GetNcoeffs();
    int nVariables = inarray.size();

    // General idea is that we are time-integrating the quantity (Mu), so we
    // need to multiply input by inverse mass matrix to get coefficients u,
    // and then backwards transform to physical space so we can apply the DG
    // operator.
    Array<OneD, NekDouble> tmp(nc);
    for (int i = 0; i < nVariables; ++i)
    {
        outarray[i] = Array<OneD, NekDouble>(m_fieldsALE[0]->GetNpoints());
        m_fieldsALE[i]->MultiplyByElmtInvMass(inarray[i], tmp);
        m_fieldsALE[i]->BwdTrans(tmp, outarray[i]);
    }
}

void ALEHelper::MoveMesh(const NekDouble &time,
                         Array<OneD, Array<OneD, NekDouble>> &traceNormals)
{
    // Only move if timestepped
    if (time == m_prevStageTime)
    {
        return;
    }

    LibUtilities::Timer timer;
    timer.Start();
    m_fieldsALE[0]->GetGraph()->GetMovement()->PerformMovement(
        time); // @TODO: Moved out of loop!
    timer.Stop();
    timer.AccumulateRegion("Movement::PerformMovement");

    ResetMatricesNormal(traceNormals);
    UpdateNormalsFlag();
    // Recompute grid velocity.
    v_UpdateGridVelocity(time);

    // Updates trace grid velocity
    for (int i = 0; i < m_gridVelocityTrace.size(); ++i)
    {
        m_fieldsALE[0]->ExtractTracePhys(m_gridVelocity[i],
                                         m_gridVelocityTrace[i], true);
    }

    // Set the flag to exchange coords in InterfaceMapDG to true
    m_fieldsALE[0]->GetGraph()->GetMovement()->GetCoordExchangeFlag() = true;

    m_prevStageTime = time;
}

void ALEHelper::ResetMatricesNormal(
    Array<OneD, Array<OneD, NekDouble>> &traceNormals)
{
    // Now update matrices and normal for each movement zone
    for (auto &ALE : m_ALEs)
    {
        ALE->ResetMatricesNormal(traceNormals, m_fieldsALE);
    }
}

void ALEHelper::UpdateNormalsFlag()
{
    // check all update nromal definitions and return true if any are true
    for (auto &ALE : m_ALEs)
    {
        if (ALE->UpdateNormalsFlag())
        {
            m_updateNormals = ALE->UpdateNormalsFlag();
            break;
        }
    }
}

const Array<OneD, const Array<OneD, NekDouble>> &ALEHelper::
    GetGridVelocityTrace()
{
    return m_gridVelocityTrace;
}

ALEFixed::ALEFixed(SpatialDomains::ZoneBaseShPtr zone)
    : m_zone(std::static_pointer_cast<SpatialDomains::ZoneFixed>(zone))
{
}

void ALEFixed::v_UpdateGridVel(
    [[maybe_unused]] NekDouble time,
    [[maybe_unused]] Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    [[maybe_unused]] Array<OneD, Array<OneD, NekDouble>> &gridVelocity)
{
}

void ALEFixed::v_ResetMatricesNormal(
    [[maybe_unused]] Array<OneD, Array<OneD, NekDouble>> &traceNormals,
    [[maybe_unused]] Array<OneD, MultiRegions::ExpListSharedPtr> &fields)
{
}

ALETranslate::ALETranslate(SpatialDomains::ZoneBaseShPtr zone)
    : m_zone(std::static_pointer_cast<SpatialDomains::ZoneTranslate>(zone))
{
}

void ALETranslate::v_UpdateGridVel(
    [[maybe_unused]] NekDouble time,
    Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    Array<OneD, Array<OneD, NekDouble>> &gridVelocity)
{
    auto vel = m_zone->GetVel(time);
    auto exp = fields[0]->GetExp();

    auto elements = m_zone->GetElements();
    for (auto &el : elements)
    {
        int indx       = fields[0]->GetElmtToExpId(el->GetGlobalID());
        int offset     = fields[0]->GetPhys_Offset(indx);
        auto expansion = (*exp)[indx];

        int nq = expansion->GetTotPoints();
        for (int i = 0; i < nq; ++i)
        {
            for (int j = 0; j < gridVelocity.size(); ++j)
            {
                gridVelocity[j][offset + i] += vel[j];
            }
        }
    }
}

void ALETranslate::v_ResetMatricesNormal(
    [[maybe_unused]] Array<OneD, Array<OneD, NekDouble>> &traceNormals,
    Array<OneD, MultiRegions::ExpListSharedPtr> &fields)
{
    auto &curvedEdges = fields[0]->GetGraph()->GetCurvedEdges();
    auto &curvedFaces = fields[0]->GetGraph()->GetCurvedFaces();

    // The order of the resets below is v important to avoid errors
    if (m_meshDistorted)
    {
        for (auto &field : fields)
        {
            field->ResetMatrices();
        }
    }

    // Loop over all elements and faces and edges and reset geometry
    // information. Only need to do this on the first field as the geometry
    // information is shared.

    if (m_zone->GetMoved())
    {
        auto conEl = m_zone->GetConstituentElements();
        for (const auto &i : conEl)
        {
            for (const auto &j : i)
            {
                j->ResetNonRecursive(curvedEdges, curvedFaces);
            }
        }

        // We need to rebuild geometric factors on the trace elements
        for (const auto &i : conEl[fields[0]->GetShapeDimension() -
                                   1]) // This only takes the trace elements
        {
            fields[0]->GetTrace()->GetExpFromGeomId(i->GetGlobalID())->Reset();
        }
    }

    if (m_zone->GetMoved())
    {
        auto conEl = m_zone->GetConstituentElements();
        // Loop over zone elements expansions and rebuild geometric
        // factors
        for (const auto &i :
             conEl[0]) // This only takes highest dimensioned elements
        {
            fields[0]->GetExpFromGeomId(i->GetGlobalID())->Reset();
        }
    }

    for (auto &field : fields)
    {
        // Reset collections (despite the default being eNoCollection it does
        // remember the last auto-tuned values), eNoImpType gives lots of output
        field->CreateCollections(Collections::eNoCollection);
    }
}

ALERotate::ALERotate(SpatialDomains::ZoneBaseShPtr zone)
    : m_zone(std::static_pointer_cast<SpatialDomains::ZoneRotate>(zone))
{
}

void ALERotate::v_UpdateGridVel(
    [[maybe_unused]] NekDouble time,
    [[maybe_unused]] Array<OneD, MultiRegions::ExpListSharedPtr> &fields,
    [[maybe_unused]] Array<OneD, Array<OneD, NekDouble>> &gridVelocity)
{

    auto angVel = m_zone->GetAngularVel(time);
    auto axis   = m_zone->GetAxis();
    auto origin = m_zone->GetOrigin();

    auto exp = fields[0]->GetExp();

    auto elements = m_zone->GetElements();
    for (auto &el : elements)
    {
        int indx       = fields[0]->GetElmtToExpId(el->GetGlobalID());
        int offset     = fields[0]->GetPhys_Offset(indx);
        auto expansion = (*exp)[indx];

        int nq = expansion->GetTotPoints();

        Array<OneD, NekDouble> xc(nq, 0.0), yc(nq, 0.0), zc(nq, 0.0);
        Array<OneD, NekDouble> norm(3, 0.0);
        expansion->GetCoords(xc, yc, zc);
        for (int i = 0; i < nq; ++i)
        {
            // Vector from origin to point
            NekDouble xpointMinOrigin = xc[i] - origin(0);
            NekDouble ypointMinOrigin = yc[i] - origin(1);
            NekDouble zpointMinOrigin = zc[i] - origin(2);

            // Vector orthogonal to plane formed by axis and point
            // We negate angVel here as by convention a positive angular
            // velocity is counter-clockwise
            norm[0] = (ypointMinOrigin * axis[2] - zpointMinOrigin * axis[1]) *
                      (-angVel);
            norm[1] = (zpointMinOrigin * axis[0] - xpointMinOrigin * axis[2]) *
                      (-angVel);
            norm[2] = (xpointMinOrigin * axis[1] - ypointMinOrigin * axis[0]) *
                      (-angVel);
            for (int j = 0; j < gridVelocity.size(); ++j)
            {
                gridVelocity[j][offset + i] = norm[j];
            }
        }
    }
}

void ALERotate::v_ResetMatricesNormal(
    Array<OneD, Array<OneD, NekDouble>> &traceNormals,
    Array<OneD, MultiRegions::ExpListSharedPtr> &fields)
{

    auto &curvedEdges = fields[0]->GetGraph()->GetCurvedEdges();
    auto &curvedFaces = fields[0]->GetGraph()->GetCurvedFaces();

    // The order of the resets below is v important to avoid errors
    if (m_meshDistorted)
    {
        for (auto &field : fields)
        {
            field->ResetMatrices();
        }
    }

    // Loop over all elements and faces and edges and reset geometry
    // information. Only need to do this on the first field as the geometry
    // information is shared.
    if (m_zone->GetMoved() && m_zone->v_GetAngle() != 0)
    {
        auto conEl = m_zone->GetConstituentElements();
        for (const auto &i : conEl)
        {
            for (const auto &j : i)
            {
                j->ResetNonRecursive(curvedEdges, curvedFaces);
            }
        }

        // We need to rebuild geometric factors on the trace elements
        for (const auto &i : conEl[fields[0]->GetShapeDimension() -
                                   1]) // This only takes the trace elements
        {
            fields[0]->GetTrace()->GetExpFromGeomId(i->GetGlobalID())->Reset();
        }
    }

    if (m_zone->GetMoved() && m_zone->v_GetAngle() != 0)
    {
        auto conEl = m_zone->GetConstituentElements();
        // Loop over zone elements expansions and rebuild geometric
        // factors
        for (const auto &i :
             conEl[0]) // This only takes highest dimensioned elements
        {
            fields[0]->GetExpFromGeomId(i->GetGlobalID())->Reset();
        }
    }

    if (m_zone->GetMoved() && m_zone->v_GetAngle() != 0)
    {
        auto conEl = m_zone->GetConstituentElements();
        // Loop over zone elements expansions and rebuild geometric factors
        // and recalc trace normals
        for (const auto &i :
             conEl[0]) // This only takes highest dimensioned elements
        {
            int nfaces =
                fields[0]->GetExpFromGeomId(i->GetGlobalID())->GetNtraces();
            for (int j = 0; j < nfaces; ++j)
            {
                fields[0]
                    ->GetExpFromGeomId(i->GetGlobalID())
                    ->ComputeTraceNormal(j);
            }
        }
    }

    for (auto &field : fields)
    {
        // Reset collections (despite the default being eNoCollection it does
        // remember the last auto-tuned values), eNoImpType gives lots of output
        field->CreateCollections(Collections::eNoCollection);
    }

    // Reload new trace normals in to the solver cache
    fields[0]->GetTrace()->GetNormals(traceNormals);
}

void ALEHelper::ExtraFldOutputGridVelocity(
    std::vector<Array<OneD, NekDouble>> &fieldcoeffs,
    std::vector<std::string> &variables)

{
    int nCoeffs = m_fieldsALE[0]->GetNcoeffs();
    // Adds extra output variables for grid velocity
    std::string gridVarName[3] = {"gridVx", "gridVy", "gridVz"};
    for (int i = 0; i < m_spaceDim; ++i)
    {
        Array<OneD, NekDouble> gridVel(nCoeffs, 0.0);
        m_fieldsALE[0]->FwdTransLocalElmt(m_gridVelocity[i], gridVel);
        fieldcoeffs.emplace_back(gridVel);
        variables.emplace_back(gridVarName[i]);
    }
}

void ALEHelper::ExtraFldOutputGrid(
    std::vector<Array<OneD, NekDouble>> &fieldcoeffs,
    std::vector<std::string> &variables)

{
    int nCoeffs = m_fieldsALE[0]->GetNcoeffs();
    int nPhys   = m_fieldsALE[0]->GetTotPoints();
    // Adds extra output variables for grid velocity
    std::string gridXName[3] = {"gridX", "gridY", "gridZ"};
    Array<OneD, Array<OneD, NekDouble>> Xc(m_spaceDim);
    for (int i = 0; i < m_spaceDim; ++i)
    {
        Xc[i] = Array<OneD, NekDouble>(nPhys, 0.0);
    }

    switch (m_spaceDim)
    {
        case 1:
        {
            m_fieldsALE[0]->GetCoords(Xc[0]);
            break;
        }
        case 2:
        {
            m_fieldsALE[0]->GetCoords(Xc[0], Xc[1]);
            break;
        }
        case 3:
        {
            m_fieldsALE[0]->GetCoords(Xc[0], Xc[1], Xc[2]);
            break;
        }
    }

    for (int i = 0; i < m_spaceDim; ++i)
    {
        Array<OneD, NekDouble> coeffs(nCoeffs);
        m_fieldsALE[0]->FwdTransLocalElmt(Xc[i], coeffs);
        fieldcoeffs.emplace_back(coeffs);
        variables.emplace_back(gridXName[i]);
    }
}

} // namespace Nektar::SolverUtils
