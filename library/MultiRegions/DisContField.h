///////////////////////////////////////////////////////////////////////////////
//
// File: DisContField.h
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
// Description: Field definition in one-dimension for a discontinuous
// LDG-H expansion
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIBS_MULTIREGIONS_DISCONTFIELD1D_H
#define NEKTAR_LIBS_MULTIREGIONS_DISCONTFIELD1D_H

#include <MultiRegions/AssemblyMap/AssemblyMapDG.h>
#include <MultiRegions/AssemblyMap/InterfaceMapDG.h>
#include <MultiRegions/AssemblyMap/LocTraceToTraceMap.h>
#include <MultiRegions/ExpList.h>
#include <MultiRegions/GlobalLinSys.h>
#include <MultiRegions/MultiRegions.hpp>
#include <MultiRegions/MultiRegionsDeclspec.h>
#include <SpatialDomains/Conditions.h>
#include <boost/algorithm/string.hpp>

namespace Nektar::MultiRegions
{

/// This class is the abstractio  n of a global discontinuous two-
/// dimensional spectral/hp element expansion which approximates the
/// solution of a set of partial differential equations.
class DisContField : public ExpList
{
public:
    Array<OneD, int> m_BCtoElmMap;
    Array<OneD, int> m_BCtoTraceMap;

    /// Default constructor.
    MULTI_REGIONS_EXPORT DisContField();

    /// Constructs a 1D discontinuous field based on a mesh and boundary
    /// conditions.
    MULTI_REGIONS_EXPORT DisContField(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &graph,
        const std::string &variable, const bool SetUpJustDG = true,
        const bool DeclareCoeffPhysArrays             = true,
        const Collections::ImplementationType ImpType = Collections::eNoImpType,
        const std::string bcvariable                  = "NotSet");

    /// Constructor for a DisContField from a List of subdomains
    /// New Constructor for arterial network
    MULTI_REGIONS_EXPORT DisContField(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &graph1D,
        const SpatialDomains::CompositeMap &domain,
        const SpatialDomains::BoundaryConditions &Allbcs,
        const std::string &variable, const LibUtilities::CommSharedPtr &comm,
        bool SetToOneSpaceDimensions = false,
        const Collections::ImplementationType ImpType =
            Collections::eNoImpType);

    /// Constructs a 1D discontinuous field based on an existing field.
    MULTI_REGIONS_EXPORT DisContField(const DisContField &In,
                                      const bool DeclareCoeffPhysArrays = true);

    MULTI_REGIONS_EXPORT DisContField(
        const DisContField &In, const SpatialDomains::MeshGraphSharedPtr &graph,
        const std::string &variable, const bool SetUpJustDG = true,
        const bool DeclareCoeffPhysArrays = true);

    /// Constructs a 1D discontinuous field based on an
    /// existing field.  (needed in order to use ContField(
    /// const ExpList &In) constructor
    MULTI_REGIONS_EXPORT DisContField(const ExpList &In);

    /// Destructor.
    MULTI_REGIONS_EXPORT ~DisContField() override;

    /// For a given key, returns the associated global linear system.
    MULTI_REGIONS_EXPORT GlobalLinSysSharedPtr
    GetGlobalBndLinSys(const GlobalLinSysKey &mkey);

    /// Check to see if expansion has the same BCs as In
    MULTI_REGIONS_EXPORT bool SameTypeOfBoundaryConditions(
        const DisContField &In);

    // Return the internal vector which directs whether the normal flux
    // at the trace defined by Left and Right Adjacent elements
    // is negated with respect to the segment normal
    MULTI_REGIONS_EXPORT std::vector<bool> &GetNegatedFluxNormal(void);

    MULTI_REGIONS_EXPORT NekDouble
    L2_DGDeriv(const int dir, const Array<OneD, const NekDouble> &coeffs,
               const Array<OneD, const NekDouble> &soln);
    MULTI_REGIONS_EXPORT void EvaluateHDGPostProcessing(
        const Array<OneD, const NekDouble> &coeffs,
        Array<OneD, NekDouble> &outarray);

    MULTI_REGIONS_EXPORT void GetLocTraceToTraceMap(
        LocTraceToTraceMapSharedPtr &LocTraceToTraceMap)
    {
        LocTraceToTraceMap = m_locTraceToTraceMap;
    }

    MULTI_REGIONS_EXPORT void GetFwdBwdTracePhys(
        const Array<OneD, const NekDouble> &field, Array<OneD, NekDouble> &Fwd,
        Array<OneD, NekDouble> &Bwd,
        const Array<OneD, const SpatialDomains::BoundaryConditionShPtr>
            &bndCond,
        const Array<OneD, const ExpListSharedPtr> &BndCondExp);

    MULTI_REGIONS_EXPORT ExpListSharedPtr &GetLocElmtTrace()
    {
        return m_locElmtTrace;
    }

protected:
    /// An array which contains the information about the boundary
    /// condition structure definition on the different boundary regions.
    Array<OneD, SpatialDomains::BoundaryConditionShPtr> m_bndConditions;

    /**
     * @brief An object which contains the discretised boundary
     * conditions.
     *
     * It is an array of size equal to the number of boundary
     * regions and consists of entries of the type
     * MultiRegions#ExpList. Every entry corresponds to the
     * spectral/hp expansion on a single boundary region.  The
     * values of the boundary conditions are stored as the
     * coefficients of the one-dimensional expansion.
     */
    Array<OneD, MultiRegions::ExpListSharedPtr> m_bndCondExpansions;

    Array<OneD, NekDouble> m_bndCondBndWeight;

    /// Interfaces mapping for trace space.
    InterfaceMapDGSharedPtr m_interfaceMap;

    /// Global boundary matrix.
    GlobalLinSysMapShPtr m_globalBndMat;

    /// Trace space storage for points between elements.
    ExpListSharedPtr m_trace;

    /// Local Elemental trace expansions
    MultiRegions::ExpListSharedPtr m_locElmtTrace;

    /// Local to global DG mapping for trace space.
    AssemblyMapDGSharedPtr m_traceMap;

    /**
     * @brief A set storing the global IDs of any boundary Verts.
     */
    std::set<int> m_boundaryTraces;

    /**
     * @brief A map which identifies groups of periodic vertices.
     */
    PeriodicMap m_periodicVerts;

    /**
     * @brief A map which identifies pairs of periodic edges.
     */
    PeriodicMap m_periodicEdges;

    /**
     * @brief A map which identifies pairs of periodic faces.
     */
    PeriodicMap m_periodicFaces;

    /**
     * @brief A vector indicating degress of freedom which need to be
     * copied from forwards to backwards space in case of a periodic
     * boundary condition.
     */
    std::vector<int> m_periodicFwdCopy;
    std::vector<int> m_periodicBwdCopy;

    /*
     * @brief A map identifying which traces are left- and
     * right-adjacent for DG.
     */
    std::vector<bool> m_leftAdjacentTraces;

    /**
     * Map of local trace (the points at the edge,face of
     * the element) to the trace space discretisation
     */
    LocTraceToTraceMapSharedPtr m_locTraceToTraceMap;

    /// Discretises the boundary conditions.
    void GenerateBoundaryConditionExpansion(
        const SpatialDomains::MeshGraphSharedPtr &graph1D,
        const SpatialDomains::BoundaryConditions &bcs,
        const std::string variable, const bool DeclareCoeffPhysArrays = true);

    /// Generate a associative map of periodic vertices in a mesh.
    void FindPeriodicTraces(const SpatialDomains::BoundaryConditions &bcs,
                            const std::string variable);

    void SetUpDG(const std::string = "DefaultVar",
                 const Collections::ImplementationType ImpType =
                     Collections::eNoImpType);

    bool IsLeftAdjacentTrace(const int n, const int e);

    ExpListSharedPtr &v_GetTrace() override;

    AssemblyMapDGSharedPtr &v_GetTraceMap(void) override;
    InterfaceMapDGSharedPtr &v_GetInterfaceMap(void) override;

    const LocTraceToTraceMapSharedPtr &v_GetLocTraceToTraceMap(
        void) const override;

    std::vector<bool> &v_GetLeftAdjacentTraces(void) override;

    void v_AddTraceIntegral(const Array<OneD, const NekDouble> &Fn,
                            Array<OneD, NekDouble> &outarray) override;

    void v_AddFwdBwdTraceIntegral(const Array<OneD, const NekDouble> &Fwd,
                                  const Array<OneD, const NekDouble> &Bwd,
                                  Array<OneD, NekDouble> &outarray) override;

    void v_AddTraceQuadPhysToField(const Array<OneD, const NekDouble> &Fwd,
                                   const Array<OneD, const NekDouble> &Bwd,
                                   Array<OneD, NekDouble> &field) override;

    void v_ExtractTracePhys(const Array<OneD, const NekDouble> &inarray,
                            Array<OneD, NekDouble> &outarray,
                            bool gridVelocity = false) override;

    void v_ExtractTracePhys(Array<OneD, NekDouble> &outarray) override;

    void v_GetLocTraceFromTracePts(
        const Array<OneD, const NekDouble> &Fwd,
        const Array<OneD, const NekDouble> &Bwd,
        Array<OneD, NekDouble> &locTraceFwd,
        Array<OneD, NekDouble> &locTraceBwd) override;

    void GenerateFieldBnd1D(SpatialDomains::BoundaryConditions &bcs,
                            const std::string variable);

    std::map<int, RobinBCInfoSharedPtr> v_GetRobinBCInfo() override;

    const Array<OneD, const MultiRegions::ExpListSharedPtr> &
    v_GetBndCondExpansions() override;

    const Array<OneD, const SpatialDomains::BoundaryConditionShPtr> &
    v_GetBndConditions() override;

    MultiRegions::ExpListSharedPtr &v_UpdateBndCondExpansion(int i) override;

    Array<OneD, SpatialDomains::BoundaryConditionShPtr> &v_UpdateBndConditions()
        override;

    void v_GetBoundaryToElmtMap(Array<OneD, int> &ElmtID,
                                Array<OneD, int> &TraceID) override;
    void v_GetBndElmtExpansion(int i, std::shared_ptr<ExpList> &result,
                               const bool DeclareCoeffPhysArrays) override;

    void v_Reset() override;

    /// Evaluate all boundary conditions at a given time..
    void v_EvaluateBoundaryConditions(
        const NekDouble time = 0.0, const std::string varName = "",
        const NekDouble x2_in = NekConstants::kNekUnsetDouble,
        const NekDouble x3_in = NekConstants::kNekUnsetDouble) override;

    /// Solve the Helmholtz equation.
    GlobalLinSysKey v_HelmSolve(const Array<OneD, const NekDouble> &inarray,
                                Array<OneD, NekDouble> &outarray,
                                const StdRegions::ConstFactorMap &factors,
                                const StdRegions::VarCoeffMap &varcoeff,
                                const MultiRegions::VarFactorsMap &varfactors,
                                const Array<OneD, const NekDouble> &dirForcing,
                                const bool PhysSpaceForcing) override;

    void v_PeriodicBwdCopy(const Array<OneD, const NekDouble> &Fwd,
                           Array<OneD, NekDouble> &Bwd) override;
    void v_PeriodicBwdRot(Array<OneD, Array<OneD, NekDouble>> &Bwd) override;

    void v_PeriodicDeriveBwdRot(TensorOfArray3D<NekDouble> &Bwd) override;

    void v_RotLocalBwdTrace(Array<OneD, Array<OneD, NekDouble>> &Bwd) override;
    void v_RotLocalBwdDeriveTrace(TensorOfArray3D<NekDouble> &Bwd) override;

    void v_FillBwdWithBwdWeight(Array<OneD, NekDouble> &weightave,
                                Array<OneD, NekDouble> &weightjmp) override;

    void v_GetFwdBwdTracePhys(Array<OneD, NekDouble> &Fwd,
                              Array<OneD, NekDouble> &Bwd) override;

    void v_GetFwdBwdTracePhys(const Array<OneD, const NekDouble> &field,
                              Array<OneD, NekDouble> &Fwd,
                              Array<OneD, NekDouble> &Bwd, bool FillBnd = true,
                              bool PutFwdInBwdOnBCs = false,
                              bool DoExchange       = true) override;

    void v_FillBwdWithBoundCond(const Array<OneD, NekDouble> &Fwd,
                                Array<OneD, NekDouble> &Bwd,
                                bool PutFwdInBwdOnBCs) override;

    void FillBwdWithBoundCond(
        const Array<OneD, NekDouble> &Fwd, Array<OneD, NekDouble> &Bwd,
        const Array<OneD, const SpatialDomains::BoundaryConditionShPtr>
            &bndConditions,
        const Array<OneD, const ExpListSharedPtr> &BndCondExpansions,
        bool PutFwdInBwdOnBCs);

    const Array<OneD, const NekDouble> &v_GetBndCondBwdWeight() override;

    void v_SetBndCondBwdWeight(const int index, const NekDouble value) override;

    void v_GetPeriodicEntities(PeriodicMap &periodicVerts,
                               PeriodicMap &periodicEdges,
                               PeriodicMap &periodicFaces) override;

    void v_AddTraceIntegralToOffDiag(
        const Array<OneD, const NekDouble> &FwdFlux,
        const Array<OneD, const NekDouble> &BwdFlux,
        Array<OneD, NekDouble> &outarray) override;

    void Rotate(Array<OneD, Array<OneD, NekDouble>> &Bwd, const int dir,
                const NekDouble angle, const int offset, const int npts);

    void DeriveRotate(TensorOfArray3D<NekDouble> &Bwd, const int dir,
                      const NekDouble angle, const int offset, const int npts);

private:
    std::vector<bool> m_negatedFluxNormal;

    SpatialDomains::BoundaryConditionsSharedPtr GetDomainBCs(
        const SpatialDomains::CompositeMap &domain,
        const SpatialDomains::BoundaryConditions &Allbcs,
        const std::string &variable);
};

typedef std::shared_ptr<DisContField> DisContFieldSharedPtr;

} // namespace Nektar::MultiRegions

#endif // NEKTAR_LIBS_MULTIREGIONS_DISCONTFIELD1D_H
