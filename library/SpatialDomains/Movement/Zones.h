////////////////////////////////////////////////////////////////////////////////
//
//  File: Zones.h
//
//  For more information, please see: http://www.nektar.info/
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following s:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description: Zones used in the non-conformal interfaces
//               and ALE implementations
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SPATIALDOMAINS_ZONES_H
#define NEKTAR_SPATIALDOMAINS_ZONES_H

#include <LibUtilities/BasicUtils/Equation.h>
#include <SpatialDomains/MeshGraph.h>

namespace Nektar::SpatialDomains
{

/// Enum of zone movement type
enum MovementType
{
    eNone,
    eFixed,
    eRotate,
    eTranslate,
    SIZE_MovementType
};

/// Map of zone movement type to movement type string
const std::string MovementTypeStr[] = {"None", "Fixed", "Rotated",
                                       "Translated"};

/// Zone base: Contains the shared functions and variables
struct ZoneBase
{
    /// Constructor
    SPATIAL_DOMAINS_EXPORT ZoneBase(MovementType type, int indx, int domainID,
                                    CompositeMap domain, int coordDim);

    /// Default destructor
    virtual ~ZoneBase() = default;

    /// Returns the type of movement
    inline MovementType GetMovementType() const
    {
        return m_type;
    }

    /// Returns the domain the zone is on
    inline CompositeMap GetDomain() const
    {
        return m_domain;
    }

    /// Returns the zone ID
    inline int &GetId()
    {
        return m_id;
    }

    /// Returns the ID of the domain making up this Zone
    inline int &GetDomainID()
    {
        return m_domainID;
    }

    /// Performs the movement of the zone at @param time
    inline bool Move(NekDouble time)
    {
        return v_Move(time);
    }

    /// Returns all highest dimension elements in the zone
    inline std::vector<Geometry *> const &GetElements() const
    {
        return m_elements;
    }

    /// Returns the flag which states if the zone has moved in this timestep
    inline bool &GetMoved()
    {
        return m_moved;
    }

    /// Clears all bounding boxes associated with the zones elements
    void ClearBoundingBoxes();

    /// Returns constituent elements, i.e. faces + edges
    inline std::array<std::set<Geometry *>, 3> &GetConstituentElements()
    {
        return m_constituentElements;
    }

    /// Returns all points in the zone at initialisation
    inline std::vector<PointGeom> &GetOriginalVertex()
    {
        return m_origVerts;
    }

    /// Returns zone displacment
    SPATIAL_DOMAINS_EXPORT virtual std::vector<NekDouble> v_GetDisp() const
    {
        std::vector<NekDouble> disp(m_coordDim);
        for (int i = 0; i < m_coordDim; ++i)
        {
            disp[i] = 0.0;
        }

        return disp;
    }

    /// Returns zone rotate angle
    SPATIAL_DOMAINS_EXPORT virtual NekDouble v_GetAngle() const
    {
        return 0.0;
    }

protected:
    /// Type of zone movement
    MovementType m_type = MovementType::eNone;
    /// Zone ID
    int m_id;
    /// ID for the composite making up this zone.
    int m_domainID;
    /// Zone domain
    CompositeMap m_domain;
    /// Vector of highest dimension zone elements
    std::vector<Geometry *> m_elements;
    /// Array of all dimension elements i.e. faces = [2], edges = [1], geom =
    /// [0]
    std::array<std::set<Geometry *>, 3> m_constituentElements;
    /// Moved flag
    bool m_moved = true;
    /// Coordinate dimension
    int m_coordDim;
    /// Vector of all points in the zone
    std::vector<PointGeomUniquePtr> m_verts;
    /// Vector of all curves in the zone
    std::vector<Curve *> m_curves;
    /// Vector of all points in the zone at initialisation
    std::vector<PointGeom> m_origVerts;

    /// Virtual function for movement of the zone at @param time
    inline virtual bool v_Move([[maybe_unused]] NekDouble time)
    {
        return false;
    }
};

typedef std::shared_ptr<ZoneBase> ZoneBaseShPtr;

/// Rotating zone: Motion of every point around a given axis on an origin
struct ZoneRotate final : public ZoneBase
{
    /**
     * Constructor for rotating zones
     *
     * @param id Zone ID
     * @param domainID ID associated with the the domain making up
     *        the zone
     * @param domain Domain that the zone consists of
     * @param coordDim Coordinate dimension
     * @param origin Origin that the zone rotates about
     * @param axis Axis that the zone rotates about
     * @param angularVelEqn Equation for the angular velocity of rotation
     */
    SPATIAL_DOMAINS_EXPORT ZoneRotate(
        int id, int domainID, const CompositeMap &domain, const int coordDim,
        const NekPoint<NekDouble> &origin, const DNekVec &axis,
        const LibUtilities::EquationSharedPtr &angularVelEqn,
        const NekDouble rampTime, const NekDouble sector,
        const Array<OneD, NekDouble> &base);

    /// Default destructor
    ~ZoneRotate() override = default;

    /// Return the angular velocity of the zone at @param time
    SPATIAL_DOMAINS_EXPORT NekDouble GetAngularVel(const NekDouble &time) const;

    /// Returns the rotation angle of the zone
    SPATIAL_DOMAINS_EXPORT NekDouble GetAngle(const NekDouble &time);

    /// Rotates the given global coordinate by the given angle
    SPATIAL_DOMAINS_EXPORT void Rotate(Array<OneD, NekDouble> &gloCoord,
                                       const NekDouble &angle);

    /// Returns the origin the zone rotates about
    inline const NekPoint<NekDouble> &GetOrigin() const
    {
        return m_origin;
    }

    /// Returns the axis the zone rotates about
    inline const DNekVec &GetAxis() const
    {
        return m_axis;
    }

    /// Returns the equation for the angular velocity of the rotation
    inline LibUtilities::EquationSharedPtr GetAngularVelEqn() const
    {
        return m_angularVelEqn;
    }

    /// Returns zone rotate angle
    NekDouble v_GetAngle() const override
    {
        return m_angle;
    }

    /// Returns the angle of rotating sector.
    inline const NekDouble &GetSectorAngle() const
    {
        return m_sector;
    }

    /// Returns the angle of rotating sector.
    inline const Array<OneD, NekDouble> &GetSectorBase() const
    {
        return m_base;
    }

protected:
    ///  Origin point rotation is performed around
    NekPoint<NekDouble> m_origin;
    /// Axis rotation is performed around
    DNekVec m_axis;
    /// Equation defining angular velocity as a function of time
    LibUtilities::EquationSharedPtr m_angularVelEqn;
    /// Rotate angle
    NekDouble m_angle;
    /// Ramp time
    NekDouble m_rampTime;
    /// Sector angle
    NekDouble m_sector;
    /// Base axis of sector
    Array<OneD, NekDouble> m_base;
    /// W matrix Rodrigues' rotation formula, cross product of axis
    DNekMat m_W = DNekMat(3, 3, 0.0);
    /// W^2 matrix Rodrigues' rotation formula, cross product of axis squared
    DNekMat m_W2 = DNekMat(3, 3, 0.0);

    /// Virtual function for movement of the zone at @param time
    SPATIAL_DOMAINS_EXPORT bool v_Move(NekDouble time) final;
};

/// Translating zone: addition of a constant vector to every point
struct ZoneTranslate final : public ZoneBase
{
    /**
     * Constructor for translating zone
     *
     * @param id Zone ID
     * @param domainID ID associated with the the domain making up
     *        the zone
     * @param domain Domain that the zone consists of
     * @param coordDim Coordinate dimension
     * @param velocity Vector of translation velocity in x,y,z direction
     */
    SPATIAL_DOMAINS_EXPORT ZoneTranslate(
        int id, int domainID, const CompositeMap &domain, const int coordDim,
        const Array<OneD, LibUtilities::EquationSharedPtr> &velocityEqns,
        const Array<OneD, LibUtilities::EquationSharedPtr> &displacementEqns);

    /// Default destructor
    ~ZoneTranslate() override = default;

    /// Returns the velocity of the zone
    SPATIAL_DOMAINS_EXPORT std::vector<NekDouble> GetVel(NekDouble &time) const;

    /// Returns the displacement of the zone
    SPATIAL_DOMAINS_EXPORT std::vector<NekDouble> GetDisp(NekDouble &time);

    std::vector<NekDouble> v_GetDisp() const override
    {
        return m_disp;
    }

    /// Returns the equation for the velocity of the translation
    inline Array<OneD, LibUtilities::EquationSharedPtr> GetVelocityEquation()
        const
    {
        return m_velocityEqns;
    }

    /// Returns the equation for the displacement of the translation
    inline Array<OneD, LibUtilities::EquationSharedPtr> GetDisplacementEquation()
        const
    {
        return m_displacementEqns;
    }

    SPATIAL_DOMAINS_EXPORT inline const Array<OneD, NekDouble> &GetZoneBox()
        const
    {
        return m_ZoneBox;
    }

    SPATIAL_DOMAINS_EXPORT inline const Array<OneD, NekDouble> &GetZoneLength()
        const
    {
        return m_ZoneLength;
    }

    SPATIAL_DOMAINS_EXPORT void UpdateZoneBox(
        const Array<OneD, NekDouble> &ZoneBox,
        const Array<OneD, NekDouble> &ZoneLength)
    {
        m_ZoneBox    = ZoneBox;
        m_ZoneLength = ZoneLength;
    }

protected:
    Array<OneD, LibUtilities::EquationSharedPtr> m_velocityEqns;
    Array<OneD, LibUtilities::EquationSharedPtr> m_displacementEqns;
    std::vector<NekDouble> m_disp;
    Array<OneD, NekDouble> m_ZoneBox;    // Domain box
    Array<OneD, NekDouble> m_ZoneLength; // Lenghth of domain

    /// Virtual function for movement of the zone at @param time
    SPATIAL_DOMAINS_EXPORT bool v_Move(NekDouble time) final;
};

/// Fixed zone: does not move
struct ZoneFixed final : public ZoneBase
{
    /// Constructor
    ZoneFixed(int id, int domainID, const CompositeMap &domain,
              const int coordDim)
        : ZoneBase(MovementType::eFixed, id, domainID, domain, coordDim)
    {
    }

    /// Default destructor
    ~ZoneFixed() override = default;

protected:
    /// Virtual function for movement of the zone at @param time
    SPATIAL_DOMAINS_EXPORT bool v_Move(NekDouble time) final;

    /// Returns the displacement of the zone
    SPATIAL_DOMAINS_EXPORT std::vector<NekDouble> v_GetDisp() const override;

    /// Returns the displacement of the zone
    SPATIAL_DOMAINS_EXPORT NekDouble v_GetAngle() const override;
};

typedef std::shared_ptr<ZoneRotate> ZoneRotateShPtr;
typedef std::shared_ptr<ZoneTranslate> ZoneTranslateShPtr;
typedef std::shared_ptr<ZoneFixed> ZoneFixedShPtr;

} // namespace Nektar::SpatialDomains

#endif // NEKTAR_SPATIALDOMAINS_ZONES_H
