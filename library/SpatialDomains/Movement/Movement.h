////////////////////////////////////////////////////////////////////////////////
//
//  File: Movement.h
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
//  Software is furnished to do so, subject to the following:
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
//  Description: This file contains the base class for implementing
//               non-conformal geometry using the Movement object
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SPATIALDOMAINS_MOVEMENT_H
#define NEKTAR_SPATIALDOMAINS_MOVEMENT_H

#include <LibUtilities/BasicUtils/FieldIO.h>

#include <SpatialDomains/MeshGraph.h>
#include <SpatialDomains/Movement/InterfaceInterpolation.h>
#include <SpatialDomains/Movement/Zones.h>

namespace Nektar::SpatialDomains
{

typedef std::map<std::pair<int, std::string>, InterfacePairShPtr>
    InterfaceCollection;

class Movement
{
public:
    /// Default constructor
    SPATIAL_DOMAINS_EXPORT Movement() = default;

    /// Constructor to read from XML file
    SPATIAL_DOMAINS_EXPORT Movement(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        MeshGraph *meshGraph);

    /// Default destructor
    SPATIAL_DOMAINS_EXPORT ~Movement() = default;

    /// Write the MOVEMENT section of the XML file
    SPATIAL_DOMAINS_EXPORT void WriteMovement(TiXmlElement *root);

    inline const InterfaceCollection &GetInterfaces() const
    {
        return m_interfaces;
    }

    inline const std::map<int, ZoneBaseShPtr> &GetZones() const
    {
        return m_zones;
    }

    SPATIAL_DOMAINS_EXPORT void PerformMovement(NekDouble timeStep);

    inline const bool &GetMoveFlag() const
    {
        return m_moveFlag;
    }

    inline bool &GetCoordExchangeFlag()
    {
        return m_coordExchangeFlag;
    }

    inline const bool &GetMovedFlag() const
    {
        return m_moved;
    }

    inline const bool &GetTranslateFlag() const
    {
        return m_translate;
    }

    inline const bool &GetRotateFlag() const
    {
        return m_rotate;
    }

    inline const bool &GetSectorRotateFlag() const
    {
        return m_sectorRotate;
    }

    inline const bool &GetImplicitALESolverFlag() const
    {
        return m_implicitALESolver;
    }

    inline const bool &GetMeshDistortedFlag() const
    {
        return m_meshDistorted;
    }

    void SetImplicitALEFlag(bool &ImplicitALE)
    {
        m_implicitALESolver = ImplicitALE;
    }

    void SetMeshDistortedFlag(bool &meshDistorted)
    {
        m_moved = meshDistorted;
    }

    // Methods for manipulating the MOVEMENT data programatically
    /// Add a zone object to this Movement data.
    SPATIAL_DOMAINS_EXPORT void AddZone(ZoneBaseShPtr zone);

    /// Add pair of interfaces to this data
    SPATIAL_DOMAINS_EXPORT void AddInterface(std::string name,
                                             InterfaceShPtr left,
                                             InterfaceShPtr right);

protected:
    InterfaceCollection m_interfaces;
    std::map<int, ZoneBaseShPtr> m_zones;
    bool m_moveFlag          = false; // Flags presence of moving zones
    bool m_translate         = false; // Flags for translation
    bool m_rotate            = false; // Flags for rotation
    bool m_sectorRotate      = false; // Flags for sector rotation
    bool m_moved             = false; // Flags to check if domain moved
    bool m_implicitALESolver = false; // Flags for implicit ALE solver
    bool m_meshDistorted     = false; // Flags to check if mesh is distorted
    bool m_coordExchangeFlag =
        true; // Flags if missing coordinates need to be calculated

private:
    /// Read zones given TiXmlDocument
    void ReadZones(TiXmlElement *zonesTag, MeshGraph *meshGraph,
                   const LibUtilities::SessionReaderSharedPtr &pSession);
    /// Read interfaces given TiXmlDocument
    void ReadInterfaces(TiXmlElement *interfacesTag, MeshGraph *meshGraph);
    /// Update the box of translate zones
    void UpdateTransZoneBox(
        const LibUtilities::SessionReaderSharedPtr &pSession);
};

typedef std::shared_ptr<Movement> MovementSharedPtr;

} // namespace Nektar::SpatialDomains

#endif // NEKTAR_SPATIALDOMAINS_MOVEMENT_H
