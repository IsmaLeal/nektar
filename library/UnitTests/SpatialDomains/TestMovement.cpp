///////////////////////////////////////////////////////////////////////////////
//
// File: TestMovement.cpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/ParseUtils.h>
#include <SpatialDomains/MeshGraph.h>
#include <SpatialDomains/Movement/InterfaceInterpolation.h>
#include <SpatialDomains/Movement/Movement.h>
#include <SpatialDomains/Movement/Zones.h>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

namespace Nektar::MovementTests
{

LibUtilities::SessionReaderSharedPtr CreateSession()
{
    char arg     = ' ';
    char *args[] = {&arg};
    return LibUtilities::SessionReader::CreateInstance(1, args);
}

const std::string angVelStr = "0.1*t";

std::vector<std::string> velocityStr     = {"1.0", "2.0", "3.0"};
std::vector<std::string> displacementStr = {"1.0", "2.0", "3.0"};
const NekPoint<NekDouble> origin         = {1., 2., 3.};
const DNekVec axis                       = {1., 2., 3.};

/// Produce dummy Zone objects, containing empty domain pointers
SpatialDomains::ZoneBaseShPtr CreateZone(
    SpatialDomains::MovementType type, int zoneID, int domainID,
    LibUtilities::InterpreterSharedPtr interpreter)
{
    SpatialDomains::CompositeMap domain;
    switch (type)
    {
        case SpatialDomains::MovementType::eFixed:
        {
            return SpatialDomains::ZoneFixedShPtr(
                MemoryManager<SpatialDomains::ZoneFixed>::AllocateSharedPtr(
                    zoneID, domainID, domain, 3));
        }
        break;
        case SpatialDomains::MovementType::eRotate:
        {
            LibUtilities::EquationSharedPtr angularVelEqn =
                std::make_shared<LibUtilities::Equation>(interpreter,
                                                         angVelStr);
            return SpatialDomains::ZoneRotateShPtr(
                MemoryManager<SpatialDomains::ZoneRotate>::AllocateSharedPtr(
                    zoneID, domainID, domain, 3, origin, axis, angularVelEqn));
        }
        break;
        case SpatialDomains::MovementType::eTranslate:
        {
            Array<OneD, LibUtilities::EquationSharedPtr> velocityEqns(3);
            Array<OneD, LibUtilities::EquationSharedPtr> displacementEqns(3);
            for (int i = 0; i < 3; ++i)
            {
                velocityEqns[i] = std::make_shared<LibUtilities::Equation>(
                    interpreter, velocityStr[i]);
                displacementEqns[i] = std::make_shared<LibUtilities::Equation>(
                    interpreter, displacementStr[i]);
            }

            return SpatialDomains::ZoneTranslateShPtr(
                MemoryManager<SpatialDomains::ZoneTranslate>::AllocateSharedPtr(
                    zoneID, domainID, domain, 3, velocityEqns,
                    displacementEqns));
        }
        break;
        default:
        {
        }
        break;
    }
    return nullptr;
}

/// Produce dummy Interface objects, containing empty domain pointers
SpatialDomains::InterfaceShPtr CreateInterface(int interfaceID,
                                               std::vector<int> compositeIDs)
{
    SpatialDomains::CompositeMap edge;
    for (auto &id : compositeIDs)
    {
        edge[id] =
            MemoryManager<SpatialDomains::Composite>::AllocateSharedPtr();
    }
    return SpatialDomains::InterfaceShPtr(
        MemoryManager<SpatialDomains::Interface>::AllocateSharedPtr(
            interfaceID, edge, false));
}

BOOST_AUTO_TEST_CASE(TestAddGetZones)
{
    LibUtilities::InterpreterSharedPtr interpreter =
        MemoryManager<LibUtilities::Interpreter>::AllocateSharedPtr();
    SpatialDomains::Movement m;
    SpatialDomains::ZoneBaseShPtr zone1 = CreateZone(
                                      SpatialDomains::MovementType::eFixed, 0,
                                      0, interpreter),
                                  zone2 = CreateZone(
                                      SpatialDomains::MovementType::eRotate, 2,
                                      2, interpreter);
    m.AddZone(zone1);
    m.AddZone(zone2);
    std::map<int, SpatialDomains::ZoneBaseShPtr> zones = m.GetZones();
    BOOST_TEST(zones.size() == 2);
    BOOST_TEST(zones.at(0).get() == zone1.get());
    BOOST_TEST(zones.at(2).get() == zone2.get());
    BOOST_TEST(zones.at(0) == zone1);
    BOOST_TEST(zones.at(2) == zone2);
}

BOOST_AUTO_TEST_CASE(TestAddGetInterfaces)
{
    SpatialDomains::Movement m;
    SpatialDomains::InterfaceShPtr interface1 = CreateInterface(0, {0}),
                                   interface2 = CreateInterface(1, {1, 2, 3}),
                                   interface3 = CreateInterface(3, {4});
    m.AddInterface("north", interface1, interface2);
    m.AddInterface("east", interface2, interface3);
    m.AddInterface("south", interface3, interface1);
    SpatialDomains::InterfaceCollection interfaces = m.GetInterfaces();
    BOOST_TEST(interfaces.size() == 3);
    SpatialDomains::InterfacePairShPtr north = interfaces.at(
                                           std::make_pair(0, "north")),
                                       east = interfaces.at(
                                           std::make_pair(1, "east")),
                                       south = interfaces.at(
                                           std::make_pair(2, "south"));
    BOOST_TEST(north->m_leftInterface == interface1);
    BOOST_TEST(north->m_rightInterface == interface2);
    BOOST_TEST(east->m_leftInterface == interface2);
    BOOST_TEST(east->m_rightInterface == interface3);
    BOOST_TEST(south->m_leftInterface == interface3);
    BOOST_TEST(south->m_rightInterface == interface1);
}

BOOST_AUTO_TEST_CASE(TestWriteMovement)
{
    LibUtilities::InterpreterSharedPtr interpreter =
        MemoryManager<LibUtilities::Interpreter>::AllocateSharedPtr();
    SpatialDomains::Movement m;
    m.AddZone(
        CreateZone(SpatialDomains::MovementType::eFixed, 0, 10, interpreter));
    m.AddZone(CreateZone(SpatialDomains::MovementType::eTranslate, 3, 13,
                         interpreter));
    m.AddZone(
        CreateZone(SpatialDomains::MovementType::eRotate, 1, 11, interpreter));
    m.AddInterface("north", CreateInterface(0, {0}),
                   CreateInterface(1, {1, 2, 3, 4}));
    m.AddInterface("south", CreateInterface(2, {5, 6, 7, 8, 9, 11}),
                   CreateInterface(3, {12}));

    SpatialDomains::InterfaceCollection interfaces     = m.GetInterfaces();
    std::map<int, SpatialDomains::ZoneBaseShPtr> zones = m.GetZones();
    TiXmlElement *nektar = new TiXmlElement("NEKTAR");
    m.WriteMovement(nektar);
    TiXmlElement *movement = nektar->FirstChildElement("MOVEMENT");
    BOOST_TEST(movement);
    TiXmlElement *xmlZones      = movement->FirstChildElement("ZONES"),
                 *xmlInterfaces = movement->FirstChildElement("INTERFACES");
    BOOST_TEST(xmlZones);
    BOOST_TEST(xmlInterfaces);

    int id, err;
    std::string attr;
    std::vector<NekDouble> vec;

    // Check the fixed zone
    TiXmlElement *zone = xmlZones->FirstChildElement();
    BOOST_TEST(zone->Value() == "F");
    err = zone->QueryIntAttribute("ID", &id);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(id == 0);
    err = zone->QueryStringAttribute("DOMAIN", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(attr == "D[10]");

    // Check the rotating zone
    zone = xmlZones->IterateChildren(zone)->ToElement();
    BOOST_TEST(zone->Value() == "R");
    err = zone->QueryIntAttribute("ID", &id);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(id == 1);
    err = zone->QueryStringAttribute("DOMAIN", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(attr == "D[11]");
    err = zone->QueryStringAttribute("ORIGIN", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    ParseUtils::GenerateVector(attr, vec);
    BOOST_TEST(NekPoint<NekDouble>(vec.at(0), vec.at(1), vec.at(2)) == origin);
    err = zone->QueryStringAttribute("AXIS", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    DNekVec xmlAxis(attr);
    BOOST_TEST(xmlAxis.GetDimension() == 3);
    for (int i = 0; i < 3; i++)
    {
        BOOST_TEST(xmlAxis[i] == axis[i]);
    }
    err = zone->QueryStringAttribute("ANGVEL", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(attr == angVelStr);

    // Check the translating zone
    zone = xmlZones->IterateChildren(zone)->ToElement();
    BOOST_TEST(zone->Value() == "T");
    err = zone->QueryIntAttribute("ID", &id);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(id == 3);
    err = zone->QueryStringAttribute("DOMAIN", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(attr == "D[13]");
    err = zone->QueryStringAttribute("XVELOCITY", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(attr == velocityStr[0]);
    err = zone->QueryStringAttribute("YVELOCITY", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(attr == velocityStr[1]);
    err = zone->QueryStringAttribute("ZVELOCITY", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(attr == velocityStr[2]);
    err = zone->QueryStringAttribute("XDISPLACEMENT", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(attr == displacementStr[0]);
    err = zone->QueryStringAttribute("YDISPLACEMENT", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(attr == displacementStr[1]);
    err = zone->QueryStringAttribute("ZDISPLACEMENT", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(attr == displacementStr[2]);

    BOOST_TEST(xmlZones->LastChild() == zone);

    TiXmlElement *intr;

    // Check the north interface
    TiXmlElement *interface = xmlInterfaces->FirstChildElement();
    BOOST_TEST(interface->Value() == "INTERFACE");
    err = interface->QueryStringAttribute("NAME", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(attr == "north");
    intr = interface->FirstChildElement();
    BOOST_TEST(intr->Value() == "L");
    err = intr->QueryIntAttribute("ID", &id);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(id == 0);
    err = intr->QueryStringAttribute("BOUNDARY", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(attr == "C[0]");
    intr = interface->IterateChildren(intr)->ToElement();
    BOOST_TEST(intr->Value() == "R");
    err = intr->QueryIntAttribute("ID", &id);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(id == 1);
    err = intr->QueryStringAttribute("BOUNDARY", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(attr == "C[1-4]");

    // Check the south interface
    interface = xmlInterfaces->IterateChildren(interface)->ToElement();
    BOOST_TEST(interface->Value() == "INTERFACE");
    err = interface->QueryStringAttribute("NAME", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(attr == "south");
    intr = interface->FirstChildElement();
    BOOST_TEST(intr->Value() == "L");
    err = intr->QueryIntAttribute("ID", &id);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(id == 2);
    err = intr->QueryStringAttribute("BOUNDARY", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(attr == "C[5-9,11]");
    intr = interface->IterateChildren(intr)->ToElement();
    BOOST_TEST(intr->Value() == "R");
    err = intr->QueryIntAttribute("ID", &id);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(id == 3);
    err = intr->QueryStringAttribute("BOUNDARY", &attr);
    BOOST_TEST(err == TIXML_SUCCESS);
    BOOST_TEST(attr == "C[12]");

    BOOST_TEST(xmlInterfaces->LastChild() == interface);

    delete nektar;
}
} // namespace Nektar::MovementTests
