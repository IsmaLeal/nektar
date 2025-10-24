////////////////////////////////////////////////////////////////////////////////
//
//  File: Movement.cpp
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
//  Software is furnished to do so, subject to the following conditions:
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

#include <algorithm>
#include <memory>
#include <string>

#include <LibUtilities/BasicUtils/ParseUtils.h>
#include <LibUtilities/Interpreter/Interpreter.h>
#include <SpatialDomains/MeshGraph.h>
#include <SpatialDomains/Movement/Movement.h>
#include <tinyxml.h>

namespace Nektar::SpatialDomains
{

std::string static inline ReadTag(std::string &tagStr)
{
    std::string::size_type indxBeg = tagStr.find_first_of('[') + 1;
    std::string::size_type indxEnd = tagStr.find_last_of(']') - 1;

    ASSERTL0(
        indxBeg <= indxEnd,
        (std::string("Error reading interface region definition:") + tagStr)
            .c_str());

    std::string indxStr = tagStr.substr(indxBeg, indxEnd - indxBeg + 1);

    return indxStr;
}

std::string static inline StripParentheses(const std::string &str)
{
    auto length = str.length();
    return str.substr(1, length - 2);
}

Movement::Movement(const LibUtilities::SessionReaderSharedPtr &pSession,
                   MeshGraph *meshGraph)
{
    TiXmlNode *nektar = pSession->GetElement("NEKTAR");
    if (nektar == nullptr)
    {
        return;
    }

    TiXmlNode *movement = nektar->FirstChild("MOVEMENT");
    if (movement != nullptr)
    {
        bool zones = movement->FirstChild("ZONES") != nullptr;
        if (zones)
        {
            TiXmlElement *zonesTag =
                pSession->GetElement("NEKTAR/MOVEMENT/ZONES");
            ReadZones(zonesTag, meshGraph, pSession);
        }

        bool interfaces = movement->FirstChild("INTERFACES") != nullptr;
        if (interfaces)
        {
            TiXmlElement *interfacesTag =
                pSession->GetElement("NEKTAR/MOVEMENT/INTERFACES");
            ReadInterfaces(interfacesTag, meshGraph);
        }

        if (m_translate)
        {
            UpdateTransZoneBox(pSession);
        }
    }

    // DEBUG COMMENTS
    if (movement != nullptr && pSession->DefinesCmdLineArgument("verbose"))
    {
        auto comm = pSession->GetComm();
        if (comm->TreatAsRankZero())
        {
            std::cout << "Movement Info:\n";
            std::cout << "\tNum zones: " << m_zones.size() << "\n";
        }

        for (auto &zone : m_zones)
        {
            auto numEl = zone.second->GetElements().size();
            comm->GetSpaceComm()->AllReduce(numEl, LibUtilities::ReduceSum);

            // Find shape type if not on this proc
            int shapeType =
                !zone.second->GetElements().empty()
                    ? zone.second->GetElements().front()->GetShapeType()
                    : -1;
            comm->GetSpaceComm()->AllReduce(shapeType, LibUtilities::ReduceMax);

            if (comm->TreatAsRankZero())
            {
                std::cout << "\t- " << zone.first << " "
                          << MovementTypeStr[static_cast<int>(
                                 zone.second->GetMovementType())]
                          << ": " << numEl << " "
                          << LibUtilities::ShapeTypeMap[shapeType] << "s\n";
            }
        }

        if (comm->TreatAsRankZero())
        {
            std::cout << "\tNum interfaces: " << m_interfaces.size() << "\n";
        }

        for (auto &interface : m_interfaces)
        {
            auto numLeft =
                interface.second->GetLeftInterface()->GetEdge().size();
            auto numRight =
                interface.second->GetRightInterface()->GetEdge().size();
            comm->GetSpaceComm()->AllReduce(numLeft, LibUtilities::ReduceSum);
            comm->GetSpaceComm()->AllReduce(numRight, LibUtilities::ReduceSum);

            // Find shape type if not on this proc
            int shapeTypeLeft =
                !interface.second->GetLeftInterface()->GetEdge().empty()
                    ? interface.second->GetLeftInterface()
                          ->GetEdge()
                          .begin()
                          ->second->GetShapeType()
                    : -1;
            comm->GetSpaceComm()->AllReduce(shapeTypeLeft,
                                            LibUtilities::ReduceMax);
            int shapeTypeRight =
                !interface.second->GetRightInterface()->GetEdge().empty()
                    ? interface.second->GetRightInterface()
                          ->GetEdge()
                          .begin()
                          ->second->GetShapeType()
                    : -1;
            comm->GetSpaceComm()->AllReduce(shapeTypeRight,
                                            LibUtilities::ReduceMax);

            if (comm->TreatAsRankZero())
            {
                std::cout << "\t- \"" << interface.first.second << "\": "
                          << interface.second->GetLeftInterface()->GetId()
                          << " (" << numLeft << " "
                          << LibUtilities::ShapeTypeMap[shapeTypeLeft]
                          << "s) <-> "
                          << interface.second->GetRightInterface()->GetId()
                          << " (" << numRight << " "
                          << LibUtilities::ShapeTypeMap[shapeTypeRight]
                          << "s)\n";
            }
        }

        comm->GetSpaceComm()->Block();
        if (comm->TreatAsRankZero())
        {
            std::cout << std::endl;
        }
    }
}

void Movement::ReadZones(TiXmlElement *zonesTag, MeshGraph *meshGraph,
                         const LibUtilities::SessionReaderSharedPtr &pSession)
{
    int coordDim = meshGraph->GetSpaceDimension();

    ASSERTL0(zonesTag, "Unable to find ZONES tag in file.");
    TiXmlElement *zonesElement = zonesTag->FirstChildElement();
    while (zonesElement)
    {
        std::string zoneType = zonesElement->Value();

        int err;
        int indx;

        err = zonesElement->QueryIntAttribute("ID", &indx);
        ASSERTL0(err == TIXML_SUCCESS, "Unable to read zone ID.");

        std::string interfaceDomainStr;
        err = zonesElement->QueryStringAttribute("DOMAIN", &interfaceDomainStr);
        ASSERTL0(err == TIXML_SUCCESS, "Unable to read zone domain.");

        auto &domains = meshGraph->GetDomain();
        auto domFind  = std::stoi(ReadTag(interfaceDomainStr));
        std::map<int, CompositeSharedPtr> domain;
        if (domains.find(domFind) != domains.end())
        {
            domain = domains.at(domFind);
        }

        ZoneBaseShPtr zone;
        if (zoneType == "F" || zoneType == "FIXED")
        {
            zone = ZoneFixedShPtr(MemoryManager<ZoneFixed>::AllocateSharedPtr(
                indx, domFind, domain, coordDim));
        }
        else if (zoneType == "R" || zoneType == "ROTATE" ||
                 zoneType == "ROTATING")
        {
            std::string originStr;
            err = zonesElement->QueryStringAttribute("ORIGIN", &originStr);
            ASSERTL0(err == TIXML_SUCCESS, "Unable to read origin.");
            std::vector<NekDouble> originVec;
            ParseUtils::GenerateVector(originStr, originVec);
            NekPoint<NekDouble> origin =
                NekPoint<NekDouble>(originVec[0], originVec[1], originVec[2]);

            std::string axisStr;
            err = zonesElement->QueryStringAttribute("AXIS", &axisStr);
            ASSERTL0(err == TIXML_SUCCESS, "Unable to read axis.");
            DNekVec axis(axisStr);
            axis.Normalize();

            std::string angularVelStr;
            err = zonesElement->QueryStringAttribute("ANGVEL", &angularVelStr);
            ASSERTL0(err == TIXML_SUCCESS, "Unable to read angular velocity.");

            LibUtilities::EquationSharedPtr angularVelEqn =
                MemoryManager<LibUtilities::Equation>::AllocateSharedPtr(
                    pSession->GetInterpreter(), angularVelStr);

            std::string rampTimeStr;
            err = zonesElement->QueryStringAttribute("RAMPTIME", &rampTimeStr);
            NekDouble rampTime = 0.0;
            if (err == TIXML_SUCCESS)
            {
                LibUtilities::Interpreter expEvaluator;
                int expr_id = expEvaluator.DefineFunction("", rampTimeStr);
                rampTime    = expEvaluator.Evaluate(expr_id);
            }

            std::string sectorStr;
            err = zonesElement->QueryStringAttribute("SECTOR", &sectorStr);
            NekDouble sector = 0.0;
            Array<OneD, NekDouble> base(3);
            if (err == TIXML_SUCCESS)
            {

                LibUtilities::Interpreter expEvaluator;
                int expr_id = expEvaluator.DefineFunction("", sectorStr);
                sector      = expEvaluator.Evaluate(expr_id);

                std::string baseStr;
                err = zonesElement->QueryStringAttribute("BASE", &baseStr);
                ASSERTL0(err == TIXML_SUCCESS, "Unable to read base.");
                std::vector<NekDouble> baseVec;
                ParseUtils::GenerateVector(baseStr, baseVec);
                // Ensure the vector has exactly 3 elements before assignment
                ASSERTL0(baseVec.size() == 3,
                         "BASE attribute must have exactly 3 elements.");
                for (int i = 0; i < 3; ++i)
                {
                    base[i] = baseVec[i];
                }
                m_sectorRotate = true;
            }

            zone = ZoneRotateShPtr(MemoryManager<ZoneRotate>::AllocateSharedPtr(
                indx, domFind, domain, coordDim, origin, axis, angularVelEqn,
                rampTime, sector, base));

            m_moveFlag = true;
            m_rotate   = true;
        }
        else if (zoneType == "T" || zoneType == "TRANSLATE" ||
                 zoneType == "TRANSLATING")
        {
            // Read velocity information
            std::string velocityStr;
            err = zonesElement->QueryStringAttribute("VELOCITY", &velocityStr);
            ASSERTL0(err == TIXML_SUCCESS,
                     "Unable to read VELOCITY for translating zone " +
                         std::to_string(indx));

            std::vector<std::string> velocityStrSplit;
            ParseUtils::GenerateVector(velocityStr, velocityStrSplit);

            ASSERTL0(velocityStrSplit.size() >= coordDim,
                     "VELOCITY dimensions must be greater than or equal to the "
                     "coordinate dimension for translating zone " +
                         std::to_string(indx));

            Array<OneD, LibUtilities::EquationSharedPtr> velocityEqns(coordDim);
            for (int i = 0; i < coordDim; ++i)
            {
                velocityEqns[i] =
                    MemoryManager<LibUtilities::Equation>::AllocateSharedPtr(
                        pSession->GetInterpreter(), velocityStrSplit[i]);
            }

            // Read displacement information
            std::string displacementStr;
            err = zonesElement->QueryStringAttribute("DISPLACEMENT",
                                                     &displacementStr);
            ASSERTL0(err == TIXML_SUCCESS,
                     "Unable to read DISPLACEMENT for translating zone " +
                         std::to_string(indx));

            std::vector<std::string> displacementStrSplit;
            ParseUtils::GenerateVector(displacementStr, displacementStrSplit);

            ASSERTL0(
                displacementStrSplit.size() >= coordDim,
                "DISPLACEMENT dimensions must be greater than or equal to the "
                "coordinate dimension for translating zone " +
                    std::to_string(indx));

            Array<OneD, LibUtilities::EquationSharedPtr> displacementEqns(
                coordDim);
            for (int i = 0; i < coordDim; ++i)
            {
                displacementEqns[i] =
                    MemoryManager<LibUtilities::Equation>::AllocateSharedPtr(
                        pSession->GetInterpreter(), displacementStrSplit[i]);
            }

            zone = ZoneTranslateShPtr(
                MemoryManager<ZoneTranslate>::AllocateSharedPtr(
                    indx, domFind, domain, coordDim, velocityEqns,
                    displacementEqns));

            m_moveFlag  = true;
            m_translate = true;
        }
        else
        {
            WARNINGL0(false, "Zone type '" + zoneType +
                                 "' is unsupported. Valid types are: 'Fixed', "
                                 "'Rotate', or 'Translate'.")
        }

        m_zones[indx] = zone;
        zonesElement  = zonesElement->NextSiblingElement();
    }
}

void Movement::ReadInterfaces(TiXmlElement *interfacesTag, MeshGraph *meshGraph)
{
    ASSERTL0(interfacesTag, "Unable to find INTERFACES tag in file.");
    TiXmlElement *interfaceElement = interfacesTag->FirstChildElement();

    while (interfaceElement)
    {
        ASSERTL0(
            "INTERFACE" == (std::string)interfaceElement->Value(),
            "Only INTERFACE tags may be present inside the INTERFACES block.")

        int err;

        std::string name;
        err = interfaceElement->QueryStringAttribute("NAME", &name);
        ASSERTL0(err == TIXML_SUCCESS, "Unable to read interface name.");
        TiXmlElement *sideElement = interfaceElement->FirstChildElement();

        Array<OneD, InterfaceShPtr> interfaces(2);
        std::vector<int> cnt;
        while (sideElement)
        {
            ASSERTL0(
                cnt.size() < 2,
                "In INTERFACE NAME " + name +
                    ", only two sides may be present in each INTERFACE block.")

            // Read ID
            int indx;
            err = sideElement->QueryIntAttribute("ID", &indx);
            ASSERTL0(err == TIXML_SUCCESS, "Unable to read interface ID.");

            // Read boundary
            std::string boundaryStr;
            err = sideElement->QueryStringAttribute("BOUNDARY", &boundaryStr);
            ASSERTL0(err == TIXML_SUCCESS,
                     "Unable to read BOUNDARY in interface " +
                         std::to_string(indx));

            CompositeMap boundaryEdge;
            std::string indxStr = ReadTag(boundaryStr);
            meshGraph->GetCompositeList(indxStr, boundaryEdge);

            // Read SKIPCHECK flag
            bool skipCheck = false;
            err = sideElement->QueryBoolAttribute("SKIPCHECK", &skipCheck);

            // Sets location in interface pair to 0 for left, and 1 for right
            auto sideElVal = sideElement->ValueStr();
            if (sideElVal == "LEFT" || sideElVal == "L")
            {
                cnt.emplace_back(0);
            }
            else if (sideElVal == "RIGHT" || sideElVal == "R")
            {
                cnt.emplace_back(1);
            }
            else
            {
                NEKERROR(ErrorUtil::efatal,
                         sideElement->ValueStr() +
                             " is not a valid interface side for interface "
                             "NAME " +
                             name + ". Please only use LEFT or RIGHT.")
            }

            interfaces[cnt[cnt.size() - 1]] =
                InterfaceShPtr(MemoryManager<Interface>::AllocateSharedPtr(
                    indx, boundaryEdge, skipCheck));

            sideElement = sideElement->NextSiblingElement();
        }

        ASSERTL0(std::accumulate(cnt.begin(), cnt.end(), 0) == 1,
                 "You must have only one LEFT and one RIGHT side"
                 " present in interface NAME " +
                     name)

        m_interfaces[std::make_pair(m_interfaces.size(), name)] =
            InterfacePairShPtr(MemoryManager<InterfacePair>::AllocateSharedPtr(
                interfaces[0], interfaces[1]));
        interfaceElement = interfaceElement->NextSiblingElement();
    }
}

/// Export this Movement information to a Nektar++ XML file.
void Movement::WriteMovement(TiXmlElement *root)
{
    if (m_zones.size() == 0 && m_interfaces.size() == 0)
    {
        return;
    }
    TiXmlElement *movement = new TiXmlElement("MOVEMENT");
    root->LinkEndChild(movement);

    TiXmlElement *zones = new TiXmlElement("ZONES");
    for (auto &i : m_zones)
    {
        const ZoneBaseShPtr z    = i.second;
        const MovementType mtype = z->GetMovementType();
        std::string label        = MovementTypeStr[static_cast<int>(mtype)];
        TiXmlElement *e          = new TiXmlElement(label.substr(0, 1));
        e->SetAttribute("ID", i.first);
        std::stringstream s;
        s << "D[" << z->GetDomainID() << "]";
        e->SetAttribute("DOMAIN", s.str());

        switch (mtype)
        {
            case MovementType::eRotate:
            {
                auto rotate = std::static_pointer_cast<ZoneRotate>(z);
                e->SetAttribute(
                    "ORIGIN", StripParentheses(rotate->GetOrigin().AsString()));
                e->SetAttribute("AXIS",
                                StripParentheses(rotate->GetAxis().AsString()));
                e->SetAttribute("ANGVEL",
                                rotate->GetAngularVelEqn()->GetExpression());
            }
            break;
            case MovementType::eTranslate:
            {
                auto translate = std::static_pointer_cast<ZoneTranslate>(z);
                e->SetAttribute(
                    "XVELOCITY",
                    translate->GetVelocityEquation()[0]->GetExpression());
                e->SetAttribute(
                    "XDISPLACEMENT",
                    translate->GetDisplacementEquation()[0]->GetExpression());
                e->SetAttribute(
                    "YVELOCITY",
                    translate->GetVelocityEquation()[1]->GetExpression());
                e->SetAttribute(
                    "YDISPLACEMENT",
                    translate->GetDisplacementEquation()[1]->GetExpression());
                e->SetAttribute(
                    "ZVELOCITY",
                    translate->GetVelocityEquation()[2]->GetExpression());
                e->SetAttribute(
                    "ZDISPLACEMENT",
                    translate->GetDisplacementEquation()[2]->GetExpression());
            }
            break;
            default:
                break;
        }
        zones->LinkEndChild(e);
    }
    movement->LinkEndChild(zones);

    TiXmlElement *interfaces = new TiXmlElement("INTERFACES");
    for (auto &i : m_interfaces)
    {
        const std::string interfaceName = i.first.second;
        TiXmlElement *e                 = new TiXmlElement("INTERFACE");
        e->SetAttribute("NAME", interfaceName);
        const InterfaceShPtr left  = i.second->GetLeftInterface();
        const InterfaceShPtr right = i.second->GetRightInterface();
        if (left)
        {
            TiXmlElement *left_e = new TiXmlElement("L");
            left_e->SetAttribute("ID", left->GetId());
            left_e->SetAttribute(
                "BOUNDARY",
                "C[" + ParseUtils::GenerateSeqString(left->GetCompositeIDs()) +
                    "]");
            e->LinkEndChild(left_e);
        }
        if (right)
        {
            TiXmlElement *right_e = new TiXmlElement("R");
            right_e->SetAttribute("ID", right->GetId());
            right_e->SetAttribute(
                "BOUNDARY",
                "C[" + ParseUtils::GenerateSeqString(right->GetCompositeIDs()) +
                    "]");
            e->LinkEndChild(right_e);
        }
        interfaces->LinkEndChild(e);
    }
    movement->LinkEndChild(interfaces);
}

// Acts as a placeholder for when ALE function and moving geometry capability
// is added. Currently unused.
void Movement::PerformMovement(NekDouble timeStep)
{
    std::set<int> movedZoneIds;
    for (auto &zone : m_zones)
    {
        if (zone.second->Move(timeStep))
        {
            movedZoneIds.insert(zone.first);
        }
    }

    // If zone has moved, set all interfaces on that zone to moved.
    for (auto &interPair : m_interfaces)
    {
        int leftId  = interPair.second->GetLeftInterface()->GetId();
        int rightId = interPair.second->GetRightInterface()->GetId();

        if (movedZoneIds.find(leftId) != movedZoneIds.end() ||
            movedZoneIds.find(rightId) != movedZoneIds.end())
        {
            m_zones[leftId]->GetMoved()  = true;
            m_zones[rightId]->GetMoved() = true;
        }
    }
    m_moved = true;
}

/// Store a zone object with this Movement data
void Movement::AddZone(ZoneBaseShPtr zone)
{
    m_zones[zone->GetId()] = zone;
    MovementType mtype     = zone->GetMovementType();
    if (mtype != MovementType::eFixed && mtype != MovementType::eNone)
    {
        m_moveFlag = true;
    }
}

/// Store an interface pair with this Movement data
void Movement::AddInterface(std::string name, InterfaceShPtr left,
                            InterfaceShPtr right)
{
    m_interfaces[std::make_pair(m_interfaces.size(), name)] =
        InterfacePairShPtr(
            MemoryManager<InterfacePair>::AllocateSharedPtr(left, right));
}

/// Generate domain box for translation mesh
void Movement::UpdateTransZoneBox(
    const LibUtilities::SessionReaderSharedPtr &pSession)
{
    // Loop over all translate zones to get the box of the whole zone
    for (auto &zones : m_zones)
    {
        if (zones.second->GetMovementType() == eTranslate)
        {
            auto zone = std::static_pointer_cast<ZoneTranslate>(zones.second);
            Array<OneD, NekDouble> ZoneBox(6);
            Array<OneD, NekDouble> ZoneLength(3);
            ZoneBox    = zone->GetZoneBox();
            ZoneLength = zone->GetZoneLength();
            auto comm  = pSession->GetComm();
            for (int i = 0; i < 3; ++i)
            {
                auto &min = ZoneBox[i];
                auto &max = ZoneBox[i + 3];
                comm->GetSpaceComm()->AllReduce(min, LibUtilities::ReduceMin);
                comm->GetSpaceComm()->AllReduce(max, LibUtilities::ReduceMax);
                ZoneLength[i] = max - min;
            }
            // Update the zone box for all translate zones
            zone->UpdateZoneBox(ZoneBox, ZoneLength);
        }
    }
}

} // namespace Nektar::SpatialDomains
