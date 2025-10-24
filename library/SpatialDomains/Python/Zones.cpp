////////////////////////////////////////////////////////////////////////////////
//
//  File: Zones.cpp
//
//  For more information, please see: http://www.nektar.info/
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
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
//  Description: Python wrapper for Zones.
//
////////////////////////////////////////////////////////////////////////////////

#include <SpatialDomains/Movement/Zones.h>

#include <LibUtilities/Python/BasicUtils/SharedArray.hpp>
#include <LibUtilities/Python/NekPyConfig.hpp>
#include <SpatialDomains/Python/SpatialDomains.h>

using namespace Nektar;
using namespace Nektar::SpatialDomains;

ZoneRotateShPtr ZoneRotate_Init(
    int id, int domainID, const CompositeMap &domain, const int coordDim,
    const NekPoint<NekDouble> &origin, const DNekVec &axis,
    const LibUtilities::EquationSharedPtr &angularVelEqn,
    const NekDouble rampTime, const NekDouble sector,
    const Array<OneD, NekDouble> &base)
{
    return std::make_shared<ZoneRotate>(id, domainID, domain, coordDim, origin,
                                        axis, angularVelEqn, rampTime, sector,
                                        base);
}

ZoneTranslateShPtr ZoneTranslate_Init(
    int id, int domainID, const CompositeMap &domain, const int coordDim,
    const Array<OneD, LibUtilities::EquationSharedPtr> &velocityEqns,
    const Array<OneD, LibUtilities::EquationSharedPtr> &displacementEqns)
{
    return std::make_shared<ZoneTranslate>(id, domainID, domain, coordDim,
                                           velocityEqns, displacementEqns);
}

ZoneFixedShPtr ZoneFixed_Init(int id, int domainID, const CompositeMap &domain,
                              const int coordDim)
{
    return std::make_shared<ZoneFixed>(id, domainID, domain, coordDim);
}

void export_Zones(py::module &m)
{
    NEKPY_WRAP_ENUM_STRING(m, MovementType, MovementTypeStr);

    py::class_<ZoneBase, std::shared_ptr<ZoneBase>>(m, "ZoneBase")
        .def("GetMovementType", &ZoneBase::GetMovementType,
             py::return_value_policy::reference_internal)
        .def("GetDomain", &ZoneBase::GetDomain,
             py::return_value_policy::reference_internal)
        .def("GetId", &ZoneBase::GetId, py::return_value_policy::copy)
        .def("GetDomainID", &ZoneBase::GetDomainID,
             py::return_value_policy::copy)
        .def("Move", &ZoneBase::Move)
        .def("GetElements", &ZoneBase::GetElements,
             py::return_value_policy::reference_internal)
        .def("GetMoved", &ZoneBase::GetMoved, py::return_value_policy::copy)
        .def("ClearBoundingBoxes", &ZoneBase::ClearBoundingBoxes);

    py::class_<ZoneRotate, ZoneBase, std::shared_ptr<ZoneRotate>>(m,
                                                                  "ZoneRotate")
        .def(py::init<>(&ZoneRotate_Init))
        .def("GetAngualrVel", &ZoneRotate::GetAngularVel)
        .def("GetOrigin", &ZoneRotate::GetOrigin, py::return_value_policy::copy)
        .def("GetAxis", &ZoneRotate::GetAxis, py::return_value_policy::copy)
        .def("GetAngularVelEqn", &ZoneRotate::GetAngularVelEqn);

    py::class_<ZoneTranslate, ZoneBase, std::shared_ptr<ZoneTranslate>>(
        m, "ZoneTranslate")
        .def(py::init<>(&ZoneTranslate_Init))
        .def("GetVelocityEquation", &ZoneTranslate::GetVelocityEquation)
        .def("GetDisplacementEqn", &ZoneTranslate::GetDisplacementEquation);

    py::class_<ZoneFixed, ZoneBase, std::shared_ptr<ZoneFixed>>(m, "ZoneFixed")
        .def(py::init<>(&ZoneFixed_Init));
}

//
