////////////////////////////////////////////////////////////////////////////////
//
//  File: Curve.cpp
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
//  Description: Python wrapper for Curve.
//
////////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Python/NekPyConfig.hpp>
#include <SpatialDomains/Curve.hpp>
#include <SpatialDomains/MeshGraph.h>

using namespace Nektar;
using namespace Nektar::SpatialDomains;

unique_ptr_objpool<Curve> Curve_Init(int curveID, LibUtilities::PointsType type)
{
    return ObjPoolManager<Curve>::AllocateUniquePtr(curveID, type);
}

py::list Curve_GetPoints(Curve *curve)
{
    py::list ret;
    for (auto &pt : curve->m_points)
    {
        ret.append(pt);
    }
    return ret;
}

void Curve_SetPoints(Curve *curve, py::list &pts)
{
    auto n = pts.size();

    for (py::ssize_t i = 0; i < n; ++i)
    {
        curve->m_points.push_back(py::cast<PointGeom *>(pts[i]));
    }
}

void export_Curve(py::module &m)
{
    py::classh<Curve>(m, "Curve")

        .def(py::init<>(&Curve_Init), py::arg("curveID"), py::arg("type"))

        .def_readwrite("curveID", &Curve::m_curveID)
        .def_readwrite("ptype", &Curve::m_ptype)
        .def_property("points", &Curve_GetPoints, &Curve_SetPoints);
}
