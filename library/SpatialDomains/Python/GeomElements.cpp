////////////////////////////////////////////////////////////////////////////////
//
//  File: GeomElements.cpp
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
//  Description: Python wrapper for GeomElements.
//
////////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Python/NekPyConfig.hpp>
#include <SpatialDomains/Python/SpatialDomains.h>

#include <SpatialDomains/HexGeom.h>
#include <SpatialDomains/MeshGraph.h>
#include <SpatialDomains/PointGeom.h>
#include <SpatialDomains/PrismGeom.h>
#include <SpatialDomains/PyrGeom.h>
#include <SpatialDomains/QuadGeom.h>
#include <SpatialDomains/SegGeom.h>
#include <SpatialDomains/TetGeom.h>
#include <SpatialDomains/TriGeom.h>

using namespace Nektar;
using namespace Nektar::SpatialDomains;

using NekError = Nektar::ErrorUtil::NekError;

template <class T, class S>
unique_ptr_objpool<T> Geometry_Init(int id, py::list &facets)
{
    std::array<S *, T::kNfacets> geomArr;

    if (py::len(facets) != T::kNfacets)
    {
        throw new NekError("Incorrect number of facets for this geometry");
    }

    for (int i = 0; i < T::kNfacets; ++i)
    {
        geomArr[i] = py::cast<S *>(facets[i]);
    }

    return ObjPoolManager<T>::AllocateUniquePtr(id, geomArr);
}

template <class T, class S>
unique_ptr_objpool<T> Geometry_Init_Curved(int id, py::list &facets,
                                           Curve *curve)
{
    std::array<S *, T::kNfacets> geomArr;

    if (py::len(facets) != T::kNfacets)
    {
        throw new NekError("Incorrect number of facets for this geometry");
    }

    for (int i = 0; i < T::kNfacets; ++i)
    {
        geomArr[i] = py::cast<S *>(facets[i]);
    }

    return ObjPoolManager<T>::AllocateUniquePtr(id, geomArr, curve);
}

template <class T, class S> void export_Geom_2d(py::module &m, const char *name)
{
    py::classh<T, Geometry2D>(m, name)
        .def(py::init<>(&Geometry_Init<T, S>), py::arg("id"),
             py::arg("segments") = py::list())
        .def(py::init<>(&Geometry_Init_Curved<T, S>), py::arg("id"),
             py::arg("segments"), py::arg("curve"));
}

template <class T, class S> void export_Geom_3d(py::module &m, const char *name)
{
    py::classh<T, Geometry3D>(m, name).def(py::init<>(&Geometry_Init<T, S>),
                                           py::arg("id"),
                                           py::arg("segments") = py::list());
}

unique_ptr_objpool<SegGeom> SegGeom_Init(int id, int coordim, py::list &points,
                                         Curve *curve)
{
    std::array<PointGeom *, 2> geomArr;

    if (py::len(points) != 2)
    {
        throw new NekError("Incorrect number of facets for this geometry");
    }

    for (int i = 0; i < 2; ++i)
    {
        geomArr[i] = py::cast<PointGeom *>(points[i]);
    }

    if (!curve)
    {
        return ObjPoolManager<SegGeom>::AllocateUniquePtr(id, coordim, geomArr);
    }
    else
    {
        return ObjPoolManager<SegGeom>::AllocateUniquePtr(id, coordim, geomArr,
                                                          curve);
    }
}

unique_ptr_objpool<PointGeom> PointGeom_Init(int coordim, int id, NekDouble x,
                                             NekDouble y, NekDouble z)
{
    return ObjPoolManager<PointGeom>::AllocateUniquePtr(coordim, id, x, y, z);
}

py::tuple PointGeom_GetCoordinates(const PointGeom &self)
{
    return py::make_tuple(self.x(), self.y(), self.z());
}

void export_GeomElements(py::module &m)
{
    // Geometry dimensioned base classes
    py::classh<Geometry1D, Geometry>(m, "Geometry1D");
    py::classh<Geometry2D, Geometry>(m, "Geometry2D")
        .def("GetCurve", &Geometry2D::GetCurve,
             py::return_value_policy::reference);
    py::classh<Geometry3D, Geometry>(m, "Geometry3D");

    // Point geometries
    py::classh<PointGeom, Geometry>(m, "PointGeom")
        .def(py::init<>(&PointGeom_Init), py::arg("coordim"), py::arg("id"),
             py::arg("x"), py::arg("y"), py::arg("z"))
        .def("GetCoordinates", &PointGeom_GetCoordinates);

    // Segment geometries
    py::classh<SegGeom, Geometry>(m, "SegGeom")
        .def(py::init<>(&SegGeom_Init), py::arg("id"), py::arg("coordim"),
             py::arg("points") = py::list(), py::arg("curve") = nullptr)
        .def("GetCurve", &SegGeom::GetCurve,
             py::return_value_policy::reference);

    export_Geom_2d<TriGeom, SegGeom>(m, "TriGeom");
    export_Geom_2d<QuadGeom, SegGeom>(m, "QuadGeom");
    export_Geom_3d<TetGeom, TriGeom>(m, "TetGeom");
    export_Geom_3d<PrismGeom, Geometry2D>(m, "PrismGeom");
    export_Geom_3d<PyrGeom, Geometry2D>(m, "PyrGeom");
    export_Geom_3d<HexGeom, QuadGeom>(m, "HexGeom");
}
