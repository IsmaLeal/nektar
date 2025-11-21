///////////////////////////////////////////////////////////////////////////////
//
// File: TestGetCoords.cpp
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

#include "LocalRegions/PointExp.h"
#include <LocalRegions/HexExp.h>
#include <SpatialDomains/MeshGraph.h>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

namespace Nektar::HexExpTests
{
SpatialDomains::SegGeomUniquePtr CreateSegGeom(unsigned int id,
                                               SpatialDomains::PointGeom *v0,
                                               SpatialDomains::PointGeom *v1)
{
    std::array<SpatialDomains::PointGeom *, 2> vertices = {v0, v1};
    SpatialDomains::SegGeomUniquePtr result(
        new SpatialDomains::SegGeom(id, 3, vertices));
    return result;
}

SpatialDomains::HexGeomUniquePtr CreateHex(
    std::array<SpatialDomains::PointGeom *, 8> v,
    std::array<SpatialDomains::SegGeomUniquePtr, 12> &segVec,
    std::array<SpatialDomains::QuadGeomUniquePtr, 6> &faceVec)
{
    std::array<std::array<int, 2>, 12> edgeVerts = {{{{0, 1}},
                                                     {{1, 2}},
                                                     {{2, 3}},
                                                     {{3, 0}},
                                                     {{0, 4}},
                                                     {{1, 5}},
                                                     {{2, 6}},
                                                     {{3, 7}},
                                                     {{4, 5}},
                                                     {{5, 6}},
                                                     {{6, 7}},
                                                     {{7, 4}}}};
    std::array<std::array<int, 4>, 6> faceEdges  = {{{{0, 1, 2, 3}},
                                                     {{0, 5, 8, 4}},
                                                     {{1, 6, 9, 5}},
                                                     {{2, 6, 10, 7}},
                                                     {{3, 7, 11, 4}},
                                                     {{8, 9, 10, 11}}}};

    // Create segments from vertices
    for (int i = 0; i < 12; ++i)
    {
        segVec[i] = CreateSegGeom(i, v[edgeVerts[i][0]], v[edgeVerts[i][1]]);
    }

    // Create faces from edges
    std::array<SpatialDomains::QuadGeom *, 6> faces;
    for (int i = 0; i < 6; ++i)
    {
        std::array<SpatialDomains::SegGeom *, 4> face;
        for (int j = 0; j < 4; ++j)
        {
            face[j] = segVec[faceEdges[i][j]].get();
        }
        faceVec[i] = SpatialDomains::QuadGeomUniquePtr(
            new SpatialDomains::QuadGeom(i, face));
        faces[i] = faceVec[i].get();
    }

    SpatialDomains::HexGeomUniquePtr hexGeom(
        new SpatialDomains::HexGeom(0, faces));
    return hexGeom;
}

BOOST_AUTO_TEST_CASE(TestHexExpThatIsStdRegion)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.0, -1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v1(
        new SpatialDomains::PointGeom(3u, 1u, 1.0, -1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v2(
        new SpatialDomains::PointGeom(3u, 2u, 1.0, 1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v3(
        new SpatialDomains::PointGeom(3u, 3u, -1.0, 1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v4(
        new SpatialDomains::PointGeom(3u, 4u, -1.0, -1.0, 1.0));
    SpatialDomains::PointGeomUniquePtr v5(
        new SpatialDomains::PointGeom(3u, 5u, 1.0, -1.0, 1.0));
    SpatialDomains::PointGeomUniquePtr v6(
        new SpatialDomains::PointGeom(3u, 6u, 1.0, 1.0, 1.0));
    SpatialDomains::PointGeomUniquePtr v7(
        new SpatialDomains::PointGeom(3u, 7u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::PointGeom *, 8> v = {
        v0.get(), v1.get(), v2.get(), v3.get(),
        v4.get(), v5.get(), v6.get(), v7.get()};
    std::array<SpatialDomains::SegGeomUniquePtr, 12> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 6> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom = CreateHex(v, segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr hexExp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Array<OneD, NekDouble> c0 = Array<OneD, NekDouble>(hexExp->GetTotPoints());
    Array<OneD, NekDouble> c1 = Array<OneD, NekDouble>(hexExp->GetTotPoints());
    Array<OneD, NekDouble> c2 = Array<OneD, NekDouble>(hexExp->GetTotPoints());
    hexExp->GetCoords(c0, c1, c2);
    std::shared_ptr<StdRegions::StdHexExp> stdHex =
        std::dynamic_pointer_cast<StdRegions::StdHexExp>(hexExp);
    stdHex->GetCoords(c0, c1, c2);
    double epsilon = 1.0e-8;
    BOOST_CHECK_CLOSE(c0[0], -1.0, epsilon);
    BOOST_CHECK_CLOSE(c0[1], -0.76505532392946474, epsilon);
    BOOST_CHECK_CLOSE(c0[2], -0.28523151648064510, epsilon);
    BOOST_CHECK_CLOSE(c0[3], 0.28523151648064510, epsilon);
    BOOST_CHECK_CLOSE(c0[4], 0.76505532392946474, epsilon);
    BOOST_CHECK_CLOSE(c0[5], 1.0, epsilon);
}

BOOST_AUTO_TEST_CASE(TestScaledAndTranslatedHexExp)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, 0.0, 0.0, 0.0));
    SpatialDomains::PointGeomUniquePtr v1(
        new SpatialDomains::PointGeom(3u, 1u, 0.5, 0.0, 0.0));
    SpatialDomains::PointGeomUniquePtr v2(
        new SpatialDomains::PointGeom(3u, 2u, 0.5, 0.5, 0.0));
    SpatialDomains::PointGeomUniquePtr v3(
        new SpatialDomains::PointGeom(3u, 3u, 0.0, 0.5, 0.0));
    SpatialDomains::PointGeomUniquePtr v4(
        new SpatialDomains::PointGeom(3u, 4u, 0.0, 0.0, 0.5));
    SpatialDomains::PointGeomUniquePtr v5(
        new SpatialDomains::PointGeom(3u, 5u, 0.5, 0.0, 0.5));
    SpatialDomains::PointGeomUniquePtr v6(
        new SpatialDomains::PointGeom(3u, 6u, 0.5, 0.5, 0.5));
    SpatialDomains::PointGeomUniquePtr v7(
        new SpatialDomains::PointGeom(3u, 7u, 0.0, 0.5, 0.5));

    std::array<SpatialDomains::PointGeom *, 8> v = {
        v0.get(), v1.get(), v2.get(), v3.get(),
        v4.get(), v5.get(), v6.get(), v7.get()};
    std::array<SpatialDomains::SegGeomUniquePtr, 12> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 6> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom = CreateHex(v, segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr hexExp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Array<OneD, NekDouble> c0 = Array<OneD, NekDouble>(hexExp->GetTotPoints());
    Array<OneD, NekDouble> c1 = Array<OneD, NekDouble>(hexExp->GetTotPoints());
    Array<OneD, NekDouble> c2 = Array<OneD, NekDouble>(hexExp->GetTotPoints());
    hexExp->GetCoords(c0, c1, c2);

    double epsilon = 1.0e-8;
    BOOST_CHECK_EQUAL(c0[0], 0.0);
    BOOST_CHECK_CLOSE(c0[1], .05873616902, epsilon);
    BOOST_CHECK_CLOSE(c0[2], .17869212088, epsilon);
    BOOST_CHECK_CLOSE(c0[3], .32130787912, epsilon);
    BOOST_CHECK_CLOSE(c0[4], .44126383098, epsilon);
    BOOST_CHECK_CLOSE(c0[5], .5, epsilon);
}

BOOST_AUTO_TEST_CASE(TestPointExpThatIsStdRegion)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.0, -1.0, -1.0));

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 1;
    unsigned int numModes      = 1;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::PointExpSharedPtr pointExp =
        MemoryManager<Nektar::LocalRegions::PointExp>::AllocateSharedPtr(
            v0.get());

    NekDouble c0 = 0, c1 = 0, c2 = 0;
    pointExp->GetCoords(c0, c1, c2);

    std::shared_ptr<StdRegions::StdPointExp> stdPoint =
        std::dynamic_pointer_cast<StdRegions::StdPointExp>(pointExp);

    Array<OneD, NekDouble> c0_arr =
        Array<OneD, NekDouble>(pointExp->GetTotPoints());
    Array<OneD, NekDouble> c1_arr =
        Array<OneD, NekDouble>(pointExp->GetTotPoints());
    Array<OneD, NekDouble> c2_arr =
        Array<OneD, NekDouble>(pointExp->GetTotPoints());
    stdPoint->GetCoords(c0_arr, c1_arr, c2_arr);

    double epsilon = 1.0e-8;
    BOOST_CHECK_CLOSE(c0, -1.0, epsilon);
    BOOST_CHECK_CLOSE(c0_arr[0], -1.0, epsilon);

    // Get GeomFactors and check it's a nullptr for pointExp
    BOOST_CHECK_EQUAL(pointExp->GetGeomFactors(), nullptr);
}

} // namespace Nektar::HexExpTests
