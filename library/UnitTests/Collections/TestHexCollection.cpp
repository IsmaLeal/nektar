///////////////////////////////////////////////////////////////////////////////
//
// File: TestHexCollection.cpp
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
/// The above copyright notice and this permission notice shall be included
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

#include <Collections/Collection.h>
#include <Collections/CollectionOptimisation.h>
#include <LocalRegions/HexExp.h>
#include <SpatialDomains/MeshGraph.h>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
namespace Nektar::HexCollectionTests
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

BOOST_AUTO_TEST_CASE(TestHexBwdTrans_StdMat_UniformP)
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

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eStdMat);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(Exp->GetNcoeffs(), 1.0), tmp;
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> phys1(Exp->GetTotPoints());
    Array<OneD, NekDouble> phys2(Exp->GetTotPoints());

    Exp->BwdTrans(coeffs, phys1);
    c.ApplyOperator(Collections::eBwdTrans, coeffs, phys2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < phys1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(phys1[i], phys2[i], epsilon);
    }
}
#if 0
BOOST_AUTO_TEST_CASE(TestHexBwdTrans_StdMat_VariableP)
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(7,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(9,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eStdMat);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(Exp->GetNcoeffs(), 1.0), tmp;
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> phys1(Exp->GetTotPoints());
    Array<OneD, NekDouble> phys2(Exp->GetTotPoints());

    Exp->BwdTrans(coeffs, phys1);
    c.ApplyOperator(Collections::eBwdTrans, coeffs, phys2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < phys1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(phys1[i], phys2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexBwdTrans_IterPerExp_UniformP)
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(Exp->GetNcoeffs(), 1.0), tmp;
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> phys1(Exp->GetTotPoints());
    Array<OneD, NekDouble> phys2(Exp->GetTotPoints());

    Exp->BwdTrans(coeffs, phys1);
    c.ApplyOperator(Collections::eBwdTrans, coeffs, phys2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < phys1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(phys1[i], phys2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexBwdTrans_IterPerExp_VariableP)
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(Exp->GetNcoeffs(), 1.0), tmp;
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> phys1(Exp->GetTotPoints());
    Array<OneD, NekDouble> phys2(Exp->GetTotPoints());

    Exp->BwdTrans(coeffs, phys1);
    c.ApplyOperator(Collections::eBwdTrans, coeffs, phys2);
    c.Initialise(Collections::eBwdTrans);

    double epsilon = 1.0e-8;
    for (int i = 0; i < phys1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(phys1[i], phys2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexBwdTrans_IterPerExp_VariableP_MultiElmt)
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(nelmts * Exp->GetNcoeffs(), 1.0), tmp;
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> phys1(nelmts * Exp->GetTotPoints());
    Array<OneD, NekDouble> phys2(nelmts * Exp->GetTotPoints());

    for (int i = 0; i < nelmts; ++i)
    {
        Exp->BwdTrans(coeffs + i * Exp->GetNcoeffs(),
                      tmp = phys1 + i * Exp->GetTotPoints());
    }

    c.ApplyOperator(Collections::eBwdTrans, coeffs, phys2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < phys1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(phys1[i], phys2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexBwdTrans_NoCollection_VariableP)
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eNoCollection);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(Exp->GetNcoeffs(), 1.0), tmp;
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> phys1(Exp->GetTotPoints());
    Array<OneD, NekDouble> phys2(Exp->GetTotPoints());

    Exp->BwdTrans(coeffs, phys1);

    c.ApplyOperator(Collections::eBwdTrans, coeffs, phys2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < phys1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(phys1[i], phys2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexBwdTrans_SumFac_UniformP)
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    int nelmts = 1;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(nelmts * Exp->GetNcoeffs(), 1.0), tmp;
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> phys1(nelmts * Exp->GetTotPoints());
    Array<OneD, NekDouble> phys2(nelmts * Exp->GetTotPoints());

    for (int i = 0; i < nelmts; ++i)
    {
        Exp->BwdTrans(coeffs + i * Exp->GetNcoeffs(),
                      tmp = phys1 + i * Exp->GetTotPoints());
    }
    c.ApplyOperator(Collections::eBwdTrans, coeffs, phys2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < phys1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(phys1[i], phys2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexBwdTrans_SumFac_UniformP_MultiElmt)
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(nelmts * Exp->GetNcoeffs(), 1.0), tmp;
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> phys1(nelmts * Exp->GetTotPoints());
    Array<OneD, NekDouble> phys2(nelmts * Exp->GetTotPoints());

    for (int i = 0; i < nelmts; ++i)
    {
        Exp->BwdTrans(coeffs + i * Exp->GetNcoeffs(),
                      tmp = phys1 + i * Exp->GetTotPoints());
    }
    c.ApplyOperator(Collections::eBwdTrans, coeffs, phys2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < phys1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(phys1[i], phys2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexBwdTrans_SumFac_VariableP)
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(7,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(9,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    int nelmts = 1;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(nelmts * Exp->GetNcoeffs(), 1.0), tmp;
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> phys1(nelmts * Exp->GetTotPoints());
    Array<OneD, NekDouble> phys2(nelmts * Exp->GetTotPoints());

    for (int i = 0; i < nelmts; ++i)
    {
        Exp->BwdTrans(coeffs + i * Exp->GetNcoeffs(),
                      tmp = phys1 + i * Exp->GetTotPoints());
    }
    c.ApplyOperator(Collections::eBwdTrans, coeffs, phys2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < phys1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(phys1[i], phys2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexBwdTrans_SumFac_VariableP_MultiElmt)
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(7,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(9,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(nelmts * Exp->GetNcoeffs(), 1.0), tmp;
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> phys1(nelmts * Exp->GetTotPoints());
    Array<OneD, NekDouble> phys2(nelmts * Exp->GetTotPoints());

    for (int i = 0; i < nelmts; ++i)
    {
        Exp->BwdTrans(coeffs + i * Exp->GetNcoeffs(),
                      tmp = phys1 + i * Exp->GetTotPoints());
    }
    c.ApplyOperator(Collections::eBwdTrans, coeffs, phys2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < phys1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(phys1[i], phys2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexBwdTrans_MatrixFree_UniformP)
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    unsigned int numModes      = 4;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eMatrixFree);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(Exp->GetNcoeffs(), 1.0), tmp;
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> physRef(Exp->GetTotPoints());
    Array<OneD, NekDouble> phys(Exp->GetTotPoints());

    Exp->BwdTrans(coeffs, physRef);
    c.ApplyOperator(Collections::eBwdTrans, coeffs, phys);

    double epsilon = 1.0e-8;
    for (int i = 0; i < physRef.size(); ++i)
    {
        BOOST_CHECK_CLOSE(physRef[i], phys[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTBase_StdMat_UniformP)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eStdMat);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nq);
    Array<OneD, NekDouble> coeffs1(Exp->GetNcoeffs());
    Array<OneD, NekDouble> coeffs2(Exp->GetNcoeffs());

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }

    Exp->IProductWRTBase(phys, coeffs1);
    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTBase_MatrixFree_UniformP_Undeformed)
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 5;
    unsigned int numModes      = 4;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eMatrixFree);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nq);
    Array<OneD, NekDouble> coeffsRef(Exp->GetNcoeffs());
    Array<OneD, NekDouble> coeffs(Exp->GetNcoeffs());

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }

    Exp->IProductWRTBase(phys, coeffsRef);
    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffsRef.size(); ++i)
    {
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTBase_MatrixFree_UniformP_Deformed)
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
        new SpatialDomains::PointGeom(3u, 6u, 2.0, 3.0, 4.0));
    SpatialDomains::PointGeomUniquePtr v7(
        new SpatialDomains::PointGeom(3u, 7u, -1.0, 1.0, 1.0));

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 5;
    unsigned int numModes      = 4;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eMatrixFree);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nq);
    Array<OneD, NekDouble> coeffsRef(Exp->GetNcoeffs());
    Array<OneD, NekDouble> coeffs(Exp->GetNcoeffs());

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }

    Exp->IProductWRTBase(phys, coeffsRef);
    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffsRef.size(); ++i)
    {
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(
    TestHexIProductWRTBase_MatrixFree_UniformP_Deformed_OverInt)
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
        new SpatialDomains::PointGeom(3u, 6u, 2.0, 3.0, 4.0));
    SpatialDomains::PointGeomUniquePtr v7(
        new SpatialDomains::PointGeom(3u, 7u, -1.0, 1.0, 1.0));

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 8;
    unsigned int numModes      = 4;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eMatrixFree);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nq);
    Array<OneD, NekDouble> coeffsRef(Exp->GetNcoeffs());
    Array<OneD, NekDouble> coeffs(Exp->GetNcoeffs());

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }

    Exp->IProductWRTBase(phys, coeffsRef);
    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffsRef.size(); ++i)
    {
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTBase_StdMat_VariableP)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(7,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(9,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eStdMat);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nq);
    Array<OneD, NekDouble> coeffs1(Exp->GetNcoeffs());
    Array<OneD, NekDouble> coeffs2(Exp->GetNcoeffs());

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }

    Exp->IProductWRTBase(phys, coeffs1);
    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs2);
    c.Initialise(Collections::eIProductWRTBase);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTBase_NoCollection_VariableP)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(7,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(9,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eNoCollection);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nq);
    Array<OneD, NekDouble> coeffs1(Exp->GetNcoeffs());
    Array<OneD, NekDouble> coeffs2(Exp->GetNcoeffs());

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }

    Exp->IProductWRTBase(phys, coeffs1);
    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs2);
    c.Initialise(Collections::eIProductWRTBase);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTBase_SumFac_VariableP_CollAll)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eGLL_Lagrange;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(4,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(6,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(8,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nq);
    Array<OneD, NekDouble> coeffs1(Exp->GetNcoeffs());
    Array<OneD, NekDouble> coeffs2(Exp->GetNcoeffs());

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }

    Exp->IProductWRTBase(phys, coeffs1);
    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTBase_SumFac_VariableP_CollDir02)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eGLL_Lagrange;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(4,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(7,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(8,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nq);
    Array<OneD, NekDouble> coeffs1(Exp->GetNcoeffs());
    Array<OneD, NekDouble> coeffs2(Exp->GetNcoeffs());

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }

    Exp->IProductWRTBase(phys, coeffs1);
    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTBase_SumFac_VariableP_CollDir12)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eGLL_Lagrange;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(6,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(8,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nq);
    Array<OneD, NekDouble> coeffs1(Exp->GetNcoeffs());
    Array<OneD, NekDouble> coeffs2(Exp->GetNcoeffs());

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }

    Exp->IProductWRTBase(phys, coeffs1);
    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTBase_StdMat_VariableP_MultiElmt)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(7,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(9,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eStdMat);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nelmts * nq), tmp;
    Array<OneD, NekDouble> coeffs1(nelmts * Exp->GetNcoeffs());
    Array<OneD, NekDouble> coeffs2(nelmts * Exp->GetNcoeffs());

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }
    Exp->IProductWRTBase(phys, coeffs1);

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, &phys[0], 1, &phys[i * nq], 1);
        Exp->IProductWRTBase(phys + i * nq,
                             tmp = coeffs1 + i * Exp->GetNcoeffs());
    }

    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTBase_IterPerExp_VariableP_MultiElmt)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(7,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(9,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nelmts * nq), tmp;
    Array<OneD, NekDouble> coeffs1(nelmts * Exp->GetNcoeffs());
    Array<OneD, NekDouble> coeffs2(nelmts * Exp->GetNcoeffs());

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }
    Exp->IProductWRTBase(phys, coeffs1);

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, &phys[0], 1, &phys[i * nq], 1);
        Exp->IProductWRTBase(phys + i * nq,
                             tmp = coeffs1 + i * Exp->GetNcoeffs());
    }

    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTBase_SumFac_UniformP)
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nq);
    Array<OneD, NekDouble> coeffs1(Exp->GetNcoeffs());
    Array<OneD, NekDouble> coeffs2(Exp->GetNcoeffs());

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }

    Exp->IProductWRTBase(phys, coeffs1);
    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs2);

    double epsilon = 1.0e-6;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        // clamp values below 1e-16 to zero
        coeffs1[i] = (std::abs(coeffs1[i]) < 1e-16) ? 0.0 : coeffs1[i];
        coeffs2[i] = (std::abs(coeffs2[i]) < 1e-16) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTBase_SumFac_VariableP)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(7,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(9,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nq);
    Array<OneD, NekDouble> coeffs1(Exp->GetNcoeffs());
    Array<OneD, NekDouble> coeffs2(Exp->GetNcoeffs());

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }

    Exp->IProductWRTBase(phys, coeffs1);
    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs2);

    double epsilon = 1.0e-6;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        // clamp values below 1e-16 to zero
        coeffs1[i] = (std::abs(coeffs1[i]) < 1e-16) ? 0.0 : coeffs1[i];
        coeffs2[i] = (std::abs(coeffs2[i]) < 1e-16) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTBase_SumFac_UniformP_MultiElmt)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nelmts * nq), tmp;
    Array<OneD, NekDouble> coeffs1(nelmts * Exp->GetNcoeffs());
    Array<OneD, NekDouble> coeffs2(nelmts * Exp->GetNcoeffs());

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }
    Exp->IProductWRTBase(phys, coeffs1);

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, &phys[0], 1, &phys[i * nq], 1);
        Exp->IProductWRTBase(phys + i * nq,
                             tmp = coeffs1 + i * Exp->GetNcoeffs());
    }
    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs2);

    double epsilon = 1.0e-6;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        // clamp values below 1e-16 to zero
        coeffs1[i] = (std::abs(coeffs1[i]) < 1e-16) ? 0.0 : coeffs1[i];
        coeffs2[i] = (std::abs(coeffs2[i]) < 1e-16) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTBase_SumFac_VariableP_MultiElmt)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(7,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(9,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nelmts * nq), tmp;
    Array<OneD, NekDouble> coeffs1(nelmts * Exp->GetNcoeffs());
    Array<OneD, NekDouble> coeffs2(nelmts * Exp->GetNcoeffs());

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }
    Exp->IProductWRTBase(phys, coeffs1);

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, &phys[0], 1, &phys[i * nq], 1);
        Exp->IProductWRTBase(phys + i * nq,
                             tmp = coeffs1 + i * Exp->GetNcoeffs());
    }
    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs2);

    double epsilon = 1.0e-4;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        // clamp values below 1e-14 to zero
        coeffs1[i] = (std::abs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (std::abs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(
    TestHexIProductWRTBase_SumFac_VariableP_MultiElmt_CollDir02)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eGLL_Lagrange;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(4,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(6,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(8,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nelmts * nq), tmp;
    Array<OneD, NekDouble> coeffs1(nelmts * Exp->GetNcoeffs());
    Array<OneD, NekDouble> coeffs2(nelmts * Exp->GetNcoeffs());

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }
    Exp->IProductWRTBase(phys, coeffs1);

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, &phys[0], 1, &phys[i * nq], 1);
        Exp->IProductWRTBase(phys + i * nq,
                             tmp = coeffs1 + i * Exp->GetNcoeffs());
    }
    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs2);

    double epsilon = 1.0e-4;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        // clamp values below 1e-14 to zero
        coeffs1[i] = (std::abs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (std::abs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(
    TestHexIProductWRTBase_SumFac_VariableP_MultiElmt_CollDir12)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eGLL_Lagrange;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(6,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(8,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nelmts * nq), tmp;
    Array<OneD, NekDouble> coeffs1(nelmts * Exp->GetNcoeffs());
    Array<OneD, NekDouble> coeffs2(nelmts * Exp->GetNcoeffs());

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }
    Exp->IProductWRTBase(phys, coeffs1);

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, &phys[0], 1, &phys[i * nq], 1);
        Exp->IProductWRTBase(phys + i * nq,
                             tmp = coeffs1 + i * Exp->GetNcoeffs());
    }
    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs2);

    double epsilon = 1.0e-4;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        // clamp values below 1e-14 to zero
        coeffs1[i] = (std::abs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (std::abs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexPhysDeriv_IterPerExp_UniformP)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::ePhysDeriv);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);
    Array<OneD, NekDouble> phys(nq), tmp, tmp1, tmp2;
    Array<OneD, NekDouble> diff1(3 * nq);
    Array<OneD, NekDouble> diff2(3 * nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }

    Exp->PhysDeriv(phys, diff1, tmp = diff1 + nq, tmp1 = diff1 + 2 * nq);
    c.ApplyOperator(Collections::ePhysDeriv, phys, diff2, tmp = diff2 + nq,
                    tmp2 = diff2 + 2 * nq);

    double epsilon = 1.0e-8;
    for (int i = 0; i < diff1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(diff1[i], diff2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexPhysDeriv_MatrixFree_UniformP_Undeformed)
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    unsigned int numModes      = 2;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eMatrixFree);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::ePhysDeriv);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);
    Array<OneD, NekDouble> phys(nq), tmp, tmp1, tmp2;
    Array<OneD, NekDouble> diffRef(3 * nq);
    Array<OneD, NekDouble> diff(3 * nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }

    Exp->PhysDeriv(phys, diffRef, tmp = diffRef + nq, tmp1 = diffRef + 2 * nq);
    c.ApplyOperator(Collections::ePhysDeriv, phys, diff, tmp = diff + nq,
                    tmp2 = diff + 2 * nq);

    double epsilon = 1.0e-8;
    for (int i = 0; i < diffRef.size(); ++i)
    {
        diffRef[i] = (std::abs(diffRef[i]) < 1e-14) ? 0.0 : diffRef[i];
        diff[i]    = (std::abs(diff[i]) < 1e-14) ? 0.0 : diff[i];
        BOOST_CHECK_CLOSE(diffRef[i], diff[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexPhysDeriv_MatrixFree_UniformP_Deformed)
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
        new SpatialDomains::PointGeom(3u, 6u, 2.0, 3.0, 4.0));
    SpatialDomains::PointGeomUniquePtr v7(
        new SpatialDomains::PointGeom(3u, 7u, -1.0, 1.0, 1.0));

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 5;
    unsigned int numModes      = 2;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eMatrixFree);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::ePhysDeriv);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);
    Array<OneD, NekDouble> phys(nq), tmp, tmp1, tmp2;
    Array<OneD, NekDouble> diffRef(3 * nq);
    Array<OneD, NekDouble> diff(3 * nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }

    Exp->PhysDeriv(phys, diffRef, tmp = diffRef + nq, tmp1 = diffRef + 2 * nq);
    c.ApplyOperator(Collections::ePhysDeriv, phys, diff, tmp = diff + nq,
                    tmp2 = diff + 2 * nq);

    double epsilon = 1.0e-8;
    for (int i = 0; i < diffRef.size(); ++i)
    {
        diffRef[i] = (std::abs(diffRef[i]) < 1e-14) ? 0.0 : diffRef[i];
        diff[i]    = (std::abs(diff[i]) < 1e-14) ? 0.0 : diff[i];
        BOOST_CHECK_CLOSE(diffRef[i], diff[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexPhysDeriv_IterPerExp_VariableP_MultiElmt)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(6,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(8,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::ePhysDeriv);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);
    Array<OneD, NekDouble> phys(nelmts * nq), tmp, tmp1, tmp2;
    Array<OneD, NekDouble> diff1(3 * nelmts * nq);
    Array<OneD, NekDouble> diff2(3 * nelmts * nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }
    Exp->PhysDeriv(phys, tmp = diff1, tmp1 = diff1 + (nelmts)*nq,
                   tmp2 = diff1 + (2 * nelmts) * nq);
    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, phys, 1, tmp = phys + i * nq, 1);
        Exp->PhysDeriv(phys, tmp = diff1 + i * nq,
                       tmp1 = diff1 + (nelmts + i) * nq,
                       tmp2 = diff1 + (2 * nelmts + i) * nq);
    }

    c.ApplyOperator(Collections::ePhysDeriv, phys, diff2,
                    tmp = diff2 + nelmts * nq, tmp2 = diff2 + 2 * nelmts * nq);

    double epsilon = 1.0e-8;
    for (int i = 0; i < diff1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(diff1[i], diff2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexPhysDeriv_NoCollection_VariableP_MultiElmt)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(6,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(8,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eNoCollection);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::ePhysDeriv);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);
    Array<OneD, NekDouble> phys(nelmts * nq), tmp, tmp1, tmp2;
    Array<OneD, NekDouble> diff1(3 * nelmts * nq);
    Array<OneD, NekDouble> diff2(3 * nelmts * nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }
    Exp->PhysDeriv(phys, tmp = diff1, tmp1 = diff1 + (nelmts)*nq,
                   tmp2 = diff1 + (2 * nelmts) * nq);

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, phys, 1, tmp = phys + i * nq, 1);
        Exp->PhysDeriv(phys, tmp = diff1 + i * nq,
                       tmp1 = diff1 + (nelmts + i) * nq,
                       tmp2 = diff1 + (2 * nelmts + i) * nq);
    }

    c.ApplyOperator(Collections::ePhysDeriv, phys, diff2,
                    tmp = diff2 + nelmts * nq, tmp2 = diff2 + 2 * nelmts * nq);

    double epsilon = 1.0e-8;
    for (int i = 0; i < diff1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(diff1[i], diff2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexPhysDeriv_StdMat_UniformP)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 4;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 3,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eStdMat);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::ePhysDeriv);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);
    Array<OneD, NekDouble> phys(nq), tmp, tmp1, tmp2;
    Array<OneD, NekDouble> diff1(3 * nq);
    Array<OneD, NekDouble> diff2(3 * nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }

    Exp->PhysDeriv(phys, diff1, tmp = diff1 + nq, tmp1 = diff1 + 2 * nq);
    c.ApplyOperator(Collections::ePhysDeriv, phys, diff2, tmp = diff2 + nq,
                    tmp2 = diff2 + 2 * nq);

    double epsilon = 1.0e-8;
    for (int i = 0; i < diff1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(diff1[i], diff2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexPhysDeriv_StdMat_VariableP_MultiElmt)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(6,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(8,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eStdMat);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::ePhysDeriv);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);
    Array<OneD, NekDouble> phys(nelmts * nq), tmp, tmp1, tmp2;
    Array<OneD, NekDouble> diff1(3 * nelmts * nq);
    Array<OneD, NekDouble> diff2(3 * nelmts * nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }
    Exp->PhysDeriv(phys, tmp = diff1, tmp1 = diff1 + (nelmts)*nq,
                   tmp2 = diff1 + (2 * nelmts) * nq);
    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, phys, 1, tmp = phys + i * nq, 1);
        Exp->PhysDeriv(phys, tmp = diff1 + i * nq,
                       tmp1 = diff1 + (nelmts + i) * nq,
                       tmp2 = diff1 + (2 * nelmts + i) * nq);
    }

    c.ApplyOperator(Collections::ePhysDeriv, phys, diff2,
                    tmp = diff2 + nelmts * nq, tmp2 = diff2 + 2 * nelmts * nq);

    double epsilon = 1.0e-8;
    for (int i = 0; i < diff1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(diff1[i], diff2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexPhysDeriv_SumFac_UniformP)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::ePhysDeriv);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);
    Array<OneD, NekDouble> phys(nq), tmp, tmp1, tmp2;
    Array<OneD, NekDouble> diff1(3 * nq);
    Array<OneD, NekDouble> diff2(3 * nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }

    Exp->PhysDeriv(phys, diff1, tmp = diff1 + nq, tmp1 = diff1 + 2 * nq);
    c.ApplyOperator(Collections::ePhysDeriv, phys, diff2, tmp = diff2 + nq,
                    tmp2 = diff2 + 2 * nq);

    double epsilon = 1.0e-8;
    for (int i = 0; i < diff1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(diff1[i], diff2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexPhysDeriv_SumFac_VariableP_MultiElmt)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(6,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(8,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::ePhysDeriv);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);
    Array<OneD, NekDouble> phys(nelmts * nq), tmp, tmp1, tmp2;
    Array<OneD, NekDouble> diff1(3 * nelmts * nq);
    Array<OneD, NekDouble> diff2(3 * nelmts * nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }
    Exp->PhysDeriv(phys, tmp = diff1, tmp1 = diff1 + (nelmts)*nq,
                   tmp2 = diff1 + (2 * nelmts) * nq);
    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, phys, 1, tmp = phys + i * nq, 1);
        Exp->PhysDeriv(phys, tmp = diff1 + i * nq,
                       tmp1 = diff1 + (nelmts + i) * nq,
                       tmp2 = diff1 + (2 * nelmts + i) * nq);
    }

    c.ApplyOperator(Collections::ePhysDeriv, phys, diff2,
                    tmp = diff2 + nelmts * nq, tmp2 = diff2 + 2 * nelmts * nq);

    double epsilon = 1.0e-8;
    for (int i = 0; i < diff1.size(); ++i)
    {
        BOOST_CHECK_CLOSE(diff1[i], diff2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTDerivBase_Iterperexp_UniformP)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nq, 0.0);
    Array<OneD, NekDouble> phys2(nq, 0.0);
    Array<OneD, NekDouble> phys3(nq, 0.0);
    Array<OneD, NekDouble> coeffs1(nm, 0.0);
    Array<OneD, NekDouble> coeffs2(nm, 0.0);

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
    }

    // Standard routines
    Exp->IProductWRTDerivBase(0, phys1, coeffs1);
    Exp->IProductWRTDerivBase(1, phys2, coeffs2);
    Vmath::Vadd(nm, coeffs1, 1, coeffs2, 1, coeffs1, 1);
    Exp->IProductWRTDerivBase(2, phys3, coeffs2);
    Vmath::Vadd(nm, coeffs1, 1, coeffs2, 1, coeffs1, 1);

    c.ApplyOperator(Collections::eIProductWRTDerivBase, phys1, phys2, phys3,
                    coeffs2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < nm; ++i)
    {
        coeffs1[i] = (std::abs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (std::abs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTDerivBase_MatrixFree_UniformP_Undeformed)
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 5;
    unsigned int numModes      = 4;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eMatrixFree);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nq, 0.0);
    Array<OneD, NekDouble> phys2(nq, 0.0);
    Array<OneD, NekDouble> phys3(nq, 0.0);
    Array<OneD, NekDouble> coeffsRef(nm, 0.0);
    Array<OneD, NekDouble> coeffs(nm, 0.0);

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
    }

    // Standard routines
    Exp->IProductWRTDerivBase(0, phys1, coeffsRef);
    Exp->IProductWRTDerivBase(1, phys2, coeffs);
    Vmath::Vadd(nm, coeffsRef, 1, coeffs, 1, coeffsRef, 1);
    Exp->IProductWRTDerivBase(2, phys3, coeffs);
    Vmath::Vadd(nm, coeffsRef, 1, coeffs, 1, coeffsRef, 1);

    c.ApplyOperator(Collections::eIProductWRTDerivBase, phys1, phys2, phys3,
                    coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < nm; ++i)
    {
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTDerivBase_MatrixFree_UniformP_Deformed)
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
        new SpatialDomains::PointGeom(3u, 6u, 2.0, 3.0, 4.0));
    SpatialDomains::PointGeomUniquePtr v7(
        new SpatialDomains::PointGeom(3u, 7u, -1.0, 1.0, 1.0));

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 5;
    unsigned int numModes      = 4;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eMatrixFree);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nq, 0.0);
    Array<OneD, NekDouble> phys2(nq, 0.0);
    Array<OneD, NekDouble> phys3(nq, 0.0);
    Array<OneD, NekDouble> coeffsRef(nm, 0.0);
    Array<OneD, NekDouble> coeffs(nm, 0.0);

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
    }

    // Standard routines
    Exp->IProductWRTDerivBase(0, phys1, coeffsRef);
    Exp->IProductWRTDerivBase(1, phys2, coeffs);
    Vmath::Vadd(nm, coeffsRef, 1, coeffs, 1, coeffsRef, 1);
    Exp->IProductWRTDerivBase(2, phys3, coeffs);
    Vmath::Vadd(nm, coeffsRef, 1, coeffs, 1, coeffsRef, 1);

    c.ApplyOperator(Collections::eIProductWRTDerivBase, phys1, phys2, phys3,
                    coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < nm; ++i)
    {
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(
    TestHexIProductWRTDerivBase_MatrixFree_UniformP_Deformed_OverInt)
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
        new SpatialDomains::PointGeom(3u, 6u, 2.0, 3.0, 4.0));
    SpatialDomains::PointGeomUniquePtr v7(
        new SpatialDomains::PointGeom(3u, 7u, -1.0, 1.0, 1.0));

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 8;
    unsigned int numModes      = 4;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eMatrixFree);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nq, 0.0);
    Array<OneD, NekDouble> phys2(nq, 0.0);
    Array<OneD, NekDouble> phys3(nq, 0.0);
    Array<OneD, NekDouble> coeffsRef(nm, 0.0);
    Array<OneD, NekDouble> coeffs(nm, 0.0);

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
    }

    // Standard routines
    Exp->IProductWRTDerivBase(0, phys1, coeffsRef);
    Exp->IProductWRTDerivBase(1, phys2, coeffs);
    Vmath::Vadd(nm, coeffsRef, 1, coeffs, 1, coeffsRef, 1);
    Exp->IProductWRTDerivBase(2, phys3, coeffs);
    Vmath::Vadd(nm, coeffsRef, 1, coeffs, 1, coeffsRef, 1);

    c.ApplyOperator(Collections::eIProductWRTDerivBase, phys1, phys2, phys3,
                    coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < nm; ++i)
    {
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTDerivBase_IterPerExp_VariableP_MultiElmt)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(6,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(8,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nelmts * nq, 0.0);
    Array<OneD, NekDouble> phys2(nelmts * nq, 0.0);
    Array<OneD, NekDouble> phys3(nelmts * nq, 0.0);
    Array<OneD, NekDouble> coeffs1(nelmts * nm, 0.0);
    Array<OneD, NekDouble> coeffs2(nelmts * nm, 0.0);
    Array<OneD, NekDouble> tmp;

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
    }

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, phys1, 1, tmp = phys1 + i * nq, 1);
        Vmath::Vcopy(nq, phys2, 1, tmp = phys2 + i * nq, 1);
        Vmath::Vcopy(nq, phys3, 1, tmp = phys3 + i * nq, 1);
    }

    // Standard routines
    for (int i = 0; i < nelmts; ++i)
    {

        Exp->IProductWRTDerivBase(0, phys1 + i * nq, tmp = coeffs1 + i * nm);
        Exp->IProductWRTDerivBase(1, phys2 + i * nq, tmp = coeffs2 + i * nm);
        Vmath::Vadd(nm, coeffs1 + i * nm, 1, coeffs2 + i * nm, 1,
                    tmp = coeffs1 + i * nm, 1);
        Exp->IProductWRTDerivBase(2, phys3 + i * nq, tmp = coeffs2 + i * nm);
        Vmath::Vadd(nm, coeffs1 + i * nm, 1, coeffs2 + i * nm, 1,
                    tmp = coeffs1 + i * nm, 1);
    }

    c.ApplyOperator(Collections::eIProductWRTDerivBase, phys1, phys2, phys3,
                    coeffs2);

    double epsilon = 1.0e-6;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        coeffs1[i] = (std::abs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (std::abs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTDerivBase_StdMat_UniformP)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eStdMat);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nq, 0.0);
    Array<OneD, NekDouble> phys2(nq, 0.0);
    Array<OneD, NekDouble> phys3(nq, 0.0);
    Array<OneD, NekDouble> coeffs1(nm, 0.0);
    Array<OneD, NekDouble> coeffs2(nm, 0.0);

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
    }

    // Standard routines
    Exp->IProductWRTDerivBase(0, phys1, coeffs1);
    Exp->IProductWRTDerivBase(1, phys2, coeffs2);
    Vmath::Vadd(nm, coeffs1, 1, coeffs2, 1, coeffs1, 1);
    Exp->IProductWRTDerivBase(2, phys3, coeffs2);
    Vmath::Vadd(nm, coeffs1, 1, coeffs2, 1, coeffs1, 1);

    c.ApplyOperator(Collections::eIProductWRTDerivBase, phys1, phys2, phys3,
                    coeffs2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        coeffs1[i] = (std::abs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (std::abs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTDerivBase_StdMat_VariableP_MultiElmt)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(6,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(8,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eStdMat);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nelmts * nq, 0.0);
    Array<OneD, NekDouble> phys2(nelmts * nq, 0.0);
    Array<OneD, NekDouble> phys3(nelmts * nq, 0.0);
    Array<OneD, NekDouble> coeffs1(nelmts * nm, 0.0);
    Array<OneD, NekDouble> coeffs2(nelmts * nm, 0.0);
    Array<OneD, NekDouble> tmp;

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
    }

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, phys1, 1, tmp = phys1 + i * nq, 1);
        Vmath::Vcopy(nq, phys2, 1, tmp = phys2 + i * nq, 1);
        Vmath::Vcopy(nq, phys3, 1, tmp = phys3 + i * nq, 1);
    }

    // Standard routines
    for (int i = 0; i < nelmts; ++i)
    {

        Exp->IProductWRTDerivBase(0, phys1 + i * nq, tmp = coeffs1 + i * nm);
        Exp->IProductWRTDerivBase(1, phys2 + i * nq, tmp = coeffs2 + i * nm);
        Vmath::Vadd(nm, coeffs1 + i * nm, 1, coeffs2 + i * nm, 1,
                    tmp = coeffs1 + i * nm, 1);
        Exp->IProductWRTDerivBase(2, phys3 + i * nq, tmp = coeffs2 + i * nm);
        Vmath::Vadd(nm, coeffs1 + i * nm, 1, coeffs2 + i * nm, 1,
                    tmp = coeffs1 + i * nm, 1);
    }

    c.ApplyOperator(Collections::eIProductWRTDerivBase, phys1, phys2, phys3,
                    coeffs2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        coeffs1[i] = (std::abs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (std::abs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(
    TestHexIProductWRTDerivBase_NoCollection_VariableP_MultiElmt)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(6,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(8,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eNoCollection);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nelmts * nq, 0.0);
    Array<OneD, NekDouble> phys2(nelmts * nq, 0.0);
    Array<OneD, NekDouble> phys3(nelmts * nq, 0.0);
    Array<OneD, NekDouble> coeffs1(nelmts * nm, 0.0);
    Array<OneD, NekDouble> coeffs2(nelmts * nm, 0.0);
    Array<OneD, NekDouble> tmp;
    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
    }

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, phys1, 1, tmp = phys1 + i * nq, 1);
        Vmath::Vcopy(nq, phys2, 1, tmp = phys2 + i * nq, 1);
        Vmath::Vcopy(nq, phys3, 1, tmp = phys3 + i * nq, 1);
    }

    // Standard routines
    for (int i = 0; i < nelmts; ++i)
    {

        Exp->IProductWRTDerivBase(0, phys1 + i * nq, tmp = coeffs1 + i * nm);
        Exp->IProductWRTDerivBase(1, phys2 + i * nq, tmp = coeffs2 + i * nm);
        Vmath::Vadd(nm, coeffs1 + i * nm, 1, coeffs2 + i * nm, 1,
                    tmp = coeffs1 + i * nm, 1);
        Exp->IProductWRTDerivBase(2, phys3 + i * nq, tmp = coeffs2 + i * nm);
        Vmath::Vadd(nm, coeffs1 + i * nm, 1, coeffs2 + i * nm, 1,
                    tmp = coeffs1 + i * nm, 1);
    }

    c.ApplyOperator(Collections::eIProductWRTDerivBase, phys1, phys2, phys3,
                    coeffs2);

    double epsilon = 1.0e-6;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        coeffs1[i] = (std::abs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (std::abs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTDerivBase_SumFac_UniformP)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 6;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nq, 0.0);
    Array<OneD, NekDouble> phys2(nq, 0.0);
    Array<OneD, NekDouble> phys3(nq, 0.0);
    Array<OneD, NekDouble> coeffs1(nm, 0.0);
    Array<OneD, NekDouble> coeffs2(nm, 0.0);

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
    }

    // Standard routines
    Exp->IProductWRTDerivBase(0, phys1, coeffs1);
    Exp->IProductWRTDerivBase(1, phys2, coeffs2);
    Vmath::Vadd(nm, coeffs1, 1, coeffs2, 1, coeffs1, 1);
    Exp->IProductWRTDerivBase(2, phys3, coeffs2);
    Vmath::Vadd(nm, coeffs1, 1, coeffs2, 1, coeffs1, 1);

    c.ApplyOperator(Collections::eIProductWRTDerivBase, phys1, phys2, phys3,
                    coeffs2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        coeffs1[i] = (std::abs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (std::abs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexIProductWRTDerivBase_SumFac_VariableP_MultiElmt)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -1.5, -1.5, -1.5));
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

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(5,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir2(6,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::PointsKey quadPointsKeyDir3(8,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      quadPointsKeyDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir1, 6,
                                                      quadPointsKeyDir2);
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir1, 8,
                                                      quadPointsKeyDir3);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nelmts * nq, 0.0);
    Array<OneD, NekDouble> phys2(nelmts * nq, 0.0);
    Array<OneD, NekDouble> phys3(nelmts * nq, 0.0);
    Array<OneD, NekDouble> coeffs1(nelmts * nm, 0.0);
    Array<OneD, NekDouble> coeffs2(nelmts * nm, 0.0);
    Array<OneD, NekDouble> tmp;

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
    }

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, phys1, 1, tmp = phys1 + i * nq, 1);
        Vmath::Vcopy(nq, phys2, 1, tmp = phys2 + i * nq, 1);
        Vmath::Vcopy(nq, phys3, 1, tmp = phys3 + i * nq, 1);
    }

    // Standard routines
    for (int i = 0; i < nelmts; ++i)
    {
        Exp->IProductWRTDerivBase(0, phys1 + i * nq, tmp = coeffs1 + i * nm);
        Exp->IProductWRTDerivBase(1, phys2 + i * nq, tmp = coeffs2 + i * nm);
        Vmath::Vadd(nm, coeffs1 + i * nm, 1, coeffs2 + i * nm, 1,
                    tmp = coeffs1 + i * nm, 1);
        Exp->IProductWRTDerivBase(2, phys3 + i * nq, tmp = coeffs2 + i * nm);
        Vmath::Vadd(nm, coeffs1 + i * nm, 1, coeffs2 + i * nm, 1,
                    tmp = coeffs1 + i * nm, 1);
    }

    c.ApplyOperator(Collections::eIProductWRTDerivBase, phys1, phys2, phys3,
                    coeffs2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffs1.size(); ++i)
    {
        coeffs1[i] = (std::abs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (std::abs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexHelmholtz_NoCollection_UniformP)
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
        new SpatialDomains::PointGeom(3u, 6u, 2.0, 3.0, 4.0));
    SpatialDomains::PointGeomUniquePtr v7(
        new SpatialDomains::PointGeom(3u, 7u, -1.0, 1.0, 1.0));

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 5;
    unsigned int numModes      = 4;

    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eNoCollection);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    StdRegions::ConstFactorMap factors;
    factors[StdRegions::eFactorLambda] = 1.5;

    c.Initialise(Collections::eHelmholtz, factors);

    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> coeffsIn(nelmts * nm);
    Array<OneD, NekDouble> coeffsRef(nelmts * nm);
    Array<OneD, NekDouble> coeffs(nelmts * nm), tmp;

    for (int i = 0; i < nm; ++i)
    {
        coeffsIn[i] = 1.0;
    }

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nm, coeffsIn, 1, tmp = coeffsIn + i * nm, 1);
    }

    StdRegions::StdMatrixKey mkey(StdRegions::eHelmholtz, Exp->DetShapeType(),
                                  *Exp, factors);

    for (int i = 0; i < nelmts; ++i)
    {
        // Standard routines
        Exp->GeneralMatrixOp(coeffsIn + i * nm, tmp = coeffsRef + i * nm, mkey);
    }

    c.ApplyOperator(Collections::eHelmholtz, coeffsIn, coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffsRef.size(); ++i)
    {
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexHelmholtz_IterPerExp_UniformP)
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
        new SpatialDomains::PointGeom(3u, 6u, 2.0, 3.0, 4.0));
    SpatialDomains::PointGeomUniquePtr v7(
        new SpatialDomains::PointGeom(3u, 7u, -1.0, 1.0, 1.0));

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 5;
    unsigned int numModes      = 4;

    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    StdRegions::ConstFactorMap factors;
    factors[StdRegions::eFactorLambda] = 1.5;

    c.Initialise(Collections::eHelmholtz, factors);

    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> coeffsIn(nelmts * nm);
    Array<OneD, NekDouble> coeffsRef(nelmts * nm);
    Array<OneD, NekDouble> coeffs(nelmts * nm), tmp;

    for (int i = 0; i < nm; ++i)
    {
        coeffsIn[i] = 1.0;
    }

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nm, coeffsIn, 1, tmp = coeffsIn + i * nm, 1);
    }

    StdRegions::StdMatrixKey mkey(StdRegions::eHelmholtz, Exp->DetShapeType(),
                                  *Exp, factors);

    for (int i = 0; i < nelmts; ++i)
    {
        // Standard routines
        Exp->GeneralMatrixOp(coeffsIn + i * nm, tmp = coeffsRef + i * nm, mkey);
    }

    c.ApplyOperator(Collections::eHelmholtz, coeffsIn, coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffsRef.size(); ++i)
    {
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexHelmholtz_IterPerExp_UniformP_ConstVarDiff)
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
        new SpatialDomains::PointGeom(3u, 6u, 2.0, 3.0, 4.0));
    SpatialDomains::PointGeomUniquePtr v7(
        new SpatialDomains::PointGeom(3u, 7u, -1.0, 1.0, 1.0));

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 5;
    unsigned int numModes      = 4;

    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    StdRegions::ConstFactorMap factors;
    factors[StdRegions::eFactorLambda]   = 1.5;
    factors[StdRegions::eFactorCoeffD00] = 1.25;
    factors[StdRegions::eFactorCoeffD01] = 0.25;
    factors[StdRegions::eFactorCoeffD11] = 1.25;
    factors[StdRegions::eFactorCoeffD02] = 0.25;
    factors[StdRegions::eFactorCoeffD12] = 0.25;
    factors[StdRegions::eFactorCoeffD22] = 1.25;

    c.Initialise(Collections::eHelmholtz, factors);

    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> coeffsIn(nelmts * nm);
    Array<OneD, NekDouble> coeffsRef(nelmts * nm);
    Array<OneD, NekDouble> coeffs(nelmts * nm), tmp;

    for (int i = 0; i < nm; ++i)
    {
        coeffsIn[i] = 1.0;
    }

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nm, coeffsIn, 1, tmp = coeffsIn + i * nm, 1);
    }

    StdRegions::StdMatrixKey mkey(StdRegions::eHelmholtz, Exp->DetShapeType(),
                                  *Exp, factors);

    for (int i = 0; i < nelmts; ++i)
    {
        // Standard routines
        Exp->GeneralMatrixOp(coeffsIn + i * nm, tmp = coeffsRef + i * nm, mkey);
    }

    c.ApplyOperator(Collections::eHelmholtz, coeffsIn, coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffsRef.size(); ++i)
    {
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexHelmholtz_MatrixFree_UniformP)
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
        new SpatialDomains::PointGeom(3u, 6u, 2.0, 3.0, 4.0));
    SpatialDomains::PointGeomUniquePtr v7(
        new SpatialDomains::PointGeom(3u, 7u, -1.0, 1.0, 1.0));

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 5;
    unsigned int numModes      = 4;

    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eMatrixFree);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    StdRegions::ConstFactorMap factors;
    factors[StdRegions::eFactorLambda] = 0.0;

    c.Initialise(Collections::eHelmholtz, factors);

    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> coeffsIn(nelmts * nm);
    Array<OneD, NekDouble> coeffsRef(nelmts * nm);
    Array<OneD, NekDouble> coeffs(nelmts * nm), tmp;

    for (int i = 0; i < nm; ++i)
    {
        coeffsIn[i] = 1.0;
    }

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nm, coeffsIn, 1, tmp = coeffsIn + i * nm, 1);
    }

    StdRegions::StdMatrixKey mkey(StdRegions::eHelmholtz, Exp->DetShapeType(),
                                  *Exp, factors);

    for (int i = 0; i < nelmts; ++i)
    {
        // Standard routines
        Exp->GeneralMatrixOp(coeffsIn + i * nm, tmp = coeffsRef + i * nm, mkey);
    }

    c.ApplyOperator(Collections::eHelmholtz, coeffsIn, coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffsRef.size(); ++i)
    {
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexHelmholtz_MatrixFree_UniformP_Deformed_OverInt)
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
        new SpatialDomains::PointGeom(3u, 6u, 2.0, 3.0, 4.0));
    SpatialDomains::PointGeomUniquePtr v7(
        new SpatialDomains::PointGeom(3u, 7u, -1.0, 1.0, 1.0));

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 8;
    unsigned int numModes      = 4;

    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eMatrixFree);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    StdRegions::ConstFactorMap factors;
    factors[StdRegions::eFactorLambda] = 0.0;

    c.Initialise(Collections::eHelmholtz, factors);

    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> coeffsIn(nelmts * nm);
    Array<OneD, NekDouble> coeffsRef(nelmts * nm);
    Array<OneD, NekDouble> coeffs(nelmts * nm), tmp;

    for (int i = 0; i < nm; ++i)
    {
        coeffsIn[i] = 1.0;
    }

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nm, coeffsIn, 1, tmp = coeffsIn + i * nm, 1);
    }

    StdRegions::StdMatrixKey mkey(StdRegions::eHelmholtz, Exp->DetShapeType(),
                                  *Exp, factors);

    for (int i = 0; i < nelmts; ++i)
    {
        // Standard routines
        Exp->GeneralMatrixOp(coeffsIn + i * nm, tmp = coeffsRef + i * nm, mkey);
    }

    c.ApplyOperator(Collections::eHelmholtz, coeffsIn, coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffsRef.size(); ++i)
    {
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexHelmholtz_MatrixFree_UniformP_ConstVarDiff)
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
        new SpatialDomains::PointGeom(3u, 6u, 2.0, 3.0, 4.0));
    SpatialDomains::PointGeomUniquePtr v7(
        new SpatialDomains::PointGeom(3u, 7u, -1.0, 1.0, 1.0));

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 5;
    unsigned int numModes      = 4;

    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eMatrixFree);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(stdExp);
    Collections::Collection c(CollExp, impTypes);
    StdRegions::ConstFactorMap factors;
    factors[StdRegions::eFactorLambda]   = 1.0;
    factors[StdRegions::eFactorCoeffD00] = 1.25;
    factors[StdRegions::eFactorCoeffD01] = 0.25;
    factors[StdRegions::eFactorCoeffD11] = 1.25;
    factors[StdRegions::eFactorCoeffD02] = 0.25;
    factors[StdRegions::eFactorCoeffD12] = 0.25;
    factors[StdRegions::eFactorCoeffD22] = 1.25;

    c.Initialise(Collections::eHelmholtz, factors);

    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> coeffsIn(nelmts * nm);
    Array<OneD, NekDouble> coeffsRef(nelmts * nm);
    Array<OneD, NekDouble> coeffs(nelmts * nm), tmp;

    for (int i = 0; i < nm; ++i)
    {
        coeffsIn[i] = 1.0;
    }

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nm, coeffsIn, 1, tmp = coeffsIn + i * nm, 1);
    }

    StdRegions::StdMatrixKey mkey(StdRegions::eHelmholtz, Exp->DetShapeType(),
                                  *Exp, factors);

    for (int i = 0; i < nelmts; ++i)
    {
        // Standard routines
        Exp->GeneralMatrixOp(coeffsIn + i * nm, tmp = coeffsRef + i * nm, mkey);
    }

    c.ApplyOperator(Collections::eHelmholtz, coeffsIn, coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffsRef.size(); ++i)
    {
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexPhysInterp1D_NoCollection_UniformP)
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
        new SpatialDomains::PointGeom(3u, 6u, 2.0, 3.0, 4.0));
    SpatialDomains::PointGeomUniquePtr v7(
        new SpatialDomains::PointGeom(3u, 7u, -1.0, 1.0, 1.0));

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 5;
    unsigned int numModes      = 4;

    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eNoCollection);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);

    StdRegions::ConstFactorMap factors;
    factors[StdRegions::eFactorConst] = 1.5;
    c.Initialise(Collections::ePhysInterp1DScaled, factors);

    const int nq = Exp->GetTotPoints();

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);
    Array<OneD, NekDouble> phys(nq), tmp;

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = pow(xc[i], 3) + pow(yc[i], 3) + pow(zc[i], 3);
    }

    const int nq1 = c.GetOutputSize(Collections::ePhysInterp1DScaled);
    Array<OneD, NekDouble> xc1(nq1);
    Array<OneD, NekDouble> yc1(nq1);
    Array<OneD, NekDouble> zc1(nq1);
    Array<OneD, NekDouble> phys1(nq1);

    c.ApplyOperator(Collections::ePhysInterp1DScaled, xc, xc1);
    c.ApplyOperator(Collections::ePhysInterp1DScaled, yc, yc1);
    c.ApplyOperator(Collections::ePhysInterp1DScaled, zc, zc1);
    c.ApplyOperator(Collections::ePhysInterp1DScaled, phys, phys1);

    double epsilon = 1.0e-8;
    // since solution is a polynomial should be able to compare soln directly
    for (int i = 0; i < nq1; ++i)
    {
        NekDouble exact = pow(xc1[i], 3) + pow(yc1[i], 3) + pow(zc1[i], 3);
        phys1[i]        = (fabs(phys1[i]) < 1e-14) ? 0.0 : phys1[i];
        exact           = (fabs(exact) < 1e-14) ? 0.0 : exact;
        BOOST_CHECK_CLOSE(phys1[i], exact, epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestHexPhysInterp1D_MatrixFree_UniformP)
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
        new SpatialDomains::PointGeom(3u, 6u, 2.0, 3.0, 4.0));
    SpatialDomains::PointGeomUniquePtr v7(
        new SpatialDomains::PointGeom(3u, 7u, -1.0, 1.0, 1.0));

    std::vector<SpatialDomains::SegGeomUniquePtr> segVec;
    std::vector<SpatialDomains::QuadGeomUniquePtr> faceVec;
    SpatialDomains::HexGeomUniquePtr hexGeom =
        CreateHex(v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get(), v6.get(), v7.get(), segVec, faceVec);

    Nektar::LibUtilities::PointsType quadPointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    unsigned int numQuadPoints = 5;
    unsigned int numModes      = 4;

    const Nektar::LibUtilities::PointsKey quadPointsKeyDir1(numQuadPoints,
                                                            quadPointsTypeDir1);
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      quadPointsKeyDir1);

    Nektar::LocalRegions::HexExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::HexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1, hexGeom.get());

    Nektar::StdRegions::StdHexExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir1, basisKeyDir1);

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    CollExp.push_back(Exp);

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eMatrixFree);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);

    StdRegions::ConstFactorMap factors;
    factors[StdRegions::eFactorConst] = 1.5;
    c.Initialise(Collections::ePhysInterp1DScaled, factors);

    const int nq = Exp->GetTotPoints();

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);
    Array<OneD, NekDouble> phys(nq), tmp;

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = pow(xc[i], 3) + pow(yc[i], 3) + pow(zc[i], 3);
    }

    const int nq1 = c.GetOutputSize(Collections::ePhysInterp1DScaled);
    Array<OneD, NekDouble> xc1(nq1);
    Array<OneD, NekDouble> yc1(nq1);
    Array<OneD, NekDouble> zc1(nq1);
    Array<OneD, NekDouble> phys1(nq1);

    c.ApplyOperator(Collections::ePhysInterp1DScaled, xc, xc1);
    c.ApplyOperator(Collections::ePhysInterp1DScaled, yc, yc1);
    c.ApplyOperator(Collections::ePhysInterp1DScaled, zc, zc1);
    c.ApplyOperator(Collections::ePhysInterp1DScaled, phys, phys1);

    double epsilon = 1.0e-8;
    // since solution is a polynomial should be able to compare soln directly
    for (int i = 0; i < nq1; ++i)
    {
        NekDouble exact = pow(xc1[i], 3) + pow(yc1[i], 3) + pow(zc1[i], 3);
        phys1[i]        = (fabs(phys1[i]) < 1e-14) ? 0.0 : phys1[i];
        exact           = (fabs(exact) < 1e-14) ? 0.0 : exact;
        BOOST_CHECK_CLOSE(phys1[i], exact, epsilon);
    }
}
#endif
} // namespace Nektar::HexCollectionTests
