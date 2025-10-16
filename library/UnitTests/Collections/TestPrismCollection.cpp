///////////////////////////////////////////////////////////////////////////////
//
// File: TestPrismCollection.cpp
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

#include <Collections/Collection.h>
#include <Collections/CollectionOptimisation.h>
#include <LocalRegions/PrismExp.h>
#include <SpatialDomains/MeshGraph.h>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

namespace Nektar::PrismCollectionTests
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

SpatialDomains::PrismGeomUniquePtr CreatePrism(
    std::array<SpatialDomains::PointGeom *, 6> v,
    std::array<SpatialDomains::SegGeomUniquePtr, 9> &segVec,
    std::array<SpatialDomains::TriGeomUniquePtr, 2> &triVec,
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> &quadVec)
{
    std::array<std::array<int, 2>, 9> edgeVerts = {{{{0, 1}},
                                                    {{1, 2}},
                                                    {{3, 2}},
                                                    {{0, 3}},
                                                    {{0, 4}},
                                                    {{1, 4}},
                                                    {{2, 5}},
                                                    {{3, 5}},
                                                    {{4, 5}}}};
    std::array<std::array<int, 4>, 5> faceEdges = {{{{0, 1, 2, 3}},
                                                    {{0, 5, 4, -1}},
                                                    {{1, 6, 8, 5}},
                                                    {{2, 6, 7, -1}},
                                                    {{3, 7, 8, 4}}}};

    // Create segments from vertices
    for (int i = 0; i < 9; ++i)
    {
        segVec[i] = CreateSegGeom(i, v[edgeVerts[i][0]], v[edgeVerts[i][1]]);
    }

    // Create faces from edges
    std::array<SpatialDomains::Geometry2D *, 5> faces;
    for (int i = 0; i < 5; ++i)
    {
        if (i % 2 == 0)
        {
            // Quad faces
            std::array<SpatialDomains::SegGeom *, 4> face;
            for (int j = 0; j < 4; ++j)
            {
                face[j] = segVec[faceEdges[i][j]].get();
            }
            quadVec[i / 2] = SpatialDomains::QuadGeomUniquePtr(
                new SpatialDomains::QuadGeom(i, face));
            faces[i] = quadVec[i / 2].get();
        }
        else
        {
            // Tri faces
            std::array<SpatialDomains::SegGeom *, 3> face;
            for (int j = 0; j < 3; ++j)
            {
                face[j] = segVec[faceEdges[i][j]].get();
            }
            triVec[(i - 1) / 2] = SpatialDomains::TriGeomUniquePtr(
                new SpatialDomains::TriGeom(i, face));
            faces[i] = triVec[(i - 1) / 2].get();
        }
    }

    SpatialDomains::PrismGeomUniquePtr prismGeom(
        new SpatialDomains::PrismGeom(0, faces));
    return prismGeom;
}

BOOST_AUTO_TEST_CASE(TestPrismBwdTrans_IterPerExp_UniformP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(5, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 4,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(4, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 4,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
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

BOOST_AUTO_TEST_CASE(TestPrismBwdTrans_StdMat_UniformP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(5, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 4,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(4, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 4,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eStdMat);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(nelmts * Exp->GetNcoeffs(), 1.0);
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> phys1(nelmts * Exp->GetTotPoints(), 0.0);
    Array<OneD, NekDouble> phys2(nelmts * Exp->GetTotPoints(), 0.0);
    Array<OneD, NekDouble> tmp;

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

BOOST_AUTO_TEST_CASE(TestPrismBwdTrans_SumFac_UniformP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(5, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 4,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(4, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 4,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(nelmts * Exp->GetNcoeffs(), 1.0);
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> phys1(nelmts * Exp->GetTotPoints(), 0.0);
    Array<OneD, NekDouble> phys2(nelmts * Exp->GetTotPoints(), 0.0);
    Array<OneD, NekDouble> tmp;

    for (int i = 0; i < nelmts; ++i)
    {
        Exp->BwdTrans(coeffs + i * Exp->GetNcoeffs(),
                      tmp = phys1 + i * Exp->GetTotPoints());
    }
    c.ApplyOperator(Collections::eBwdTrans, coeffs, phys2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < phys1.size(); ++i)
    {
        phys1[i] = (fabs(phys1[i]) < 1e-14) ? 0.0 : phys1[i];
        phys2[i] = (fabs(phys1[i]) < 1e-14) ? 0.0 : phys2[i];
        BOOST_CHECK_CLOSE(phys1[i], phys2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismBwdTrans_IterPerExp_VariableP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(7, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 6,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(8, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 8,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(nelmts * Exp->GetNcoeffs(), 1.0);
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> phys1(nelmts * Exp->GetTotPoints(), 0.0);
    Array<OneD, NekDouble> phys2(nelmts * Exp->GetTotPoints(), 0.0);
    Array<OneD, NekDouble> tmp;

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

BOOST_AUTO_TEST_CASE(TestPrismBwdTrans_MatrixFree_UniformP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    unsigned int numQuadPoints = 5;
    unsigned int numModes      = 4;

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(numQuadPoints,
                                                        PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(numQuadPoints,
                                                        PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, numModes,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(numQuadPoints - 1,
                                                        PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, numModes,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    unsigned int nelmts = 2;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (unsigned int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);

    // ... only one op at the time ...
    impTypes[Collections::eBwdTrans] = Collections::eMatrixFree;
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(nelmts * Exp->GetNcoeffs(), 1.0);
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> physRef(nelmts * Exp->GetTotPoints(), 0.0);
    Array<OneD, NekDouble> phys(nelmts * Exp->GetTotPoints(), 0.0);
    Array<OneD, NekDouble> tmp;

    for (unsigned int i = 0; i < nelmts; ++i)
    {
        Exp->BwdTrans(coeffs + i * Exp->GetNcoeffs(),
                      tmp = physRef + i * Exp->GetTotPoints());
    }
    c.ApplyOperator(Collections::eBwdTrans, coeffs, phys);

    double epsilon = 1.0e-8;
    for (unsigned int i = 0; i < physRef.size(); ++i)
    {
        BOOST_CHECK_CLOSE(physRef[i], phys[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismBwdTrans_MatrixFree_UniformP_OverInt_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    unsigned int numQuadPoints = 8;
    unsigned int numModes      = 4;

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(numQuadPoints,
                                                        PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(numQuadPoints,
                                                        PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, numModes,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(numQuadPoints - 1,
                                                        PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, numModes,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    unsigned int nelmts = 2;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (unsigned int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);

    // ... only one op at the time ...
    impTypes[Collections::eBwdTrans] = Collections::eMatrixFree;
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(nelmts * Exp->GetNcoeffs(), 1.0);
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> physRef(nelmts * Exp->GetTotPoints(), 0.0);
    Array<OneD, NekDouble> phys(nelmts * Exp->GetTotPoints(), 0.0);
    Array<OneD, NekDouble> tmp;

    for (unsigned int i = 0; i < nelmts; ++i)
    {
        Exp->BwdTrans(coeffs + i * Exp->GetNcoeffs(),
                      tmp = physRef + i * Exp->GetTotPoints());
    }
    c.ApplyOperator(Collections::eBwdTrans, coeffs, phys);

    double epsilon = 1.0e-8;
    for (unsigned int i = 0; i < physRef.size(); ++i)
    {
        BOOST_CHECK_CLOSE(physRef[i], phys[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismBwdTrans_StdMat_VariableP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(7, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 6,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(8, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 8,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eStdMat);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(nelmts * Exp->GetNcoeffs(), 1.0);
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> phys1(nelmts * Exp->GetTotPoints(), 0.0);
    Array<OneD, NekDouble> phys2(nelmts * Exp->GetTotPoints(), 0.0);
    Array<OneD, NekDouble> tmp;

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
BOOST_AUTO_TEST_CASE(TestPrismBwdTrans_SumFac_VariableP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(7, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 6,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(8, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 8,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eBwdTrans);

    Array<OneD, NekDouble> coeffs(nelmts * Exp->GetNcoeffs(), 1.0);
    for (int i = 0; i < coeffs.size(); ++i)
    {
        coeffs[i] = i + 1;
    }
    Array<OneD, NekDouble> phys1(nelmts * Exp->GetTotPoints(), 0.0);
    Array<OneD, NekDouble> phys2(nelmts * Exp->GetTotPoints(), 0.0);
    Array<OneD, NekDouble> tmp;

    for (int i = 0; i < nelmts; ++i)
    {
        Exp->BwdTrans(coeffs + i * Exp->GetNcoeffs(),
                      tmp = phys1 + i * Exp->GetTotPoints());
    }
    c.ApplyOperator(Collections::eBwdTrans, coeffs, phys2);

    double epsilon = 1.0e-8;
    for (int i = 0; i < phys1.size(); ++i)
    {
        phys1[i] = (fabs(phys1[i]) < 1e-14) ? 0.0 : phys1[i];
        phys2[i] = (fabs(phys1[i]) < 1e-14) ? 0.0 : phys2[i];
        BOOST_CHECK_CLOSE(phys1[i], phys2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismIProductWRTBase_IterPerExp_UniformP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(5, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 4,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(4, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 4,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nelmts * nq, 0.0);
    Array<OneD, NekDouble> coeffs1(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> coeffs2(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> tmp;
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
        // clamp values below 1e-14 to zero
        coeffs1[i] = (fabs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (fabs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismIProductWRTBase_StdMat_UniformP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(5, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 4,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(4, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 4,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eStdMat);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nelmts * nq, 0.0);
    Array<OneD, NekDouble> coeffs1(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> coeffs2(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> tmp;
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
        // clamp values below 1e-14 to zero
        coeffs1[i] = (fabs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (fabs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismIProductWRTBase_SumFac_UniformP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(5, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 4,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(4, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 4,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nelmts * nq, 0.0);
    Array<OneD, NekDouble> coeffs1(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> coeffs2(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> tmp;
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
        // clamp values below 1e-14 to zero
        coeffs1[i] = (fabs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (fabs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(
    TestPrismIProductWRTBase_MatrixFree_UniformP_Undeformed_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    unsigned int numQuadPoints = 5;
    unsigned int numModes      = 4;

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(numQuadPoints,
                                                        PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(numQuadPoints,
                                                        PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, numModes,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(numQuadPoints - 1,
                                                        PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, numModes,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    unsigned int nelmts = 2;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (unsigned int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);

    // ... only one op at the time ...
    impTypes[Collections::eIProductWRTBase] = Collections::eMatrixFree;
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nelmts * nq, 0.0);
    Array<OneD, NekDouble> coeffsRef(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> coeffs(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> tmp;
    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }
    Exp->IProductWRTBase(phys, coeffsRef);

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, &phys[0], 1, &phys[i * nq], 1);
        Exp->IProductWRTBase(phys + i * nq,
                             tmp = coeffsRef + i * Exp->GetNcoeffs());
    }

    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffsRef.size(); ++i)
    {
        // clamp values below 1e-14 to zero
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(
    TestPrismIProductWRTBase_MatrixFree_UniformP_Deformed_MultiElmt)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -2.0, -3.0, -4.0));
    SpatialDomains::PointGeomUniquePtr v1(
        new SpatialDomains::PointGeom(3u, 1u, 1.0, -1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v2(
        new SpatialDomains::PointGeom(3u, 2u, 1.0, 1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v3(
        new SpatialDomains::PointGeom(3u, 3u, -1.0, 1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v4(
        new SpatialDomains::PointGeom(3u, 4u, -1.0, -1.0, 1.0));
    SpatialDomains::PointGeomUniquePtr v5(
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    unsigned int numQuadPoints = 5;
    unsigned int numModes      = 4;

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(numQuadPoints,
                                                        PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(numQuadPoints,
                                                        PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, numModes,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(numQuadPoints - 1,
                                                        PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, numModes,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    unsigned int nelmts = 2;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (unsigned int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);

    // ... only one op at the time ...
    impTypes[Collections::eIProductWRTBase] = Collections::eMatrixFree;
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nelmts * nq, 0.0);
    Array<OneD, NekDouble> coeffsRef(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> coeffs(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> tmp;
    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }
    Exp->IProductWRTBase(phys, coeffsRef);

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, &phys[0], 1, &phys[i * nq], 1);
        Exp->IProductWRTBase(phys + i * nq,
                             tmp = coeffsRef + i * Exp->GetNcoeffs());
    }

    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffsRef.size(); ++i)
    {
        // clamp values below 1e-14 to zero
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(
    TestPrismIProductWRTBase_MatrixFree_UniformP_Deformed_OverInt_MultiElmt)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -2.0, -3.0, -4.0));
    SpatialDomains::PointGeomUniquePtr v1(
        new SpatialDomains::PointGeom(3u, 1u, 1.0, -1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v2(
        new SpatialDomains::PointGeom(3u, 2u, 1.0, 1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v3(
        new SpatialDomains::PointGeom(3u, 3u, -1.0, 1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v4(
        new SpatialDomains::PointGeom(3u, 4u, -1.0, -1.0, 1.0));
    SpatialDomains::PointGeomUniquePtr v5(
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    unsigned int numQuadPoints = 8;
    unsigned int numModes      = 4;

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(numQuadPoints,
                                                        PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(numQuadPoints,
                                                        PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, numModes,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(numQuadPoints - 1,
                                                        PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, numModes,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    unsigned int nelmts = 2;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (unsigned int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);

    // ... only one op at the time ...
    impTypes[Collections::eIProductWRTBase] = Collections::eMatrixFree;
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nelmts * nq, 0.0);
    Array<OneD, NekDouble> coeffsRef(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> coeffs(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> tmp;
    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }
    Exp->IProductWRTBase(phys, coeffsRef);

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, &phys[0], 1, &phys[i * nq], 1);
        Exp->IProductWRTBase(phys + i * nq,
                             tmp = coeffsRef + i * Exp->GetNcoeffs());
    }

    c.ApplyOperator(Collections::eIProductWRTBase, phys, coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffsRef.size(); ++i)
    {
        // clamp values below 1e-14 to zero
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismIProductWRTBase_IterPerExp_VariableP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(7, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 6,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(8, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 8,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nelmts * nq, 0.0);
    Array<OneD, NekDouble> coeffs1(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> coeffs2(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> tmp;
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
        // clamp values below 1e-14 to zero
        coeffs1[i] = (fabs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (fabs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismIProductWRTBase_StdMat_VariableP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(7, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 6,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(8, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 8,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eStdMat);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nelmts * nq, 0.0);
    Array<OneD, NekDouble> coeffs1(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> coeffs2(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> tmp;
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
        // clamp values below 1e-14 to zero
        coeffs1[i] = (fabs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (fabs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismIProductWRTBase_SumFac_VariableP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(7, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 6,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(8, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 8,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTBase);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> phys(nelmts * nq, 0.0);
    Array<OneD, NekDouble> coeffs1(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> coeffs2(nelmts * Exp->GetNcoeffs(), 0.0);
    Array<OneD, NekDouble> tmp;
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
        // clamp values below 1e-14 to zero
        coeffs1[i] = (fabs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (fabs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismPhysDeriv_IterPerExp_UniformP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(5, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 4,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(4, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 4,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
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
    Exp->PhysDeriv(phys, diff1, tmp1 = diff1 + (nelmts)*nq,
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

BOOST_AUTO_TEST_CASE(TestPrismPhysDeriv_StdMat_UniformP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(5, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 4,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(4, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 4,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eStdMat);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
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
    Exp->PhysDeriv(phys, diff1, tmp1 = diff1 + (nelmts)*nq,
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
        // clamp values below 1e-14 to zero
        diff1[i] = (fabs(diff1[i]) < 1e-14) ? 0.0 : diff1[i];
        diff2[i] = (fabs(diff2[i]) < 1e-14) ? 0.0 : diff2[i];
        BOOST_CHECK_CLOSE(diff1[i], diff2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismPhysDeriv_SumFac_UniformP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(5, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 4,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(4, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 4,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 2;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
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
    Exp->PhysDeriv(phys, diff1, tmp1 = diff1 + (nelmts)*nq,
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
        diff1[i] = (fabs(diff1[i]) < 1e-14) ? 0.0 : diff1[i];
        diff2[i] = (fabs(diff1[i]) < 1e-14) ? 0.0 : diff2[i];
        BOOST_CHECK_CLOSE(diff1[i], diff2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismPhysDeriv_IterPerExp_VariableP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(7, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 6,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(8, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 8,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
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
    Exp->PhysDeriv(phys, diff1, tmp1 = diff1 + (nelmts)*nq,
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
        // clamp values below 1e-14 to zero
        diff1[i] = (fabs(diff1[i]) < 1e-14) ? 0.0 : diff1[i];
        diff2[i] = (fabs(diff2[i]) < 1e-14) ? 0.0 : diff2[i];
        BOOST_CHECK_CLOSE(diff1[i], diff2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismPhysDeriv_SumFac_VariableP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(7, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 6,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(8, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 8,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
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
        diff1[i] = (fabs(diff1[i]) < 1e-14) ? 0.0 : diff1[i];
        diff2[i] = (fabs(diff1[i]) < 1e-14) ? 0.0 : diff2[i];
        BOOST_CHECK_CLOSE(diff1[i], diff2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismPhysDeriv_MatrixFree_UniformP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    unsigned int numQuadPoints = 5;
    unsigned int numModes      = 2;

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(numQuadPoints,
                                                        PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(numQuadPoints,
                                                        PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, numModes,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(numQuadPoints - 1,
                                                        PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, numModes,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 2;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);

    // ... only one op at the time ...
    impTypes[Collections::ePhysDeriv] = Collections::eMatrixFree;
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::ePhysDeriv);

    const int nq = Exp->GetTotPoints();
    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);
    Array<OneD, NekDouble> phys(nelmts * nq), tmp, tmp1, tmp2;
    Array<OneD, NekDouble> diffRef(3 * nelmts * nq);
    Array<OneD, NekDouble> diff(3 * nelmts * nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
    }
    Exp->PhysDeriv(phys, diffRef, tmp1 = diffRef + (nelmts)*nq,
                   tmp2 = diffRef + (2 * nelmts) * nq);

    for (int i = 1; i < nelmts; ++i)
    {
        Vmath::Vcopy(nq, phys, 1, tmp = phys + i * nq, 1);
        Exp->PhysDeriv(phys, tmp = diffRef + i * nq,
                       tmp1 = diffRef + (nelmts + i) * nq,
                       tmp2 = diffRef + (2 * nelmts + i) * nq);
    }

    c.ApplyOperator(Collections::ePhysDeriv, phys, diff,
                    tmp = diff + nelmts * nq, tmp2 = diff + 2 * nelmts * nq);

    double epsilon = 1.0e-8;
    for (int i = 0; i < diffRef.size(); ++i)
    {
        diffRef[i] = (std::abs(diffRef[i]) < 1e-14) ? 0.0 : diffRef[i];
        diff[i]    = (std::abs(diff[i]) < 1e-14) ? 0.0 : diff[i];
        BOOST_CHECK_CLOSE(diffRef[i], diff[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(
    TestPrismIProductWRTDerivBase_IterPerExp_UniformP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(5, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 4,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(4, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 4,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nelmts * nq), tmp;
    Array<OneD, NekDouble> phys2(nelmts * nq);
    Array<OneD, NekDouble> phys3(nelmts * nq);
    Array<OneD, NekDouble> coeffs1(nelmts * nm);
    Array<OneD, NekDouble> coeffs2(nelmts * nm);

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys3[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
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
        coeffs1[i] = (fabs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (fabs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismIProductWRTDerivBase_StdMat_UniformP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(5, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 4,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(4, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 4,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eStdMat);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nelmts * nq), tmp;
    Array<OneD, NekDouble> phys2(nelmts * nq);
    Array<OneD, NekDouble> phys3(nelmts * nq);
    Array<OneD, NekDouble> coeffs1(nelmts * nm);
    Array<OneD, NekDouble> coeffs2(nelmts * nm);

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys3[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
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
        coeffs1[i] = (fabs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (fabs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismIProductWRTDerivBase_SumFac_UniformP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(5, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 4,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(4, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 4,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nelmts * nq), tmp;
    Array<OneD, NekDouble> phys2(nelmts * nq);
    Array<OneD, NekDouble> phys3(nelmts * nq);
    Array<OneD, NekDouble> coeffs1(nelmts * nm);
    Array<OneD, NekDouble> coeffs2(nelmts * nm);

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys3[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
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
        coeffs1[i] = (fabs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (fabs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(
    TestPrismIProductWRTDerivBase_IterPerExp_VariableP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(7, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 6,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(8, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 8,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nelmts * nq), tmp;
    Array<OneD, NekDouble> phys2(nelmts * nq);
    Array<OneD, NekDouble> phys3(nelmts * nq);
    Array<OneD, NekDouble> coeffs1(nelmts * nm);
    Array<OneD, NekDouble> coeffs2(nelmts * nm);

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys3[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
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
        coeffs1[i] = (fabs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (fabs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismIProductWRTDerivBase_StdMat_VariableP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(7, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 6,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(8, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 8,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eStdMat);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nelmts * nq), tmp;
    Array<OneD, NekDouble> phys2(nelmts * nq);
    Array<OneD, NekDouble> phys3(nelmts * nq);
    Array<OneD, NekDouble> coeffs1(nelmts * nm);
    Array<OneD, NekDouble> coeffs2(nelmts * nm);

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys3[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
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
        coeffs1[i] = (fabs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (fabs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismIProductWRTDerivBase_SumFac_VariableP_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(5, PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, 4,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(7, PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, 6,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(8, PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, 8,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    int nelmts = 10;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 3,
                                               Collections::eSumFac);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nelmts * nq), tmp;
    Array<OneD, NekDouble> phys2(nelmts * nq);
    Array<OneD, NekDouble> phys3(nelmts * nq);
    Array<OneD, NekDouble> coeffs1(nelmts * nm);
    Array<OneD, NekDouble> coeffs2(nelmts * nm);

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys3[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
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
        coeffs1[i] = (fabs(coeffs1[i]) < 1e-14) ? 0.0 : coeffs1[i];
        coeffs2[i] = (fabs(coeffs2[i]) < 1e-14) ? 0.0 : coeffs2[i];
        BOOST_CHECK_CLOSE(coeffs1[i], coeffs2[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(
    TestPrismIProductWRTDerivBase_MatriFree_UniformP_Undeformed_MultiElmt)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    unsigned int numQuadPoints = 7;
    unsigned int numModes      = 6;

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(numQuadPoints,
                                                        PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(numQuadPoints,
                                                        PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, numModes,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(numQuadPoints - 1,
                                                        PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, numModes,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    unsigned int nelmts = 1;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (unsigned int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);

    // ... only one op at the time ...
    impTypes[Collections::eIProductWRTDerivBase] = Collections::eMatrixFree;
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nelmts * nq), tmp;
    Array<OneD, NekDouble> phys2(nelmts * nq);
    Array<OneD, NekDouble> phys3(nelmts * nq);
    Array<OneD, NekDouble> coeffsRef(nelmts * nm);
    Array<OneD, NekDouble> coeffs(nelmts * nm);

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys3[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
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
        Exp->IProductWRTDerivBase(0, phys1 + i * nq, tmp = coeffsRef + i * nm);
        Exp->IProductWRTDerivBase(1, phys2 + i * nq, tmp = coeffs + i * nm);
        Vmath::Vadd(nm, coeffsRef + i * nm, 1, coeffs + i * nm, 1,
                    tmp = coeffsRef + i * nm, 1);
        Exp->IProductWRTDerivBase(2, phys3 + i * nq, tmp = coeffs + i * nm);
        Vmath::Vadd(nm, coeffsRef + i * nm, 1, coeffs + i * nm, 1,
                    tmp = coeffsRef + i * nm, 1);
    }

    c.ApplyOperator(Collections::eIProductWRTDerivBase, phys1, phys2, phys3,
                    coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffsRef.size(); ++i)
    {
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(
    TestPrismIProductWRTDerivBase_MatriFree_UniformP_Deformed_MultiElmt)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -2.0, -3.0, -4.0));
    SpatialDomains::PointGeomUniquePtr v1(
        new SpatialDomains::PointGeom(3u, 1u, 1.0, -1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v2(
        new SpatialDomains::PointGeom(3u, 2u, 1.0, 1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v3(
        new SpatialDomains::PointGeom(3u, 3u, -1.0, 1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v4(
        new SpatialDomains::PointGeom(3u, 4u, -1.0, -1.0, 1.0));
    SpatialDomains::PointGeomUniquePtr v5(
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    unsigned int numQuadPoints = 7;
    unsigned int numModes      = 6;

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(numQuadPoints,
                                                        PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(numQuadPoints,
                                                        PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, numModes,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(numQuadPoints - 1,
                                                        PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, numModes,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    unsigned int nelmts = 1;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (unsigned int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);

    // ... only one op at the time ...
    impTypes[Collections::eIProductWRTDerivBase] = Collections::eMatrixFree;
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nelmts * nq), tmp;
    Array<OneD, NekDouble> phys2(nelmts * nq);
    Array<OneD, NekDouble> phys3(nelmts * nq);
    Array<OneD, NekDouble> coeffsRef(nelmts * nm);
    Array<OneD, NekDouble> coeffs(nelmts * nm);

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys3[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
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
        Exp->IProductWRTDerivBase(0, phys1 + i * nq, tmp = coeffsRef + i * nm);
        Exp->IProductWRTDerivBase(1, phys2 + i * nq, tmp = coeffs + i * nm);
        Vmath::Vadd(nm, coeffsRef + i * nm, 1, coeffs + i * nm, 1,
                    tmp = coeffsRef + i * nm, 1);
        Exp->IProductWRTDerivBase(2, phys3 + i * nq, tmp = coeffs + i * nm);
        Vmath::Vadd(nm, coeffsRef + i * nm, 1, coeffs + i * nm, 1,
                    tmp = coeffsRef + i * nm, 1);
    }

    c.ApplyOperator(Collections::eIProductWRTDerivBase, phys1, phys2, phys3,
                    coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffsRef.size(); ++i)
    {
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(
    TestPrismIProductWRTDerivBase_MatriFree_UniformP_Deformed_OverInt_MultiElmt)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -2.0, -3.0, -4.0));
    SpatialDomains::PointGeomUniquePtr v1(
        new SpatialDomains::PointGeom(3u, 1u, 1.0, -1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v2(
        new SpatialDomains::PointGeom(3u, 2u, 1.0, 1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v3(
        new SpatialDomains::PointGeom(3u, 3u, -1.0, 1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v4(
        new SpatialDomains::PointGeom(3u, 4u, -1.0, -1.0, 1.0));
    SpatialDomains::PointGeomUniquePtr v5(
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    unsigned int numQuadPoints = 12;
    unsigned int numModes      = 6;

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(numQuadPoints,
                                                        PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(numQuadPoints,
                                                        PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, numModes,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(numQuadPoints - 1,
                                                        PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, numModes,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    unsigned int nelmts = 1;

    std::vector<StdRegions::StdExpansionSharedPtr> CollExp;
    for (unsigned int i = 0; i < nelmts; ++i)
    {
        CollExp.push_back(Exp);
    }

    LibUtilities::SessionReaderSharedPtr dummySession;
    Collections::CollectionOptimisation colOpt(dummySession, 2,
                                               Collections::eIterPerExp);
    Collections::OperatorImpMap impTypes = colOpt.GetOperatorImpMap(Exp);

    // ... only one op at the time ...
    impTypes[Collections::eIProductWRTDerivBase] = Collections::eMatrixFree;
    Collections::Collection c(CollExp, impTypes);
    c.Initialise(Collections::eIProductWRTDerivBase);

    const int nq = Exp->GetTotPoints();
    const int nm = Exp->GetNcoeffs();
    Array<OneD, NekDouble> phys1(nelmts * nq), tmp;
    Array<OneD, NekDouble> phys2(nelmts * nq);
    Array<OneD, NekDouble> phys3(nelmts * nq);
    Array<OneD, NekDouble> coeffsRef(nelmts * nm);
    Array<OneD, NekDouble> coeffs(nelmts * nm);

    Array<OneD, NekDouble> xc(nq), yc(nq), zc(nq);

    Exp->GetCoords(xc, yc, zc);

    for (int i = 0; i < nq; ++i)
    {
        phys1[i] = sin(xc[i]) * cos(yc[i]) * sin(zc[i]);
        phys2[i] = cos(xc[i]) * sin(yc[i]) * cos(zc[i]);
        phys3[i] = cos(xc[i]) * sin(yc[i]) * sin(zc[i]);
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
        Exp->IProductWRTDerivBase(0, phys1 + i * nq, tmp = coeffsRef + i * nm);
        Exp->IProductWRTDerivBase(1, phys2 + i * nq, tmp = coeffs + i * nm);
        Vmath::Vadd(nm, coeffsRef + i * nm, 1, coeffs + i * nm, 1,
                    tmp = coeffsRef + i * nm, 1);
        Exp->IProductWRTDerivBase(2, phys3 + i * nq, tmp = coeffs + i * nm);
        Vmath::Vadd(nm, coeffsRef + i * nm, 1, coeffs + i * nm, 1,
                    tmp = coeffsRef + i * nm, 1);
    }

    c.ApplyOperator(Collections::eIProductWRTDerivBase, phys1, phys2, phys3,
                    coeffs);

    double epsilon = 1.0e-8;
    for (int i = 0; i < coeffsRef.size(); ++i)
    {
        coeffsRef[i] = (std::abs(coeffsRef[i]) < 1e-14) ? 0.0 : coeffsRef[i];
        coeffs[i]    = (std::abs(coeffs[i]) < 1e-14) ? 0.0 : coeffs[i];
        BOOST_CHECK_CLOSE(coeffsRef[i], coeffs[i], epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismHelmholtz_IterPerExp_UniformP_ConstVarDiff)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    unsigned int numQuadPoints = 7;
    unsigned int numModes      = 6;

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(numQuadPoints,
                                                        PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(numQuadPoints,
                                                        PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, numModes,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(numQuadPoints - 1,
                                                        PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, numModes,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    Nektar::StdRegions::StdPrismExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdPrismExp>::AllocateSharedPtr(
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
    factors[StdRegions::eFactorLambda]   = 0.0;
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

BOOST_AUTO_TEST_CASE(TestPrismHelmholtz_MatrixFree_UniformP)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    unsigned int numQuadPoints = 7;
    unsigned int numModes      = 6;

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(numQuadPoints,
                                                        PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(numQuadPoints,
                                                        PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, numModes,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(numQuadPoints - 1,
                                                        PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, numModes,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    Nektar::StdRegions::StdPrismExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdPrismExp>::AllocateSharedPtr(
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

BOOST_AUTO_TEST_CASE(TestPrismHelmholtz_MatrixFree_Deformed_OverInt)
{
    SpatialDomains::PointGeomUniquePtr v0(
        new SpatialDomains::PointGeom(3u, 0u, -2.0, -3.0, -4.0));
    SpatialDomains::PointGeomUniquePtr v1(
        new SpatialDomains::PointGeom(3u, 1u, 1.0, -1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v2(
        new SpatialDomains::PointGeom(3u, 2u, 1.0, 1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v3(
        new SpatialDomains::PointGeom(3u, 3u, -1.0, 1.0, -1.0));
    SpatialDomains::PointGeomUniquePtr v4(
        new SpatialDomains::PointGeom(3u, 4u, -1.0, -1.0, 1.0));
    SpatialDomains::PointGeomUniquePtr v5(
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    unsigned int numQuadPoints = 10;
    unsigned int numModes      = 6;

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(numQuadPoints,
                                                        PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(numQuadPoints,
                                                        PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, numModes,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(numQuadPoints - 1,
                                                        PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, numModes,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    Nektar::StdRegions::StdPrismExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdPrismExp>::AllocateSharedPtr(
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

BOOST_AUTO_TEST_CASE(TestPrismHelmholtz_MatrixFree_UniformP_ConstVarDiff)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    unsigned int numQuadPoints = 7;
    unsigned int numModes      = 6;

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(numQuadPoints,
                                                        PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(numQuadPoints,
                                                        PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, numModes,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(numQuadPoints - 1,
                                                        PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, numModes,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    Nektar::StdRegions::StdPrismExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdPrismExp>::AllocateSharedPtr(
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
    factors[StdRegions::eFactorLambda]   = 0.0;
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

BOOST_AUTO_TEST_CASE(TestPrismPhsyInterp1DScaled_NoCollection_UniformP)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    unsigned int numQuadPoints = 7;
    unsigned int numModes      = 6;

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(numQuadPoints,
                                                        PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(numQuadPoints,
                                                        PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, numModes,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(numQuadPoints - 1,
                                                        PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, numModes,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    Nektar::StdRegions::StdPrismExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdPrismExp>::AllocateSharedPtr(
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
    Array<OneD, NekDouble> phys(nq);

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

    double epsilon = 2.0e-8;
    for (int i = 0; i < nq1; ++i)
    {
        NekDouble exact = pow(xc1[i], 3) + pow(yc1[i], 3) + pow(zc1[i], 3);
        phys1[i]        = (fabs(phys1[i]) < 1e-14) ? 0.0 : phys1[i];
        exact           = (fabs(exact) < 1e-14) ? 0.0 : exact;
        BOOST_CHECK_CLOSE(phys1[i], exact, epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestPrismPhsyInterp1DScaled_MatrixFree_UniformP)
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
        new SpatialDomains::PointGeom(3u, 5u, -1.0, 1.0, 1.0));

    std::array<SpatialDomains::SegGeomUniquePtr, 9> segVec;
    std::array<SpatialDomains::QuadGeomUniquePtr, 3> quadVec;
    std::array<SpatialDomains::TriGeomUniquePtr, 2> triVec;
    std::array<SpatialDomains::PointGeom *, 6> v = {
        v0.get(), v1.get(), v2.get(), v3.get(), v4.get(), v5.get()};
    SpatialDomains::PrismGeomUniquePtr prismGeom =
        CreatePrism(v, segVec, triVec, quadVec);

    unsigned int numQuadPoints = 7;
    unsigned int numModes      = 6;

    Nektar::LibUtilities::PointsType PointsTypeDir1 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir1(numQuadPoints,
                                                        PointsTypeDir1);
    Nektar::LibUtilities::BasisType basisTypeDir1 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir1(basisTypeDir1, numModes,
                                                      PointsKeyDir1);

    Nektar::LibUtilities::PointsType PointsTypeDir2 =
        Nektar::LibUtilities::eGaussLobattoLegendre;
    const Nektar::LibUtilities::PointsKey PointsKeyDir2(numQuadPoints,
                                                        PointsTypeDir2);
    Nektar::LibUtilities::BasisType basisTypeDir2 =
        Nektar::LibUtilities::eModified_A;
    const Nektar::LibUtilities::BasisKey basisKeyDir2(basisTypeDir2, numModes,
                                                      PointsKeyDir2);

    Nektar::LibUtilities::PointsType PointsTypeDir3 =
        Nektar::LibUtilities::eGaussRadauMAlpha1Beta0;
    const Nektar::LibUtilities::PointsKey PointsKeyDir3(numQuadPoints - 1,
                                                        PointsTypeDir3);
    Nektar::LibUtilities::BasisType basisTypeDir3 =
        Nektar::LibUtilities::eModified_B;
    const Nektar::LibUtilities::BasisKey basisKeyDir3(basisTypeDir3, numModes,
                                                      PointsKeyDir3);

    Nektar::LocalRegions::PrismExpSharedPtr Exp =
        MemoryManager<Nektar::LocalRegions::PrismExp>::AllocateSharedPtr(
            basisKeyDir1, basisKeyDir2, basisKeyDir3, prismGeom.get());

    Nektar::StdRegions::StdPrismExpSharedPtr stdExp =
        MemoryManager<Nektar::StdRegions::StdPrismExp>::AllocateSharedPtr(
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
    Array<OneD, NekDouble> phys(nq);

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
    for (int i = 0; i < nq1; ++i)
    {
        NekDouble exact = pow(xc1[i], 3) + pow(yc1[i], 3) + pow(zc1[i], 3);
        phys1[i]        = (fabs(phys1[i]) < 1e-14) ? 0.0 : phys1[i];
        exact           = (fabs(exact) < 1e-14) ? 0.0 : exact;
        BOOST_CHECK_CLOSE(phys1[i], exact, epsilon);
    }
}

} // namespace Nektar::PrismCollectionTests
