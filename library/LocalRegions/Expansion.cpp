///////////////////////////////////////////////////////////////////////////////
//
// File: Expansion.cpp
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
// Description: File for Expansion routines
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Foundations/Interp.h>
#include <LocalRegions/Expansion.h>
#include <LocalRegions/MatrixKey.h>

using namespace std;

namespace Nektar::LocalRegions
{
Expansion::Expansion(SpatialDomains::Geometry *pGeom)
    : m_indexMapManager(
          std::bind(&Expansion::CreateIndexMap, this, std::placeholders::_1),
          std::string("ExpansionIndexMap")),
      m_geom(pGeom), m_elementTraceLeft(-1), m_elementTraceRight(-1)
{
    GenGeomFactors();

    if (!m_geomFactors)
    {
        return;
    }

    if (!m_geomFactors->IsValid())
    {
        int nDim    = m_base.size();
        string type = "regular";
        if (m_geomFactors->GetGtype() == SpatialDomains::eDeformed)
        {
            type = "deformed";
        }

        stringstream err;
        err << nDim << "D " << type << " Jacobian not positive "
            << "(element ID = " << m_geom->GetGlobalID() << ") "
            << "(first vertex ID = " << m_geom->GetVid(0) << ")";
        NEKERROR(ErrorUtil::ewarning, err.str());
    }

    m_traceExp.clear();
}

Expansion::Expansion(const Expansion &pSrc)
    : StdExpansion(pSrc), m_indexMapManager(pSrc.m_indexMapManager),
      m_geom(pSrc.m_geom)
{
    m_geomFactors =
        ObjPoolManager<SpatialDomains::GeomFactors>::AllocateUniquePtr(
            *(pSrc.m_geomFactors));
}

Expansion::~Expansion()
{
}

DNekScalMatSharedPtr Expansion::GetLocMatrix(
    const LocalRegions::MatrixKey &mkey)
{
    return v_GetLocMatrix(mkey);
}

void Expansion::DropLocMatrix(const LocalRegions::MatrixKey &mkey)
{
    return v_DropLocMatrix(mkey);
}

DNekMatSharedPtr Expansion::BuildTransformationMatrix(
    const DNekScalMatSharedPtr &r_bnd, const StdRegions::MatrixType matrixType)
{
    return v_BuildTransformationMatrix(r_bnd, matrixType);
}

DNekMatSharedPtr Expansion::BuildVertexMatrix(const DNekScalMatSharedPtr &r_bnd)
{
    return v_BuildVertexMatrix(r_bnd);
}

void Expansion::ExtractDataToCoeffs(
    const NekDouble *data, const std::vector<unsigned int> &nummodes,
    const int nmodes_offset, NekDouble *coeffs,
    std::vector<LibUtilities::BasisType> &fromType)
{
    v_ExtractDataToCoeffs(data, nummodes, nmodes_offset, coeffs, fromType);
}

void Expansion::AddEdgeNormBoundaryInt(
    const int edge, const std::shared_ptr<Expansion> &EdgeExp,
    const Array<OneD, const NekDouble> &Fx,
    const Array<OneD, const NekDouble> &Fy, Array<OneD, NekDouble> &outarray)
{
    v_AddEdgeNormBoundaryInt(edge, EdgeExp, Fx, Fy, outarray);
}

void Expansion::AddEdgeNormBoundaryInt(
    const int edge, const std::shared_ptr<Expansion> &EdgeExp,
    const Array<OneD, const NekDouble> &Fn, Array<OneD, NekDouble> &outarray)
{
    v_AddEdgeNormBoundaryInt(edge, EdgeExp, Fn, outarray);
}

void Expansion::AddFaceNormBoundaryInt(
    const int face, const std::shared_ptr<Expansion> &FaceExp,
    const Array<OneD, const NekDouble> &Fn, Array<OneD, NekDouble> &outarray)
{
    v_AddFaceNormBoundaryInt(face, FaceExp, Fn, outarray);
}

void Expansion::DGDeriv(const int dir,
                        const Array<OneD, const NekDouble> &inarray,
                        Array<OneD, ExpansionSharedPtr> &EdgeExp,
                        Array<OneD, Array<OneD, NekDouble>> &coeffs,
                        Array<OneD, NekDouble> &outarray)
{
    v_DGDeriv(dir, inarray, EdgeExp, coeffs, outarray);
}

NekDouble Expansion::VectorFlux(const Array<OneD, Array<OneD, NekDouble>> &vec)
{
    return v_VectorFlux(vec);
}

void Expansion::NormalTraceDerivFactors(
    Array<OneD, Array<OneD, NekDouble>> &factors,
    Array<OneD, Array<OneD, NekDouble>> &d0factors,
    Array<OneD, Array<OneD, NekDouble>> &d1factors)
{
    return v_NormalTraceDerivFactors(factors, d0factors, d1factors);
}

DNekScalMatSharedPtr Expansion::GetLocMatrix(
    const StdRegions::MatrixType mtype,
    const StdRegions::ConstFactorMap &factors,
    const StdRegions::VarCoeffMap &varcoeffs)
{
    MatrixKey mkey(mtype, DetShapeType(), *this, factors, varcoeffs);
    return GetLocMatrix(mkey);
}

SpatialDomains::Geometry *Expansion::GetGeom() const
{
    return m_geom;
}

void Expansion::Reset()
{
    // Clear metrics
    m_metrics.clear();

    // Regenerate geometry factors
    GenGeomFactors();
}

IndexMapValuesSharedPtr Expansion::CreateIndexMap(const IndexMapKey &ikey)
{
    IndexMapValuesSharedPtr returnval;

    IndexMapType itype = ikey.GetIndexMapType();

    int entity = ikey.GetIndexEntity();

    StdRegions::Orientation orient = ikey.GetIndexOrientation();

    Array<OneD, unsigned int> map;
    Array<OneD, int> sign;

    switch (itype)
    {
        case eEdgeToElement:
        {
            GetTraceToElementMap(entity, map, sign, orient);
        }
        break;
        case eFaceToElement:
        {
            GetTraceToElementMap(entity, map, sign, orient);
        }
        break;
        case eEdgeInterior:
        {
            ASSERTL0(false, "Boundary Index Map not implemented yet.");
            // v_GetEdgeInteriorMap(entity,orient,map,sign);
        }
        break;
        case eFaceInterior:
        {
            ASSERTL0(false, "Boundary Index Map not implemented yet.");
            // v_GetFaceInteriorMap(entity,orient,map,sign);
        }
        break;
        case eBoundary:
        {
            ASSERTL0(false, "Boundary Index Map not implemented yet.");
        }
        break;
        case eVertex:
        {
            ASSERTL0(false, "Vertex Index Map not implemented yet.");
        }
        break;
        default:
        {
            ASSERTL0(false, "The Index Map you are requiring "
                            "is not between the possible options.");
        }
    }

    returnval = MemoryManager<IndexMapValues>::AllocateSharedPtr(map.size());

    for (int i = 0; i < map.size(); i++)
    {
        (*returnval)[i].index = map[i];
        (*returnval)[i].sign  = sign[i];
    }

    return returnval;
}

const NormalVector &Expansion::GetTraceNormal(const int id)
{
    std::map<int, NormalVector>::const_iterator x;
    x = m_traceNormals.find(id);

    // if edge normal not defined compute it
    if (x == m_traceNormals.end())
    {
        v_ComputeTraceNormal(id);
        x = m_traceNormals.find(id);
    }
    return x->second;
}

DNekScalMatSharedPtr Expansion::v_GetLocMatrix(
    [[maybe_unused]] const LocalRegions::MatrixKey &mkey)
{
    NEKERROR(ErrorUtil::efatal, "This function is only valid for LocalRegions");
    return NullDNekScalMatSharedPtr;
}

DNekScalBlkMatSharedPtr Expansion::CreateStaticCondMatrix(const MatrixKey &mkey)
{
    DNekScalBlkMatSharedPtr returnval;

    ASSERTL2(m_geomFactors->GetGtype() != SpatialDomains::eNoGeomType,
             "Geometric information is not set up");

    // set up block matrix system
    unsigned int nbdry      = NumBndryCoeffs();
    unsigned int nint       = (unsigned int)(m_ncoeffs - nbdry);
    unsigned int exp_size[] = {nbdry, nint};
    unsigned int nblks      = 2;
    returnval               = MemoryManager<DNekScalBlkMat>::AllocateSharedPtr(
        nblks, nblks, exp_size, exp_size);
    // Really need a constructor which takes Arrays
    NekDouble factor = 1.0;

    switch (mkey.GetMatrixType())
    {
        // this can only use stdregions statically condensed system
        // for mass matrix
        case StdRegions::eMass:
            if ((m_geomFactors->GetGtype() == SpatialDomains::eDeformed) ||
                (mkey.GetNVarCoeff()))
            {
                factor = 1.0;
                goto UseLocRegionsMatrix;
            }
            else
            {
                factor = (m_geomFactors->GetJac())[0];
                goto UseStdRegionsMatrix;
            }
            break;
        default: // use Deformed case for both
                 // regular and deformed geometries
            factor = 1.0;
            goto UseLocRegionsMatrix;
            break;
        UseStdRegionsMatrix:
        {
            NekDouble invfactor     = 1.0 / factor;
            NekDouble one           = 1.0;
            DNekBlkMatSharedPtr mat = GetStdStaticCondMatrix(mkey);
            DNekScalMatSharedPtr Atmp;
            DNekMatSharedPtr Asubmat;

            returnval->SetBlock(
                0, 0,
                Atmp = MemoryManager<DNekScalMat>::AllocateSharedPtr(
                    factor, Asubmat = mat->GetBlock(0, 0)));
            returnval->SetBlock(
                0, 1,
                Atmp = MemoryManager<DNekScalMat>::AllocateSharedPtr(
                    one, Asubmat = mat->GetBlock(0, 1)));
            returnval->SetBlock(
                1, 0,
                Atmp = MemoryManager<DNekScalMat>::AllocateSharedPtr(
                    factor, Asubmat = mat->GetBlock(1, 0)));
            returnval->SetBlock(
                1, 1,
                Atmp = MemoryManager<DNekScalMat>::AllocateSharedPtr(
                    invfactor, Asubmat = mat->GetBlock(1, 1)));
        }
        break;
        UseLocRegionsMatrix:
        {
            int i, j;
            NekDouble invfactor = 1.0 / factor;
            NekDouble one       = 1.0;
            DNekScalMat &mat    = *GetLocMatrix(mkey);
            DNekMatSharedPtr A =
                MemoryManager<DNekMat>::AllocateSharedPtr(nbdry, nbdry);
            DNekMatSharedPtr B =
                MemoryManager<DNekMat>::AllocateSharedPtr(nbdry, nint);
            DNekMatSharedPtr C =
                MemoryManager<DNekMat>::AllocateSharedPtr(nint, nbdry);
            DNekMatSharedPtr D =
                MemoryManager<DNekMat>::AllocateSharedPtr(nint, nint);

            Array<OneD, unsigned int> bmap(nbdry);
            Array<OneD, unsigned int> imap(nint);
            GetBoundaryMap(bmap);
            GetInteriorMap(imap);

            for (i = 0; i < nbdry; ++i)
            {
                for (j = 0; j < nbdry; ++j)
                {
                    (*A)(i, j) = mat(bmap[i], bmap[j]);
                }

                for (j = 0; j < nint; ++j)
                {
                    (*B)(i, j) = mat(bmap[i], imap[j]);
                }
            }

            for (i = 0; i < nint; ++i)
            {
                for (j = 0; j < nbdry; ++j)
                {
                    (*C)(i, j) = mat(imap[i], bmap[j]);
                }

                for (j = 0; j < nint; ++j)
                {
                    (*D)(i, j) = mat(imap[i], imap[j]);
                }
            }

            // Calculate static condensed system
            if (nint)
            {
                D->Invert();
                (*B) = (*B) * (*D);
                (*A) = (*A) - (*B) * (*C);
            }

            DNekScalMatSharedPtr Atmp;

            returnval->SetBlock(
                0, 0,
                Atmp =
                    MemoryManager<DNekScalMat>::AllocateSharedPtr(factor, A));
            returnval->SetBlock(
                0, 1,
                Atmp = MemoryManager<DNekScalMat>::AllocateSharedPtr(one, B));
            returnval->SetBlock(
                1, 0,
                Atmp =
                    MemoryManager<DNekScalMat>::AllocateSharedPtr(factor, C));
            returnval->SetBlock(
                1, 1,
                Atmp = MemoryManager<DNekScalMat>::AllocateSharedPtr(invfactor,
                                                                     D));

            // Remove the local matrix from manager if using this option since
            // we assume it is only created to generate static condensed system
            v_DropLocMatrix(mkey);
        }
    }
    return returnval;
}

void Expansion::v_DropLocMatrix(
    [[maybe_unused]] const LocalRegions::MatrixKey &mkey)
{
    NEKERROR(ErrorUtil::efatal, "This function is only valid for LocalRegions");
}

void Expansion::v_MultiplyByQuadratureMetric(
    const Array<OneD, const NekDouble> &inarray,
    Array<OneD, NekDouble> &outarray)
{
    const int nqtot = GetTotPoints();

    if (m_metrics.count(eMetricQuadrature) == 0)
    {
        ComputeQuadratureMetric();
    }

    Vmath::Vmul(nqtot, m_metrics[eMetricQuadrature], 1, inarray, 1, outarray,
                1);
}

void Expansion::v_DivideByQuadratureMetric(
    const Array<OneD, const NekDouble> &inarray,
    Array<OneD, NekDouble> &outarray)
{
    const int nqtot = GetTotPoints();

    if (m_metrics.count(eMetricQuadrature) == 0)
    {
        ComputeQuadratureMetric();
    }

    Vmath::Vdiv(nqtot, inarray, 1, m_metrics[eMetricQuadrature], 1, outarray,
                1);
    // remove NaN or Inf values and set to zero
    for (int i = 0; i < nqtot; ++i)
    {
        if (std::isnan(outarray[i]) || std::isinf(outarray[i]))
        {
            outarray[i] = 0.0;
        }
    }
}

void Expansion::ComputeLaplacianMetric()
{
    v_ComputeLaplacianMetric();
}

void Expansion::ComputeQuadratureMetric()
{
    unsigned int nqtot            = GetTotPoints();
    SpatialDomains::GeomType type = m_geomFactors->GetGtype();
    if (type == SpatialDomains::eRegular ||
        type == SpatialDomains::eMovingRegular)
    {
        m_metrics[eMetricQuadrature] =
            Array<OneD, NekDouble>(nqtot, m_geomFactors->GetJac()[0]);
    }
    else
    {
        m_metrics[eMetricQuadrature] = m_geomFactors->GetJac();
    }

    v_MultiplyByStdQuadratureMetric(m_metrics[eMetricQuadrature],
                                    m_metrics[eMetricQuadrature]);
}

void Expansion::StdDerivBaseOnTraceMat(Array<OneD, DNekMatSharedPtr> &DerivMat)
{
    int nquad   = GetTotPoints();
    int ntraces = GetNtraces();
    int ndir    = m_base.size();

    Array<OneD, NekDouble> coeffs(m_ncoeffs);
    Array<OneD, NekDouble> phys(nquad);

    Array<OneD, Array<OneD, int>> traceids(ntraces);

    int tottracepts = 0;
    for (int i = 0; i < ntraces; ++i)
    {
        GetTracePhysMap(i, traceids[i]);
        tottracepts += GetTraceNumPoints(i);
    }

    // initialise array to null so can call for
    // differnt dimensions
    Array<OneD, Array<OneD, NekDouble>> Deriv(3, NullNekDouble1DArray);

    DerivMat = Array<OneD, DNekMatSharedPtr>(ndir);

    for (int i = 0; i < ndir; ++i)
    {
        Deriv[i] = Array<OneD, NekDouble>(nquad);
        DerivMat[i] =
            MemoryManager<DNekMat>::AllocateSharedPtr(m_ncoeffs, tottracepts);
    }

    for (int i = 0; i < m_ncoeffs; ++i)
    {
        Vmath::Zero(m_ncoeffs, coeffs, 1);
        coeffs[i] = 1.0;
        BwdTrans(coeffs, phys);

        // dphi_i/d\xi_1,  dphi_i/d\xi_2  dphi_i/d\xi_3
        StdPhysDeriv(phys, Deriv[0], Deriv[1], Deriv[2]);

        int cnt = 0;
        for (int j = 0; j < ntraces; ++j)
        {
            int nTracePts = GetTraceNumPoints(j);
            for (int k = 0; k < nTracePts; ++k)
            {
                for (int d = 0; d < ndir; ++d)
                {
                    (*DerivMat[d])(i, cnt + k) = Deriv[d][traceids[j][k]];
                }
            }
            cnt += nTracePts;
        }
    }
}

void Expansion::PhysDerivBaseOnTraceMat(const int traceid,
                                        Array<OneD, DNekMatSharedPtr> &DerivMat)
{
    int nquad = GetTotPoints();
    int ndir  = m_base.size();

    Array<OneD, NekDouble> coeffs(m_ncoeffs);
    Array<OneD, NekDouble> phys(nquad);

    Array<OneD, int> tracePhysIds;
    GetTracePhysMap(traceid, tracePhysIds);

    int nTracePts = GetTraceNumPoints(traceid);

    // initialise array to null so can call for
    // differnt dimensions
    Array<OneD, Array<OneD, NekDouble>> Deriv(3, NullNekDouble1DArray);
    DerivMat = Array<OneD, DNekMatSharedPtr>(ndir);
    for (int d = 0; d < ndir; ++d)
    {
        Deriv[d]    = Array<OneD, NekDouble>(nquad);
        DerivMat[d] = MemoryManager<DNekMat>::AllocateSharedPtr(m_ncoeffs,
                                                                nTracePts, 0.0);
    }

    for (int i = 0; i < m_ncoeffs; ++i)
    {
        Vmath::Zero(m_ncoeffs, coeffs, 1);
        coeffs[i] = 1.0;
        BwdTrans(coeffs, phys);

        // dphi_i/d\xi_1,  dphi_i/d\xi_2  dphi_i/d\xi_3
        PhysDeriv(phys, Deriv[0], Deriv[1], Deriv[2]);

        for (int k = 0; k < nTracePts; ++k)
        {
            for (int d = 0; d < ndir; ++d)
            {
                (*DerivMat[d])(i, k) = Deriv[d][tracePhysIds[k]];
            }
        }
    }
}

void Expansion::PhysBaseOnTraceMat(const int traceid,
                                   DNekMatSharedPtr &BdataMat)
{
    int nquad = GetTotPoints();

    Array<OneD, NekDouble> coeffs(m_ncoeffs);
    Array<OneD, NekDouble> phys(nquad);

    Array<OneD, int> tracePhysIds;
    GetTracePhysMap(traceid, tracePhysIds);

    int nTracePts = GetTraceNumPoints(traceid);

    BdataMat =
        MemoryManager<DNekMat>::AllocateSharedPtr(m_ncoeffs, nTracePts, 0.0);

    for (int i = 0; i < m_ncoeffs; ++i)
    {
        Vmath::Zero(m_ncoeffs, coeffs, 1);
        coeffs[i] = 1.0;
        BwdTrans(coeffs, phys);

        for (int k = 0; k < nTracePts; ++k)
        {
            (*BdataMat)(i, k) = phys[tracePhysIds[k]];
        }
    }
}

void Expansion::v_GetCoords(Array<OneD, NekDouble> &coords_0,
                            Array<OneD, NekDouble> &coords_1,
                            Array<OneD, NekDouble> &coords_2)
{
    ASSERTL1(m_geom, "m_geom not defined");

    // get physical points defined in Geom
    m_geom->FillGeom();

    const int expDim = m_base.size();
    int nqGeom       = 1;
    bool doCopy      = true;

    Array<OneD, LibUtilities::BasisSharedPtr> CBasis(expDim);
    Array<OneD, Array<OneD, NekDouble>> tmp(3);

    for (int i = 0; i < expDim; ++i)
    {
        CBasis[i] = m_geom->GetXmap()->GetBasis(i);
        nqGeom *= CBasis[i]->GetNumPoints();
        doCopy = doCopy &&
                 m_base[i]->GetBasisKey().SamePoints(CBasis[i]->GetBasisKey());
    }

    tmp[0] = coords_0;
    tmp[1] = coords_1;
    tmp[2] = coords_2;

    if (doCopy)
    {
        for (int i = 0; i < m_geom->GetCoordim(); ++i)
        {
            m_geom->GetXmap()->BwdTrans(m_geom->GetCoeffs(i), tmp[i]);
        }
    }
    else
    {
        for (int i = 0; i < m_geom->GetCoordim(); ++i)
        {
            Array<OneD, NekDouble> tmpGeom(nqGeom);
            m_geom->GetXmap()->BwdTrans(m_geom->GetCoeffs(i), tmpGeom);

            switch (expDim)
            {
                case 1:
                {
                    LibUtilities::Interp1D(
                        CBasis[0]->GetPointsKey(), &tmpGeom[0],
                        m_base[0]->GetPointsKey(), &tmp[i][0]);
                    break;
                }
                case 2:
                {
                    LibUtilities::Interp2D(
                        CBasis[0]->GetPointsKey(), CBasis[1]->GetPointsKey(),
                        &tmpGeom[0], m_base[0]->GetPointsKey(),
                        m_base[1]->GetPointsKey(), &tmp[i][0]);
                    break;
                }
                case 3:
                {
                    LibUtilities::Interp3D(
                        CBasis[0]->GetPointsKey(), CBasis[1]->GetPointsKey(),
                        CBasis[2]->GetPointsKey(), &tmpGeom[0],
                        m_base[0]->GetPointsKey(), m_base[1]->GetPointsKey(),
                        m_base[2]->GetPointsKey(), &tmp[i][0]);
                    break;
                }
            }
        }
    }
}

void Expansion::ComputeGmatcdotMF(const Array<TwoD, const NekDouble> &df,
                                  const Array<OneD, const NekDouble> &direction,
                                  Array<OneD, Array<OneD, NekDouble>> &dfdir)
{
    int shapedim = dfdir.size();
    int coordim  = GetCoordim();
    int nqtot    = direction.size() / coordim;

    for (int j = 0; j < shapedim; j++)
    {
        dfdir[j] = Array<OneD, NekDouble>(nqtot, 0.0);
        for (int k = 0; k < coordim; k++)
        {
            if (m_geomFactors->GetGtype() == SpatialDomains::eDeformed)
            {
                Vmath::Vvtvp(nqtot, &df[shapedim * k + j][0], 1,
                             &direction[k * nqtot], 1, &dfdir[j][0], 1,
                             &dfdir[j][0], 1);
            }
            else
            {
                Vmath::Svtvp(nqtot, df[shapedim * k + j][0],
                             &direction[k * nqtot], 1, &dfdir[j][0], 1,
                             &dfdir[j][0], 1);
            }
        }
    }
}

// Get Moving frames
Array<OneD, NekDouble> Expansion::GetMF(
    const int dir, const int shapedim, const StdRegions::VarCoeffMap &varcoeffs)
{
    int coordim = GetCoordim();

    int nquad0, nquad1, nquad2;
    int nqtot = 1;
    switch (shapedim)
    {
        case 2:
        {
            nquad0 = m_base[0]->GetNumPoints();
            nquad1 = m_base[1]->GetNumPoints();
            nqtot  = nquad0 * nquad1;
            break;
        }
        case 3:
        {
            nquad0 = m_base[0]->GetNumPoints();
            nquad1 = m_base[1]->GetNumPoints();
            nquad2 = m_base[2]->GetNumPoints();
            nqtot  = nquad0 * nquad1 * nquad2;
            break;
        }
        default:
            break;
    }

    StdRegions::VarCoeffType MMFCoeffs[15] = {
        StdRegions::eVarCoeffMF1x,   StdRegions::eVarCoeffMF1y,
        StdRegions::eVarCoeffMF1z,   StdRegions::eVarCoeffMF1Div,
        StdRegions::eVarCoeffMF1Mag, StdRegions::eVarCoeffMF2x,
        StdRegions::eVarCoeffMF2y,   StdRegions::eVarCoeffMF2z,
        StdRegions::eVarCoeffMF2Div, StdRegions::eVarCoeffMF2Mag,
        StdRegions::eVarCoeffMF3x,   StdRegions::eVarCoeffMF3y,
        StdRegions::eVarCoeffMF3z,   StdRegions::eVarCoeffMF3Div,
        StdRegions::eVarCoeffMF3Mag};

    Array<OneD, NekDouble> MF(coordim * nqtot);
    Array<OneD, NekDouble> tmp(nqtot);
    for (int k = 0; k < coordim; k++)
    {
        StdRegions::VarCoeffMap::const_iterator MFdir =
            varcoeffs.find(MMFCoeffs[5 * dir + k]);
        tmp = MFdir->second.GetValue();

        Vmath::Vcopy(nqtot, &tmp[0], 1, &MF[k * nqtot], 1);
    }

    return MF;
}

// Get magnitude of MF
Array<OneD, NekDouble> Expansion::GetMFDiv(
    const int dir, const StdRegions::VarCoeffMap &varcoeffs)
{
    int indxDiv = 3;

    StdRegions::VarCoeffType MMFCoeffs[15] = {
        StdRegions::eVarCoeffMF1x,   StdRegions::eVarCoeffMF1y,
        StdRegions::eVarCoeffMF1z,   StdRegions::eVarCoeffMF1Div,
        StdRegions::eVarCoeffMF1Mag, StdRegions::eVarCoeffMF2x,
        StdRegions::eVarCoeffMF2y,   StdRegions::eVarCoeffMF2z,
        StdRegions::eVarCoeffMF2Div, StdRegions::eVarCoeffMF2Mag,
        StdRegions::eVarCoeffMF3x,   StdRegions::eVarCoeffMF3y,
        StdRegions::eVarCoeffMF3z,   StdRegions::eVarCoeffMF3Div,
        StdRegions::eVarCoeffMF3Mag};

    StdRegions::VarCoeffMap::const_iterator MFdir =
        varcoeffs.find(MMFCoeffs[5 * dir + indxDiv]);
    Array<OneD, NekDouble> MFDiv = MFdir->second.GetValue();

    return MFDiv;
}

// Get magnitude of MF
Array<OneD, NekDouble> Expansion::GetMFMag(
    const int dir, const StdRegions::VarCoeffMap &varcoeffs)
{
    int indxMag = 4;

    StdRegions::VarCoeffType MMFCoeffs[15] = {
        StdRegions::eVarCoeffMF1x,   StdRegions::eVarCoeffMF1y,
        StdRegions::eVarCoeffMF1z,   StdRegions::eVarCoeffMF1Div,
        StdRegions::eVarCoeffMF1Mag, StdRegions::eVarCoeffMF2x,
        StdRegions::eVarCoeffMF2y,   StdRegions::eVarCoeffMF2z,
        StdRegions::eVarCoeffMF2Div, StdRegions::eVarCoeffMF2Mag,
        StdRegions::eVarCoeffMF3x,   StdRegions::eVarCoeffMF3y,
        StdRegions::eVarCoeffMF3z,   StdRegions::eVarCoeffMF3Div,
        StdRegions::eVarCoeffMF3Mag};

    StdRegions::VarCoeffMap::const_iterator MFdir =
        varcoeffs.find(MMFCoeffs[5 * dir + indxMag]);
    Array<OneD, NekDouble> MFmag = MFdir->second.GetValue();

    return MFmag;
}

DNekMatSharedPtr Expansion::v_BuildTransformationMatrix(
    [[maybe_unused]] const DNekScalMatSharedPtr &r_bnd,
    [[maybe_unused]] const StdRegions::MatrixType matrixType)
{
    NEKERROR(ErrorUtil::efatal, "This function is only valid for LocalRegions");
    return NullDNekMatSharedPtr;
}

DNekMatSharedPtr Expansion::v_BuildVertexMatrix(
    [[maybe_unused]] const DNekScalMatSharedPtr &r_bnd)
{
    NEKERROR(ErrorUtil::efatal, "This function is only valid for LocalRegions");
    return NullDNekMatSharedPtr;
}

void Expansion::v_ExtractDataToCoeffs(
    [[maybe_unused]] const NekDouble *data,
    [[maybe_unused]] const std::vector<unsigned int> &nummodes,
    [[maybe_unused]] const int nmodes_offset,
    [[maybe_unused]] NekDouble *coeffs,
    [[maybe_unused]] std::vector<LibUtilities::BasisType> &fromType)
{
    NEKERROR(ErrorUtil::efatal, "This function is only valid for LocalRegions");
}

void Expansion::v_AddEdgeNormBoundaryInt(
    [[maybe_unused]] const int edge,
    [[maybe_unused]] const std::shared_ptr<Expansion> &EdgeExp,
    [[maybe_unused]] const Array<OneD, const NekDouble> &Fx,
    [[maybe_unused]] const Array<OneD, const NekDouble> &Fy,
    [[maybe_unused]] Array<OneD, NekDouble> &outarray)
{
    NEKERROR(ErrorUtil::efatal, "This function is only valid for LocalRegions");
}

void Expansion::v_AddEdgeNormBoundaryInt(
    [[maybe_unused]] const int edge,
    [[maybe_unused]] const std::shared_ptr<Expansion> &EdgeExp,
    [[maybe_unused]] const Array<OneD, const NekDouble> &Fn,
    [[maybe_unused]] Array<OneD, NekDouble> &outarray)
{
    NEKERROR(ErrorUtil::efatal, "This function is only valid for LocalRegions");
}

void Expansion::v_AddFaceNormBoundaryInt(
    [[maybe_unused]] const int face,
    [[maybe_unused]] const std::shared_ptr<Expansion> &FaceExp,
    [[maybe_unused]] const Array<OneD, const NekDouble> &Fn,
    [[maybe_unused]] Array<OneD, NekDouble> &outarray)
{
    NEKERROR(ErrorUtil::efatal, "This function is only valid for LocalRegions");
}

void Expansion::v_DGDeriv(
    [[maybe_unused]] const int dir,
    [[maybe_unused]] const Array<OneD, const NekDouble> &inarray,
    [[maybe_unused]] Array<OneD, ExpansionSharedPtr> &EdgeExp,
    [[maybe_unused]] Array<OneD, Array<OneD, NekDouble>> &coeffs,
    [[maybe_unused]] Array<OneD, NekDouble> &outarray)
{
    NEKERROR(ErrorUtil::efatal, "This function is only valid for LocalRegions");
}

NekDouble Expansion::v_VectorFlux(
    [[maybe_unused]] const Array<OneD, Array<OneD, NekDouble>> &vec)
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for "
             "shape expansion in LocalRegions, not parant class");
    return 0.0;
}

void Expansion::v_NormalTraceDerivFactors(
    [[maybe_unused]] Array<OneD, Array<OneD, NekDouble>> &factors,
    [[maybe_unused]] Array<OneD, Array<OneD, NekDouble>> &d0factors,
    [[maybe_unused]] Array<OneD, Array<OneD, NekDouble>> &d1factors)
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for "
             "shape expansion in LocalRegions, not parant class");
}

StdRegions::Orientation Expansion::v_GetTraceOrient([[maybe_unused]] int trace)
{
    return StdRegions::eForwards;
}

void Expansion::v_SetCoeffsToOrientation(
    [[maybe_unused]] StdRegions::Orientation dir,
    [[maybe_unused]] Array<OneD, const NekDouble> &inarray,
    [[maybe_unused]] Array<OneD, NekDouble> &outarray)
{
    NEKERROR(ErrorUtil::efatal, "This function is not defined for this shape");
}

void Expansion::v_GetTraceQFactors(
    [[maybe_unused]] const int trace,
    [[maybe_unused]] Array<OneD, NekDouble> &outarray)
{
    NEKERROR(ErrorUtil::efatal,
             "Method does not exist for this shape or library");
}

void Expansion::v_GetTracePhysVals(
    [[maybe_unused]] const int trace,
    [[maybe_unused]] const StdRegions::StdExpansionSharedPtr &TraceExp,
    [[maybe_unused]] const Array<OneD, const NekDouble> &inarray,
    [[maybe_unused]] Array<OneD, NekDouble> &outarray,
    [[maybe_unused]] StdRegions::Orientation orient)
{
    NEKERROR(ErrorUtil::efatal,
             "Method does not exist for this shape or library");
}

void Expansion::v_GetTracePhysMap([[maybe_unused]] const int edge,
                                  [[maybe_unused]] Array<OneD, int> &outarray)
{
    NEKERROR(ErrorUtil::efatal,
             "Method does not exist for this shape or library");
}

void Expansion::v_ReOrientTracePhysMap(
    [[maybe_unused]] const StdRegions::Orientation orient,
    [[maybe_unused]] Array<OneD, int> &idmap, [[maybe_unused]] const int nq0,
    [[maybe_unused]] const int nq1)
{
    NEKERROR(ErrorUtil::efatal,
             "Method does not exist for this shape or library");
}

void Expansion::v_GenTraceExp([[maybe_unused]] const int traceid,
                              [[maybe_unused]] ExpansionSharedPtr &exp)
{
    NEKERROR(ErrorUtil::efatal,
             "Method does not exist for this shape or library");
}

void Expansion::v_ComputeTraceNormal([[maybe_unused]] const int id)
{
    ASSERTL0(false, "Cannot compute trace normal for this expansion.");
}

const Array<OneD, const NekDouble> &Expansion::v_GetPhysNormals(void)
{
    NEKERROR(ErrorUtil::efatal, "This function is not valid for this class");
    return NullNekDouble1DArray;
}

void Expansion::v_SetPhysNormals(
    [[maybe_unused]] Array<OneD, const NekDouble> &normal)
{
    NEKERROR(ErrorUtil::efatal, "This function is not valid for this class");
}

void Expansion::v_SetUpPhysNormals([[maybe_unused]] const int edge)
{
    NEKERROR(ErrorUtil::efatal, "This function is not valid for this class");
}

void Expansion::v_AddRobinMassMatrix(
    [[maybe_unused]] const int trace,
    [[maybe_unused]] const Array<OneD, const NekDouble> &primCoeffs,
    [[maybe_unused]] DNekMatSharedPtr &inoutmat)
{
    NEKERROR(ErrorUtil::efatal, "This function is not valid for this class");
}

void Expansion::v_AddRobinTraceContribution(
    [[maybe_unused]] const int traceid,
    [[maybe_unused]] const Array<OneD, const NekDouble> &primCoeffs,
    [[maybe_unused]] const Array<OneD, NekDouble> &incoeffs,
    [[maybe_unused]] Array<OneD, NekDouble> &coeffs)
{
    NEKERROR(ErrorUtil::efatal, "This function is not valid for this class");
}

void Expansion::v_TraceNormLen([[maybe_unused]] const int traceid,
                               [[maybe_unused]] NekDouble &h,
                               [[maybe_unused]] NekDouble &p)
{
    NEKERROR(ErrorUtil::efatal, "This method has not been defined");
}

const Array<OneD, const NekDouble> &Expansion::GetElmtBndNormDirElmtLen(
    const int nbnd) const
{
    auto x = m_elmtBndNormDirElmtLen.find(nbnd);
    ASSERTL0(x != m_elmtBndNormDirElmtLen.end(),
             "m_elmtBndNormDirElmtLen normal not computed.");
    return x->second;
}

void Expansion::v_AlignVectorToCollapsedDir(
    [[maybe_unused]] const int dir,
    [[maybe_unused]] const Array<OneD, const NekDouble> &inarray,
    [[maybe_unused]] Array<OneD, Array<OneD, NekDouble>> &outarray)
{
    NEKERROR(ErrorUtil::efatal, "v_AlignVectorToCollapsedDir is not defined");
}

void GetTraceQuadRange(const LibUtilities::ShapeType shapeType,
                       const LibUtilities::BasisKeyVector &bkeys, int traceid,
                       std::vector<int> &q_begin, std::vector<int> &q_end)
{
    auto DIM = LibUtilities::ShapeTypeDimMap[shapeType];

    if (DIM == 1)
    {
        q_begin.resize(1);
        q_end.resize(1);
        LibUtilities::GetEffectiveQuadRange(bkeys[0].GetPointsKey(), q_begin[0],
                                            q_end[0]);
        return;
    }
    else if (DIM == 2)
    {
        q_begin.resize(2);
        q_end.resize(2);
        LibUtilities::GetEffectiveQuadRange(bkeys[0].GetPointsKey(), q_begin[0],
                                            q_end[0]);
        LibUtilities::GetEffectiveQuadRange(bkeys[1].GetPointsKey(), q_begin[1],
                                            q_end[1]);
    }
    else if (DIM == 3)
    {
        q_begin.resize(3);
        q_end.resize(3);
        LibUtilities::GetEffectiveQuadRange(bkeys[0].GetPointsKey(), q_begin[0],
                                            q_end[0]);
        LibUtilities::GetEffectiveQuadRange(bkeys[1].GetPointsKey(), q_begin[1],
                                            q_end[1]);
        LibUtilities::GetEffectiveQuadRange(bkeys[2].GetPointsKey(), q_begin[2],
                                            q_end[2]);
    }

    switch (shapeType)
    {
        case LibUtilities::eTriangle:
        {
            switch (traceid)
            {
                case 0:
                {
                    // assume it always includes the end points
                    q_begin[1] = 0;
                    q_end[1]   = 1;
                    return;
                }
                case 1:
                {
                    // assume it always includes the end points
                    q_begin[0] = bkeys[0].GetNumPoints() - 1;
                    q_end[0]   = bkeys[0].GetNumPoints();
                    return;
                }
                case 2:
                {
                    q_begin[0] = 0;
                    q_end[0]   = 1;
                    return;
                }
                default:
                {
                    NEKERROR(ErrorUtil::efatal, "Invalid trace id");
                    break;
                }
            }
        }
        break;
        case LibUtilities::eQuadrilateral:
        {
            switch (traceid)
            {
                case 0:
                {
                    q_begin[1] = 0;
                    q_end[1]   = 1;
                    return;
                }
                case 1:
                {
                    q_begin[0] = bkeys[0].GetNumPoints() - 1;
                    q_end[0]   = bkeys[0].GetNumPoints();
                    return;
                }
                case 2:
                {
                    q_begin[1] = bkeys[1].GetNumPoints() - 1;
                    q_end[1]   = bkeys[1].GetNumPoints();
                    return;
                }
                case 3:
                {
                    q_begin[0] = 0;
                    q_end[0]   = 1;
                    return;
                }
                default:
                {
                    NEKERROR(ErrorUtil::efatal, "Invalid trace id");
                    break;
                }
            }
        }
        break;
        case LibUtilities::eHexahedron:
        {
            switch (traceid)
            {
                case 0:
                {
                    q_begin[2] = 0;
                    q_end[2]   = 1;
                    return;
                }
                case 1:
                {
                    q_begin[1] = 0;
                    q_end[1]   = 1;
                    return;
                }
                case 2:
                {
                    q_begin[0] = bkeys[0].GetNumPoints() - 1;
                    q_end[0]   = bkeys[0].GetNumPoints();
                    return;
                }
                case 3:
                {
                    q_begin[1] = bkeys[1].GetNumPoints() - 1;
                    q_end[1]   = bkeys[1].GetNumPoints();
                    return;
                }
                case 4:
                {
                    q_begin[0] = 0;
                    q_end[0]   = 1;
                    return;
                }
                case 5:
                {
                    q_begin[2] = bkeys[2].GetNumPoints() - 1;
                    q_end[2]   = bkeys[2].GetNumPoints();
                    return;
                }
                default:
                {
                    NEKERROR(ErrorUtil::efatal, "Invalid trace id");
                    break;
                }
            }
        }
        break;
        case LibUtilities::eTetrahedron:
        {
            switch (traceid)
            {
                case 0:
                {
                    q_begin[2] = 0;
                    q_end[2]   = 1;
                    return;
                }
                case 1:
                {
                    q_begin[1] = 0;
                    q_end[1]   = 1;
                    return;
                }
                case 2:
                {
                    q_begin[0] = bkeys[0].GetNumPoints() - 1;
                    q_end[0]   = bkeys[0].GetNumPoints();
                    return;
                }
                case 3:
                {
                    q_begin[0] = 0;
                    q_end[0]   = 1;
                    return;
                }
                default:
                {
                    NEKERROR(ErrorUtil::efatal, "Invalid trace id");
                    break;
                }
            }
        }
        break;
        case LibUtilities::ePrism:
        {
            switch (traceid)
            {
                case 0:
                {
                    q_begin[2] = 0;
                    q_end[2]   = 1;
                    return;
                }
                case 1:
                {
                    q_begin[1] = 0;
                    q_end[1]   = 1;
                    return;
                }
                case 2:
                {
                    q_begin[0] = bkeys[0].GetNumPoints() - 1;
                    q_end[0]   = bkeys[0].GetNumPoints();
                    return;
                }
                case 3:
                {
                    q_begin[1] = bkeys[1].GetNumPoints() - 1;
                    q_end[1]   = bkeys[1].GetNumPoints();
                    return;
                }
                case 4:
                {
                    q_begin[0] = 0;
                    q_end[0]   = 1;
                    return;
                }
                default:
                {
                    NEKERROR(ErrorUtil::efatal, "Invalid trace id");
                    break;
                }
            }
        }
        break;
        case LibUtilities::ePyramid:
        {
            switch (traceid)
            {
                case 0:
                {
                    q_begin[2] = 0;
                    q_end[2]   = 1;
                    return;
                }
                case 1:
                {
                    q_begin[1] = 0;
                    q_end[1]   = 1;
                    return;
                }
                case 2:
                {
                    q_begin[0] = bkeys[0].GetNumPoints() - 1;
                    q_end[0]   = bkeys[0].GetNumPoints();
                    return;
                }
                case 3:
                {
                    q_begin[0] = bkeys[1].GetNumPoints() - 1;
                    q_end[0]   = bkeys[1].GetNumPoints();
                    return;
                }
                case 4:
                {
                    q_begin[1] = 0;
                    q_end[1]   = 1;
                    return;
                }
                default:
                {
                    NEKERROR(ErrorUtil::efatal, "Invalid trace id");
                    break;
                }
            }
        }
        break;
        default:
            return;
    }
}

} // namespace Nektar::LocalRegions
