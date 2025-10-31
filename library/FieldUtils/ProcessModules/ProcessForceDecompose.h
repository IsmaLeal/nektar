////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessForceDecompose.h
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
//  Description: Perform force decomposition.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef FIELDUTILS_PROCESSFORCEDECOMPOSE_H
#define FIELDUTILS_PROCESSFORCEDECOMPOSE_H

#include "../Module.h"

namespace Nektar::FieldUtils
{

/**
 * @brief This processing module calculates the Q Criterion and adds it
 * as an extra-field to the output file.
 */
class ProcessForceDecompose : public ProcessModule
{
public:
    ProcessForceDecompose(){};
    ProcessForceDecompose(FieldSharedPtr f);

protected:
    NekDouble PhysIntegral(MultiRegions::ExpListSharedPtr exp,
                           Array<OneD, NekDouble> value);
    void GetQ(const Array<OneD, MultiRegions::ExpListSharedPtr> exp,
              Array<OneD, NekDouble> &Q);
    void QFromField(const Array<OneD, MultiRegions::ExpListSharedPtr> exp,
                    Array<OneD, NekDouble> &Q);
    void QFromPressure(const Array<OneD, MultiRegions::ExpListSharedPtr> exp,
                       Array<OneD, NekDouble> &Q);
    NekDouble GetViscosity();
    void GetGradPressure(const Array<OneD, MultiRegions::ExpListSharedPtr> exp,
                         Array<OneD, Array<OneD, NekDouble>> &gradp);
    void GetVelocity(const Array<OneD, MultiRegions::ExpListSharedPtr> exp,
                     Array<OneD, Array<OneD, NekDouble>> &vel);
    void GetLaplaceVelocity(
        const Array<OneD, MultiRegions::ExpListSharedPtr> exp,
        Array<OneD, Array<OneD, NekDouble>> &lapvel);
    void GetStressTensor(const Array<OneD, MultiRegions::ExpListSharedPtr> exp,
                         Array<OneD, Array<OneD, NekDouble>> &shear);
    void GetPhi(const Array<OneD, MultiRegions::ExpListSharedPtr> exp,
                Array<OneD, Array<OneD, NekDouble>> &phi,
                std::map<int, std::string> &Iphi);
    void GetInfoPhi(std::map<int, std::string> &Iphi);
    void VolumeIntegrateForce(const MultiRegions::ExpListSharedPtr &field,
                              const Array<OneD, Array<OneD, NekDouble>> &data,
                              const LibUtilities::CommSharedPtr &comm,
                              std::map<int, std::string> &Iphi,
                              Array<OneD, NekDouble> &BoundBox, int dir);
    int FindVariable(const std::string &var);
    int m_spacedim;
    int m_expdim;
};
} // namespace Nektar::FieldUtils

#endif
