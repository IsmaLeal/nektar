///////////////////////////////////////////////////////////////////////////////
//
// File: SwitchNodesPoints.h
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
// Description: Switch statement for different shape types over
// polynomial order and quadrature order to initialisation operator
// kernals
//
///////////////////////////////////////////////////////////////////////////////

// This code produces a mombo switch statement that is used in
// mutliple operators usign BOOST_PP. It also uses preprocessor directives
// based on the shape type and dimension to limit the code inclusion.

// This header is included in the OperatorType.h file. Ideally the
// operator() method would be a template in the Operator base class
// but becasue the operator{1,23}D is both a function and template that
// need to be in the inherited class it is not possible.

#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/arithmetic/mul.hpp>
#include <boost/preprocessor/comparison/not_equal.hpp>
#include <boost/preprocessor/repetition/for.hpp>
#include <boost/preprocessor/tuple/elem.hpp>

#include "SwitchLimits.h"

/** Macro tests the values of the tuple 'state' to see if the first
   element, given by BOOST_PP_TUPLE_ELEM(2, 0, state), is not equal to
   the second element plus one, given by
   BOOST_PP_INC(BOOST_PP_TUPLE_ELEM(2, 1, state)) and returns 1 if not
   equal otherwise zero */
#define TEST(r, state)                                                         \
    BOOST_PP_NOT_EQUAL(BOOST_PP_TUPLE_ELEM(3, 0, state),                       \
                       BOOST_PP_INC(BOOST_PP_TUPLE_ELEM(3, 2, state)))

#define TEST1(r, state)                                                        \
    BOOST_PP_NOT_EQUAL(BOOST_PP_TUPLE_ELEM(3, 1, state),                       \
                       BOOST_PP_INC(BOOST_PP_TUPLE_ELEM(3, 2, state)))

/** Macro returns an updated tuple where the first element, given by
   BOOST_PP_TUPLE_ELEM(2, 0, state), is incremented by one and the
   second element, given by BOOST_PP_TUPLE_ELEM(2, 1, state) remains
   the same*/
#define UPDATE(r, state)                                                       \
    (BOOST_PP_INC(BOOST_PP_TUPLE_ELEM(3, 0, state)),                           \
     BOOST_PP_TUPLE_ELEM(3, 1, state), BOOST_PP_TUPLE_ELEM(3, 2, state))

#define UPDATE1(r, state)                                                      \
    (BOOST_PP_TUPLE_ELEM(3, 0, state),                                         \
     BOOST_PP_INC(BOOST_PP_TUPLE_ELEM(3, 1, state)),                           \
     BOOST_PP_TUPLE_ELEM(3, 2, state))

#if defined(SHAPE_DIMENSION_1D)

/* Define the switch statement for 1D shape operators and min, max range */
#define OPERATOR1D_Q(r, state)                                                 \
    case BOOST_PP_TUPLE_ELEM(3, 1, state):                                     \
        operator1D<BOOST_PP_TUPLE_ELEM(3, 0, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 1, state)>(input, output);           \
        break;

#define OPERATOR1D_M(r, state)                                                 \
    case BOOST_PP_TUPLE_ELEM(3, 0, state):                                     \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r(                                                  \
                (BOOST_PP_TUPLE_ELEM(3, 0, state),                             \
                 BOOST_PP_TUPLE_ELEM(3, 0, state),                             \
                 BOOST_PP_MUL(2, BOOST_PP_TUPLE_ELEM(3, 0, state))),           \
                TEST1, UPDATE1, OPERATOR1D_Q) default                          \
                : operator1D(input, output);                                   \
            break;                                                             \
        }                                                                      \
        break;

/* start switch definition for operator 1D*/
{
#if defined(SHAPE_TYPE_SEG)

    switch (nm0)
    {
        /*
          expand switch statement for mm0  from 2 to 8 with an inner switch
          of quadrature modes from nm0 to 2*nm0
          , i.e.
              ..
              case 4:
              switch(nq0)
              {
              ...
                 case 5:
                 operator1D<4,5>(input,output);
                 break;
                 ...
                 }
        */

        BOOST_PP_FOR((MIN1D, MIN1D, MAX1D), TEST, UPDATE, OPERATOR1D_M);
        default:
            operator1D(input, output);
            break;
    }

#endif
}

#elif defined(SHAPE_DIMENSION_2D)

/* Define the switch statement for 2D shape operators and min, max range */
#define OPERATOR2D_Q_TRI(r, state)                                             \
    case BOOST_PP_TUPLE_ELEM(3, 1, state):                                     \
        operator2D<BOOST_PP_TUPLE_ELEM(3, 0, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 0, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 1, state),                           \
                   BOOST_PP_DEC(BOOST_PP_TUPLE_ELEM(3, 1, state))>(input,      \
                                                                   output);    \
        break;

#define OPERATOR2D_M_TRI(r, state)                                             \
    case BOOST_PP_TUPLE_ELEM(3, 0, state):                                     \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r(                                                  \
                (BOOST_PP_TUPLE_ELEM(3, 0, state),                             \
                 BOOST_PP_INC(BOOST_PP_TUPLE_ELEM(3, 0, state)),               \
                 BOOST_PP_MUL(2, BOOST_PP_TUPLE_ELEM(3, 0, state))),           \
                TEST1, UPDATE1, OPERATOR2D_Q_TRI) default                      \
                : operator2D(input, output);                                   \
            break;                                                             \
        }                                                                      \
        break;

#define OPERATOR2D_Q_QUAD(r, state)                                            \
    case BOOST_PP_TUPLE_ELEM(3, 1, state):                                     \
        operator2D<BOOST_PP_TUPLE_ELEM(3, 0, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 0, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 1, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 1, state)>(input, output);           \
        break;

#define OPERATOR2D_M_QUAD(r, state)                                            \
    case BOOST_PP_TUPLE_ELEM(3, 0, state):                                     \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r(                                                  \
                (BOOST_PP_TUPLE_ELEM(3, 0, state),                             \
                 BOOST_PP_TUPLE_ELEM(3, 0, state),                             \
                 BOOST_PP_MUL(2, BOOST_PP_TUPLE_ELEM(3, 0, state))),           \
                TEST1, UPDATE1, OPERATOR2D_Q_QUAD) default                     \
                : operator2D(input, output);                                   \
            break;                                                             \
        }                                                                      \
        break;

/* start switch definition for operator 2D*/
{

#if defined(SHAPE_TYPE_TRI)

    if (nm0 == nm1 && nq0 == nq1 + 1)
    {
        switch (nm0)
        {
            BOOST_PP_FOR((MIN2D, MIN2D, MAX2D), TEST, UPDATE, OPERATOR2D_M_TRI);
            default:
                operator2D(input, output);
                break;
        }
    }

#elif defined(SHAPE_TYPE_QUAD)

    if (nm0 == nm1 && nq0 == nq1)
    {
        switch (nm0)
        {
            BOOST_PP_FOR((MIN2D, MIN2D, MAX2D), TEST, UPDATE,
                         OPERATOR2D_M_QUAD);
            default:
                operator2D(input, output);
                break;
        }
    }

#endif // SHAPE_TYPE

    else
    {
        operator2D(input, output);
    }
}

#elif defined(SHAPE_DIMENSION_3D)

/* Define the switch statement for 3D shape operators and min, max range */

#define OPERATOR3D_Q_HEX(r, state)                                             \
    case BOOST_PP_TUPLE_ELEM(3, 1, state):                                     \
        operator3D<BOOST_PP_TUPLE_ELEM(3, 0, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 0, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 0, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 1, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 1, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 1, state)>(input, output);           \
        break;

#define OPERATOR3D_M_HEX(r, state)                                             \
    case BOOST_PP_TUPLE_ELEM(3, 0, state):                                     \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r(                                                  \
                (BOOST_PP_TUPLE_ELEM(3, 0, state),                             \
                 BOOST_PP_TUPLE_ELEM(3, 0, state),                             \
                 BOOST_PP_MUL(2, BOOST_PP_TUPLE_ELEM(3, 0, state))),           \
                TEST1, UPDATE1, OPERATOR3D_Q_HEX) default                      \
                : operator3D(input, output);                                   \
            break;                                                             \
        }                                                                      \
        break;

#define OPERATOR3D_Q_TET(r, state)                                             \
    case BOOST_PP_TUPLE_ELEM(3, 1, state):                                     \
        operator3D<BOOST_PP_TUPLE_ELEM(3, 0, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 0, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 0, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 1, state),                           \
                   BOOST_PP_DEC(BOOST_PP_TUPLE_ELEM(3, 1, state)),             \
                   BOOST_PP_DEC(BOOST_PP_TUPLE_ELEM(3, 1, state))>(input,      \
                                                                   output);    \
        break;

#define OPERATOR3D_M_TET(r, state)                                             \
    case BOOST_PP_TUPLE_ELEM(3, 0, state):                                     \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r(                                                  \
                (BOOST_PP_TUPLE_ELEM(3, 0, state),                             \
                 BOOST_PP_INC(BOOST_PP_TUPLE_ELEM(3, 0, state)),               \
                 BOOST_PP_MUL(2, BOOST_PP_TUPLE_ELEM(3, 0, state))),           \
                TEST1, UPDATE1, OPERATOR3D_Q_TET) default                      \
                : operator3D(input, output);                                   \
            break;                                                             \
        }                                                                      \
        break;

#define OPERATOR3D_Q_PYR_PRISM(r, state)                                       \
    case BOOST_PP_TUPLE_ELEM(3, 1, state):                                     \
        operator3D<BOOST_PP_TUPLE_ELEM(3, 0, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 0, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 0, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 1, state),                           \
                   BOOST_PP_TUPLE_ELEM(3, 1, state),                           \
                   BOOST_PP_DEC(BOOST_PP_TUPLE_ELEM(3, 1, state))>(input,      \
                                                                   output);    \
        break;

#define OPERATOR3D_M_PYR_PRISM(r, state)                                       \
    case BOOST_PP_TUPLE_ELEM(3, 0, state):                                     \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r(                                                  \
                (BOOST_PP_TUPLE_ELEM(3, 0, state),                             \
                 BOOST_PP_INC(BOOST_PP_TUPLE_ELEM(3, 0, state)),               \
                 BOOST_PP_MUL(2, BOOST_PP_TUPLE_ELEM(3, 0, state))),           \
                TEST1, UPDATE1, OPERATOR3D_Q_PYR_PRISM) default                \
                : operator3D(input, output);                                   \
            break;                                                             \
        }                                                                      \
        break;

/* start switch definition for operator 3D*/
{
#if defined(SHAPE_TYPE_HEX)

    if (nm0 == nm1 && nm0 == nm2 && nq0 == nq1 && nq0 == nq2)
    {
        switch (nm0)
        {
            BOOST_PP_FOR((MIN3D, MIN3D, MAX3D), TEST, UPDATE, OPERATOR3D_M_HEX);
            default:
                operator3D(input, output);
                break;
        }
    }

#elif defined(SHAPE_TYPE_TET)

    if (nm0 == nm1 && nm0 == nm2 && nq0 == nq1 + 1 && nq0 == nq2 + 1)
    {
        switch (nm0)
        {
            BOOST_PP_FOR((MIN3D, MIN3D, MAX3D), TEST, UPDATE, OPERATOR3D_M_TET);
            default:
                operator3D(input, output);
                break;
        }
    }

#elif defined(SHAPE_TYPE_PYR) || defined(SHAPE_TYPE_PRISM)

    if (nm0 == nm1 && nm0 == nm2 && nq0 == nq1 && nq0 == nq2 + 1)
    {
        switch (nm0)
        {
            BOOST_PP_FOR((MIN3D, MIN3D, MAX3D), TEST, UPDATE,
                         OPERATOR3D_M_PYR_PRISM);
            default:
                operator3D(input, output);
                break;
        }
    }

#endif // SHAPE_TYPE

    else
    {
        operator3D(input, output);
    }
}

#endif // SHAPE_DIMENSION
