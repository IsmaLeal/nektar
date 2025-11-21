////////////////////////////////////////////////////////////////////////////////
//
//  File: MeshGraphIOXmlCompressed.cpp
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
//  Description:
//
//
////////////////////////////////////////////////////////////////////////////////

#include "MeshGraphIOXmlCompressed.h"

#include <LibUtilities/BasicUtils/CompressData.h>
#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <LibUtilities/BasicUtils/ParseUtils.h>

#include <LibUtilities/Interpreter/Interpreter.h>
#include <SpatialDomains/MeshEntities.hpp>

// These are required for the Write(...) and Import(...) functions.
#include <boost/algorithm/string/predicate.hpp>
#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <boost/format.hpp>

#include <tinyxml.h>

namespace Nektar::SpatialDomains
{

std::string MeshGraphIOXmlCompressed::className =
    GetMeshGraphIOFactory().RegisterCreatorFunction(
        "XmlCompressed", MeshGraphIOXmlCompressed::create,
        "IO with compressed Xml geometry");

void MeshGraphIOXmlCompressed::v_ReadVertices()
{
    int spaceDimension = m_meshGraph->GetSpaceDimension();

    // Now read the vertices
    TiXmlElement *element = m_xmlGeom->FirstChildElement("VERTEX");
    ASSERTL0(element, "Unable to find mesh VERTEX tag in file.");

    NekDouble xscale, yscale, zscale;

    // check to see if any scaling parameters are in
    // attributes and determine these values
    LibUtilities::Interpreter expEvaluator;
    const char *xscal = element->Attribute("XSCALE");
    if (!xscal)
    {
        xscale = 1.0;
    }
    else
    {
        std::string xscalstr = xscal;
        int expr_id          = expEvaluator.DefineFunction("", xscalstr);
        xscale               = expEvaluator.Evaluate(expr_id);
    }

    const char *yscal = element->Attribute("YSCALE");
    if (!yscal)
    {
        yscale = 1.0;
    }
    else
    {
        std::string yscalstr = yscal;
        int expr_id          = expEvaluator.DefineFunction("", yscalstr);
        yscale               = expEvaluator.Evaluate(expr_id);
    }

    const char *zscal = element->Attribute("ZSCALE");
    if (!zscal)
    {
        zscale = 1.0;
    }
    else
    {
        std::string zscalstr = zscal;
        int expr_id          = expEvaluator.DefineFunction("", zscalstr);
        zscale               = expEvaluator.Evaluate(expr_id);
    }

    NekDouble xmove, ymove, zmove;

    // check to see if any moving parameters are in
    // attributes and determine these values

    const char *xmov = element->Attribute("XMOVE");
    if (!xmov)
    {
        xmove = 0.0;
    }
    else
    {
        std::string xmovstr = xmov;
        int expr_id         = expEvaluator.DefineFunction("", xmovstr);
        xmove               = expEvaluator.Evaluate(expr_id);
    }

    const char *ymov = element->Attribute("YMOVE");
    if (!ymov)
    {
        ymove = 0.0;
    }
    else
    {
        std::string ymovstr = ymov;
        int expr_id         = expEvaluator.DefineFunction("", ymovstr);
        ymove               = expEvaluator.Evaluate(expr_id);
    }

    const char *zmov = element->Attribute("ZMOVE");
    if (!zmov)
    {
        zmove = 0.0;
    }
    else
    {
        std::string zmovstr = zmov;
        int expr_id         = expEvaluator.DefineFunction("", zmovstr);
        zmove               = expEvaluator.Evaluate(expr_id);
    }

    NekDouble zrotate;

    const char *zrot = element->Attribute("ZROT");
    if (!zrot)
    {
        zrotate = 0.0;
    }
    else
    {
        std::string zrotstr = zrot;
        int expr_id         = expEvaluator.DefineFunction("", zrotstr);
        zrotate             = expEvaluator.Evaluate(expr_id);
    }

    std::string IsCompressed;
    element->QueryStringAttribute("COMPRESSED", &IsCompressed);

    if (boost::iequals(IsCompressed,
                       LibUtilities::CompressData::GetCompressString()))
    {
        // Extract the vertex body
        TiXmlNode *vertexChild = element->FirstChild();
        ASSERTL0(vertexChild, "Unable to extract the data from the compressed "
                              "vertex tag.");

        std::string vertexStr;
        if (vertexChild->Type() == TiXmlNode::TINYXML_TEXT)
        {
            vertexStr += vertexChild->ToText()->ValueStr();
        }

        std::vector<SpatialDomains::MeshVertex> vertData;
        LibUtilities::CompressData::ZlibDecodeFromBase64Str(vertexStr,
                                                            vertData);

        int indx;
        NekDouble xval, yval, zval;
        for (int i = 0; i < vertData.size(); ++i)
        {
            indx = vertData[i].id;
            xval = vertData[i].x;
            yval = vertData[i].y;
            zval = vertData[i].z;

            xval = xval * xscale + xmove;
            yval = yval * yscale + ymove;
            zval = zval * zscale + zmove;

            if (zrotate != 0.0)
            {
                NekDouble xval_tmp = xval * cos(zrotate) - yval * sin(zrotate);
                yval               = xval * sin(zrotate) + yval * cos(zrotate);
                xval               = xval_tmp;
            }

            m_meshGraph->CreatePointGeom(spaceDimension, indx, xval, yval,
                                         zval);
        }
    }
    else
    {
        ASSERTL0(false, "Compressed formats do not match. Expected :" +
                            LibUtilities::CompressData::GetCompressString() +
                            " but got " + IsCompressed);
    }
}

void MeshGraphIOXmlCompressed::v_ReadCurves()
{
    auto &curvedEdges  = m_meshGraph->GetCurvedEdges();
    auto &curvedFaces  = m_meshGraph->GetCurvedFaces();
    auto &curveNodes   = m_meshGraph->GetAllCurveNodes();
    int spaceDimension = m_meshGraph->GetSpaceDimension();

    // check to see if any scaling parameters are in
    // attributes and determine these values
    TiXmlElement *element = m_xmlGeom->FirstChildElement("VERTEX");
    ASSERTL0(element, "Unable to find mesh VERTEX tag in file.");

    NekDouble xscale, yscale, zscale;

    LibUtilities::Interpreter expEvaluator;
    const char *xscal = element->Attribute("XSCALE");
    if (!xscal)
    {
        xscale = 1.0;
    }
    else
    {
        std::string xscalstr = xscal;
        int expr_id          = expEvaluator.DefineFunction("", xscalstr);
        xscale               = expEvaluator.Evaluate(expr_id);
    }

    const char *yscal = element->Attribute("YSCALE");
    if (!yscal)
    {
        yscale = 1.0;
    }
    else
    {
        std::string yscalstr = yscal;
        int expr_id          = expEvaluator.DefineFunction("", yscalstr);
        yscale               = expEvaluator.Evaluate(expr_id);
    }

    const char *zscal = element->Attribute("ZSCALE");
    if (!zscal)
    {
        zscale = 1.0;
    }
    else
    {
        std::string zscalstr = zscal;
        int expr_id          = expEvaluator.DefineFunction("", zscalstr);
        zscale               = expEvaluator.Evaluate(expr_id);
    }

    NekDouble xmove, ymove, zmove;

    // check to see if any moving parameters are in
    // attributes and determine these values

    const char *xmov = element->Attribute("XMOVE");
    if (!xmov)
    {
        xmove = 0.0;
    }
    else
    {
        std::string xmovstr = xmov;
        int expr_id         = expEvaluator.DefineFunction("", xmovstr);
        xmove               = expEvaluator.Evaluate(expr_id);
    }

    const char *ymov = element->Attribute("YMOVE");
    if (!ymov)
    {
        ymove = 0.0;
    }
    else
    {
        std::string ymovstr = ymov;
        int expr_id         = expEvaluator.DefineFunction("", ymovstr);
        ymove               = expEvaluator.Evaluate(expr_id);
    }

    const char *zmov = element->Attribute("ZMOVE");
    if (!zmov)
    {
        zmove = 0.0;
    }
    else
    {
        std::string zmovstr = zmov;
        int expr_id         = expEvaluator.DefineFunction("", zmovstr);
        zmove               = expEvaluator.Evaluate(expr_id);
    }

    NekDouble zrotate;

    const char *zrot = element->Attribute("ZROT");
    if (!zrot)
    {
        zrotate = 0.0;
    }
    else
    {
        std::string zrotstr = zrot;
        int expr_id         = expEvaluator.DefineFunction("", zrotstr);
        zrotate             = expEvaluator.Evaluate(expr_id);
    }

    /// Look for elements in CURVE block.
    TiXmlElement *field = m_xmlGeom->FirstChildElement("CURVED");

    if (!field) // return if no curved entities
    {
        return;
    }

    std::string IsCompressed;
    field->QueryStringAttribute("COMPRESSED", &IsCompressed);

    if (IsCompressed.size() == 0)
    {
        // this could be that the curved tag is empty
        // in this case we dont want to read it
        return;
    }

    ASSERTL0(boost::iequals(IsCompressed,
                            LibUtilities::CompressData::GetCompressString()),
             "Compressed formats do not match. Expected :" +
                 LibUtilities::CompressData::GetCompressString() + " but got " +
                 IsCompressed);

    std::vector<SpatialDomains::MeshCurvedInfo> edginfo;
    std::vector<SpatialDomains::MeshCurvedInfo> facinfo;
    SpatialDomains::MeshCurvedPts cpts;

    // read edge, face info and curved poitns.
    TiXmlElement *x = field->FirstChildElement();
    while (x)
    {
        const char *entitytype = x->Value();
        // read in edge or face info
        if (boost::iequals(entitytype, "E"))
        {
            // read in data
            std::string elmtStr;
            TiXmlNode *child = x->FirstChild();

            if (child->Type() == TiXmlNode::TINYXML_TEXT)
            {
                elmtStr += child->ToText()->ValueStr();
            }

            LibUtilities::CompressData::ZlibDecodeFromBase64Str(elmtStr,
                                                                edginfo);
        }
        else if (boost::iequals(entitytype, "F"))
        {
            // read in data
            std::string elmtStr;
            TiXmlNode *child = x->FirstChild();

            if (child->Type() == TiXmlNode::TINYXML_TEXT)
            {
                elmtStr += child->ToText()->ValueStr();
            }

            LibUtilities::CompressData::ZlibDecodeFromBase64Str(elmtStr,
                                                                facinfo);
        }
        else if (boost::iequals(entitytype, "DATAPOINTS"))
        {
            int32_t id;
            ASSERTL0(x->Attribute("ID", &id),
                     "Failed to get ID from PTS section");
            cpts.id = id;

            // read in data
            std::string elmtStr;

            TiXmlElement *DataIdx = x->FirstChildElement("INDEX");
            ASSERTL0(DataIdx, "Cannot read data index tag in compressed "
                              "curved section");

            TiXmlNode *child = DataIdx->FirstChild();
            if (child->Type() == TiXmlNode::TINYXML_TEXT)
            {
                elmtStr = child->ToText()->ValueStr();
            }

            LibUtilities::CompressData::ZlibDecodeFromBase64Str(elmtStr,
                                                                cpts.index);

            TiXmlElement *DataPts = x->FirstChildElement("POINTS");
            ASSERTL0(DataPts, "Cannot read data pts tag in compressed "
                              "curved section");

            child = DataPts->FirstChild();
            if (child->Type() == TiXmlNode::TINYXML_TEXT)
            {
                elmtStr = child->ToText()->ValueStr();
            }

            LibUtilities::CompressData::ZlibDecodeFromBase64Str(elmtStr,
                                                                cpts.pts);
        }
        else
        {
            ASSERTL0(false, "Unknown tag in curved section");
        }
        x = x->NextSiblingElement();
    }

    // rescale (x,y,z) points;
    for (int i = 0; i < cpts.pts.size(); ++i)
    {
        cpts.pts[i].x = xscale * cpts.pts[i].x + xmove;
        cpts.pts[i].y = yscale * cpts.pts[i].y + ymove;
        cpts.pts[i].z = zscale * cpts.pts[i].z + zmove;

        if (zrotate != 0.0)
        {
            NekDouble xval_tmp =
                cpts.pts[i].x * cos(zrotate) - cpts.pts[i].y * sin(zrotate);
            cpts.pts[i].y =
                cpts.pts[i].x * sin(zrotate) + cpts.pts[i].y * cos(zrotate);
            cpts.pts[i].x = xval_tmp;
        }
    }

    for (int i = 0; i < edginfo.size(); ++i)
    {
        int edgeid = edginfo[i].entityid;
        LibUtilities::PointsType ptype;

        curvedEdges[edgeid] = ObjPoolManager<Curve>::AllocateUniquePtr(
            edgeid, ptype = (LibUtilities::PointsType)edginfo[i].ptype);

        // load points
        int offset = edginfo[i].ptoffset;
        for (int j = 0; j < edginfo[i].npoints; ++j)
        {
            int idx = cpts.index[offset + j];
            curveNodes.emplace_back(
                ObjPoolManager<PointGeom>::AllocateUniquePtr(
                    spaceDimension, edginfo[i].id, cpts.pts[idx].x,
                    cpts.pts[idx].y, cpts.pts[idx].z));
            curvedEdges[edgeid]->m_points.emplace_back(curveNodes.back().get());
        }
    }

    for (int i = 0; i < facinfo.size(); ++i)
    {
        int faceid = facinfo[i].entityid;
        LibUtilities::PointsType ptype;

        curvedFaces[faceid] = ObjPoolManager<Curve>::AllocateUniquePtr(
            faceid, ptype = (LibUtilities::PointsType)facinfo[i].ptype);

        int offset = facinfo[i].ptoffset;
        for (int j = 0; j < facinfo[i].npoints; ++j)
        {
            int idx = cpts.index[offset + j];
            curveNodes.emplace_back(
                ObjPoolManager<PointGeom>::AllocateUniquePtr(
                    spaceDimension, facinfo[i].id, cpts.pts[idx].x,
                    cpts.pts[idx].y, cpts.pts[idx].z));
            curvedFaces[faceid]->m_points.emplace_back(curveNodes.back().get());
        }
    }
}

void MeshGraphIOXmlCompressed::v_ReadEdges()
{
    auto &curvedEdges  = m_meshGraph->GetCurvedEdges();
    int spaceDimension = m_meshGraph->GetSpaceDimension();

    CurveMap::iterator it;

    /// Look for elements in ELEMENT block.
    TiXmlElement *field = m_xmlGeom->FirstChildElement("EDGE");

    ASSERTL0(field, "Unable to find EDGE tag in file.");

    std::string IsCompressed;
    field->QueryStringAttribute("COMPRESSED", &IsCompressed);

    ASSERTL0(boost::iequals(IsCompressed,
                            LibUtilities::CompressData::GetCompressString()),
             "Compressed formats do not match. Expected :" +
                 LibUtilities::CompressData::GetCompressString() + " but got " +
                 IsCompressed);
    // Extract the edge body
    TiXmlNode *edgeChild = field->FirstChild();
    ASSERTL0(edgeChild, "Unable to extract the data from "
                        "the compressed edge tag.");

    std::string edgeStr;
    if (edgeChild->Type() == TiXmlNode::TINYXML_TEXT)
    {
        edgeStr += edgeChild->ToText()->ValueStr();
    }

    std::vector<SpatialDomains::MeshEdge> edgeData;
    LibUtilities::CompressData::ZlibDecodeFromBase64Str(edgeStr, edgeData);

    int indx;
    for (int i = 0; i < edgeData.size(); ++i)
    {
        indx                                = edgeData[i].id;
        std::array<PointGeom *, 2> vertices = {
            m_meshGraph->GetPointGeom(edgeData[i].v0),
            m_meshGraph->GetPointGeom(edgeData[i].v1)};
        SegGeomUniquePtr edge;
        it = curvedEdges.find(indx);
        if (it == curvedEdges.end())
        {
            m_meshGraph->AddGeom(indx,
                                 ObjPoolManager<SegGeom>::AllocateUniquePtr(
                                     indx, spaceDimension, vertices));
        }
        else
        {
            m_meshGraph->AddGeom(
                indx, ObjPoolManager<SegGeom>::AllocateUniquePtr(
                          indx, spaceDimension, vertices, it->second.get()));
        }
    }
}

void MeshGraphIOXmlCompressed::v_ReadFaces()
{
    auto &curvedFaces = m_meshGraph->GetCurvedFaces();

    /// Look for elements in FACE block.
    TiXmlElement *field = m_xmlGeom->FirstChildElement("FACE");

    ASSERTL0(field, "Unable to find FACE tag in file.");

    /// All faces are of the form: "<? ID="#"> ... </?>", with
    /// ? being an element type (either Q or T).
    /// They might be in compressed format and so then need upacking.

    TiXmlElement *element = field->FirstChildElement();
    CurveMap::iterator it;

    while (element)
    {
        std::string elementType(element->ValueStr());

        ASSERTL0(elementType == "Q" || elementType == "T",
                 (std::string("Unknown 3D face type: ") + elementType).c_str());

        std::string IsCompressed;
        element->QueryStringAttribute("COMPRESSED", &IsCompressed);

        ASSERTL0(
            boost::iequals(IsCompressed,
                           LibUtilities::CompressData::GetCompressString()),
            "Compressed formats do not match. Expected :" +
                LibUtilities::CompressData::GetCompressString() + " but got " +
                IsCompressed);

        // Extract the face body
        TiXmlNode *faceChild = element->FirstChild();
        ASSERTL0(faceChild, "Unable to extract the data from "
                            "the compressed face tag.");

        std::string faceStr;
        if (faceChild->Type() == TiXmlNode::TINYXML_TEXT)
        {
            faceStr += faceChild->ToText()->ValueStr();
        }

        int indx;
        if (elementType == "T")
        {
            std::vector<SpatialDomains::MeshTri> faceData;
            LibUtilities::CompressData::ZlibDecodeFromBase64Str(faceStr,
                                                                faceData);

            for (int i = 0; i < faceData.size(); ++i)
            {
                indx = faceData[i].id;

                /// See if this face has curves.
                it = curvedFaces.find(indx);

                /// Create a TriGeom to hold the new definition.
                std::array<SegGeom *, TriGeom::kNedges> edges = {
                    m_meshGraph->GetSegGeom(faceData[i].e[0]),
                    m_meshGraph->GetSegGeom(faceData[i].e[1]),
                    m_meshGraph->GetSegGeom(faceData[i].e[2])};

                if (it == curvedFaces.end())
                {
                    m_meshGraph->AddGeom(
                        indx, ObjPoolManager<TriGeom>::AllocateUniquePtr(
                                  indx, edges));
                }
                else
                {
                    m_meshGraph->AddGeom(
                        indx, ObjPoolManager<TriGeom>::AllocateUniquePtr(
                                  indx, edges, it->second.get()));
                }
            }
        }
        else if (elementType == "Q")
        {
            std::vector<SpatialDomains::MeshQuad> faceData;
            LibUtilities::CompressData::ZlibDecodeFromBase64Str(faceStr,
                                                                faceData);

            for (int i = 0; i < faceData.size(); ++i)
            {
                indx = faceData[i].id;

                /// See if this face has curves.
                it = curvedFaces.find(indx);

                /// Create a QuadGeom to hold the new definition.
                std::array<SegGeom *, QuadGeom::kNedges> edges = {
                    m_meshGraph->GetSegGeom(faceData[i].e[0]),
                    m_meshGraph->GetSegGeom(faceData[i].e[1]),
                    m_meshGraph->GetSegGeom(faceData[i].e[2]),
                    m_meshGraph->GetSegGeom(faceData[i].e[3])};

                if (it == curvedFaces.end())
                {
                    m_meshGraph->AddGeom(
                        indx, ObjPoolManager<QuadGeom>::AllocateUniquePtr(
                                  indx, edges));
                }
                else
                {
                    m_meshGraph->AddGeom(
                        indx, ObjPoolManager<QuadGeom>::AllocateUniquePtr(
                                  indx, edges, it->second.get()));
                }
            }
        }
        /// Keep looking
        element = element->NextSiblingElement();
    }
}

void MeshGraphIOXmlCompressed::v_ReadElements1D()
{
    auto &curvedEdges  = m_meshGraph->GetCurvedEdges();
    int spaceDimension = m_meshGraph->GetSpaceDimension();

    TiXmlElement *field = nullptr;

    /// Look for elements in ELEMENT block.
    field = m_xmlGeom->FirstChildElement("ELEMENT");

    ASSERTL0(field, "Unable to find ELEMENT tag in file.");

    /// All elements are of the form: "<S ID = n> ... </S>", with
    /// ? being the element type.

    TiXmlElement *segment = field->FirstChildElement("S");
    CurveMap::iterator it;

    while (segment)
    {
        std::string IsCompressed;
        segment->QueryStringAttribute("COMPRESSED", &IsCompressed);
        ASSERTL0(
            boost::iequals(IsCompressed,
                           LibUtilities::CompressData::GetCompressString()),
            "Compressed formats do not match. Expected :" +
                LibUtilities::CompressData::GetCompressString() + " but got " +
                IsCompressed);

        // Extract the face body
        TiXmlNode *child = segment->FirstChild();
        ASSERTL0(child, "Unable to extract the data from "
                        "the compressed face tag.");

        std::string str;
        if (child->Type() == TiXmlNode::TINYXML_TEXT)
        {
            str += child->ToText()->ValueStr();
        }

        int indx;

        std::vector<SpatialDomains::MeshEdge> data;
        LibUtilities::CompressData::ZlibDecodeFromBase64Str(str, data);

        for (int i = 0; i < data.size(); ++i)
        {
            indx = data[i].id;

            /// See if this face has curves.
            it = curvedEdges.find(indx);

            std::array<PointGeom *, 2> vertices = {
                m_meshGraph->GetPointGeom(data[i].v0),
                m_meshGraph->GetPointGeom(data[i].v1)};
            SegGeomUniquePtr seg;

            if (it == curvedEdges.end())
            {
                m_meshGraph->AddGeom(indx,
                                     ObjPoolManager<SegGeom>::AllocateUniquePtr(
                                         indx, spaceDimension, vertices));
            }
            else
            {
                m_meshGraph->AddGeom(
                    indx,
                    ObjPoolManager<SegGeom>::AllocateUniquePtr(
                        indx, spaceDimension, vertices, it->second.get()));
            }
        }
        /// Keep looking for additional segments
        segment = segment->NextSiblingElement("S");
    }
}

void MeshGraphIOXmlCompressed::v_ReadElements2D()
{
    auto &curvedFaces = m_meshGraph->GetCurvedFaces();

    /// Look for elements in ELEMENT block.
    TiXmlElement *field = m_xmlGeom->FirstChildElement("ELEMENT");

    ASSERTL0(field, "Unable to find ELEMENT tag in file.");

    // Set up curve map for curved elements on an embedded manifold.
    CurveMap::iterator it;

    /// All elements are of the form: "<? ID="#"> ... </?>", with
    /// ? being the element type.

    TiXmlElement *element = field->FirstChildElement();

    while (element)
    {
        std::string elementType(element->ValueStr());

        ASSERTL0(
            elementType == "Q" || elementType == "T",
            (std::string("Unknown 2D element type: ") + elementType).c_str());

        std::string IsCompressed;
        element->QueryStringAttribute("COMPRESSED", &IsCompressed);

        ASSERTL0(
            boost::iequals(IsCompressed,
                           LibUtilities::CompressData::GetCompressString()),
            "Compressed formats do not match. Expected :" +
                LibUtilities::CompressData::GetCompressString() + " but got " +
                IsCompressed);

        // Extract the face body
        TiXmlNode *faceChild = element->FirstChild();
        ASSERTL0(faceChild, "Unable to extract the data from "
                            "the compressed face tag.");

        std::string faceStr;
        if (faceChild->Type() == TiXmlNode::TINYXML_TEXT)
        {
            faceStr += faceChild->ToText()->ValueStr();
        }

        int indx;
        if (elementType == "T")
        {
            std::vector<SpatialDomains::MeshTri> faceData;
            LibUtilities::CompressData::ZlibDecodeFromBase64Str(faceStr,
                                                                faceData);

            for (int i = 0; i < faceData.size(); ++i)
            {
                indx = faceData[i].id;

                /// See if this face has curves.
                it = curvedFaces.find(indx);

                /// Create a TriGeom to hold the new definition.
                std::array<SegGeom *, TriGeom::kNedges> edges = {
                    m_meshGraph->GetSegGeom(faceData[i].e[0]),
                    m_meshGraph->GetSegGeom(faceData[i].e[1]),
                    m_meshGraph->GetSegGeom(faceData[i].e[2])};

                if (it == curvedFaces.end())
                {
                    m_meshGraph->AddGeom(
                        indx, ObjPoolManager<TriGeom>::AllocateUniquePtr(
                                  indx, edges));
                }
                else
                {
                    m_meshGraph->AddGeom(
                        indx, ObjPoolManager<TriGeom>::AllocateUniquePtr(
                                  indx, edges, it->second.get()));
                }
            }
        }
        else if (elementType == "Q")
        {
            std::vector<SpatialDomains::MeshQuad> faceData;
            LibUtilities::CompressData::ZlibDecodeFromBase64Str(faceStr,
                                                                faceData);

            for (int i = 0; i < faceData.size(); ++i)
            {
                indx = faceData[i].id;

                /// See if this face has curves.
                it = curvedFaces.find(indx);

                /// Create a QuadGeom to hold the new definition.
                std::array<SegGeom *, QuadGeom::kNedges> edges = {
                    m_meshGraph->GetSegGeom(faceData[i].e[0]),
                    m_meshGraph->GetSegGeom(faceData[i].e[1]),
                    m_meshGraph->GetSegGeom(faceData[i].e[2]),
                    m_meshGraph->GetSegGeom(faceData[i].e[3])};

                if (it == curvedFaces.end())
                {
                    m_meshGraph->AddGeom(
                        indx, ObjPoolManager<QuadGeom>::AllocateUniquePtr(
                                  indx, edges));
                }
                else
                {
                    m_meshGraph->AddGeom(
                        indx, ObjPoolManager<QuadGeom>::AllocateUniquePtr(
                                  indx, edges, it->second.get()));
                }
            }
        }
        /// Keep looking
        element = element->NextSiblingElement();
    }
}

void MeshGraphIOXmlCompressed::v_ReadElements3D()
{
    /// Look for elements in ELEMENT block.
    TiXmlElement *field = m_xmlGeom->FirstChildElement("ELEMENT");

    ASSERTL0(field, "Unable to find ELEMENT tag in file.");

    /// All elements are of the form: "<? ID="#"> ... </?>", with
    /// ? being the element type.

    TiXmlElement *element = field->FirstChildElement();

    while (element)
    {
        std::string elementType(element->ValueStr());

        // A - tet, P - pyramid, R - prism, H - hex
        ASSERTL0(
            elementType == "A" || elementType == "P" || elementType == "R" ||
                elementType == "H",
            (std::string("Unknown 3D element type: ") + elementType).c_str());

        std::string IsCompressed;
        element->QueryStringAttribute("COMPRESSED", &IsCompressed);

        ASSERTL0(
            boost::iequals(IsCompressed,
                           LibUtilities::CompressData::GetCompressString()),
            "Compressed formats do not match. Expected :" +
                LibUtilities::CompressData::GetCompressString() + " but got " +
                IsCompressed);

        // Extract the face body
        TiXmlNode *child = element->FirstChild();
        ASSERTL0(child, "Unable to extract the data from "
                        "the compressed face tag.");

        std::string str;
        if (child->Type() == TiXmlNode::TINYXML_TEXT)
        {
            str += child->ToText()->ValueStr();
        }

        int indx;
        if (elementType == "A")
        {
            std::vector<SpatialDomains::MeshTet> data;
            LibUtilities::CompressData::ZlibDecodeFromBase64Str(str, data);
            std::array<TriGeom *, 4> tfaces;
            for (int i = 0; i < data.size(); ++i)
            {
                indx = data[i].id;
                for (int j = 0; j < 4; ++j)
                {
                    Geometry2D *face = m_meshGraph->GetGeometry2D(data[i].f[j]);
                    tfaces[j]        = static_cast<TriGeom *>(face);
                }

                auto tetGeom =
                    ObjPoolManager<TetGeom>::AllocateUniquePtr(indx, tfaces);
                m_meshGraph->PopulateFaceToElMap(tetGeom.get(), 4);
                m_meshGraph->AddGeom(indx, std::move(tetGeom));
            }
        }
        else if (elementType == "P")
        {
            std::vector<SpatialDomains::MeshPyr> data;
            LibUtilities::CompressData::ZlibDecodeFromBase64Str(str, data);
            std::array<Geometry2D *, 5> faces;
            for (int i = 0; i < data.size(); ++i)
            {
                indx        = data[i].id;
                int Ntfaces = 0;
                int Nqfaces = 0;
                for (int j = 0; j < 5; ++j)
                {
                    Geometry2D *face = m_meshGraph->GetGeometry2D(data[i].f[j]);

                    if (face == nullptr ||
                        (face->GetShapeType() != LibUtilities::eTriangle &&
                         face->GetShapeType() != LibUtilities::eQuadrilateral))
                    {
                        std::stringstream errorstring;
                        errorstring << "Element " << indx
                                    << " has invalid face: " << j;
                        ASSERTL0(false, errorstring.str().c_str());
                    }
                    else if (face->GetShapeType() == LibUtilities::eTriangle)
                    {
                        faces[j] = static_cast<TriGeom *>(face);
                        Ntfaces++;
                    }
                    else if (face->GetShapeType() ==
                             LibUtilities::eQuadrilateral)
                    {
                        faces[j] = static_cast<QuadGeom *>(face);
                        Nqfaces++;
                    }
                }
                ASSERTL0((Ntfaces == 4) && (Nqfaces == 1),
                         "Did not identify the correct number of "
                         "triangular and quadrilateral faces for a "
                         "pyramid");

                auto pyrGeom =
                    ObjPoolManager<PyrGeom>::AllocateUniquePtr(indx, faces);
                m_meshGraph->PopulateFaceToElMap(pyrGeom.get(), 5);
                m_meshGraph->AddGeom(indx, std::move(pyrGeom));
            }
        }
        else if (elementType == "R")
        {
            std::vector<SpatialDomains::MeshPrism> data;
            LibUtilities::CompressData::ZlibDecodeFromBase64Str(str, data);
            std::array<Geometry2D *, 5> faces;
            for (int i = 0; i < data.size(); ++i)
            {
                indx        = data[i].id;
                int Ntfaces = 0;
                int Nqfaces = 0;
                for (int j = 0; j < 5; ++j)
                {
                    Geometry2D *face = m_meshGraph->GetGeometry2D(data[i].f[j]);
                    if (face == nullptr ||
                        (face->GetShapeType() != LibUtilities::eTriangle &&
                         face->GetShapeType() != LibUtilities::eQuadrilateral))
                    {
                        std::stringstream errorstring;
                        errorstring << "Element " << indx
                                    << " has invalid face: " << j;
                        ASSERTL0(false, errorstring.str().c_str());
                    }
                    else if (face->GetShapeType() == LibUtilities::eTriangle)
                    {
                        faces[j] = static_cast<TriGeom *>(face);
                        Ntfaces++;
                    }
                    else if (face->GetShapeType() ==
                             LibUtilities::eQuadrilateral)
                    {
                        faces[j] = static_cast<QuadGeom *>(face);
                        Nqfaces++;
                    }
                }
                ASSERTL0((Ntfaces == 2) && (Nqfaces == 3),
                         "Did not identify the correct number of "
                         "triangular and quadrilateral faces for a "
                         "prism");

                auto prismGeom =
                    ObjPoolManager<PrismGeom>::AllocateUniquePtr(indx, faces);
                m_meshGraph->PopulateFaceToElMap(prismGeom.get(), 5);
                m_meshGraph->AddGeom(indx, std::move(prismGeom));
            }
        }
        else if (elementType == "H")
        {
            std::vector<SpatialDomains::MeshHex> data;
            LibUtilities::CompressData::ZlibDecodeFromBase64Str(str, data);

            std::array<QuadGeom *, 6> faces;
            for (int i = 0; i < data.size(); ++i)
            {
                indx = data[i].id;
                for (int j = 0; j < 6; ++j)
                {
                    Geometry2D *face = m_meshGraph->GetGeometry2D(data[i].f[j]);
                    faces[j]         = static_cast<QuadGeom *>(face);
                }

                auto hexGeom =
                    ObjPoolManager<HexGeom>::AllocateUniquePtr(indx, faces);
                m_meshGraph->PopulateFaceToElMap(hexGeom.get(), 6);
                m_meshGraph->AddGeom(indx, std::move(hexGeom));
            }
        }
        /// Keep looking
        element = element->NextSiblingElement();
    }
}

void WriteVert(PointGeom *vert, std::vector<MeshVertex> &vertInfo, int vertID)
{
    MeshVertex v;
    v.id = vertID;
    v.x  = vert->x();
    v.y  = vert->y();
    v.z  = vert->z();
    vertInfo.push_back(v);
}
void MeshGraphIOXmlCompressed::v_WriteVertices(TiXmlElement *geomTag,
                                               std::vector<int> keysToWrite)
{
    auto &verts = m_meshGraph->GetGeomMap<PointGeom>();
    if (m_meshGraph->GetGeomMap<PointGeom>().size() == 0)
    {
        return;
    }

    TiXmlElement *vertTag = new TiXmlElement("VERTEX");

    std::vector<MeshVertex> vertInfo;

    if (keysToWrite.empty())
    {
        for (auto [id, vert] : verts)
        {
            WriteVert(vert, vertInfo, id);
        }
    }
    else
    {
        for (int id : keysToWrite)
        {
            WriteVert(verts.at(id), vertInfo, id);
        }
    }

    vertTag->SetAttribute("COMPRESSED",
                          LibUtilities::CompressData::GetCompressString());
    vertTag->SetAttribute("BITSIZE",
                          LibUtilities::CompressData::GetBitSizeStr());

    std::string vertStr;
    LibUtilities::CompressData::ZlibEncodeToBase64Str(vertInfo, vertStr);

    vertTag->LinkEndChild(new TiXmlText(vertStr));

    geomTag->LinkEndChild(vertTag);
}

void WriteEdge(SegGeom *seg, std::vector<MeshEdge> &edgeInfo, int edgeID)
{
    MeshEdge e;
    e.id = edgeID;
    e.v0 = seg->GetVid(0);
    e.v1 = seg->GetVid(1);
    edgeInfo.push_back(e);
}
void MeshGraphIOXmlCompressed::v_WriteEdges(TiXmlElement *geomTag,
                                            std::vector<int> keysToWrite)
{
    auto &edges = m_meshGraph->GetGeomMap<SegGeom>();
    if (m_meshGraph->GetGeomMap<SegGeom>().size() == 0)
    {
        return;
    }

    int meshDimension = m_meshGraph->GetMeshDimension();

    TiXmlElement *edgeTag = new TiXmlElement(meshDimension == 1 ? "S" : "EDGE");

    std::vector<MeshEdge> edgeInfo;

    if (keysToWrite.empty())
    {
        for (auto [id, edge] : edges)
        {
            WriteEdge(edge, edgeInfo, id);
        }
    }
    else
    {
        for (int id : keysToWrite)
        {
            WriteEdge(edges.at(id), edgeInfo, id);
        }
    }

    std::string edgeStr;
    LibUtilities::CompressData::ZlibEncodeToBase64Str(edgeInfo, edgeStr);

    edgeTag->SetAttribute("COMPRESSED",
                          LibUtilities::CompressData::GetCompressString());
    edgeTag->SetAttribute("BITSIZE",
                          LibUtilities::CompressData::GetBitSizeStr());

    edgeTag->LinkEndChild(new TiXmlText(edgeStr));

    if (meshDimension == 1)
    {
        TiXmlElement *tmp = new TiXmlElement("ELEMENT");
        tmp->LinkEndChild(edgeTag);
        geomTag->LinkEndChild(tmp);
    }
    else
    {
        geomTag->LinkEndChild(edgeTag);
    }
}

void WriteTri(TriGeom *tri, std::vector<MeshTri> &triInfo, int triID)
{
    MeshTri t;
    t.id   = triID;
    t.e[0] = tri->GetEid(0);
    t.e[1] = tri->GetEid(1);
    t.e[2] = tri->GetEid(2);
    triInfo.push_back(t);
}
void MeshGraphIOXmlCompressed::v_WriteTris(TiXmlElement *faceTag,
                                           std::vector<int> keysToWrite)
{
    auto &tris = m_meshGraph->GetGeomMap<TriGeom>();

    if (tris.size() == 0)
    {
        return;
    }

    std::string tag = "T";

    std::vector<MeshTri> triInfo;

    if (keysToWrite.empty())
    {
        for (auto [id, tri] : tris)
        {
            WriteTri(tri, triInfo, id);
        }
    }
    else
    {
        for (int id : keysToWrite)
        {
            WriteTri(tris.at(id), triInfo, id);
        }
    }

    TiXmlElement *x = new TiXmlElement(tag);
    std::string triStr;
    LibUtilities::CompressData::ZlibEncodeToBase64Str(triInfo, triStr);

    x->SetAttribute("COMPRESSED",
                    LibUtilities::CompressData::GetCompressString());
    x->SetAttribute("BITSIZE", LibUtilities::CompressData::GetBitSizeStr());

    x->LinkEndChild(new TiXmlText(triStr));

    faceTag->LinkEndChild(x);
}

void WriteQuad(QuadGeom *quad, std::vector<MeshQuad> &quadInfo, int quadID)
{
    MeshQuad q;
    q.id   = quadID;
    q.e[0] = quad->GetEid(0);
    q.e[1] = quad->GetEid(1);
    q.e[2] = quad->GetEid(2);
    q.e[3] = quad->GetEid(3);
    quadInfo.push_back(q);
}
void MeshGraphIOXmlCompressed::v_WriteQuads(TiXmlElement *faceTag,
                                            std::vector<int> keysToWrite)
{
    auto &quads = m_meshGraph->GetGeomMap<QuadGeom>();

    if (quads.size() == 0)
    {
        return;
    }

    std::string tag = "Q";

    std::vector<MeshQuad> quadInfo;

    if (keysToWrite.empty())
    {
        for (auto [id, quad] : quads)
        {
            WriteQuad(quad, quadInfo, id);
        }
    }
    else
    {
        for (int id : keysToWrite)
        {
            WriteQuad(quads.at(id), quadInfo, id);
        }
    }

    TiXmlElement *x = new TiXmlElement(tag);
    std::string quadStr;
    LibUtilities::CompressData::ZlibEncodeToBase64Str(quadInfo, quadStr);

    x->SetAttribute("COMPRESSED",
                    LibUtilities::CompressData::GetCompressString());
    x->SetAttribute("BITSIZE", LibUtilities::CompressData::GetBitSizeStr());

    x->LinkEndChild(new TiXmlText(quadStr));

    faceTag->LinkEndChild(x);
}

void WriteHex(HexGeom *hex, std::vector<MeshHex> &elementInfo, int hexID)
{
    MeshHex e;
    e.id   = hexID;
    e.f[0] = hex->GetFid(0);
    e.f[1] = hex->GetFid(1);
    e.f[2] = hex->GetFid(2);
    e.f[3] = hex->GetFid(3);
    e.f[4] = hex->GetFid(4);
    e.f[5] = hex->GetFid(5);
    elementInfo.push_back(e);
}
void MeshGraphIOXmlCompressed::v_WriteHexs(TiXmlElement *elmtTag,
                                           std::vector<int> keysToWrite)
{
    auto &hexes = m_meshGraph->GetGeomMap<HexGeom>();

    if (hexes.size() == 0)
    {
        return;
    }

    std::string tag = "H";

    std::vector<MeshHex> elementInfo;

    if (keysToWrite.empty())
    {
        for (auto [id, hex] : hexes)
        {
            WriteHex(hex, elementInfo, id);
        }
    }
    else
    {
        for (int id : keysToWrite)
        {
            WriteHex(hexes.at(id), elementInfo, id);
        }
    }

    TiXmlElement *x = new TiXmlElement(tag);
    std::string elStr;
    LibUtilities::CompressData::ZlibEncodeToBase64Str(elementInfo, elStr);

    x->SetAttribute("COMPRESSED",
                    LibUtilities::CompressData::GetCompressString());
    x->SetAttribute("BITSIZE", LibUtilities::CompressData::GetBitSizeStr());

    x->LinkEndChild(new TiXmlText(elStr));

    elmtTag->LinkEndChild(x);
}

void WritePrism(PrismGeom *prism, std::vector<MeshPrism> &elementInfo,
                int prismID)
{
    MeshPrism e;
    e.id   = prismID;
    e.f[0] = prism->GetFid(0);
    e.f[1] = prism->GetFid(1);
    e.f[2] = prism->GetFid(2);
    e.f[3] = prism->GetFid(3);
    e.f[4] = prism->GetFid(4);
    elementInfo.push_back(e);
}
void MeshGraphIOXmlCompressed::v_WritePrisms(TiXmlElement *elmtTag,
                                             std::vector<int> keysToWrite)
{
    auto &prisms = m_meshGraph->GetGeomMap<PrismGeom>();

    if (prisms.size() == 0)
    {
        return;
    }

    std::string tag = "R";

    std::vector<MeshPrism> elementInfo;

    if (keysToWrite.empty())
    {
        for (auto [id, prism] : prisms)
        {
            WritePrism(prism, elementInfo, id);
        }
    }
    else
    {
        for (int id : keysToWrite)
        {
            WritePrism(prisms.at(id), elementInfo, id);
        }
    }

    TiXmlElement *x = new TiXmlElement(tag);
    std::string elStr;
    LibUtilities::CompressData::ZlibEncodeToBase64Str(elementInfo, elStr);

    x->SetAttribute("COMPRESSED",
                    LibUtilities::CompressData::GetCompressString());
    x->SetAttribute("BITSIZE", LibUtilities::CompressData::GetBitSizeStr());

    x->LinkEndChild(new TiXmlText(elStr));

    elmtTag->LinkEndChild(x);
}

void WritePyr(PyrGeom *pyr, std::vector<MeshPyr> &elementInfo, int pyrID)
{
    MeshPyr e;
    e.id   = pyrID;
    e.f[0] = pyr->GetFid(0);
    e.f[1] = pyr->GetFid(1);
    e.f[2] = pyr->GetFid(2);
    e.f[3] = pyr->GetFid(3);
    e.f[4] = pyr->GetFid(4);
    elementInfo.push_back(e);
}
void MeshGraphIOXmlCompressed::v_WritePyrs(TiXmlElement *elmtTag,
                                           std::vector<int> keysToWrite)
{
    auto &pyrs = m_meshGraph->GetGeomMap<PyrGeom>();

    if (pyrs.size() == 0)
    {
        return;
    }

    std::string tag = "P";

    std::vector<MeshPyr> elementInfo;

    if (keysToWrite.empty())
    {
        for (auto [id, pyr] : pyrs)
        {
            WritePyr(pyr, elementInfo, id);
        }
    }
    else
    {
        for (int id : keysToWrite)
        {
            WritePyr(pyrs.at(id), elementInfo, id);
        }
    }

    TiXmlElement *x = new TiXmlElement(tag);
    std::string elStr;
    LibUtilities::CompressData::ZlibEncodeToBase64Str(elementInfo, elStr);

    x->SetAttribute("COMPRESSED",
                    LibUtilities::CompressData::GetCompressString());
    x->SetAttribute("BITSIZE", LibUtilities::CompressData::GetBitSizeStr());

    x->LinkEndChild(new TiXmlText(elStr));

    elmtTag->LinkEndChild(x);
}

void WriteTet(TetGeom *tet, std::vector<MeshTet> &elementInfo, int tetID)
{
    MeshTet e;
    e.id   = tetID;
    e.f[0] = tet->GetFid(0);
    e.f[1] = tet->GetFid(1);
    e.f[2] = tet->GetFid(2);
    e.f[3] = tet->GetFid(3);
    elementInfo.push_back(e);
}
void MeshGraphIOXmlCompressed::v_WriteTets(TiXmlElement *elmtTag,
                                           std::vector<int> keysToWrite)
{
    auto &tets = m_meshGraph->GetGeomMap<TetGeom>();

    if (tets.size() == 0)
    {
        return;
    }

    std::string tag = "A";

    std::vector<MeshTet> elementInfo;

    if (keysToWrite.empty())
    {
        for (auto [id, tet] : tets)
        {
            WriteTet(tet, elementInfo, id);
        }
    }
    else
    {
        for (int id : keysToWrite)
        {
            WriteTet(tets.at(id), elementInfo, id);
        }
    }

    TiXmlElement *x = new TiXmlElement(tag);
    std::string elStr;
    LibUtilities::CompressData::ZlibEncodeToBase64Str(elementInfo, elStr);

    x->SetAttribute("COMPRESSED",
                    LibUtilities::CompressData::GetCompressString());
    x->SetAttribute("BITSIZE", LibUtilities::CompressData::GetBitSizeStr());

    x->LinkEndChild(new TiXmlText(elStr));

    elmtTag->LinkEndChild(x);
}

void WriteCurvedEdge(CurveUniquePtr &curve, int key,
                     std::vector<MeshCurvedInfo> &edgeInfo, int &edgeCnt,
                     int &ptOffset, int &newIdx, MeshCurvedPts &curvedPts)
{
    MeshCurvedInfo cinfo;
    cinfo.id       = edgeCnt++;
    cinfo.entityid = key;
    cinfo.npoints  = curve->m_points.size();
    cinfo.ptype    = curve->m_ptype;
    cinfo.ptid     = 0;
    cinfo.ptoffset = ptOffset;

    edgeInfo.push_back(cinfo);

    for (int j = 0; j < curve->m_points.size(); j++)
    {
        MeshVertex v;
        v.id = newIdx;
        v.x  = curve->m_points[j]->x();
        v.y  = curve->m_points[j]->y();
        v.z  = curve->m_points[j]->z();
        curvedPts.pts.push_back(v);
        curvedPts.index.push_back(newIdx);
        newIdx++;
    }
    ptOffset += cinfo.npoints;
}
void WriteCurvedFace(CurveUniquePtr &curve, int key,
                     std::vector<MeshCurvedInfo> &faceInfo, int &faceCnt,
                     int &ptOffset, int &newIdx, MeshCurvedPts &curvedPts)
{
    MeshCurvedInfo cinfo;
    cinfo.id       = faceCnt++;
    cinfo.entityid = key;
    cinfo.npoints  = curve->m_points.size();
    cinfo.ptype    = curve->m_ptype;
    cinfo.ptid     = 0;
    cinfo.ptoffset = ptOffset;

    faceInfo.push_back(cinfo);

    for (int j = 0; j < curve->m_points.size(); j++)
    {
        MeshVertex v;
        v.id = newIdx;
        v.x  = curve->m_points[j]->x();
        v.y  = curve->m_points[j]->y();
        v.z  = curve->m_points[j]->z();
        curvedPts.pts.push_back(v);
        curvedPts.index.push_back(newIdx);
        newIdx++;
    }
    ptOffset += cinfo.npoints;
}
void MeshGraphIOXmlCompressed::v_WriteCurves(TiXmlElement *geomTag,
                                             CurveMap &edges, CurveMap &faces,
                                             std::vector<int> *keysToWriteEdges,
                                             std::vector<int> *keysToWriteFaces)
{
    if (edges.size() == 0 && faces.size() == 0)
    {
        return;
    }
    else if (keysToWriteEdges != nullptr && keysToWriteFaces != nullptr)
    {
        if (keysToWriteEdges->size() == 0 && keysToWriteFaces->size() == 0)
        {
            return;
        }
    }

    TiXmlElement *curveTag = new TiXmlElement("CURVED");

    std::vector<MeshCurvedInfo> edgeInfo;
    std::vector<MeshCurvedInfo> faceInfo;
    MeshCurvedPts curvedPts;
    curvedPts.id = 0;
    int ptOffset = 0;
    int newIdx   = 0;
    int edgeCnt  = 0;
    int faceCnt  = 0;

    if (keysToWriteEdges == nullptr)
    {
        for (auto &i : edges)
        {
            WriteCurvedEdge(i.second, i.first, edgeInfo, edgeCnt, ptOffset,
                            newIdx, curvedPts);
        }
    }
    else
    {
        for (int key : *keysToWriteEdges)
        {
            WriteCurvedEdge(edges[key], key, edgeInfo, edgeCnt, ptOffset,
                            newIdx, curvedPts);
        }
    }

    if (keysToWriteFaces == nullptr)
    {
        for (auto &i : faces)
        {
            WriteCurvedFace(i.second, i.first, faceInfo, faceCnt, ptOffset,
                            newIdx, curvedPts);
        }
    }
    else
    {
        for (int key : *keysToWriteFaces)
        {
            WriteCurvedFace(faces[key], key, faceInfo, faceCnt, ptOffset,
                            newIdx, curvedPts);
        }
    }

    curveTag->SetAttribute("COMPRESSED",
                           LibUtilities::CompressData::GetCompressString());
    curveTag->SetAttribute("BITSIZE",
                           LibUtilities::CompressData::GetBitSizeStr());

    if (edgeInfo.size())
    {
        TiXmlElement *x = new TiXmlElement("E");
        std::string dataStr;
        LibUtilities::CompressData::ZlibEncodeToBase64Str(edgeInfo, dataStr);

        x->LinkEndChild(new TiXmlText(dataStr));
        curveTag->LinkEndChild(x);
    }

    if (faceInfo.size())
    {
        TiXmlElement *x = new TiXmlElement("F");
        std::string dataStr;
        LibUtilities::CompressData::ZlibEncodeToBase64Str(faceInfo, dataStr);

        x->LinkEndChild(new TiXmlText(dataStr));
        curveTag->LinkEndChild(x);
    }

    if (edgeInfo.size() || faceInfo.size())
    {
        TiXmlElement *x = new TiXmlElement("DATAPOINTS");
        x->SetAttribute("ID", curvedPts.id);
        TiXmlElement *subx = new TiXmlElement("INDEX");
        std::string dataStr;
        LibUtilities::CompressData::ZlibEncodeToBase64Str(curvedPts.index,
                                                          dataStr);
        subx->LinkEndChild(new TiXmlText(dataStr));
        x->LinkEndChild(subx);

        subx = new TiXmlElement("POINTS");
        LibUtilities::CompressData::ZlibEncodeToBase64Str(curvedPts.pts,
                                                          dataStr);
        subx->LinkEndChild(new TiXmlText(dataStr));
        x->LinkEndChild(subx);
        curveTag->LinkEndChild(x);
    }

    geomTag->LinkEndChild(curveTag);
}
} // namespace Nektar::SpatialDomains
