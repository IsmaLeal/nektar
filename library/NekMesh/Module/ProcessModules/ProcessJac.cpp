////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessJac.cpp
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
//  Description: Calculate Jacobians of elements.
//
////////////////////////////////////////////////////////////////////////////////

#include "ProcessJac.h"
#include <LibUtilities/BasicUtils/ParseUtils.h>
#include <NekMesh/MeshElements/Element.h>

using namespace Nektar::NekMesh;

namespace Nektar::NekMesh
{

ModuleKey ProcessJac::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eProcessModule, "jac"), ProcessJac::create,
    "Process elements based on values of Jacobian.");

ProcessJac::ProcessJac(MeshSharedPtr m) : ProcessModule(m)
{
    m_config["extract"] =
        ConfigOption(false, "0.0",
                     "Extract elements from mesh with scaled Jacobian under a "
                     "set threshold value (default: invalid elements)");
    m_config["list"] = ConfigOption(
        true, "0", "Print list of elements having negative Jacobian.");
    m_config["histo"] =
        ConfigOption(false, "1.0",
                     "Output the scaled Jacobians under a threshold value into "
                     "a txt file which can be used to show histogram of "
                     "Jacobian (default: Output Jacobians of all elements),"
                     "and shows the histogram in log scale on the screen."
                     "The modules takes 3 values: 1. the thershold (double);"
                     " 2.the number of bins with positive Jacobians on the "
                     "on-screen histogram; 3. The number of bins with negative "
                     "Jacobians on the on-screen histogram.");
    m_config["histofile"] =
        ConfigOption(false, "jacs.txt",
                     "Optional filename for the text file containing the "
                     "Jacobian information.");
    m_config["quality"] =
        ConfigOption(false, "0",
                     "Show the percentage of the ratio between (each histogram "
                     "bin)/(Total number of mesh elements) on the screen, and "
                     "evaluate the minimum Jacobian value and the integral of "
                     "the scaled Jacobian. ");
    m_config["detail"] =
        ConfigOption(false, "0",
                     "Output the composite ID and name of boundary elements "
                     "into the txt file. histo must be set to get the detail");
}

ProcessJac::~ProcessJac()
{
}

void ProcessJac::Process()
{
    bool extract   = m_config["extract"].beenSet;
    bool printList = m_config["list"].beenSet;
    bool histogram = m_config["histo"].beenSet;
    bool quality   = m_config["quality"].beenSet;
    bool detail    = m_config["detail"].beenSet;

    /// The maximum value of Jacobian extracted to output xml file.
    NekDouble thres = m_config["extract"].as<NekDouble>();

    // Get the limits of the histogram to be generated
    std::vector<NekDouble> histo_lim;
    ParseUtils::GenerateVector(m_config["histo"].as<std::string>().c_str(),
                               histo_lim);

    /// The maximum value of Jacobian shown on the histogram.
    m_histo_max_value = histo_lim[0];
    /// The number of bins with positive and negative values.
    m_ScalePos = histo_lim.size() >= 2 ? static_cast<int>(histo_lim[1]) : 10;
    m_ScaleNeg = histo_lim.size() >= 3 ? static_cast<int>(histo_lim[2]) : 1;

    // std::string filename = m_config["outfile"].as<std::string>();
    // std::cout << "testing filename - does it contain the file extension? " <<
    // filename << std::endl;

    // Set the number of bins of the histogram shown on the screen.
    if (m_ScalePos < 0)
    {
        m_log(WARNING) << "Number of bins of histogram must be greater than 0, "
                          "set to default value 10 instead"
                       << std::endl;
        m_ScalePos = 10;
    }
    if (m_ScaleNeg <= 0)
    {
        m_log(WARNING)
            << "Bins with negative values must be greater than 0, "
               "set to default value 1 instead, the negative values will be "
               "shown in one bin named \"invalid\" "
            << std::endl;
        m_ScaleNeg = 1;
    }
    m_interval = m_histo_max_value / NekDouble(m_ScalePos);

    std::vector<ElementSharedPtr> el = m_mesh->m_element[m_mesh->m_expDim];

    // Clear mesh and push back elements based on "thres".
    if (extract)
    {
        m_mesh->m_element[m_mesh->m_expDim].clear();
    }

    if (printList)
    {
        m_log << "Elements with negative Jacobian:" << std::endl;
    }

    // Outputting jacobian information into a text file which can be used to
    // generate a histogram.
    std::ofstream output_file;
    if (histogram)
    {
        // Clear output file.
        std::ofstream clear_file;
        std::string filename = m_config["histofile"].as<std::string>();
        clear_file.open(filename, std::ofstream::out | std::ofstream::trunc);
        clear_file.close();
        // open file for writing, if it doesn't exist create it.
        output_file.open(filename, std::ios::app);
        output_file << std::setw(ElementID) << "Element ID" << std::setw(Jac)
                    << "Jac" << std::setw(type) << "type";
        if (m_mesh->m_expDim == 2)
        {
            output_file << std::setw(Boundary_edge_ID) << "Boundary Edge ID"
                        << std::setw(VertexID) << "Vertex ID"
                        << std::setw(CoordX) << "Coordinates X"
                        << std::setw(CoordY) << "Coordinates Y ";
        }
        else
        {
            output_file << std::setw(Boundary_face_ID) << "Boundary face ID"
                        << std::setw(EdgeID) << "Edge ID" << std::setw(CoordX)
                        << "Coordinates X" << std::setw(CoordY)
                        << "Coordinates Y" << std::setw(CoordZ)
                        << " Coordinates Z";
        }
        if (detail)
        {
            output_file << std::setw(CompositeID) << "Composite ID"
                        << std::setw(CompositeName) << " Composite Name";
        }
        output_file << "  NeighborEl ID" << std::setw(NeighborElID) << "\n";
    }

    int nNeg = 0;
    // Store the number of elements in each Jac bin.
    Array<OneD, NekDouble> histo(m_ScaleNeg + m_ScalePos, 1.0);
    // Store the scaled Jacobians of each element
    Array<OneD, NekDouble> histo_Jac(el.size(), 0.0);
    // The integral of scaled Jacobians. (Scaled Jacobian) * (Area of element)
    // is considered as the elemental Jacobian and the integral is the sum of
    // all the elemental values
    NekDouble Integral = 0.0;
    // Area/volume of the mesh.
    NekDouble Area = 0.0;
    std::vector<element_reorder> el_reorder(el.size());
    // Iterate over list of elements of expansion dimension.
    for (int i = 0; i < el.size(); ++i)
    {
        SpatialDomains::EntityHolder holder;
        // Create elemental geometry.
        SpatialDomains::Geometry *geom =
            el[i]->GetGeom(m_mesh->m_spaceDim, holder);

        LibUtilities::PointsKeyVector p = geom->GetXmap()->GetPointsKeys();
        // Generate geometric factors.
        SpatialDomains::GeomFactorsUniquePtr gfac = geom->GenGeomFactors(p);

        SpatialDomains::DerivStorage deriv = gfac->GetDeriv(p);
        const int pts                      = deriv[0][0].size();
        Array<OneD, NekDouble> jc(pts);
        for (int k = 0; k < pts; ++k)
        {
            DNekMat jac(m_mesh->m_expDim, m_mesh->m_expDim, 0.0, eFULL);
            for (int l = 0; l < m_mesh->m_expDim; ++l)
            {
                for (int j = 0; j < m_mesh->m_expDim; ++j)
                {
                    jac(j, l) = deriv[l][j][k];
                }
            }
            if (m_mesh->m_expDim == 2)
            {
                jc[k] = jac(0, 0) * jac(1, 1) - jac(0, 1) * jac(1, 0);
            }
            else if (m_mesh->m_expDim == 3)
            {
                jc[k] =
                    jac(0, 0) *
                        (jac(1, 1) * jac(2, 2) - jac(2, 1) * jac(1, 2)) -
                    jac(0, 1) *
                        (jac(1, 0) * jac(2, 2) - jac(2, 0) * jac(1, 2)) +
                    jac(0, 2) * (jac(1, 0) * jac(2, 1) - jac(2, 0) * jac(1, 1));
            }
        }
        NekDouble scaledJac =
            Vmath::Vmin(jc.size(), jc, 1) / Vmath::Vmax(jc.size(), jc, 1);
        histo_Jac[i] = scaledJac;

        Integral += geom->GetXmap()->Integral(jc) * scaledJac;
        Area += geom->GetXmap()->Integral(jc);

        bool valid = gfac->IsValid();

        if (extract && (scaledJac < thres || !valid))
        {
            m_mesh->m_element[m_mesh->m_expDim].push_back(el[i]);
        }

        // Get the Jacobian and, if it is negative, print a warning
        // message.
        if (!valid)
        {
            nNeg++;
            if (printList)
            {
                m_log(INFO)
                    << "  - " << el[i]->GetId() << " ("
                    << LibUtilities::ShapeTypeMap[el[i]->GetConf().m_e] << ")"
                    << "  " << scaledJac << "\n";
            }
        }
        // Get the number of elements in each histogram bin based on the scaled
        // Jacobian. This will be used for outputting histogram on the screen,
        // with a scale between (-1.0 * m_ScaledNeg *
        // m_interval, 1.0 * m_ScaledPos * m_interval). The elements with
        // Jacobians below "-1.0 * m_ScaledNeg * m_interval" are classified into
        // one bin as there are not enough space to show too much bins on the
        // screen.
        if ((histogram || quality) && valid)
        {
            for (int i = 0; i < m_ScalePos; ++i)
            {
                if (scaledJac > m_interval * NekDouble(i) &&
                    scaledJac < m_interval * NekDouble(i + 1))
                {
                    histo[m_ScaleNeg + i] += 1.0;
                    break;
                }
            }
        }
        else if ((histogram || quality) && !valid)
        {
            for (int i = 0; i < m_ScaleNeg; ++i)
            {
                if (scaledJac < -1.0 * m_interval * NekDouble(i) &&
                    scaledJac > -1.0 * m_interval * NekDouble(i + 1))
                {
                    histo[m_ScaleNeg - i - 1] += 1.0;
                    break;
                }
                if (i == m_ScaleNeg - 1)
                {
                    histo[0] += 1.0;
                }
            }
        }
        // Output coordinates and boundary composite names to the text file.
        if (histogram && scaledJac <= m_histo_max_value)
        {
            // output ID of element, Jac, type of element, physical coordinates,
            // composite names.
            output_file << std::setw(ElementID) << el[i]->GetId()
                        << std::setw(Jac) << scaledJac << std::setw(type)
                        << LibUtilities::ShapeTypeMap[el[i]->GetConf().m_e];
            // Find boundary elements
            if (!el[i]->HasBoundaryLinks())
            {
                output_file << "\n";
                continue;
            }
            GetBoundaryCoordinate(el[i], output_file, detail);
            int32_t sum_enum =
                ElementID + Jac + type + CoordX + CoordY + NeighborElID;
            if (detail)
            {
                sum_enum += CompositeName + CompositeID;
            }
            if (m_mesh->m_expDim == 2)
            {
                sum_enum += Boundary_edge_ID + VertexID;
            }
            else if (m_mesh->m_expDim == 3)
            {
                sum_enum += Boundary_face_ID + EdgeID + CoordZ;
            }
            // Output a few "===" below the information of each element. This is
            // only used to help the user read the info of each element easier.
            output_file << std::string(sum_enum, '=') << "\n";
        }
        m_log(VERBOSE).Progress(i, el.size(), "Calculating Elemental Jacobian",
                                i - 1);
    }

    // Reorder the elements before push back based on the value of scaled
    // Jacobians.
    for (int i = 0; i < el.size(); ++i)
    {
        el_reorder[i].El  = el[i];
        el_reorder[i].Jac = histo_Jac[i];
    }
    std::sort(
        el_reorder.begin(), el_reorder.end(),
        [&](element_reorder a, element_reorder b) { return a.Jac < b.Jac; });

    // Extract elements whose Jacobians is lower than "thres"
    if (extract)
    {
        m_mesh->m_element[m_mesh->m_expDim - 1].clear();
        ProcessVertices();
        ProcessEdges();
        ProcessFaces();
        ProcessElements();
        ProcessComposites();
    }

    m_log(INFO) << "Total negative Jacobians: " << nNeg << std::endl;

    if (nNeg > 0)
    {
        m_log(WARNING) << "Detected " << nNeg << " element"
                       << (nNeg == 1 ? "" : "s") << " with negative Jacobian."
                       << std::endl;
    }

    // If "histo" is set by user, show histogram on the screen. If m_ScaleNeg is
    // equal to 1, the negative Jacobians in the on-screen histogram will be
    // combined to one bin and will be named as "invalid". If m_ScaleNeg is set
    // by user, the on-screen histogram will show the bins between
    // (-1.0 * m_ScaleNeg * m_interval, 1.0 * m_ScalePos * m_interval).
    if (histogram)
    {
        GetHistogram(histo);
    }

    if (quality)
    {
        Qualitycheck(histo);
        m_log(INFO) << "Worst Jacobian: "
                    << Vmath::Vmin(histo_Jac.size(), histo_Jac, 1) << std::endl;
        m_log(INFO) << "Integration of Jacobian: " << Integral / Area * 100.0
                    << "%" << std::endl;
    }
}

void ProcessJac::GetBoundaryCoordinate(const ElementSharedPtr &el,
                                       std::ofstream &output_file, bool detail)
{
    std::vector<ElementSharedPtr> Neighbor;
    switch (m_mesh->m_expDim)
    {
        case 2:
        {
            int32_t space = ElementID + Jac + type;
            std::vector<EdgeSharedPtr> B_edge; // Boundary edge
            for (auto &edge : el->GetEdgeList())
            {
                // If one edge of a boundary element link to only one element,
                // it is a boundary edge. If one edge link to two elements, it
                // is an interior edge and the other element is the neighbor of
                // this boudnary element. Get this neighbor element and further
                // output the ID of the neighbor into the text file.
                if (edge->m_elLink.size() == 1)
                {
                    B_edge.push_back(edge); // Get the boundary edge.
                    continue;
                }
                for (int j = 0; j < edge->m_elLink.size(); ++j)
                {
                    if (edge->m_elLink[j].first.lock()->GetId() != el->GetId())
                    {
                        // Get the neighbor of the boundary element
                        Neighbor.push_back(edge->m_elLink[j].first.lock());
                    }
                }
            }
            for (int j = 0; j < B_edge.size(); ++j)
            {
                if (j != 0)
                {
                    output_file << "\n";
                    output_file << std::setw(space) << " ";
                }
                // Output boundary edge ID and its two vertex IDs and
                // coordinates
                output_file << std::setw(Boundary_edge_ID)
                            << B_edge.at(j)->m_id;
                output_file << std::setw(VertexID)
                            << B_edge.at(j)->m_n1->GetID() << std::setw(CoordX)
                            << B_edge.at(j)->m_n1->m_x << std::setw(CoordY)
                            << B_edge.at(j)->m_n1->m_y;
                output_file << "\n";
                output_file << std::setw(space + Boundary_edge_ID) << " ";
                output_file << std::setw(VertexID)
                            << B_edge.at(j)->m_n2->GetID() << std::setw(CoordX)
                            << B_edge.at(j)->m_n2->m_x << std::setw(CoordY)
                            << B_edge.at(j)->m_n2->m_y;
            }
            break;
        }
        case 3:
        {
            int32_t space = ElementID + Jac + type + Boundary_face_ID;
            std::vector<FaceSharedPtr> B_face; // Boundary face
            for (auto &face : el->GetFaceList())
            {
                if (face->m_elLink.size() == 1)
                {
                    B_face.push_back(face);
                    continue;
                }
                for (int j = 0; j < face->m_elLink.size(); ++j)
                {
                    if (face->m_elLink[j].first.lock()->GetId() != el->GetId())
                    {
                        Neighbor.push_back(face->m_elLink[j].first.lock());
                    }
                }
            }
            for (int j = 0; j < B_face.size(); ++j)
            {
                if (j != 0)
                {
                    output_file << "\n";
                    output_file << std::setw(ElementID + Jac + type) << " ";
                }
                // Output face ID
                output_file << std::setw(Boundary_face_ID)
                            << B_face.at(j)->m_id;
                SpatialDomains::EntityHolder holder;
                SpatialDomains::Geometry *geom =
                    B_face.at(j)->GetGeom(m_mesh->m_spaceDim, holder);
                // List all edges of the boundary face
                for (int i = 0; i < B_face.at(j)->m_edgeList.size(); ++i)
                {
                    if (i != 0)
                    {
                        output_file << "\n";
                        output_file << std::setw(space) << " ";
                    }
                    // Output edge ID and its vertex IDs and coordinates
                    output_file << std::setw(EdgeID) << geom->GetEid(i);
                    SpatialDomains::PointGeom *point = geom->GetVertex(0);
                    Array<OneD, NekDouble> x(m_mesh->m_expDim, 0.0);
                    point->GetCoords(x);
                    output_file << std::setw(CoordX) << x[0]
                                << std::setw(CoordY) << x[1]
                                << std::setw(CoordZ) << x[2] << "\n";
                    output_file << std::setw(space + EdgeID) << " ";
                    point = geom->GetVertex(1);
                    point->GetCoords(x);
                    output_file << std::setw(CoordX) << x[0]
                                << std::setw(CoordY) << x[1]
                                << std::setw(CoordZ) << x[2];
                }
            }
            break;
        }
        default:
        {
            ASSERTL0(false, "Dimension error: can only obtain boundary "
                            "coordinates in 2D and 3D meshes");
        }
    }
    std::string Composite_Name;
    unsigned int Composite_ID = 0;
    if (detail)
    {
        GetCompositeName(el, Composite_Name, Composite_ID);
        output_file << std::setw(CompositeID) << Composite_ID
                    << std::setw(CompositeName) << Composite_Name;
    }
    output_file << " ";
    for (int j = 0; j < Neighbor.size(); ++j)
    {
        output_file << " " << Neighbor.at(j)->GetId();
    }
    output_file << std::endl;
}
bool ProcessJac::GetCompositeName(const ElementSharedPtr &el,
                                  std::string &CompositeName,
                                  unsigned int &CompositeID)
{
    std::vector<ElementSharedPtr> el_1 =
        m_mesh->m_element[m_mesh->m_expDim - 1];
    int32_t NumFaceEdge = 0;
    if (m_mesh->m_expDim == 2)
    {
        NumFaceEdge = el->GetEdgeList().size();
    }
    else if (m_mesh->m_expDim == 3)
    {
        NumFaceEdge = el->GetFaceList().size();
    }
    else
    {
        ASSERTL0(false, "Dimension error: can only get the composite names in "
                        "2D and 3D meshes");
    }
    for (int k = 0; k < NumFaceEdge; ++k)
    {
        if (el->GetBoundaryLink(k) < 0)
        {
            continue;
        }
        int32_t el_1_ID = el_1[el->GetBoundaryLink(k)]->GetId();
        std::vector<std::pair<int32_t, int32_t>> comps;
        for (auto it : m_mesh->m_composite)
        {
            comps.push_back(
                std::make_pair(it.second->m_id, it.second->m_items.size()));
        }
        std::sort(
            comps.begin(), comps.end(),
            [&](std::pair<int32_t, int32_t> a, std::pair<int32_t, int32_t> b) {
                return a.second < b.second;
            });
        for (int i = 0; i < m_mesh->m_composite.size(); ++i)
        {
            auto it = m_mesh->m_composite.at(comps.at(i).first);
            if (it->m_items[0]->GetId() == 0)
            {
                continue;
            }
            for (int j = 0; j < it->m_items.size(); ++j)
            {
                if (it->m_items[j]->GetId() == el_1_ID)
                {
                    CompositeID   = it->m_id;
                    CompositeName = it->m_label;
                    return true;
                }
            }
        }
    }
    return false;
}
void ProcessJac::GetHistogram(const Array<OneD, NekDouble> &histo)
{
    int32_t maxsize = histo.size();
    std::cout << std::string(100, '=') << "\n";
    std::cout << std::endl;
    Array<OneD, NekDouble> HistoLog(maxsize, 0.0);
    Vmath::Vlog(maxsize, histo, 1, HistoLog, 1);
    std::cout << "y (Number of Elements)"
              << "\n";
    std::cout << "^"
              << "\n";
    std::cout << "|";
    Array<OneD, NekDouble> HistoTemp(maxsize, 0.0);
    for (int j = 0; j < maxsize; ++j)
    {
        HistoTemp[j] = HistoLog[j] / Vmath::Vmax(HistoLog.size(), HistoLog, 1) *
                       HistoLog.size();
    }
    // Output the number of elements of the highest bin in screen
    // histogram.
    for (int j = 0; j < maxsize; ++j)
    {
        if (j == Vmath::Imax(HistoTemp.size(), HistoTemp, 1))
        {
            if (histo[j] - 1 > 0)
            {
                std::cout << std::setw(6) << int(histo[j] - 1.0);
            }
            std::cout << "\n";
            break;
        }
        else
        {
            std::cout << std::setw(6) << " ";
        }
    }
    for (int i = maxsize - 1; i >= 0; --i)
    {
        std::cout << "|";
        for (int j = 0; j < maxsize; ++j)
        {
            if (i < int(HistoTemp[j] + 0.5) && HistoLog[j] != 0)
            {
                std::cout << "|----|";
            }
            else if (int(histo[j] - 1.0) == 0)
            {
                std::cout << std::setw(6) << " ";
            }
            else if (i == int(HistoTemp[j] + 0.5))
            {
                std::cout << std::setw(6) << int(histo[j] - 1.0);
            }
            else
            {
                std::cout << std::setw(6) << " ";
            }
        }
        std::cout << "\n";
    }
    std::cout << " ";
    for (int i = 0; i < histo.size(); ++i)
    {
        std::cout << "-----|";
    }
    std::cout << "-> x (Jac)"
              << "\n";
    std::cout << " ";
    if (m_ScaleNeg == 1)
    {
        std::cout << std::setw(6) << "invalid";
        for (int i = 1; i < m_ScalePos; ++i)
        {
            std::cout << std::setw(6) << std::setprecision(2) << m_interval * i;
        }
    }
    else
    {
        for (int i = m_ScaleNeg - 1; i > 0; --i)
        {
            std::cout << std::setw(6) << std::setprecision(2)
                      << -1.0 * m_interval * NekDouble(i);
        }
        for (int i = 0; i < m_ScalePos; ++i)
        {
            std::cout << std::setw(6) << std::setprecision(2)
                      << m_interval * NekDouble(i);
        }
    }
    std::cout << std::setw(6) << m_histo_max_value << std::endl;
}

void ProcessJac::Qualitycheck(Array<OneD, NekDouble> histo)
{
    NekDouble El_num = Vmath::Vsum(histo.size(), histo, 1) - histo.size();
    std::cout << "\n";
    if (m_ScaleNeg == 1)
    {
        std::cout << "Elements invalid " << std::setw(18) << "<  0.0) = ";
        std::cout << std::setw(6) << std::setprecision(3) << histo[0] - 1
                  << " (";
        std::cout << std::setw(6) << std::setprecision(4)
                  << ((histo[0] - 1) / El_num) * 100 << "%)"
                  << "\n";
    }
    else
    {
        std::cout << "Elements" << std::setw(17) << "<";
        std::cout << std::setw(6) << std::setprecision(3)
                  << -1.0 * NekDouble(m_ScaleNeg - 1) * m_interval << ") = ";
        std::cout << std::setw(6) << std::setprecision(3) << histo[0] - 1
                  << " (";
        std::cout << std::setw(6) << std::setprecision(4)
                  << ((histo[0] - 1) / El_num) * 100 << "%)"
                  << "\n";
        for (int i = m_ScaleNeg - 1; i > 0; --i)
        {
            std::cout << "Elements between (";
            std::cout << std::setw(6) << std::setprecision(3)
                      << -1.0 * NekDouble(i) * m_interval << ",";
            std::cout << std::setw(6) << std::setprecision(3)
                      << -1.0 * NekDouble(i - 1) * m_interval << ") = ";
            std::cout << std::setw(6) << histo[m_ScaleNeg - i] - 1 << " (";
            std::cout << std::setw(6) << std::setprecision(4)
                      << ((histo[m_ScaleNeg - i] - 1) / El_num) * 100 << "%)"
                      << "\n";
        }
    }
    for (int i = 0; i < m_ScalePos; ++i)
    {
        std::cout << "Elements between (";
        std::cout << std::setw(6) << std::setprecision(3)
                  << 1.0 * NekDouble(i) * m_interval << ",";
        std::cout << std::setw(6) << std::setprecision(3)
                  << 1.0 * NekDouble(i + 1) * m_interval << ") = ";
        std::cout << std::setw(6) << std::setprecision(4)
                  << histo[m_ScaleNeg + i] - 1 << " (";
        std::cout << std::setw(6) << std::setprecision(4)
                  << ((histo[m_ScaleNeg + i] - 1) / El_num) * 100 << "%)"
                  << "\n";
    }
    std::cout << std::string(100, '=') << std::endl;
}
} // namespace Nektar::NekMesh
