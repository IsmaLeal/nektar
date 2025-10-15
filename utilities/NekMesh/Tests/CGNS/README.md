# Test files #

Individual elements (all 4th order with mid-volume nodes):
HEXA_125
PENTA_75
PYRA_55
TETRA_35

Whole meshes (mesh elements indicated):
HiOrderMesh_tet_hex_pyr_order4_midvol
linear_tet_prism_pyra_mesh
linear_tet_pyra_hex_mesh

Pyramid shielding tests:
ImpossiblePyramids_linear
ImpossiblePyramids_order4_midface


Not yet working:
ImpossiblePyramids,order4,midvol   (only an issue with the 4th order pyra volume nodes)
PyramidShielding_Pyr55Tet34        (same issue as above ^)
PyramidShielding_Pyr55Tet35        (same issue as above ^)
twoprisms_bad   (issue with the prism elements)