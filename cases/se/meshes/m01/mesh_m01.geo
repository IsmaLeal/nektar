// Sudden expansion geometry
Point(1) = {-5, -0.5, 0, 1.0};
Point(2) = {0, -0.5, 0, 1.0};
Point(3) = {0, -1.5, 0, 1.0};
Point(4) = {20, -1.5, 0, 1.0};
Point(5) = {-5, 0.5, 0, 1.0};
Point(6) = {0, 0.5, 0, 1.0};
Point(7) = {0, 1.5, 0, 1.0};
Point(8) = {20, 1.5, 0, 1.0};
Line(1) = {5, 1};
Line(2) = {1, 2};
Line(3) = {2, 3};
Line(4) = {3, 4};
Line(5) = {4, 8};
Line(6) = {8, 7};
Line(7) = {7, 6};
Line(8) = {6, 5};
Curve Loop(1) = {6, 7, 8, 1, 2, 3, 4, 5};
Plane Surface(1) = {1};
Physical Curve(9) = {1};
Physical Curve(10) = {8, 7, 6, 4, 3, 2};
Physical Curve(11) = {5};
Physical Surface(12) = {1};

// --- mesh grading around the step faces (recirculation zone) ---
Field[1] = Distance;
Field[1].CurvesList = {3, 7};
Field[1].Sampling = 100;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = 0.05;
Field[2].SizeMax = 1.0;
Field[2].DistMin = 0.5;
Field[2].DistMax = 5.0;

Background Field = 2;

// let the background field be the sole size authority
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
//+
Field[2].SizeMin = 0.1;
//+
Field[2].SizeMin = 0.5;
//+
Field[2].SizeMin = 0.3;
//+
Field[2].DistMin = 0.15;
//+
Field[2].DistMin = 0.5;
//+
Field[2].SizeMin = 0.5;
//+
Field[2].SizeMin = 0.4;
