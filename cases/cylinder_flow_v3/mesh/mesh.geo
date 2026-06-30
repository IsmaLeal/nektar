// Gmsh project created on Thu Feb 12 15:04:44 2026
SetFactory("OpenCASCADE");
//+
Rectangle(1) = {-5, -5, 0, 15, 10, 0};
//+
Disk(2) = {0, 0, 0, 0.5, 0.5};
//+
Curve Loop(3) = {3, 4, 1, 2};
//+
Curve Loop(4) = {5};
//+
Plane Surface(3) = {3, 4};
//+
Physical Curve(6) = {3, 1};
//+
Physical Curve(7) = {4};
//+
Physical Curve(8) = {2};
//+
Physical Curve(9) = {5};
//+
Physical Surface(10) = {3};
