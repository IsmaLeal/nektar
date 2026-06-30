SetFactory("OpenCASCADE");

pi = 3.141592653589793;
L  = 2*pi;

x0 = 0;   x1 = 1.5; x2 = 4.8; x3 = L;
y0 = 0;   y1 = 1.5; y2 = 4.8; y3 = L;

// Points (4x4 grid)
Point(1)  = {x0, y0, 0};
Point(2)  = {x1, y0, 0};
Point(3)  = {x2, y0, 0};
Point(4)  = {x3, y0, 0};

Point(5)  = {x0, y1, 0};
Point(6)  = {x1, y1, 0};
Point(7)  = {x2, y1, 0};
Point(8)  = {x3, y1, 0};

Point(9)  = {x0, y2, 0};
Point(10) = {x1, y2, 0};
Point(11) = {x2, y2, 0};
Point(12) = {x3, y2, 0};

Point(13) = {x0, y3, 0};
Point(14) = {x1, y3, 0};
Point(15) = {x2, y3, 0};
Point(16) = {x3, y3, 0};

// Horizontal lines (bottom to top)
Line(1)  = {1,2};
Line(2)  = {2,3};
Line(3)  = {3,4};

Line(4)  = {5,6};
Line(5)  = {6,7};
Line(6)  = {7,8};

Line(7)  = {9,10};
Line(8)  = {10,11};
Line(9)  = {11,12};

Line(10) = {13,14};
Line(11) = {14,15};
Line(12) = {15,16};

// Vertical lines (left to right)
Line(13) = {1,5};
Line(14) = {5,9};
Line(15) = {9,13};

Line(16) = {2,6};
Line(17) = {6,10};
Line(18) = {10,14};

Line(19) = {3,7};
Line(20) = {7,11};
Line(21) = {11,15};

Line(22) = {4,8};
Line(23) = {8,12};
Line(24) = {12,16};

// 9 surfaces (line loops)
Line Loop(1) = {1,16,-4,-13};
Plane Surface(1) = {1};

Line Loop(2) = {2,19,-5,-16};
Plane Surface(2) = {2};

Line Loop(3) = {3,22,-6,-19};
Plane Surface(3) = {3};

Line Loop(4) = {4,17,-7,-14};
Plane Surface(4) = {4};

Line Loop(5) = {5,20,-8,-17};
Plane Surface(5) = {5};

Line Loop(6) = {6,23,-9,-20};
Plane Surface(6) = {6};

Line Loop(7) = {7,18,-10,-15};
Plane Surface(7) = {7};

Line Loop(8) = {8,21,-11,-18};
Plane Surface(8) = {8};

Line Loop(9) = {9,24,-12,-21};
Plane Surface(9) = {9};

Recombine Surface "*";
Mesh.ElementOrder = 1;

// Physical groups for NekMesh
Physical Line(1) = {1,2,3};       // Bottom
Physical Line(2) = {22,23,24};    // Right
Physical Line(3) = {10,11,12};    // Top
Physical Line(4) = {13,14,15};    // Left
Physical Surface(10) = {1:9};

Transfinite Curve "*" = 2;
Transfinite Surface "*";
