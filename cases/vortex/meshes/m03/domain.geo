// 7x7 quad mesh on [0, 2pi]^2.
// Refined around the vortex centre (pi, pi):
// the centre quad [pi-0.2, pi+0.2]^2 has width 0.4 (= 2*rc), well below
// the Lamb-Oseen core scale rc=0.2.
// Vortex centre is the INTERIOR centroid of the middle quad — not on any edge.

SetFactory("OpenCASCADE");
pi = 3.141592653589793;
L  = 2*pi;

// 8 x-cuts -> 7 elements per direction
x0 = 0;
x1 = 1.0;
x2 = pi - 0.7;
x3 = pi - 0.2;
x4 = pi + 0.2;
x5 = pi + 0.7;
x6 = L - 1.0;
x7 = L;
y0 = x0; y1 = x1; y2 = x2; y3 = x3; y4 = x4; y5 = x5; y6 = x6; y7 = x7;

// Helper macro: define a single point at (x,y) with given tag
// We construct an 8x8 array of points -> 64 points total -> 49 quads.

// Points: tag = (j-1)*8 + i  (i,j run 1..8, with x_{i-1}, y_{j-1})
Point(1)  = {x0, y0, 0};   Point(2)  = {x1, y0, 0};   Point(3)  = {x2, y0, 0};   Point(4)  = {x3, y0, 0};
Point(5)  = {x4, y0, 0};   Point(6)  = {x5, y0, 0};   Point(7)  = {x6, y0, 0};   Point(8)  = {x7, y0, 0};

Point(9)  = {x0, y1, 0};   Point(10) = {x1, y1, 0};   Point(11) = {x2, y1, 0};   Point(12) = {x3, y1, 0};
Point(13) = {x4, y1, 0};   Point(14) = {x5, y1, 0};   Point(15) = {x6, y1, 0};   Point(16) = {x7, y1, 0};

Point(17) = {x0, y2, 0};   Point(18) = {x1, y2, 0};   Point(19) = {x2, y2, 0};   Point(20) = {x3, y2, 0};
Point(21) = {x4, y2, 0};   Point(22) = {x5, y2, 0};   Point(23) = {x6, y2, 0};   Point(24) = {x7, y2, 0};

Point(25) = {x0, y3, 0};   Point(26) = {x1, y3, 0};   Point(27) = {x2, y3, 0};   Point(28) = {x3, y3, 0};
Point(29) = {x4, y3, 0};   Point(30) = {x5, y3, 0};   Point(31) = {x6, y3, 0};   Point(32) = {x7, y3, 0};

Point(33) = {x0, y4, 0};   Point(34) = {x1, y4, 0};   Point(35) = {x2, y4, 0};   Point(36) = {x3, y4, 0};
Point(37) = {x4, y4, 0};   Point(38) = {x5, y4, 0};   Point(39) = {x6, y4, 0};   Point(40) = {x7, y4, 0};

Point(41) = {x0, y5, 0};   Point(42) = {x1, y5, 0};   Point(43) = {x2, y5, 0};   Point(44) = {x3, y5, 0};
Point(45) = {x4, y5, 0};   Point(46) = {x5, y5, 0};   Point(47) = {x6, y5, 0};   Point(48) = {x7, y5, 0};

Point(49) = {x0, y6, 0};   Point(50) = {x1, y6, 0};   Point(51) = {x2, y6, 0};   Point(52) = {x3, y6, 0};
Point(53) = {x4, y6, 0};   Point(54) = {x5, y6, 0};   Point(55) = {x6, y6, 0};   Point(56) = {x7, y6, 0};

Point(57) = {x0, y7, 0};   Point(58) = {x1, y7, 0};   Point(59) = {x2, y7, 0};   Point(60) = {x3, y7, 0};
Point(61) = {x4, y7, 0};   Point(62) = {x5, y7, 0};   Point(63) = {x6, y7, 0};   Point(64) = {x7, y7, 0};

// Horizontal lines: 7 per row * 8 rows = 56 lines
// Line(row*7 + col) where row=0..7, col=1..7  -> tags 1..56
ll = 0;
For r In {0:7}
  For c In {0:6}
    p1 = r*8 + c + 1;
    p2 = r*8 + c + 2;
    ll = ll + 1;
    Line(ll) = {p1, p2};
  EndFor
EndFor
// ll is now 56

// Vertical lines: 7 per column * 8 columns = 56 lines, tags 57..112
For c In {0:7}
  For r In {0:6}
    p1 = r*8 + c + 1;
    p2 = (r+1)*8 + c + 1;
    ll = ll + 1;
    Line(ll) = {p1, p2};
  EndFor
EndFor
// ll is now 112

// Surfaces: 49 quads. Quad (r,c) (r,c in 0..6) bounded by
//   bottom: horizontal line at row r, col c             -> tag r*7+c+1
//   right : vertical line at col c+1, row r             -> tag 56 + (c+1)*7 + r + 1
//   top   : horizontal line at row r+1, col c (reversed)-> -(r+1)*7 - c - 1
//   left  : vertical line at col c, row r (reversed)    -> -(56 + c*7 + r + 1)
sid = 0;
For r In {0:6}
  For c In {0:6}
    sid = sid + 1;
    lb = r*7 + c + 1;
    lr = 56 + (c+1)*7 + r + 1;
    lt = (r+1)*7 + c + 1;
    ll_ = 56 + c*7 + r + 1;
    Line Loop(sid) = {lb, lr, -lt, -ll_};
    Plane Surface(sid) = {sid};
  EndFor
EndFor

Recombine Surface "*";
Mesh.ElementOrder = 1;

// Physical groups for NekMesh
// Bottom row 0:    horizontal lines tag 1..7
// Top row 7:       horizontal lines tag 50..56
// Left col 0:      vertical lines tag 57..63
// Right col 7:     vertical lines tag 105..111
Physical Line(1) = {1,2,3,4,5,6,7};                  // Bottom  (row 0, horizontal)
Physical Line(2) = {106,107,108,109,110,111,112};    // Right   (col 7, vertical)
Physical Line(3) = {50,51,52,53,54,55,56};           // Top     (row 7, horizontal)
Physical Line(4) = {57,58,59,60,61,62,63};           // Left    (col 0, vertical)
Physical Surface(10) = {1:49};

Transfinite Curve "*" = 2;
Transfinite Surface "*";
