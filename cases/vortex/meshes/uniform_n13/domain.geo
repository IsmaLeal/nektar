// 13x13 uniform quad mesh on [0, 2pi]^2.
// N = 13 (ODD) so the vortex center (pi, pi) sits at the centroid of
// element (6, 6) (0-indexed) — strictly interior, never on an edge/vertex.
// Element width = 2*pi/13 ≈ 0.4833.

SetFactory("OpenCASCADE");
pi = 3.141592653589793;
L  = 2*pi;
N  = 13;
h  = L / N;   // 0.4833...

// Points: (N+1)^2 = 196 points, indexed (j*(N+1) + i + 1) for i=0..N, j=0..N
For j In {0:N}
  For i In {0:N}
    tag = j*(N+1) + i + 1;
    Point(tag) = {i*h, j*h, 0};
  EndFor
EndFor

// Horizontal lines: N per row * (N+1) rows = N*(N+1) lines, tags 1 .. N*(N+1)
ll = 0;
For r In {0:N}
  For c In {0:N-1}
    p1 = r*(N+1) + c + 1;
    p2 = r*(N+1) + c + 2;
    ll = ll + 1;
    Line(ll) = {p1, p2};
  EndFor
EndFor
nH = ll;   // = N*(N+1) = 13*14 = 182

// Vertical lines: N per column * (N+1) columns
For c In {0:N}
  For r In {0:N-1}
    p1 = r*(N+1) + c + 1;
    p2 = (r+1)*(N+1) + c + 1;
    ll = ll + 1;
    Line(ll) = {p1, p2};
  EndFor
EndFor

// Surfaces: N*N quads. Quad (r, c) (r, c in 0..N-1).
sid = 0;
For r In {0:N-1}
  For c In {0:N-1}
    sid = sid + 1;
    lb = r*N + c + 1;                    // bottom: horizontal at row r, col c
    lr = nH + (c+1)*N + r + 1;           // right : vertical   at col c+1, row r
    lt = (r+1)*N + c + 1;                // top   : horizontal at row r+1, col c
    ll_ = nH + c*N + r + 1;              // left  : vertical   at col c, row r
    Line Loop(sid) = {lb, lr, -lt, -ll_};
    Plane Surface(sid) = {sid};
  EndFor
EndFor

Recombine Surface "*";
Mesh.ElementOrder = 1;

// Physical groups for NekMesh (linear lines on the boundary)
// Bottom row 0: horizontal lines tags 1..N
// Top row N:    horizontal lines tags N*N+1 .. N*(N+1)
// Left col 0:   vertical lines tags nH+1 .. nH+N
// Right col N:  vertical lines tags nH+N*N+1 .. nH+N*(N+1)
bottom_lines[] = {};
For c In {0:N-1}
  bottom_lines[] += {c+1};
EndFor
Physical Line(1) = bottom_lines[];

right_lines[] = {};
For r In {0:N-1}
  right_lines[] += {nH + N*N + r + 1};
EndFor
Physical Line(2) = right_lines[];

top_lines[] = {};
For c In {0:N-1}
  top_lines[] += {N*N + c + 1};
EndFor
Physical Line(3) = top_lines[];

left_lines[] = {};
For r In {0:N-1}
  left_lines[] += {nH + r + 1};
EndFor
Physical Line(4) = left_lines[];

surf_list[] = {};
For s In {1:N*N}
  surf_list[] += {s};
EndFor
Physical Surface(10) = surf_list[];

Transfinite Curve "*" = 2;
Transfinite Surface "*";
