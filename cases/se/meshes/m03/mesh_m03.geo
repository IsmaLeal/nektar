// Sudden expansion ER=3, structured/transfinite, symmetric about y=0.
  // Verticals uniform (symmetry); horizontals graded toward the step at x=0.
  // m03: outlet extended x=20 -> x=40 (Ducimetiere et al. 2024 domain, M
  // integrates v on the axis over 0<=x<=40). Expansion-section grading
  // rebalanced (nxe 40->70, progression 1.03->1.02) so the first cell at
  // the step keeps m02's size (~0.274 vs ~0.277) while the outlet-end
  // cell grows to ~1.05.

  // --- points: 3 cols (x=-5,0,40) ---
  Point(1)={-5,-0.5,0};  Point(2)={-5,0,0};
  Point(3)={-5,0.5,0};
  Point(4)={0,-1.5,0};   Point(5)={0,-0.5,0};  Point(6)={0,0,0};
  Point(7)={0,0.5,0};    Point(8)={0,1.5,0};
  Point(9)={40,-1.5,0};  Point(10)={40,-0.5,0};
  Point(11)={40,0,0};
  Point(12)={40,0.5,0};  Point(13)={40,1.5,0};

  // --- horizontals (left->right) ---
  Line(1)={1,5}; Line(2)={2,6}; Line(3)={3,7};
  // inlet: y=-0.5, 0(axis), 0.5
  Line(4)={4,9}; Line(5)={5,10}; Line(6)={6,11};
  // exp: y=-1.5,-0.5, 0(axis)
  Line(7)={7,12}; Line(8)={8,13};
  // exp: y=0.5, 1.5

  // --- verticals (bottom->top) ---
  Line(11)={1,2}; Line(12)={2,3};
  // x=-5
  Line(13)={4,5}; Line(14)={5,6}; Line(15)={6,7};
  Line(16)={7,8};   // x=0
  Line(17)={9,10}; Line(18)={10,11}; Line(19)={11,12};
  Line(20)={12,13}; // x=40

  // --- surfaces ---
  Curve Loop(1)={1,14,-2,-11};   Plane Surface(1)={1};   // inlet_L
  Curve Loop(2)={2,15,-3,-12};   Plane Surface(2)={2};   // inlet_U
  Curve Loop(3)={4,17,-5,-13};   Plane Surface(3)={3};   // exp_LO
  Curve Loop(4)={5,18,-6,-14};   Plane Surface(4)={4};   // exp_LI
  Curve Loop(5)={6,19,-7,-15};   Plane Surface(5)={5};   // exp_UI
  Curve Loop(6)={7,20,-8,-16};   Plane Surface(6)={6};   // exp_UO

  // --- transfinite ---
  nxi=10; nxe=70; nyi=6; nyo=10;
  Transfinite Curve{1,2,3}        = nxi Using Progression 0.96;
   // inlet x: refine at x=0
  Transfinite Curve{4,5,6,7,8}    = nxe Using Progression 1.02;
   // exp x: refine at x=0
  Transfinite Curve{11,12,14,15,18,19} = nyi;
   // inner y (h=0.5), uniform
  Transfinite Curve{13,16,17,20}  = nyo;
   // outer y (h=1.0), uniform
  Transfinite Surface{1,2,3,4,5,6};
  Recombine Surface{1,2,3,4,5,6};
   // quads; delete for triangles

  // --- physical groups (IDs match your casefile composites) ---
  Physical Curve(9)  = {11,12};            // inlet  (x=-5)
  Physical Curve(11) = {17,18,19,20};      // outlet (x=40)
  Physical Curve(10) = {1,3,4,8,13,16};    // walls + step faces
  Physical Curve(13) = {2,6};              // symmetry axis y=0
  Physical Surface(12) = {1,2,3,4,5,6};    // domain
