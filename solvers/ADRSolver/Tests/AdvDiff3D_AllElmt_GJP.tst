<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>2D unsteady CG Adv Diff 3D, with All elmts P=5 with Gradient Stabilisation </description>
    <executable>ADRSolver</executable>
    <parameters>AdvDiff3D_AllElmt_GJP.xml</parameters>
    <files>
        <file description="Session File">AdvDiff3D_AllElmt_GJP.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-8">0.00442664</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-8">0.0581488</value>
        </metric>
    </metrics>
</test>

