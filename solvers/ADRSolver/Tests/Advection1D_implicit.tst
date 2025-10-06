<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>1D unsteady implicit advection MODIFIED, P=3. Note the test is not physically correct but tests the algorithm.</description>
    <executable>ADRSolver</executable>
    <parameters>Advection1D_implicit.xml</parameters>
    <files>
        <file description="Session File">Advection1D_implicit.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-12">0.230629</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-12">0.375265</value>
        </metric>
    </metrics>
</test>
