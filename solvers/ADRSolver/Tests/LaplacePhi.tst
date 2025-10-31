<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Velocity potentials around a NACA0012 airfoil at AoA =15 degrees</description>
    <executable>ADRSolver</executable>
    <parameters>NACA0012_Re400.xml LaplacePhi.xml</parameters>
    <files>
        <file description="Session File">NACA0012_Re400.xml</file>
        <file description="Session File">LaplacePhi.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="phi0" tolerance="1e-6">1.3798686e-01</value>
            <value variable="phi1" tolerance="1e-6">4.8161428e-01</value>
            <value variable="phi2" tolerance="1e-6">2.4665487e-01</value>
            <value variable="phi3" tolerance="1e-6">7.6264719e-01</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="phi0" tolerance="1e-6">1.5289871e-01</value>
            <value variable="phi1" tolerance="1e-6">4.8487220e-01</value>
            <value variable="phi2" tolerance="1e-6">2.7239127e-01</value>
            <value variable="phi3" tolerance="1e-6">1.3021036e-01</value>
        </metric>
    </metrics>
</test>
