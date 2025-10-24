<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>NS, Couette flow with sector rotating, exact solution, parallel</description>
    <executable>CompressibleFlowSolver</executable>
    <parameters>Movement_rotate_sector_couette_rotper_IM.xml</parameters>
    <processes>2</processes>
    <files>
        <file description="Session File">Movement_rotate_sector_couette_rotper_IM.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-6">7.56297e-08</value>
            <value variable="rhou" tolerance="1e-6">4.64125e-07</value>
            <value variable="rhov" tolerance="1e-6">4.12447e-07</value>
            <value variable="rhow" tolerance="1e-6">3.69947e-07</value>
            <value variable="E" tolerance="1e-4">0.000187311</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-6">1.3005e-06</value>
            <value variable="rhou" tolerance="1e-6">4.88876e-06</value>
            <value variable="rhov" tolerance="1e-6">5.66227e-06</value>
            <value variable="rhow" tolerance="1e-6">5.3949e-06</value>
            <value variable="E" tolerance="1e-4">0.00177369</value>
        </metric>
    </metrics>
</test>
