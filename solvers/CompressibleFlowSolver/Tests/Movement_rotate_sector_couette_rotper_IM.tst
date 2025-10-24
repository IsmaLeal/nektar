<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>NS, Couette flow with sector rotating, exact solution</description>
    <executable>CompressibleFlowSolver</executable>
    <parameters>Movement_rotate_sector_couette_rotper_IM.xml</parameters>
    <files>
        <file description="Session File">Movement_rotate_sector_couette_rotper_IM.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-6">7.56497e-08</value>
            <value variable="rhou" tolerance="1e-6">4.64301e-07</value>
            <value variable="rhov" tolerance="1e-6">4.1234e-07</value>
            <value variable="rhow" tolerance="1e-6">3.69765e-07</value>
            <value variable="E" tolerance="1e-4">0.000187152</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-6">1.30044e-06</value>
            <value variable="rhou" tolerance="1e-6">4.88914e-06</value>
            <value variable="rhov" tolerance="1e-6">5.57593e-06</value>
            <value variable="rhow" tolerance="1e-6">5.37951e-06</value>
            <value variable="E" tolerance="1e-4">0.00176646</value>
        </metric>
    </metrics>
</test>
