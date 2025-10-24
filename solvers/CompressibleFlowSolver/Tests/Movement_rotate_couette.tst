<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>NS, Couette flow with non-conformal circle rotating, exact solution</description>
    <executable>CompressibleFlowSolver</executable>
    <parameters>Movement_rotate_couette.xml</parameters>
    <files>
        <file description="Session File">Movement_rotate_couette.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-6">1.86712e-09</value>
            <value variable="rhou" tolerance="1e-5">1.49861e-06</value>
            <value variable="rhov" tolerance="1e-5">1.01588e-06</value>
            <value variable="E" tolerance="1e-5">0.000516945</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-6">4.80025e-08</value>
            <value variable="rhou" tolerance="1e-5">6.29942e-05</value>
            <value variable="rhov" tolerance="1e-5">2.5512e-05</value>
            <value variable="E" tolerance="1e-3">0.0130325</value>
        </metric>
    </metrics>
</test>
