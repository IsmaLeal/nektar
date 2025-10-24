<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>NS, Couette flow with non-conformal circle rotating, exact solution, parallel</description>
    <executable>CompressibleFlowSolver</executable>
    <parameters>Movement_rotate_couette.xml</parameters>
    <processes>6</processes>
    <files>
        <file description="Session File">Movement_rotate_couette.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-6">1.78417e-09</value>
            <value variable="rhou" tolerance="1e-5">1.04618e-06</value>
            <value variable="rhov" tolerance="1e-5">9.27381e-07</value>
            <value variable="E" tolerance="1e-3">0.000452425</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-6">9.49316e-08</value>
            <value variable="rhou" tolerance="2e-5">5.1062e-05</value>
            <value variable="rhov" tolerance="1e-5">1.89465e-05</value>
            <value variable="E" tolerance="1e-2">0.0215249</value>
        </metric>
    </metrics>
</test>
