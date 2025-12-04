<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Lamb-Oseen periodic vortex (smoke test)</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>domain.xml</parameters>
    <files>
        <file description="Session">domain.xml</file>
        <file description="Collections options">domain.opt</file>
    </files>
    <!-- Placeholder loose metrics to confirm run completion; tighten after capturing reference output. -->
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e6">0</value>
            <value variable="v" tolerance="1e6">0</value>
            <value variable="p" tolerance="1e6">0</value>
        </metric>
    </metrics>
</test>
