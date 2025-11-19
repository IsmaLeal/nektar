<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>3D channel flow, Pyramidic elements, using SVV</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>Pyr_channel_SVV.xml</parameters>
    <files>
        <file description="Session File">Pyr_channel_SVV.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-9">1.89957e-08</value>
            <value variable="v" tolerance="1e-9">2.09809e-08</value>
            <value variable="w" tolerance="1e-9">7.73065e-08</value>
            <value variable="p" tolerance="1e-8">8.24215e-07</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="5e-8">8.09166e-08</value>
            <value variable="v" tolerance="5e-8">1.1468e-07</value>
            <value variable="w" tolerance="1e-8">8.35e-07</value>
            <value variable="p" tolerance="1e-6">1.65227e-05</value>
        </metric>
    </metrics>
</test>
