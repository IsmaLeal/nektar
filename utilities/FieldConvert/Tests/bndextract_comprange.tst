<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>extract boundary, 3D channel flow, Hexahedral elements, P=3</description>
    <executable>FieldConvert</executable>
    <parameters>-m extract:bnd=3  Hex_channel_m3.xml:xml:comprange=5 Hex_channel_m3_0.chk bnd.fld -e -f</parameters>
    <files>
        <file description="Session File">Hex_channel_m3.xml</file>
        <file description="field File">Hex_channel_m3_0.chk</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
      	    <value variable="x" tolerance="1e-9">0</value>
            <value variable="y" tolerance="1e-9">0.57735</value>
            <value variable="z" tolerance="1e-9">0.57735</value>
            <value variable="u" tolerance="1e-9">0.00164865</value>
            <value variable="v" tolerance="1e-9">0.00153491</value>
            <value variable="w" tolerance="1e-9">0.47056</value>
            <value variable="p" tolerance="1e-9">0.00082937</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="x" tolerance="1e-9">0</value>
            <value variable="y" tolerance="1e-9">1</value>
            <value variable="z" tolerance="1e-9">1</value>
	    <value variable="u" tolerance="1e-9">8.90468e-07</value>
            <value variable="v" tolerance="1e-9">4.97516e-06</value>
            <value variable="w" tolerance="1e-9">0.138374</value>
            <value variable="p" tolerance="1e-9">2.90323e-14</value>
        </metric>
    </metrics>
</test>

