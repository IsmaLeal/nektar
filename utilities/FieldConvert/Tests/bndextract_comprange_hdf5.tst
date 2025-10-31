<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>extract boundary, 3D channel flow, Hexahedral elements with Hdf5, P=3</description>
    <executable>FieldConvert</executable>
    <parameters>-m extract:bnd=1  Hex_channel_m3_hdf5.xml:xml:comprange=6 Hex_channel_m3_0.chk bnd.fld -e -f</parameters>
    <files>
        <file description="Session File">Hex_channel_m3_hdf5.xml</file>
        <file description="Session File">Hex_channel_m3_hdf5.nekg</file>
        <file description="field File">Hex_channel_m3_0.chk</file>
    </files>
    <metrics>

       <metric type="L2" id="1">
      	    <value variable="x" tolerance="1e-9">0.57735</value>
            <value variable="y" tolerance="1e-9">0.57735</value>
            <value variable="z" tolerance="1e-9">1</value>
            <value variable="u" tolerance="1e-9">0.296847</value>
            <value variable="v" tolerance="1e-9">0.0837328</value>
            <value variable="w" tolerance="1e-9">0.484771</value>
            <value variable="p" tolerance="1e-9">0.408285</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="x" tolerance="1e-9">1</value>
            <value variable="y" tolerance="1e-9">1</value>
            <value variable="z" tolerance="1e-9">1</value>
	    <value variable="u" tolerance="1e-9">0.524745</value>
            <value variable="v" tolerance="1e-9">0.138487</value>
            <value variable="w" tolerance="1e-9">0.939717</value>
            <value variable="p" tolerance="1e-9">0.997512</value>
        </metric>
    </metrics>
</test>

