<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> NekMesh ProjectCAD with Shaft (check for detailed CAD projection info) </description>
    <executable>NekMesh</executable>
    <parameters> -m bl:nq=4:surf=1,2:layers=10:r=1.2 -m jac:list -m linkcheck 04_bl_ShaftHO.xml 04_bl_ShaftHO-out.xml:xml:test -v </parameters>
    <files>
        <file description="Input File">04_bl_ShaftHO.xml</file>
    </files>
    <metrics>
        # Mesh Quality
        <metric type="regex" id="1">
            <regex>.*Total negative Jacobians: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                </match>
            </matches>
        </metric>
        # Main BL
        <metric type="regex" id="2">
            <regex>.*Elements after\s+=\s+(\d+)</regex>
            <matches>
                <match>
                    <field id="0">474</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
