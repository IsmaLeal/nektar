<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> NekMesh ProjectCAD with Shaft (check for detailed CAD projection info) </description>
    <executable>NekMesh</executable>
    <parameters> -m bl:nq=5:surf=6,7,8:layers=4:r=3 -m jac:list -m linkcheck 05_bl_NACA0012.xml 05_bl_NACA0012-out.xml:xml:test -v </parameters>
    <files>
        <file description="Input File">05_bl_NACA0012.xml</file>
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
                    <field id="0">6901</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
