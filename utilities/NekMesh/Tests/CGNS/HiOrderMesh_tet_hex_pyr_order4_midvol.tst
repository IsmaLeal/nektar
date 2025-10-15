<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> NekMesh with CGNS input to high order mesh with mixed elements</description>
    <executable>NekMesh</executable>
    <parameters> -m jac:list HiOrderMesh_tet_hex_pyr_order4_midvol.cgns out.xml:xml:test </parameters>
    <files>
        <file description="Input File">HiOrderMesh_tet_hex_pyr_order4_midvol.cgns</file>
    </files>
    <metrics>
       <metric type="regex" id="1">
            <regex>.*Total negative Jacobians: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
