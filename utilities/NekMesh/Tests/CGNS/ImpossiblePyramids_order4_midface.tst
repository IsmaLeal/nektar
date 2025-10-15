<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> NekMesh with CGNS input to test a linear mesh with tet, prism and pyra elements</description>
    <executable>NekMesh</executable>
    <parameters> -m jac:list ImpossiblePyramids_order4_midface.cgns out.xml:xml:test </parameters>
    <files>
        <file description="Input File">ImpossiblePyramids_order4_midface.cgns</file>
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
