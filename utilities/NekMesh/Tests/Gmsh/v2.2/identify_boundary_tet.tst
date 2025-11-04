<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Gmsh (v2.2) high-order tet cube, that uses the second tag (CAD tag) to identify boundary composites</description>
    <executable>NekMesh</executable>
    <parameters>-m jac:list identify_boundary_tet.msh:msh:identifyComposite identify_boundary_tet.xml:xml:stats -v -f </parameters>
    <files>
        <file description="Input File">identify_boundary_tet.msh </file>
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

        # Number of composites
        <metric type="regex" id="2">
            <regex>.*Number of composites : (\d+)</regex>
            <matches>
                <match>
                    <field id="0">7</field>
                </match>
            </matches>
        </metric>
        # Number of elements
        <metric type="regex" id="3">
            <regex>.*Elements\s+:\s+(\d+)</regex>
            <matches>
                <match>
                    <field id="0">143</field>
                </match>
            </matches>
        </metric>
        # Number of bnd elements
        <metric type="regex" id="4">
            <regex>.*Bnd elements\s+:\s+(\d+)</regex>
            <matches>
                <match>
                    <field id="0">84</field>
                </match>
            </matches>
        </metric>

    </metrics>
</test>
