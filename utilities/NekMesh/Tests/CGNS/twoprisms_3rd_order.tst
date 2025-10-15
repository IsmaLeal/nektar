<?xml version="1.0" encoding="utf-8"?>
<test>
  <description> NekMesh with CGNS for two prisms at 3rd order</description>
    <segment type="sequential">
      <executable>NekMesh</executable>
      <parameters>  twoprisms_3rd_order.cgns twoprisms_3rd_order.xml  </parameters>
      <processes> 1 </processes>
    </segment>
    <segment type="sequential">
      <executable>../FieldConvert/FieldConvert</executable>
      <parameters>twoprisms_3rd_order.xml out.plt</parameters>
        <processes> 1 </processes>
    </segment>
    <files>
      <file description="Input File">twoprisms_3rd_order.cgns</file>
    </files>
    <metrics>
      <metric type="nowarning" id="1"> 
        <regex>.*Message :.(\w+).(\w+).*</regex>
        <matches>
          <match>
            <field id="1">3D</field>
            <field id="2">deformed</field>
          </match>
        </matches>
      </metric>
    </metrics>
</test>
