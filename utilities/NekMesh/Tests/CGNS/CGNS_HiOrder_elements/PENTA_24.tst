<?xml version="1.0" encoding="utf-8"?>
<test>
  <description> NekMesh with CGNS for two prisms at 3rd order</description>
    <segment type="sequential">
      <executable>NekMesh</executable>
      <parameters>  PENTA_24.cgns out.xml  </parameters>
      <processes> 1 </processes>
    </segment>
    <segment type="sequential">
      <executable>../FieldConvert/FieldConvert</executable>
      <parameters>out.xml out.plt</parameters>
        <processes> 1 </processes>
    </segment>
    <files>
      <file description="Input File">PENTA_24.cgns</file>
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
