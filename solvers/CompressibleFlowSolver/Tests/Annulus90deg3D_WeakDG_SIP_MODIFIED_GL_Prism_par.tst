<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>cNS, manufactured, C-shape, Prism, variable P, see issue #306</description>
    <executable>CompressibleFlowSolver</executable>
    <parameters>Annulus90deg3D_WeakDG_SIP_MODIFIED_GL_Prism.xml</parameters>
    <processes>8</processes>
    <files>
        <file description="Session File">Annulus90deg3D_WeakDG_SIP_MODIFIED_GL_Prism.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-05">6.27674e-04</value>
            <value variable="rhou" tolerance="1e-05">4.24951e-03</value>
            <value variable="rhov" tolerance="1e-05">1.77652e-03</value>
            <value variable="rhow" tolerance="1e-05">1.47034e-08</value>
            <value variable="E" tolerance="1e-01">1.35260e+02</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="5e-05">1.17333e+00</value>
            <value variable="rhou" tolerance="1e-02">1.22616e+01</value>
            <value variable="rhov" tolerance="1e-02">6.64597</value>
            <value variable="rhow" tolerance="3e-04">1.79436e-04</value>
            <value variable="E" tolerance="10">2.52674e+05</value>
        </metric>
    </metrics>
</test>
