#version 330 core
in vec4 frag_pos;

uniform vec3 light_pos;
uniform float far_plane;

void main()
{
    float lightDistance = length(frag_pos.xyz - light_pos);
    
    // map to [0;1] range by dividing by far_plane
    lightDistance = lightDistance / far_plane;
    
    // write this as modified depth
    gl_FragDepth = lightDistance;
}
