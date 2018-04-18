#version 330 core
out vec4 fragment_color;

in vec2 tex_coord;

uniform sampler2D depth_map;
uniform float near_plane;
uniform float far_plane;

// required when using a perspective projection matrix
float LinearizeDepth(float depth)
{
    float z = depth * 2.0 - 1.0; // Back to NDC 
    return (2.0 * near_plane * far_plane) / (far_plane + near_plane - z * (far_plane - near_plane));	
}

void main()
{             
    float depthValue = texture(depth_map, tex_coord).r;
    // fragment_color = vec4(vec3(LinearizeDepth(depthValue) / far_plane), 1.0); // perspective
    fragment_color = vec4(vec3(depthValue), 1.0); // orthographic
}