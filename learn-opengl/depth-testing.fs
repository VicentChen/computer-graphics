#version 330 core
in vec2 tex_coord;

out vec4 fragment_color;

uniform sampler2D texture_1;

void main() {
  fragment_color = vec4(vec3(gl_FragCoord.z), 1.0);
}