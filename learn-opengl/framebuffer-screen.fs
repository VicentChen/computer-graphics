#version 330 core

in vec2 tex_coords;

out vec4 fragment_color;

uniform sampler2D texture_1;

const float offset = 1.0f / 1280.0f;

void main() {
  // inverse
  // fragment_color = vec4(1.0f) - texture(texture_1, tex_coords);

  // gray scale
  // fragment_color = texture(texture_1, tex_coords);
  // float average = (fragment_color.r + fragment_color.g + fragment_color.b) / 3.0f;
  // fragment_color = vec4(average, average, average, 1.0f);

  // gray scale
  // fragment_color = texture(texture_1, tex_coords);
  // float average = 0.2126 * fragment_color.r + 0.7152 * fragment_color.g + 0.0722 * fragment_color.b;
  // fragment_color = vec4(average, average, average, 1.0f);

  // kernel
  vec2 offset[9] = vec2[](
    vec2(-offset,  offset), // top-left
    vec2( 0.0f,    offset), // top-center
    vec2( offset,  offset), // top-right
    vec2(-offset,  0.0f),   // center-left
    vec2( 0.0f,    0.0f),   // center-center
    vec2( offset,  0.0f),   // center-right
    vec2(-offset, -offset), // bottom-left
    vec2( 0.0f,   -offset), // bottom-center
    vec2( offset, -offset)  // bottom-right    
  );

  // sharpen
  // float kernel[9] = float[](
  //   2, 2, 2,
  //   2, -15, 2,
  //   2, 2, 2
  // );

  // blur
  // float kernel[9] = float[](
  //   1.0 / 16, 2.0 / 16, 1.0 / 16,
  //   2.0 / 16, 4.0 / 16, 2.0 / 16,
  //   1.0 / 16, 2.0 / 16, 1.0 / 16  
  // );

  // edge detection
  float kernel[9] = float[] (
    1, 1, 1,
    1, -8, 1,
    1, 1, 1
  );

  vec3 sample_tex[9];
  for (int i = 0; i < 9; i++)
    sample_tex[i] = vec3(texture(texture_1, tex_coords.st + offset[i]));
  
  vec3 col = vec3(0.0f);
  for (int i  = 0; i < 9; i++)
    col += sample_tex[i] * kernel[i];
  
  fragment_color = vec4(col, 1.0f);
}