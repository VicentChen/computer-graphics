#version 330 core
out vec4 FragColor;

in vec3 Normal;  
in vec3 FragPos;  
in vec3 specular;
in vec3 diffuse;
in vec3 ambient;
  
uniform vec3 objectColor;

void main()
{        
    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
} 