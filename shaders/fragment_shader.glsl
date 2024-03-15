#version 330
//in vec3 newColor;
in vec2 v_texture;
in vec3 v_normal;

out vec4 outColor;

uniform sampler2D samplerTex;

void main()
{
    vec3 ambientLightIntensity = vec3(0.5f, 0.5f, 0.5f);
    vec3 LightIntensity = vec3(0.9f, 0.9f, 0.9f);
    vec3 LightDirection = normalize(vec3(-0.0f, 0.0f, -1.0f));

    vec4 texel = texture(samplerTex, v_texture);
    

    vec3 lightIntensity = ambientLightIntensity + LightIntensity * max(dot(v_normal, LightDirection), 0.0f);

    outColor = vec4(texel.rgb * lightIntensity, texel.a);
}