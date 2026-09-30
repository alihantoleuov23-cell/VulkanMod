#version 450
#extension GL_ARB_separate_shader_objects : enable

precision highp float;
precision mediump sampler2D;

layout(early_fragment_tests) in;

layout(binding = 1) uniform sampler2D diffuseTex;
layout(binding = 2) uniform sampler2D lightMap;

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec2 fragLightCoord;
layout(location = 3) in float fragDist;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 baseColor = texture(diffuseTex, fragTexCoord);
    if (baseColor.a < 0.1) discard;
    outColor = baseColor * fragColor * texture(lightMap, fragLightCoord);
}
