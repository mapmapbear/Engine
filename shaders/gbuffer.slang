/* Copyright (c) 2025 Holochip Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import common_types;

struct VSInput {
    [[vk::location(0)]] float3 Position;
    [[vk::location(1)]] float3 Normal;
    [[vk::location(2)]] float2 UV;
    [[vk::location(3)]] float4 Tangent;

    [[vk::location(4)]] column_major float4x4 InstanceModelMatrix;
    [[vk::location(8)]] float4 InstanceNormal0;
    [[vk::location(9)]] float4 InstanceNormal1;
    [[vk::location(10)]] float4 InstanceNormal2;
};

struct VSOutput {
    float4 Position : SV_POSITION;
    float3 WorldPos : TEXCOORD0;
    float3 Normal : NORMAL;
    float2 UV : TEXCOORD1;
    float4 Tangent : TANGENT;
};

struct GBufferOutput {
    float4 AlbedoAlpha : SV_TARGET0;
    float4 NormalRoughness : SV_TARGET1;
    float4 EmissiveMetallic : SV_TARGET2;
    float4 AoReserved : SV_TARGET3;
};

[[vk::binding(0, 0)]] ConstantBuffer<UniformBufferObject> ubo;
[[vk::binding(1, 0)]] Sampler2D baseColorMap;
[[vk::binding(2, 0)]] Sampler2D metallicRoughnessMap;
[[vk::binding(3, 0)]] Sampler2D normalMap;
[[vk::binding(4, 0)]] Sampler2D occlusionMap;
[[vk::binding(5, 0)]] Sampler2D emissiveMap;

[[vk::push_constant]] PushConstants material;

[[shader("vertex")]]
VSOutput VSMain(VSInput input)
{
    VSOutput output;

    float4x4 instanceModelMatrix = input.InstanceModelMatrix;
    float4 worldPos = mul(ubo.model, mul(instanceModelMatrix, float4(input.Position, 1.0)));
    output.Position = mul(ubo.proj, mul(ubo.view, worldPos));
    output.WorldPos = worldPos.xyz;

    float3x3 instNormal = float3x3(input.InstanceNormal0.xyz, input.InstanceNormal1.xyz, input.InstanceNormal2.xyz);
    float3x3 model3x3 = (float3x3)ubo.model;
    float3 worldNormal = normalize(mul(model3x3, mul(instNormal, input.Normal)));
    output.Normal = worldNormal;

    float3 worldTangent = normalize(mul(model3x3, mul(instNormal, input.Tangent.xyz)));
    output.UV = input.UV;
    output.Tangent = float4(worldTangent, input.Tangent.w);
    return output;
}

[[shader("fragment")]]
GBufferOutput PSMain(VSOutput input)
{
    GBufferOutput output;

    float2 uv = float2(input.UV.x, 1.0 - input.UV.y);

    float4 baseColor = (material.baseColorTextureSet < 0)
        ? material.baseColorFactor
        : baseColorMap.Sample(uv) * material.baseColorFactor;

    if (material.alphaMask > 0.5 && baseColor.a < material.alphaMaskCutoff)
    {
        discard;
    }

    float4 mrOrSpecGloss = (material.physicalDescriptorTextureSet < 0)
        ? float4(1.0, 1.0, 1.0, 1.0)
        : metallicRoughnessMap.Sample(uv);

    float metallic = 0.0;
    float roughness = 1.0;

    if (material.useSpecGlossWorkflow != 0)
    {
        float gloss = clamp(mrOrSpecGloss.a * material.glossinessFactor, 0.0, 1.0);
        roughness = clamp(1.0 - gloss, 0.0, 1.0);
        float maxSpec = max(material.specularFactor.x, max(material.specularFactor.y, material.specularFactor.z));
        metallic = saturate(maxSpec);
    }
    else
    {
        float metallicTex = mrOrSpecGloss.b;
        float roughnessTex = mrOrSpecGloss.g;
        metallic = clamp(metallicTex * material.metallicFactor, 0.0, 1.0);
        roughness = clamp(roughnessTex * material.roughnessFactor, 0.0, 1.0);
    }

    float ao = (material.occlusionTextureSet < 0) ? 1.0 : occlusionMap.Sample(uv).r;

    float3 emissiveTex = (material.emissiveTextureSet < 0) ? float3(1.0, 1.0, 1.0) : emissiveMap.Sample(uv).rgb;
    float3 emissive = emissiveTex * material.emissiveFactor;
    if (material.hasEmissiveStrengthExt)
    {
        emissive *= material.emissiveStrength;
    }

    float3 N = normalize(input.Normal);
    if (material.normalTextureSet >= 0)
    {
        float3 tangentNormal = normalMap.Sample(uv).xyz * 2.0 - 1.0;
        float3 T = normalize(input.Tangent.xyz);
        float handedness = -input.Tangent.w;
        float3 B = normalize(cross(N, T)) * handedness;
        float3x3 TBN = float3x3(T, B, N);
        N = normalize(mul(TBN, tangentNormal));
    }

    float3 packedNormal = N * 0.5 + 0.5;

    output.AlbedoAlpha = float4(baseColor.rgb, baseColor.a);
    output.NormalRoughness = float4(packedNormal, roughness);
    output.EmissiveMetallic = float4(max(emissive, 0.0), metallic);
    output.AoReserved = float4(ao, 0.0, 0.0, 1.0);
    return output;
}
