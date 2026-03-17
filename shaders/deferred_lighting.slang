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
import pbr_utils;
import lighting_utils;

[[vk::binding(0, 0)]] Sampler2D gbufferAlbedoAlpha;
[[vk::binding(1, 0)]] Sampler2D gbufferNormalRoughness;
[[vk::binding(2, 0)]] Sampler2D gbufferEmissiveMetallic;
[[vk::binding(3, 0)]] Sampler2D gbufferAo;
[[vk::binding(4, 0)]] Sampler2D sceneDepth;
[[vk::binding(5, 0)]] StructuredBuffer<LightData> lightBuffer;
[[vk::binding(6, 0)]] RWTexture2D<float4> lightingOutput;
[[vk::binding(7, 0)]] ConstantBuffer<UniformBufferObject> ubo;
[[vk::binding(8, 0)]] StructuredBuffer<TileHeader> tileHeaders;
[[vk::binding(9, 0)]] StructuredBuffer<uint> tileLightIndices;

float3 reconstructWorldPosition(float2 uv, float depth)
{
    float4 clipPos = float4(uv * 2.0 - 1.0, depth, 1.0);
    float4x4 invViewProj = inverse(mul(ubo.proj, ubo.view));
    float4 worldPos = mul(invViewProj, clipPos);
    float invW = 1.0 / max(abs(worldPos.w), 1e-6);
    return worldPos.xyz * invW;
}

[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint width, height;
    lightingOutput.GetDimensions(width, height);
    if (dispatchThreadID.x >= width || dispatchThreadID.y >= height)
    {
        return;
    }

    float2 invResolution = float2(1.0 / max(float(width), 1.0), 1.0 / max(float(height), 1.0));
    float2 uv = (float2(dispatchThreadID.xy) + 0.5) * invResolution;

    float depth = sceneDepth.SampleLevel(uv, 0.0).r;
    if (depth >= 0.999999)
    {
        lightingOutput[dispatchThreadID.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float4 albedoAlpha = gbufferAlbedoAlpha.SampleLevel(uv, 0.0);
    float4 normalRoughness = gbufferNormalRoughness.SampleLevel(uv, 0.0);
    float4 emissiveMetallic = gbufferEmissiveMetallic.SampleLevel(uv, 0.0);
    float4 aoReserved = gbufferAo.SampleLevel(uv, 0.0);

    float3 albedo = max(albedoAlpha.rgb, 0.0);
    float3 N = normalize(normalRoughness.xyz * 2.0 - 1.0);
    float roughness = clamp(normalRoughness.w, 0.04, 1.0);
    float metallic = clamp(emissiveMetallic.w, 0.0, 1.0);
    float3 emissive = max(emissiveMetallic.rgb, 0.0);
    float ao = saturate(aoReserved.x);

    float3 worldPos = reconstructWorldPosition(uv, depth);
    float3 V = normalize(ubo.camPos.xyz - worldPos);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    float3 diffuseLighting = float3(0.0, 0.0, 0.0);
    float3 specularLighting = float3(0.0, 0.0, 0.0);

    const uint TILE = 16u;
    uint tilesX = (uint(ubo.screenDimensions.x) + TILE - 1u) / TILE;
    uint tilesY = (uint(ubo.screenDimensions.y) + TILE - 1u) / TILE;
    uint totalTiles = max(tilesX * tilesY, 1u);

    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    uint tileX = (tilesX > 0u) ? min(px / TILE, tilesX - 1u) : 0u;
    uint tileY = (tilesY > 0u) ? min(py / TILE, tilesY - 1u) : 0u;

    float dVS = abs(mul(ubo.view, float4(worldPos, 1.0)).z);
    float lnN = log(max(ubo.nearZ, 1e-4));
    float lnF = log(max(ubo.farZ, lnN + 1e-4));
    float denom = max(lnF - lnN, 1e-6);
    float slices = max(ubo.slicesZ, 1.0);
    float lambda = saturate((log(max(dVS, 1e-4)) - lnN) / denom);
    uint slice = (uint)clamp(floor(lambda * slices), 0.0, slices - 1.0);
    uint tileId = (slice * tilesY + tileY) * tilesX + tileX;

    bool useForwardPlus = (ubo.padding1 != 0.0);
    uint base = 0u;
    uint count = 0u;

    if (useForwardPlus && tileId < totalTiles * (uint)slices)
    {
        TileHeader th = tileHeaders[tileId];
        base = th.offset;
        count = th.count;
    }

    if (useForwardPlus && count > 0u)
    {
        [loop]
        for (uint li = 0u; li < count; ++li)
        {
            uint lightIndex = tileLightIndices[base + li];
            LightData light = lightBuffer[lightIndex];
            LightEvaluation eval = evaluateLight(light, worldPos, N);
            accumulateLighting(eval, N, V, albedo, metallic, roughness, F0, diffuseLighting, specularLighting);
        }
    }
    else
    {
        [loop]
        for (uint li = 0u; li < (uint)ubo.lightCount; ++li)
        {
            LightData light = lightBuffer[li];
            LightEvaluation eval = evaluateLight(light, worldPos, N);
            accumulateLighting(eval, N, V, albedo, metallic, roughness, F0, diffuseLighting, specularLighting);
        }
    }

    float3 ambient = albedo * ao * (0.1 * ubo.scaleIBLAmbient);
    float3 color = diffuseLighting + specularLighting + ambient + emissive;
    lightingOutput[dispatchThreadID.xy] = float4(max(color, 0.0), 1.0);
}
