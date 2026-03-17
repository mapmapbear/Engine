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

struct FrustumCullAABB {
    float4 minBounds;
    float4 maxBounds;
};

struct OcclusionCullParams {
    column_major float4x4 clipFromWorld;
    float4 depthPyramidInfo;
    float4 safetyInfo;
};

[[vk::binding(0, 0)]] StructuredBuffer<FrustumCullAABB> instanceAabbsRO;
[[vk::binding(1, 0)]] StructuredBuffer<uint> frustumVisibleIndicesRO;
[[vk::binding(2, 0)]] StructuredBuffer<uint> frustumVisibleCountRO;
[[vk::binding(3, 0)]] Texture2D<float> depthPyramidRO;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> occlusionVisibleIndicesRW;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> occlusionVisibleCountRW;
[[vk::binding(6, 0)]] ConstantBuffer<OcclusionCullParams> params;

float sampleDepthPyramid(uint mipLevel, uint2 baseSize, float2 pixelCoord)
{
    uint mipWidth = 0;
    uint mipHeight = 0;
    uint mipLevels = 0;
    depthPyramidRO.GetDimensions(mipLevel, mipWidth, mipHeight, mipLevels);
    if (mipWidth == 0u || mipHeight == 0u)
    {
        return 1.0;
    }

    uint2 mipSize = uint2(mipWidth, mipHeight);
    float2 mipScale = float2(baseSize) / float2(mipSize);
    float2 mipCoordF = pixelCoord / mipScale;
    float2 maxCoord = float2(mipSize) - float2(1.0, 1.0);
    int2 mipCoord = int2(clamp(mipCoordF, float2(0.0, 0.0), maxCoord));
    return depthPyramidRO.Load(int3(mipCoord, int(mipLevel)));
}

bool computeProjectedBounds(FrustumCullAABB aabb, out float2 rectMinUv, out float2 rectMaxUv, out float nearestDepth)
{
    rectMinUv = float2(1.0, 1.0);
    rectMaxUv = float2(0.0, 0.0);
    nearestDepth = 1.0;

    [unroll]
    for (uint corner = 0; corner < 8; ++corner)
    {
        float3 p = float3(
            ((corner & 1u) != 0u) ? aabb.maxBounds.x : aabb.minBounds.x,
            ((corner & 2u) != 0u) ? aabb.maxBounds.y : aabb.minBounds.y,
            ((corner & 4u) != 0u) ? aabb.maxBounds.z : aabb.minBounds.z);

        float4 clip = mul(params.clipFromWorld, float4(p, 1.0));
        if (clip.w <= 1e-5)
        {
            return false;
        }

        float invW = rcp(clip.w);
        float2 uv = clip.xy * invW * 0.5 + 0.5;
        float depth = saturate(clip.z * invW * 0.5 + 0.5);

        rectMinUv = min(rectMinUv, uv);
        rectMaxUv = max(rectMaxUv, uv);
        nearestDepth = min(nearestDepth, depth);
    }

    return true;
}

bool isVisibleConservative(FrustumCullAABB aabb)
{
    uint2 baseSize = uint2(max(1u, (uint)params.depthPyramidInfo.x), max(1u, (uint)params.depthPyramidInfo.y));
    uint mipCount = max(1u, (uint)params.depthPyramidInfo.z);

    float2 rectMinUv;
    float2 rectMaxUv;
    float nearestDepth;
    if (!computeProjectedBounds(aabb, rectMinUv, rectMaxUv, nearestDepth))
    {
        return true;
    }

    if (nearestDepth <= 0.0)
    {
        return true;
    }

    if (rectMaxUv.x <= 0.0 || rectMaxUv.y <= 0.0 || rectMinUv.x >= 1.0 || rectMinUv.y >= 1.0)
    {
        return true;
    }

    if (rectMinUv.x < 0.0 || rectMinUv.y < 0.0 || rectMaxUv.x > 1.0 || rectMaxUv.y > 1.0)
    {
        return true;
    }

    float2 baseSizeF = float2(baseSize);
    float guardBandPixels = max(0.0, params.safetyInfo.x);
    float2 rectMinPx = rectMinUv * baseSizeF - guardBandPixels.xx;
    float2 rectMaxPx = rectMaxUv * baseSizeF + guardBandPixels.xx;

    rectMinPx = clamp(rectMinPx, float2(0.0, 0.0), baseSizeF - float2(1.0, 1.0));
    rectMaxPx = clamp(rectMaxPx, float2(0.0, 0.0), baseSizeF - float2(1.0, 1.0));

    float2 rectSizePx = max(rectMaxPx - rectMinPx, float2(1.0, 1.0));
    float screenArea = max(1.0, baseSizeF.x * baseSizeF.y);
    float coverageFraction = (rectSizePx.x * rectSizePx.y) / screenArea;
    if (coverageFraction >= params.safetyInfo.y)
    {
        return true;
    }

    float maxDim = max(rectSizePx.x, rectSizePx.y);
    uint mipLevel = 0;
    if (maxDim > 1.0)
    {
        mipLevel = min((uint)floor(log2(maxDim)), mipCount - 1u);
    }

    float depthBias = max(0.0, params.depthPyramidInfo.w) + max(0.0, params.safetyInfo.z) * float(mipLevel);

    [unroll]
    for (uint y = 0; y < 3; ++y)
    {
        [unroll]
        for (uint x = 0; x < 3; ++x)
        {
            float2 t = float2(float(x) * 0.5, float(y) * 0.5);
            float2 samplePx = lerp(rectMinPx, rectMaxPx, t);
            float sampleDepth = sampleDepthPyramid(mipLevel, baseSize, samplePx);
            if (nearestDepth <= sampleDepth + depthBias)
            {
                return true;
            }
        }
    }

    return false;
}

[numthreads(64,1,1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint frustumCountBufferSize = 0;
    uint frustumCountBufferStride = 0;
    frustumVisibleCountRO.GetDimensions(frustumCountBufferSize, frustumCountBufferStride);
    if (frustumCountBufferSize == 0u)
    {
        return;
    }

    uint frustumIndexBufferSize = 0;
    uint frustumIndexBufferStride = 0;
    frustumVisibleIndicesRO.GetDimensions(frustumIndexBufferSize, frustumIndexBufferStride);
    if (frustumIndexBufferSize == 0u)
    {
        return;
    }

    uint aabbBufferSize = 0;
    uint aabbBufferStride = 0;
    instanceAabbsRO.GetDimensions(aabbBufferSize, aabbBufferStride);
    if (aabbBufferSize == 0u)
    {
        return;
    }

    uint outputCountBufferSize = 0;
    uint outputCountBufferStride = 0;
    occlusionVisibleCountRW.GetDimensions(outputCountBufferSize, outputCountBufferStride);
    if (outputCountBufferSize == 0u)
    {
        return;
    }

    uint outputIndexBufferSize = 0;
    uint outputIndexBufferStride = 0;
    occlusionVisibleIndicesRW.GetDimensions(outputIndexBufferSize, outputIndexBufferStride);
    if (outputIndexBufferSize == 0u)
    {
        return;
    }

    uint inputOffset = dispatchThreadID.x;
    uint visibleCount = min(frustumVisibleCountRO[0], frustumIndexBufferSize);
    if (inputOffset >= visibleCount)
    {
        return;
    }

    uint instanceIndex = frustumVisibleIndicesRO[inputOffset];
    if (instanceIndex >= aabbBufferSize)
    {
        return;
    }

    FrustumCullAABB aabb = instanceAabbsRO[instanceIndex];

    if (!isVisibleConservative(aabb))
    {
        return;
    }

    uint outputOffset = 0u;
    while (true)
    {
        uint currentCount = occlusionVisibleCountRW[0];
        if (currentCount >= outputIndexBufferSize)
        {
            return;
        }

        uint originalCount = 0u;
        InterlockedCompareExchange(occlusionVisibleCountRW[0], currentCount, currentCount + 1u, originalCount);
        if (originalCount == currentCount)
        {
            outputOffset = currentCount;
            break;
        }
    }

    occlusionVisibleIndicesRW[outputOffset] = instanceIndex;
}
