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

// Instance frustum culling against six clip planes.

struct FrustumCullAABB {
    float4 minBounds;
    float4 maxBounds;
};

struct FrustumCullParams {
    uint instanceCount;
    uint planeCount;
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]] StructuredBuffer<FrustumCullAABB> instanceAabbsRO;
[[vk::binding(1, 0)]] StructuredBuffer<float4> frustumPlanesRO;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> visibleIndicesRW;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> visibleCountRW;
[[vk::binding(4, 0)]] ConstantBuffer<FrustumCullParams> params;

bool aabbIntersectsFrustum(float3 worldMin, float3 worldMax) {
    if (params.planeCount < 6) {
        return false;
    }

    [unroll]
    for (uint i = 0; i < 6; ++i) {
        float4 p = frustumPlanesRO[i];
        float3 n = p.xyz;
        float3 v = float3(
            (n.x >= 0.0) ? worldMax.x : worldMin.x,
            (n.y >= 0.0) ? worldMax.y : worldMin.y,
            (n.z >= 0.0) ? worldMax.z : worldMin.z);

        if (dot(n, v) + p.w < 0.0) {
            return false;
        }
    }

    return true;
}

[numthreads(64,1,1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint instanceIndex = dispatchThreadID.x;
    if (instanceIndex >= params.instanceCount) {
        return;
    }

    FrustumCullAABB aabb = instanceAabbsRO[instanceIndex];
    if (!aabbIntersectsFrustum(aabb.minBounds.xyz, aabb.maxBounds.xyz)) {
        return;
    }

    uint visibleOffset = 0;
    InterlockedAdd(visibleCountRW[0], 1, visibleOffset);
    visibleIndicesRW[visibleOffset] = instanceIndex;
}
