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

struct DepthPyramidPushConstants
{
    uint2 srcSize;
    uint2 dstSize;
};

[[vk::binding(0, 0)]] Texture2D<float> srcDepth;
[[vk::binding(1, 0)]] RWTexture2D<float> dstDepth;
[[vk::push_constant]] DepthPyramidPushConstants pushConstants;

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 dstCoord = dispatchThreadID.xy;
    if (dstCoord.x >= pushConstants.dstSize.x || dstCoord.y >= pushConstants.dstSize.y)
    {
        return;
    }

    float minDepth = 1.0;
    uint2 srcBase = dstCoord * 2;

    [unroll]
    for (uint y = 0; y < 2; ++y)
    {
        [unroll]
        for (uint x = 0; x < 2; ++x)
        {
            uint2 srcCoord = srcBase + uint2(x, y);
            if (srcCoord.x < pushConstants.srcSize.x && srcCoord.y < pushConstants.srcSize.y)
            {
                float depth = srcDepth.Load(int3(srcCoord, 0));
                minDepth = min(minDepth, depth);
            }
        }
    }

    dstDepth[int2(dstCoord)] = minDepth;
}
