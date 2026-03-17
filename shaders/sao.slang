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

// Conservative SAO groundwork pass.
// Input: depth pyramid mip chain generated from the main depth buffer.
// Output: full-resolution AO texture for later post-process integration.

[[vk::binding(0, 0)]] Sampler2D depthPyramid;
[[vk::binding(1, 0)]] RWTexture2D<float4> aoOutput;

struct SAOPush
{
    float2 invResolution;
    float  radiusPixels;
    float  intensity;
    float  bias;
    float  contrast;
    float  mipBias;
    float2 _pad;
};

[[vk::push_constant]] SAOPush pushConsts;

[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint width, height;
    aoOutput.GetDimensions(width, height);
    if (dispatchThreadID.x >= width || dispatchThreadID.y >= height)
    {
        return;
    }

    float2 uv = (float2(dispatchThreadID.xy) + 0.5) * pushConsts.invResolution;
    float centerDepth = depthPyramid.SampleLevel(uv, 0.0).r;

    if (centerDepth >= 0.9999)
    {
        aoOutput[dispatchThreadID.xy] = float4(1.0, 1.0, 1.0, 1.0);
        return;
    }

    static const float2 kRing[8] = {
        float2( 1.0,  0.0),
        float2(-1.0,  0.0),
        float2( 0.0,  1.0),
        float2( 0.0, -1.0),
        float2( 0.7071,  0.7071),
        float2(-0.7071,  0.7071),
        float2( 0.7071, -0.7071),
        float2(-0.7071, -0.7071)
    };

    float occlusion = 0.0;
    const float radius = max(pushConsts.radiusPixels, 0.5);
    const float lod = max(pushConsts.mipBias, 0.0);

    [unroll]
    for (uint i = 0; i < 8; ++i)
    {
        float2 sampleUv = uv + kRing[i] * radius * pushConsts.invResolution;
        float sampleDepth = depthPyramid.SampleLevel(sampleUv, lod).r;
        float delta = centerDepth - sampleDepth;
        occlusion += saturate((delta - pushConsts.bias) * pushConsts.contrast);
    }

    float ao = 1.0 - (occlusion / 8.0) * saturate(pushConsts.intensity);
    ao = saturate(ao);
    aoOutput[dispatchThreadID.xy] = float4(ao, ao, ao, 1.0);
}
