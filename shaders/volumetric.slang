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

[[vk::binding(0, 0)]] Sampler2D depthPyramid;
[[vk::binding(1, 0)]] StructuredBuffer<LightData> lightBuffer;
[[vk::binding(2, 0)]] RWTexture2D<float4> volumetricOutput;

struct VolumetricPush
{
    float2 invResolution;
    float  scatteringStrength;
    float  extinction;
    float  anisotropy;
    float  heightFogDensity;
    float  maxDistance;
    float  maxIntensity;
    uint   lightCount;
    uint   stepCount;
};

[[vk::push_constant]] VolumetricPush pushConsts;

float schlickPhase(float cosTheta, float k)
{
    float denom = 1.0 - k * cosTheta;
    return (1.0 - k * k) / max(12.5663706 * denom * denom, 1e-4);
}

[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint width, height;
    volumetricOutput.GetDimensions(width, height);
    if (dispatchThreadID.x >= width || dispatchThreadID.y >= height)
    {
        return;
    }

    float2 uv = (float2(dispatchThreadID.xy) + 0.5) * pushConsts.invResolution;
    float depth = depthPyramid.SampleLevel(uv, 0.0).r;

    float3 scattering = float3(0.0, 0.0, 0.0);
    if (depth < 0.9999)
    {
        const uint maxSteps = max(pushConsts.stepCount, 1u);
        const float viewDistance = min(pushConsts.maxDistance, depth * pushConsts.maxDistance);
        const float stepLength = viewDistance / float(maxSteps);

        uint maxLightCount = min(pushConsts.lightCount, 64u);

        [loop]
        for (uint lightIndex = 0; lightIndex < maxLightCount; ++lightIndex)
        {
            LightData light = lightBuffer[lightIndex];
            if (light.lightType != 1)
            {
                continue;
            }

            float3 lightDir = normalize(-light.direction.xyz);
            float3 lightColor = max(light.color.rgb, 0.0);

            float accum = 0.0;
            [loop]
            for (uint step = 0; step < maxSteps; ++step)
            {
                float t = (float(step) + 0.5) / float(maxSteps);
                float sampleDistance = t * viewDistance;
                float opticalDepth = sampleDistance * pushConsts.extinction;
                float transmittance = exp(-opticalDepth);
                float heightFog = exp(-sampleDistance * pushConsts.heightFogDensity);
                float phase = schlickPhase(dot(float3(0.0, 0.0, -1.0), lightDir), saturate(pushConsts.anisotropy));
                accum += transmittance * heightFog * phase * stepLength;
            }

            scattering += lightColor * accum * pushConsts.scatteringStrength;
            break;
        }
    }

    scattering = min(scattering, pushConsts.maxIntensity.xxx);
    volumetricOutput[dispatchThreadID.xy] = float4(scattering, 1.0);
}
