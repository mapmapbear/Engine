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
// Shadow map sampling utilities with reusable compare and PCF kernels
// Shared between rasterization and ray query shaders

import common_types;

float shadowDepthCompare(Sampler2D shadowMap, float2 uv, float receiverDepth, float bias) {
    if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) {
        return 1.0;
    }
    if (receiverDepth <= 0.0 || receiverDepth >= 1.0) {
        return 1.0;
    }

    float sampledDepth = shadowMap.Sample(uv).r;
    return ((receiverDepth - bias) <= sampledDepth) ? 1.0 : 0.0;
}

float shadowMapPCFKernel1(Sampler2D shadowMap, float2 uv, float receiverDepth, float bias, float2 texelSize) {
    return shadowDepthCompare(shadowMap, uv, receiverDepth, bias);
}

float shadowMapPCFKernel2(Sampler2D shadowMap, float2 uv, float receiverDepth, float bias, float2 texelSize) {
    float2 baseUv = uv - texelSize * 0.5;
    float visibility = 0.0;

    [unroll]
    for (int y = 0; y < 2; ++y) {
        [unroll]
        for (int x = 0; x < 2; ++x) {
            float2 offset = float2(float(x), float(y)) * texelSize;
            visibility += shadowDepthCompare(shadowMap, baseUv + offset, receiverDepth, bias);
        }
    }

    return visibility * 0.25;
}

float shadowMapPCFKernel3(Sampler2D shadowMap, float2 uv, float receiverDepth, float bias, float2 texelSize) {
    float visibility = 0.0;

    [unroll]
    for (int y = -1; y <= 1; ++y) {
        [unroll]
        for (int x = -1; x <= 1; ++x) {
            float2 offset = float2(float(x), float(y)) * texelSize;
            visibility += shadowDepthCompare(shadowMap, uv + offset, receiverDepth, bias);
        }
    }

    return visibility * (1.0 / 9.0);
}

float shadowMapPCFKernel5(Sampler2D shadowMap, float2 uv, float receiverDepth, float bias, float2 texelSize) {
    float visibility = 0.0;

    [unroll]
    for (int y = -2; y <= 2; ++y) {
        [unroll]
        for (int x = -2; x <= 2; ++x) {
            float2 offset = float2(float(x), float(y)) * texelSize;
            visibility += shadowDepthCompare(shadowMap, uv + offset, receiverDepth, bias);
        }
    }

    return visibility * (1.0 / 25.0);
}

float sampleShadowMapPCF(Sampler2D shadowMap, float2 uv, float receiverDepth, float bias, float2 texelSize, uint kernelSize) {
    if (kernelSize <= 1u) {
        return shadowMapPCFKernel1(shadowMap, uv, receiverDepth, bias, texelSize);
    }
    if (kernelSize == 2u) {
        return shadowMapPCFKernel2(shadowMap, uv, receiverDepth, bias, texelSize);
    }
    if (kernelSize == 3u) {
        return shadowMapPCFKernel3(shadowMap, uv, receiverDepth, bias, texelSize);
    }
    return shadowMapPCFKernel5(shadowMap, uv, receiverDepth, bias, texelSize);
}
