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
#include "imgui/imgui.h"
#include "imgui_system.h"
#include "mesh_component.h"
#include "model_loader.h"
#include "renderer.h"
#include "transform_component.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glm/gtx/norm.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <ranges>
#include <sstream>
#include <stdexcept>

namespace
{
	float halton(uint32_t index, uint32_t base)
	{
		float value = 0.0f;
		float invBase = 1.0f / static_cast<float>(base);
		float fraction = invBase;
		while (index > 0u)
		{
			value += static_cast<float>(index % base) * fraction;
			index /= base;
			fraction *= invBase;
		}
		return value;
	}

	constexpr uint32_t kGpuProfileQueriesPerPass = 2u;

	constexpr uint32_t gpuProfileQueryIndex(uint32_t passIndex, bool beginMarker)
	{
		const uint32_t base = passIndex * kGpuProfileQueriesPerPass;
		return beginMarker ? base : (base + 1u);
	}

	constexpr std::array<const char*, 12> kGpuProfilePassNames = {
		"Frustum Cull",
		"Ray Query",
		"Depth Prepass",
		"Depth Pyramid",
		"SAO",
		"Volumetric",
		"Forward+ Cull",
		"Opaque",
		"TAA History",
		"Composite",
		"Transparent",
		"ImGui"
	};

	const char* gpuProfilePassName(uint32_t passIndex)
	{
		return passIndex < kGpuProfilePassNames.size() ? kGpuProfilePassNames[passIndex] : "Unknown";
	}
}

// ===================== Culling helpers implementation =====================

Renderer::FrustumPlanes Renderer::extractFrustumPlanes(const glm::mat4& vp)
{
	// Work in row-major form for standard plane extraction by transposing GLM's column-major matrix
	glm::mat4 m = glm::transpose(vp);
	FrustumPlanes fp{};
	// Left   : m[3] + m[0]
	fp.planes[0] = m[3] + m[0];
	// Right  : m[3] - m[0]
	fp.planes[1] = m[3] - m[0];
	// Bottom : m[3] + m[1]
	fp.planes[2] = m[3] + m[1];
	// Top    : m[3] - m[1]
	fp.planes[3] = m[3] - m[1];
	// Near   : m[2] (matches Vulkan [0, 1] clip range)
	fp.planes[4] = m[2];
	// Far    : m[3] - m[2]
	fp.planes[5] = m[3] - m[2];

	// Normalize planes
	for (auto& p : fp.planes)
	{
		glm::vec3 n(p.x, p.y, p.z);
		float len = glm::length(n);
		if (len > 0.0f)
		{
			p /= len;
		}
	}
	return fp;
}

void Renderer::transformAABB(const glm::mat4& M,
                             const glm::vec3& localMin,
                             const glm::vec3& localMax,
                             glm::vec3& outMin,
                             glm::vec3& outMax)
{
	// OBB (from model) to world AABB using center/extents and absolute 3x3
	const glm::vec3 c = 0.5f * (localMin + localMax);
	const glm::vec3 e = 0.5f * (localMax - localMin);

	const glm::vec3 worldCenter = glm::vec3(M * glm::vec4(c, 1.0f));
	// Upper-left 3x3
	const glm::mat3 A = glm::mat3(M);
	const glm::mat3 AbsA = glm::mat3(glm::abs(A[0]), glm::abs(A[1]), glm::abs(A[2]));
	const glm::vec3 worldExtents = AbsA * e; // component-wise combination

	outMin = worldCenter - worldExtents;
	outMax = worldCenter + worldExtents;
}

bool Renderer::aabbIntersectsFrustum(const glm::vec3& worldMin,
                                     const glm::vec3& worldMax,
                                     const FrustumPlanes& frustum)
{
	// Use the p-vertex test against each plane; if outside any plane → culled
	for (const auto& p : frustum.planes)
	{
		const glm::vec3 n(p.x, p.y, p.z);
		// Choose positive vertex (furthest in direction of normal)
		glm::vec3 v{
			n.x >= 0.0f ? worldMax.x : worldMin.x,
			n.y >= 0.0f ? worldMax.y : worldMin.y,
			n.z >= 0.0f ? worldMax.z : worldMin.z
		};

		// If the most positive vertex is still on the negative side of the plane,
		// then the entire box is on the negative side.
		// Use a small epsilon to avoid numerical issues.
		if (glm::dot(n, v) + p.w < -0.01f)
		{
			return false; // completely outside
		}
	}
	return true;
}

// This file contains rendering-related methods from the Renderer class

// Create swap chain
bool Renderer::createSwapChain()
{
	try
	{
		// Query swap chain support
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

		// Choose swap surface format, present mode, and extent
		vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		// Choose image count
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
		{
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		// Create swap chain info
		vk::SwapchainCreateInfoKHR createInfo{
			.surface = *surface,
			.minImageCount = imageCount,
			.imageFormat = surfaceFormat.format,
			.imageColorSpace = surfaceFormat.colorSpace,
			.imageExtent = extent,
			.imageArrayLayers = 1,
			.imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,
			.preTransform = swapChainSupport.capabilities.currentTransform,
			.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
			.presentMode = presentMode,
			.clipped = VK_TRUE,
			.oldSwapchain = nullptr
		};

		// Find queue families
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		std::array<uint32_t, 2> queueFamilyIndicesLoc = {indices.graphicsFamily.value(), indices.presentFamily.value()};

		// Set sharing mode
		if (indices.graphicsFamily != indices.presentFamily)
		{
			createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
			createInfo.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndicesLoc.size());
			createInfo.pQueueFamilyIndices = queueFamilyIndicesLoc.data();
		}
		else
		{
			createInfo.imageSharingMode = vk::SharingMode::eExclusive;
			createInfo.queueFamilyIndexCount = 0;
			createInfo.pQueueFamilyIndices = nullptr;
		}

		// Create swap chain
		swapChain = vk::raii::SwapchainKHR(device, createInfo);

		// Get swap chain images
		swapChainImages = swapChain.getImages();

		// Swapchain images start in UNDEFINED layout; track per-image layout for correct barriers.
		swapChainImageLayouts.assign(swapChainImages.size(), vk::ImageLayout::eUndefined);

		// Store swap chain format and extent
		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;

		return true;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to create swap chain: " << e.what() << std::endl;
		return false;
	}
}

// ===================== Planar reflections resources =====================
bool Renderer::createReflectionResources(uint32_t width, uint32_t height)
{
	try
	{
		destroyReflectionResources();
		reflections.clear();
		reflections.resize(MAX_FRAMES_IN_FLIGHT);
		reflectionVPs.clear();
		reflectionVPs.resize(MAX_FRAMES_IN_FLIGHT, glm::mat4(1.0f));
		sampleReflectionVP = glm::mat4(1.0f);

		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			auto& rt = reflections[i];
			rt.width = width;
			rt.height = height;

			// Color RT: use swapchain format to match existing PBR pipeline rendering formats
			vk::Format colorFmt = swapChainImageFormat;
			auto [colorImg, colorAlloc] = createImagePooled(
				width,
				height,
				colorFmt,
				vk::ImageTiling::eOptimal,
				// Allow sampling in glass and blitting to swapchain for diagnostics
				vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
				vk::ImageUsageFlagBits::eTransferSrc,
				vk::MemoryPropertyFlagBits::eDeviceLocal,
				/*mipLevels*/
				1,
				vk::SharingMode::eExclusive,
				{});
			rt.color = std::move(colorImg);
			rt.colorAlloc = std::move(colorAlloc);
			rt.colorView = createImageView(rt.color, colorFmt, vk::ImageAspectFlagBits::eColor, 1);
			// Simple sampler for sampling reflection texture (no mips)
			vk::SamplerCreateInfo sampInfo{
				.magFilter = vk::Filter::eLinear, .minFilter = vk::Filter::eLinear,
				.mipmapMode = vk::SamplerMipmapMode::eNearest, .addressModeU = vk::SamplerAddressMode::eClampToEdge,
				.addressModeV = vk::SamplerAddressMode::eClampToEdge,
				.addressModeW = vk::SamplerAddressMode::eClampToEdge, .minLod = 0.0f, .maxLod = 0.0f
			};
			rt.colorSampler = vk::raii::Sampler(device, sampInfo);

			// Depth RT
			vk::Format depthFmt = findDepthFormat();
			auto [depthImg, depthAlloc] = createImagePooled(
				width,
				height,
				depthFmt,
				vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eDepthStencilAttachment,
				vk::MemoryPropertyFlagBits::eDeviceLocal,
				/*mipLevels*/
				1,
				vk::SharingMode::eExclusive,
				{});
			rt.depth = std::move(depthImg);
			rt.depthAlloc = std::move(depthAlloc);
			rt.depthView = createImageView(rt.depth, depthFmt, vk::ImageAspectFlagBits::eDepth, 1);
		}

		// One-time initialization: transition all per-frame reflection color images
		// from UNDEFINED to SHADER_READ_ONLY_OPTIMAL so that the first frame can
		// legally sample the "previous" frame's image.
		if (!reflections.empty())
		{
			vk::CommandPoolCreateInfo poolInfo{
				.flags = vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
				.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()
			};
			vk::raii::CommandPool tempPool(device, poolInfo);
			vk::CommandBufferAllocateInfo allocInfo{
				.commandPool = *tempPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1
			};
			vk::raii::CommandBuffers cbs(device, allocInfo);
			vk::raii::CommandBuffer& cb = cbs[0];
			cb.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

			std::vector<vk::ImageMemoryBarrier2> barriers;
			barriers.reserve(reflections.size());
			for (auto& rt : reflections)
			{
				if (!!*rt.color)
				{
					barriers.push_back(vk::ImageMemoryBarrier2{
						.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
						.srcAccessMask = vk::AccessFlagBits2::eNone,
						.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
						.dstAccessMask = vk::AccessFlagBits2::eShaderRead,
						.oldLayout = vk::ImageLayout::eUndefined,
						.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
						.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.image = *rt.color,
						.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
					});
				}
			}
			if (!barriers.empty())
			{
				vk::DependencyInfo depInfo{
					.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
					.pImageMemoryBarriers = barriers.data()
				};
				cb.pipelineBarrier2(depInfo);
			}
			cb.end();
			vk::SubmitInfo submit{.commandBufferCount = 1, .pCommandBuffers = &*cb};
			vk::raii::Fence fence(device, vk::FenceCreateInfo{});
			{
				std::lock_guard<std::mutex> lock(queueMutex);
				graphicsQueue.submit(submit, *fence);
			}
			vk::Result result = waitForFencesSafe(*fence, VK_TRUE);
			if (result != vk::Result::eSuccess)
			{
				std::cerr << "Error: Failed to wait for reflection resource fence: " << vk::to_string(result) <<
					std::endl;
			}
		}

		return true;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to create reflection resources: " << e.what() << std::endl;
		destroyReflectionResources();
		return false;
	}
}

void Renderer::destroyReflectionResources()
{
	for (auto& rt : reflections)
	{
		rt.colorSampler = vk::raii::Sampler(nullptr);
		rt.colorView = vk::raii::ImageView(nullptr);
		rt.colorAlloc = nullptr;
		rt.color = vk::raii::Image(nullptr);
		rt.depthView = vk::raii::ImageView(nullptr);
		rt.depthAlloc = nullptr;
		rt.depth = vk::raii::Image(nullptr);
		rt.width = rt.height = 0;
	}
}

bool Renderer::createCascadedShadowResources()
{
	if (!enableCascadedShadowMaps)
	{
		csmResourcesDirty = false;
		return true;
	}

	destroyCascadedShadowResources();

	try
	{
		const uint32_t extent = std::max(512u, csmShadowMapResolution);
		const vk::Format shadowFormat = findSupportedFormat(
			{vk::Format::eD32Sfloat, vk::Format::eD24UnormS8Uint, vk::Format::eD16Unorm},
			vk::ImageTiling::eOptimal,
			vk::FormatFeatureFlagBits::eDepthStencilAttachment);

		for (uint32_t cascadeIndex = 0; cascadeIndex < CSM_CASCADE_COUNT; ++cascadeIndex)
		{
			auto [shadowImage, shadowAlloc] = createImagePooled(
				extent,
				extent,
				shadowFormat,
				vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
				vk::MemoryPropertyFlagBits::eDeviceLocal,
				1,
				vk::SharingMode::eExclusive,
				{});

			csmShadowMaps[cascadeIndex].image = std::move(shadowImage);
			csmShadowMaps[cascadeIndex].allocation = std::move(shadowAlloc);
			csmShadowMaps[cascadeIndex].view = createImageView(
				csmShadowMaps[cascadeIndex].image,
				shadowFormat,
				vk::ImageAspectFlagBits::eDepth,
				1);

			vk::SamplerCreateInfo samplerInfo{
				.magFilter = vk::Filter::eLinear,
				.minFilter = vk::Filter::eLinear,
				.mipmapMode = vk::SamplerMipmapMode::eNearest,
				.addressModeU = vk::SamplerAddressMode::eClampToBorder,
				.addressModeV = vk::SamplerAddressMode::eClampToBorder,
				.addressModeW = vk::SamplerAddressMode::eClampToBorder,
				.mipLodBias = 0.0f,
				.anisotropyEnable = VK_FALSE,
				.maxAnisotropy = 1.0f,
				.compareEnable = VK_FALSE,
				.compareOp = vk::CompareOp::eAlways,
				.minLod = 0.0f,
				.maxLod = 0.0f,
				.borderColor = vk::BorderColor::eFloatOpaqueWhite,
				.unnormalizedCoordinates = VK_FALSE
			};
			csmShadowMaps[cascadeIndex].sampler = vk::raii::Sampler(device, samplerInfo);

			transitionImageLayout(
				*csmShadowMaps[cascadeIndex].image,
				shadowFormat,
				vk::ImageLayout::eUndefined,
				vk::ImageLayout::eShaderReadOnlyOptimal,
				1);
			csmShadowMaps[cascadeIndex].currentLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		}

		csmResourceExtentWidth = extent;
		csmResourceExtentHeight = extent;
		csmResourcesDirty = false;
		return true;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to create cascaded shadow map resources: " << e.what() << std::endl;
		destroyCascadedShadowResources();
		return false;
	}
}

void Renderer::destroyCascadedShadowResources()
{
	for (auto& cascadeMap : csmShadowMaps)
	{
		cascadeMap.sampler = vk::raii::Sampler(nullptr);
		cascadeMap.view = vk::raii::ImageView(nullptr);
		cascadeMap.allocation.reset();
		cascadeMap.image = vk::raii::Image(nullptr);
		cascadeMap.currentLayout = vk::ImageLayout::eUndefined;
	}
	csmResourceExtentWidth = 0;
	csmResourceExtentHeight = 0;
	csmResourcesDirty = true;
}

void Renderer::updateCascadedShadowData(const std::vector<ExtractedLight>& lights, CameraComponent* camera)
{
	csmDataValid = false;

	if (!enableCascadedShadowMaps || !camera)
	{
		return;
	}

	if (csmResourcesDirty || csmResourceExtentWidth == 0 || csmResourceExtentHeight == 0)
	{
		if (!createCascadedShadowResources())
		{
			return;
		}
	}

	glm::vec3 lightDirection(-0.6f, -1.0f, -0.3f);
	glm::vec3 lightPosition = camera->GetPosition() - glm::normalize(lightDirection) * 100.0f;
	bool hasDirectionalLight = false;
	for (const auto& light : lights)
	{
		if (light.type == ExtractedLight::Type::Directional)
		{
			lightDirection = light.direction;
			lightPosition = light.position;
			hasDirectionalLight = true;
			break;
		}
	}

	const glm::vec3 lightDir = glm::normalize(glm::length2(lightDirection) > 1e-6f
		                                          ? lightDirection
		                                          : glm::vec3(-0.6f, -1.0f, -0.3f));
	const float nearPlane = std::max(0.01f, camera->GetNearPlane());
	const float farPlane = std::max(nearPlane + 1.0f, camera->GetFarPlane());
	const float clipRange = farPlane - nearPlane;
	const float lambda = std::clamp(csmSplitLambda, 0.0f, 1.0f);

	std::array<float, CSM_CASCADE_COUNT + 1> cascadeRanges{};
	cascadeRanges[0] = nearPlane;
	for (uint32_t cascadeIndex = 1; cascadeIndex <= CSM_CASCADE_COUNT; ++cascadeIndex)
	{
		const float p = static_cast<float>(cascadeIndex) / static_cast<float>(CSM_CASCADE_COUNT);
		const float logarithmic = nearPlane * std::pow(farPlane / nearPlane, p);
		const float uniform = nearPlane + clipRange * p;
		cascadeRanges[cascadeIndex] = lambda * logarithmic + (1.0f - lambda) * uniform;
	}
	cascadeRanges[CSM_CASCADE_COUNT] = farPlane;

	const glm::mat4 cameraView = camera->GetViewMatrix();
	const glm::mat4 cameraProj = camera->GetProjectionMatrix();
	const glm::mat4 invViewProj = glm::inverse(cameraProj * cameraView);

	const std::array<glm::vec3, 8> clipCorners = {
		glm::vec3(-1.0f, 1.0f, -1.0f),
		glm::vec3(1.0f, 1.0f, -1.0f),
		glm::vec3(1.0f, -1.0f, -1.0f),
		glm::vec3(-1.0f, -1.0f, -1.0f),
		glm::vec3(-1.0f, 1.0f, 1.0f),
		glm::vec3(1.0f, 1.0f, 1.0f),
		glm::vec3(1.0f, -1.0f, 1.0f),
		glm::vec3(-1.0f, -1.0f, 1.0f)
	};

	std::array<glm::vec3, 8> frustumCornersWorld{};
	for (size_t i = 0; i < clipCorners.size(); ++i)
	{
		const glm::vec4 corner = invViewProj * glm::vec4(clipCorners[i], 1.0f);
		frustumCornersWorld[i] = glm::vec3(corner) / corner.w;
	}

	for (uint32_t cascadeIndex = 0; cascadeIndex < CSM_CASCADE_COUNT; ++cascadeIndex)
	{
		const float cascadeNear = cascadeRanges[cascadeIndex];
		const float cascadeFar = cascadeRanges[cascadeIndex + 1];
		const float nearRatio = (cascadeNear - nearPlane) / clipRange;
		const float farRatio = (cascadeFar - nearPlane) / clipRange;

		std::array<glm::vec3, 8> cascadeCorners{};
		for (uint32_t cornerIndex = 0; cornerIndex < 4; ++cornerIndex)
		{
			const glm::vec3 cornerNear = frustumCornersWorld[cornerIndex];
			const glm::vec3 cornerFar = frustumCornersWorld[cornerIndex + 4];
			const glm::vec3 cornerRay = cornerFar - cornerNear;
			cascadeCorners[cornerIndex] = cornerNear + cornerRay * nearRatio;
			cascadeCorners[cornerIndex + 4] = cornerNear + cornerRay * farRatio;
		}

		glm::vec3 frustumCenter(0.0f);
		for (const glm::vec3& corner : cascadeCorners)
		{
			frustumCenter += corner;
		}
		frustumCenter /= static_cast<float>(cascadeCorners.size());

		float radius = 0.0f;
		for (const glm::vec3& corner : cascadeCorners)
		{
			radius = std::max(radius, glm::distance(corner, frustumCenter));
		}
		radius = std::max(radius, 1.0f);
		radius = std::ceil(radius * 16.0f) / 16.0f;

		const glm::vec3 up = (std::abs(lightDir.y) > 0.99f) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
		const glm::vec3 lightAnchor = hasDirectionalLight ? lightPosition : frustumCenter;
		glm::mat4 lightView = glm::lookAt(lightAnchor - lightDir * (radius * 2.0f), frustumCenter, up);

		glm::vec3 mins(std::numeric_limits<float>::max());
		glm::vec3 maxs(std::numeric_limits<float>::lowest());
		for (const glm::vec3& corner : cascadeCorners)
		{
			const glm::vec3 cornerLightSpace = glm::vec3(lightView * glm::vec4(corner, 1.0f));
			mins = glm::min(mins, cornerLightSpace);
			maxs = glm::max(maxs, cornerLightSpace);
		}

		const float zPad = std::max(100.0f, radius * 2.0f);
		glm::mat4 lightProj = glm::ortho(mins.x, maxs.x, mins.y, maxs.y, mins.z - zPad, maxs.z + zPad);
		csmCascadeLightSpaceMatrices[cascadeIndex] = lightProj * lightView;
		csmCascadeSplitDepths[cascadeIndex] = cascadeFar;
	}

	const float blendFraction = std::clamp(csmTransitionBlendFraction, 0.01f, 0.49f);
	csmTransitionData = glm::vec4(blendFraction, 1.0f / blendFraction, nearPlane, farPlane);
	csmDataValid = true;
}

void Renderer::renderReflectionPass(vk::raii::CommandBuffer& cmd,
                                    const glm::vec4& planeWS,
                                    CameraComponent* camera,
                                    const std::vector<RenderJob>& jobs)
{
	if (reflections.empty())
		return;
	auto& rt = reflections[currentFrame];
	if (rt.width == 0 || rt.height == 0 || !*rt.colorView || !*rt.depthView)
		return;

	// Transition reflection color to COLOR_ATTACHMENT_OPTIMAL (Sync2)
	vk::ImageMemoryBarrier2 toColor2{
		.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
		.srcAccessMask = {},
		.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
		.oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		.newLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = *rt.color,
		.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
	};
	// Transition reflection depth to DEPTH_STENCIL_ATTACHMENT_OPTIMAL (Sync2)
	vk::ImageMemoryBarrier2 toDepth2{
		.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
		.srcAccessMask = {},
		.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests,
		.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
		vk::AccessFlagBits2::eDepthStencilAttachmentRead,
		.oldLayout = vk::ImageLayout::eUndefined,
		.newLayout = vk::ImageLayout::eDepthAttachmentOptimal,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = *rt.depth,
		.subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}
	};
	std::array<vk::ImageMemoryBarrier2, 2> preBarriers{toColor2, toDepth2};
	vk::DependencyInfo depInfoToColor{
		.imageMemoryBarrierCount = static_cast<uint32_t>(preBarriers.size()), .pImageMemoryBarriers = preBarriers.data()
	};
	cmd.pipelineBarrier2(depInfoToColor);

	vk::RenderingAttachmentInfo colorAtt{
		.imageView = *rt.colorView,
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		// Clear to black so scene content dominates reflections
		.clearValue = vk::ClearValue{vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}}}
	};
	vk::RenderingAttachmentInfo depthAtt{
		.imageView = *rt.depthView,
		.imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eDontCare,
		.clearValue = vk::ClearValue{vk::ClearDepthStencilValue{1.0f, 0}}
	};
	vk::RenderingInfo rinfo{
		.renderArea = vk::Rect2D({0, 0}, {rt.width, rt.height}),
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &colorAtt,
		.pDepthAttachment = &depthAtt
	};
	cmd.beginRendering(rinfo);
	// Compute mirrored view matrix about planeWS (default Y=0 plane)
	glm::mat4 reflectM(1.0f);
	// For Y=0 plane, reflection is simply flip Y
	if (glm::length(glm::vec3(planeWS.x, planeWS.y, planeWS.z)) > 0.5f && fabsf(planeWS.y - 1.0f) < 1e-3f &&
		fabsf(planeWS.x) < 1e-3f && fabsf(planeWS.z) < 1e-3f)
	{
		reflectM[1][1] = -1.0f;
	}
	else
	{
		// General plane reflection matrix R = I - 2*n*n^T for normalized plane; ignore translation for now
		glm::vec3 n = glm::normalize(glm::vec3(planeWS));
		glm::mat3 R = glm::mat3(1.0f) - 2.0f * glm::outerProduct(n, n);
		reflectM = glm::mat4(R);
	}

	glm::mat4 viewReflected = camera ? (camera->GetViewMatrix() * reflectM) : reflectM;
	glm::mat4 projReflected = camera ? camera->GetProjectionMatrix() : glm::mat4(1.0f);
	currentReflectionVP = projReflected * viewReflected;
	currentReflectionPlane = planeWS;
	if (currentFrame < reflectionVPs.size())
	{
		reflectionVPs[currentFrame] = currentReflectionVP;
	}

	// Set viewport/scissor to reflection RT size
	vk::Viewport rv(0.0f, 0.0f, static_cast<float>(rt.width), static_cast<float>(rt.height), 0.0f, 1.0f);
	cmd.setViewport(0, rv);
	vk::Rect2D rs({0, 0}, {rt.width, rt.height});
	cmd.setScissor(0, rs);

	// Draw opaque entities with mirrored view
	// Use reflection-specific pipeline (cull none) to avoid mirrored winding issues.
	if (!!*pbrReflectionGraphicsPipeline)
	{
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pbrReflectionGraphicsPipeline);
	}
	else if (!!*pbrGraphicsPipeline)
	{
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pbrGraphicsPipeline);
	}

	// Prepare frustum for mirrored view to allow culling
	FrustumPlanes reflectFrustum = extractFrustumPlanes(currentReflectionVP);

	// Render all jobs (skip transparency)
	for (const auto& job : jobs)
	{
		Entity* entity = job.entity;
		MeshComponent* meshComponent = job.meshComp;
		EntityResources* entityRes = job.entityRes;
		MeshResources* meshRes = job.meshRes;

		if (entityRes->cachedIsBlended)
			continue;

		// Frustum culling for mirrored view
		if (meshComponent->HasLocalAABB())
		{
			const glm::mat4 model = job.transformComp ? job.transformComp->GetModelMatrix() : glm::mat4(1.0f);
			glm::vec3 wmin, wmax;
			transformAABB(model, meshComponent->GetLocalAABBMin(), meshComponent->GetLocalAABBMax(), wmin, wmax);
			if (!aabbIntersectsFrustum(wmin, wmax, reflectFrustum))
			{
				continue; // culled from reflection
			}
		}

		// Bind geometry
		std::array<vk::Buffer, 2> buffers = {*meshRes->vertexBuffer, *entityRes->instanceBuffer};
		std::array<vk::DeviceSize, 2> offsets = {0, 0};
		cmd.bindVertexBuffers(0, buffers, offsets);
		cmd.bindIndexBuffer(*meshRes->indexBuffer, 0, vk::IndexType::eUint32);

		// Populate UBO with mirrored view + clip plane and reflection flags
		UniformBufferObject ubo{};
		if (job.transformComp)
			ubo.model = job.transformComp->GetModelMatrix();
		else
			ubo.model = glm::mat4(1.0f);
		ubo.view = viewReflected;
		ubo.proj = projReflected;
		ubo.camPos = glm::vec4(camera ? camera->GetPosition() : glm::vec3(0), 1.0f);
		ubo.reflectionPass = 1;
		ubo.reflectionEnabled = 0;
		ubo.reflectionVP = currentReflectionVP;
		ubo.clipPlaneWS = planeWS;
		// Ray query shadows in reflection pass
		ubo.padding2 = enableRasterRayQueryShadows ? 1.0f : 0.0f;

		updateUniformBufferInternal(currentFrame, entity, entityRes, camera, ubo);

		// Bind descriptor set (PBR set 0)
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
		                       *pbrPipelineLayout,
		                       0,
		                       *entityRes->pbrDescriptorSets[currentFrame],
		                       nullptr);

		// Push material properties
		MaterialProperties mp = entityRes->cachedMaterialProps;
		// Transmission suppressed during reflection pass via UBO (reflectionPass=1)
		mp.transmissionFactor = 0.0f;
		pushMaterialProperties(*cmd, mp);

		// Issue draw
		uint32_t instanceCount = std::max(1u, static_cast<uint32_t>(meshComponent->GetInstanceCount()));
		cmd.drawIndexed(meshRes->indexCount, instanceCount, 0, 0, 0);
	}

	cmd.endRendering();

	// Transition reflection color to SHADER_READ_ONLY for sampling in main pass (Sync2)
	vk::ImageMemoryBarrier2 toSample2{
		.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
		.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
		.dstAccessMask = vk::AccessFlagBits2::eShaderRead,
		.oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = *rt.color,
		.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
	};
	vk::DependencyInfo depInfoToSample{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toSample2};
	cmd.pipelineBarrier2(depInfoToSample);
}

// Create image views
bool Renderer::createImageViews()
{
	try
	{
		opaqueSceneColorImages.clear();
		opaqueSceneColorImageAllocations.clear();
		opaqueSceneColorImageViews.clear();
		opaqueSceneColorImageLayouts.clear();
		opaqueSceneColorSampler.clear();
		// Resize image views vector
		swapChainImageViews.clear();
		swapChainImageViews.reserve(swapChainImages.size());

		// Create image view info template (image will be set per iteration)
		vk::ImageViewCreateInfo createInfo{
			.viewType = vk::ImageViewType::e2D,
			.format = swapChainImageFormat,
			.components = {
				.r = vk::ComponentSwizzle::eIdentity,
				.g = vk::ComponentSwizzle::eIdentity,
				.b = vk::ComponentSwizzle::eIdentity,
				.a = vk::ComponentSwizzle::eIdentity
			},
			.subresourceRange = {
				.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0,
				.layerCount = 1
			}
		};

		// Create image view for each swap chain image
		for (const auto& image : swapChainImages)
		{
			createInfo.image = image;
			swapChainImageViews.emplace_back(device, createInfo);
		}

		return true;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to create image views: " << e.what() << std::endl;
		return false;
	}
}

// Setup dynamic rendering
bool Renderer::setupDynamicRendering()
{
	try
	{
		// Create color attachment
		colorAttachments = {
			vk::RenderingAttachmentInfo{
				.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
				.loadOp = vk::AttachmentLoadOp::eClear,
				.storeOp = vk::AttachmentStoreOp::eStore,
				.clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})

			}
		};

		// Create depth attachment
		depthAttachment = vk::RenderingAttachmentInfo{
			.imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eStore,
			.clearValue = vk::ClearDepthStencilValue(1.0f, 0)
		};

		// Create rendering info
		renderingInfo = vk::RenderingInfo{
			.renderArea = vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent),
			.layerCount = 1,
			.colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
			.pColorAttachments = colorAttachments.data(),
			.pDepthAttachment = &depthAttachment
		};

		return true;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to setup dynamic rendering: " << e.what() << std::endl;
		return false;
	}
}

// Create command pool
bool Renderer::createCommandPool()
{
	try
	{
		// Find queue families
		QueueFamilyIndices queueFamilyIndicesLoc = findQueueFamilies(physicalDevice);

		// Create command pool info
		vk::CommandPoolCreateInfo poolInfo{
			.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			.queueFamilyIndex = queueFamilyIndicesLoc.graphicsFamily.value()
		};

		// Create command pool
		commandPool = vk::raii::CommandPool(device, poolInfo);

		return true;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to create command pool: " << e.what() << std::endl;
		return false;
	}
}

// Create command buffers
bool Renderer::createCommandBuffers()
{
	try
	{
		// Resize command buffers vector
		commandBuffers.clear();
		commandBuffers.reserve(MAX_FRAMES_IN_FLIGHT);

		// Create command buffer allocation info
		vk::CommandBufferAllocateInfo allocInfo{
			.commandPool = *commandPool,
			.level = vk::CommandBufferLevel::ePrimary,
			.commandBufferCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)
		};

		// Allocate command buffers
		commandBuffers = vk::raii::CommandBuffers(device, allocInfo);

		return true;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to create command buffers: " << e.what() << std::endl;
		return false;
	}
}

// Create sync objects
bool Renderer::createSyncObjects()
{
	try
	{
		// Resize semaphores and fences vectors
		imageAvailableSemaphores.clear();
		renderFinishedSemaphores.clear();
		inFlightFences.clear();

		// Semaphores per swapchain image (indexed by imageIndex from acquireNextImage)
		// The presentation engine holds semaphores until the image is re-acquired, so we need
		// one semaphore per swapchain image to avoid reuse conflicts. See Vulkan spec:
		// https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html
		const auto semaphoreCount = static_cast<uint32_t>(swapChainImages.size());
		imageAvailableSemaphores.reserve(semaphoreCount);
		renderFinishedSemaphores.reserve(semaphoreCount);

		// Fences per frame-in-flight for CPU-GPU synchronization (indexed by currentFrame)
		inFlightFences.reserve(MAX_FRAMES_IN_FLIGHT);

		// Create semaphore info
		vk::SemaphoreCreateInfo semaphoreInfo{};

		// Create semaphores per swapchain image (indexed by imageIndex for presentation sync)
		for (uint32_t i = 0; i < semaphoreCount; i++)
		{
			imageAvailableSemaphores.emplace_back(device, semaphoreInfo);
			renderFinishedSemaphores.emplace_back(device, semaphoreInfo);
		}

		// Create fences per frame-in-flight (indexed by currentFrame for CPU-GPU pacing)
		vk::FenceCreateInfo fenceInfo{
			.flags = vk::FenceCreateFlagBits::eSignaled
		};
		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			inFlightFences.emplace_back(device, fenceInfo);
		}

		// Ensure uploads timeline semaphore exists (created early in createLogicalDevice)
		// No action needed here unless reinitializing after swapchain recreation.
		initializeGpuProfiling();
		return true;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to create sync objects: " << e.what() << std::endl;
		return false;
	}
}

void Renderer::initializeGpuProfiling()
{
	teardownGpuProfiling();

	if (!queueFamilyIndices.graphicsFamily.has_value())
	{
		return;
	}

	const auto queueFamilies = physicalDevice.getQueueFamilyProperties();
	const uint32_t graphicsFamilyIndex = queueFamilyIndices.graphicsFamily.value();
	if (graphicsFamilyIndex >= queueFamilies.size())
	{
		return;
	}

	const auto& queueProps = queueFamilies[graphicsFamilyIndex];
	const auto deviceProps = physicalDevice.getProperties();
	if (queueProps.timestampValidBits == 0 || deviceProps.limits.timestampPeriod <= 0.0f)
	{
		return;
	}

	const uint32_t queryCount = static_cast<uint32_t>(GPU_PROFILE_PASS_COUNT) * kGpuProfileQueriesPerPass;
	try
	{
		gpuProfilingQueryPools.reserve(MAX_FRAMES_IN_FLIGHT);
		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			vk::QueryPoolCreateInfo queryInfo{
				.queryType = vk::QueryType::eTimestamp,
				.queryCount = queryCount
			};
			gpuProfilingQueryPools.emplace_back(device, queryInfo);
		}
		gpuProfilingQueriesPending.assign(MAX_FRAMES_IN_FLIGHT, 0u);
		gpuTimestampPeriodNs = deviceProps.limits.timestampPeriod;
		gpuProfilingSupported = true;
	}
	catch (const std::exception&)
	{
		teardownGpuProfiling();
	}
}

void Renderer::teardownGpuProfiling()
{
	gpuProfilingQueryPools.clear();
	gpuProfilingQueriesPending.clear();
	gpuProfilingSupported = false;
	gpuTimestampPeriodNs = 0.0f;
	gpuProfilingLastCompleted = GpuProfilingFrameStats{};
	gpuProfilingLastFrameMs = 0.0f;
	gpuProfilingLastFrameValid = false;
}

bool Renderer::isGpuProfilingActive() const
{
	return gpuProfilingSupported && gpuProfilingEnabled && !gpuProfilingQueryPools.empty();
}

void Renderer::resetGpuProfilingQueries(vk::raii::CommandBuffer& cmd, uint32_t frameIndex)
{
	if (!isGpuProfilingActive() || frameIndex >= gpuProfilingQueryPools.size())
	{
		return;
	}

	const uint32_t queryCount = static_cast<uint32_t>(GPU_PROFILE_PASS_COUNT) * kGpuProfileQueriesPerPass;
	cmd.resetQueryPool(*gpuProfilingQueryPools[frameIndex], 0u, queryCount);
}

void Renderer::writeGpuProfileTimestamp(vk::raii::CommandBuffer& cmd, GpuProfilePass pass, bool beginMarker)
{
	if (!isGpuProfilingActive() || currentFrame >= gpuProfilingQueryPools.size())
	{
		return;
	}

	const vk::PipelineStageFlags2 stage = beginMarker
		                                      ? vk::PipelineStageFlagBits2::eTopOfPipe
		                                      : vk::PipelineStageFlagBits2::eBottomOfPipe;
	cmd.writeTimestamp2(stage, *gpuProfilingQueryPools[currentFrame],
	                    gpuProfileQueryIndex(static_cast<uint32_t>(pass), beginMarker));
}

void Renderer::resolveGpuProfilingFrame(uint32_t frameIndex)
{
	if (!gpuProfilingSupported || frameIndex >= gpuProfilingQueryPools.size() || frameIndex >=
		gpuProfilingQueriesPending.size() || gpuProfilingQueriesPending[frameIndex] == 0u)
	{
		return;
	}

	const uint32_t queryCount = static_cast<uint32_t>(GPU_PROFILE_PASS_COUNT) * kGpuProfileQueriesPerPass;
	std::vector<uint64_t> queryData(static_cast<size_t>(queryCount) * 2u, 0u);
	const vk::Result result = (*device).getQueryPoolResults(
		*gpuProfilingQueryPools[frameIndex],
		0u,
		queryCount,
		queryData.size() * sizeof(uint64_t),
		queryData.data(),
		sizeof(uint64_t) * 2u,
		vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);

	gpuProfilingQueriesPending[frameIndex] = 0u;
	if (result != vk::Result::eSuccess)
	{
		return;
	}

	GpuProfilingFrameStats stats{};
	bool frameValid = false;
	uint64_t earliestTimestamp = std::numeric_limits<uint64_t>::max();
	uint64_t latestTimestamp = 0u;

	for (uint32_t passIndex = 0; passIndex < static_cast<uint32_t>(GPU_PROFILE_PASS_COUNT); ++passIndex)
	{
		const uint32_t startQuery = passIndex * kGpuProfileQueriesPerPass;
		const uint32_t endQuery = startQuery + 1u;

		const uint64_t startValue = queryData[static_cast<size_t>(startQuery) * 2u];
		const uint64_t startAvailable = queryData[static_cast<size_t>(startQuery) * 2u + 1u];
		const uint64_t endValue = queryData[static_cast<size_t>(endQuery) * 2u];
		const uint64_t endAvailable = queryData[static_cast<size_t>(endQuery) * 2u + 1u];

		if (startAvailable == 0u || endAvailable == 0u || endValue < startValue)
		{
			continue;
		}

		const uint64_t deltaTicks = endValue - startValue;
		stats.passMs[passIndex] = (static_cast<double>(deltaTicks) * static_cast<double>(gpuTimestampPeriodNs)) /
			1'000'000.0;
		stats.passValid[passIndex] = 1u;

		earliestTimestamp = std::min(earliestTimestamp, startValue);
		latestTimestamp = std::max(latestTimestamp, endValue);
		frameValid = true;
	}

	if (frameValid && latestTimestamp >= earliestTimestamp)
	{
		stats.totalMs = (static_cast<double>(latestTimestamp - earliestTimestamp) * static_cast<double>(
			gpuTimestampPeriodNs)) / 1'000'000.0;
		stats.totalValid = true;
		gpuProfilingLastFrameMs = static_cast<float>(stats.totalMs);
		gpuProfilingLastFrameValid = true;
	}
	else
	{
		gpuProfilingLastFrameValid = false;
		gpuProfilingLastFrameMs = 0.0f;
	}

	gpuProfilingLastCompleted = stats;
	++gpuProfilingSamplesCollected;
}

// Clean up swap chain
void Renderer::cleanupSwapChain()
{
	teardownGpuProfiling();
	destroyDepthPyramidResources();
	destroySAOResources();
	destroyCascadedShadowResources();
	destroyTAAHistoryResources();

	for (auto& f : frustumCullPerFrame)
	{
		f.instanceAabbBuffer = vk::raii::Buffer(nullptr);
		f.instanceAabbBufferAllocation.reset();
		f.instanceAabbMapped = nullptr;
		f.instanceCapacity = 0;

		f.visibleIndicesBuffer = vk::raii::Buffer(nullptr);
		f.visibleIndicesBufferAllocation.reset();
		f.visibleIndicesMapped = nullptr;
		f.visibleIndicesCapacity = 0;

		f.visibleCountBuffer = vk::raii::Buffer(nullptr);
		f.visibleCountBufferAllocation.reset();
		f.visibleCountMapped = nullptr;

		f.occlusionVisibleIndicesBuffer = vk::raii::Buffer(nullptr);
		f.occlusionVisibleIndicesBufferAllocation.reset();
		f.occlusionVisibleIndicesMapped = nullptr;
		f.occlusionVisibleIndicesCapacity = 0;

		f.occlusionVisibleCountBuffer = vk::raii::Buffer(nullptr);
		f.occlusionVisibleCountBufferAllocation.reset();
		f.occlusionVisibleCountMapped = nullptr;

		f.paramsBuffer = vk::raii::Buffer(nullptr);
		f.paramsBufferAllocation.reset();
		f.paramsMapped = nullptr;

		f.occlusionParamsBuffer = vk::raii::Buffer(nullptr);
		f.occlusionParamsBufferAllocation.reset();
		f.occlusionParamsMapped = nullptr;

		f.indirectCommandsBuffer = vk::raii::Buffer(nullptr);
		f.indirectCommandsBufferAllocation.reset();
		f.indirectCommandsMapped = nullptr;
		f.indirectCommandCapacity = 0;

		f.computeSet = nullptr;
		f.occlusionComputeSet = nullptr;
	}
	frustumCullPerFrame.clear();
	frustumCullPipeline = vk::raii::Pipeline(nullptr);
	frustumCullPipelineLayout = vk::raii::PipelineLayout(nullptr);
	frustumCullDescriptorSetLayout = vk::raii::DescriptorSetLayout(nullptr);
	lastGpuFrustumInputCount = 0;
	lastGpuFrustumVisibleCount = 0;

	// Clean up depth resources
	depthImageView = vk::raii::ImageView(nullptr);
	depthImage = vk::raii::Image(nullptr);
	depthImageAllocation = nullptr;

	// Clean up swap chain image views
	swapChainImageViews.clear();

	// Note: Keep descriptor pool alive here to ensure descriptor sets remain valid during swapchain recreation.
	// descriptorPool is preserved; it will be managed during full renderer teardown.

	// Destroy reflection render targets if present
	destroyReflectionResources();

	// Clean up pipelines
	graphicsPipeline = vk::raii::Pipeline(nullptr);
	pbrGraphicsPipeline = vk::raii::Pipeline(nullptr);
	lightingPipeline = vk::raii::Pipeline(nullptr);

	// Clean up pipeline layouts
	pipelineLayout = vk::raii::PipelineLayout(nullptr);
	pbrPipelineLayout = vk::raii::PipelineLayout(nullptr);
	lightingPipelineLayout = vk::raii::PipelineLayout(nullptr);

	// Clean up sync objects (they need to be recreated with new swap chain image count)
	imageAvailableSemaphores.clear();
	renderFinishedSemaphores.clear();
	inFlightFences.clear();

	// Clean up swap chain
	swapChain = vk::raii::SwapchainKHR(nullptr);
}

// Recreate swap chain
void Renderer::recreateSwapChain()
{
	// Prevent background uploads worker from mutating descriptors while we rebuild
	StopUploadsWorker();

	// Block descriptor writes while we rebuild swapchain and descriptor pools
	descriptorSetsValid.store(false, std::memory_order_relaxed);
	{
		// Drop any deferred descriptor updates that target old descriptor sets
		std::lock_guard<std::mutex> lk(pendingDescMutex);
		pendingDescOps.clear();
		descriptorRefreshPending.store(false, std::memory_order_relaxed);
	}

	// Wait for all frames in flight to complete before recreating the swap chain
	std::vector<vk::Fence> allFences;
	allFences.reserve(inFlightFences.size());
	for (const auto& fence : inFlightFences)
	{
		allFences.push_back(*fence);
	}
	if (!allFences.empty())
	{
		vk::Result result = waitForFencesSafe(allFences, VK_TRUE);
		if (result != vk::Result::eSuccess)
		{
			std::cerr << "Error: Failed to wait for in-flight fences during swap chain recreation: " <<
				vk::to_string(result) << std::endl;
		}
	}

	// Wait for the device to be idle before recreating the swap chain
	// External synchronization required (VVL): serialize against queue submits/present.
	WaitIdle();

	// Clean up old swap chain resources
	cleanupSwapChain();

	// Recreate swap chain and related resources
	createSwapChain();
	createImageViews();
	setupDynamicRendering();
	createDepthResources();
	createCascadedShadowResources();

	// (Re)create reflection resources if enabled
	if (enablePlanarReflections)
	{
		uint32_t rw = std::max(
			1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.width) * reflectionResolutionScale));
		uint32_t rh = std::max(
			1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.height) * reflectionResolutionScale));
		createReflectionResources(rw, rh);
	}

	// Recreate sync objects with correct sizing for new swap chain
	createSyncObjects();

	// Recreate off-screen opaque scene color and descriptor sets needed by transparent pass
	createOpaqueSceneColorResources();
	if (!createTAAHistoryResources())
	{
		std::cerr << "Warning: Failed to recreate TAA history resources after swapchain recreation\n";
	}
	createTransparentDescriptorSets();
	createTransparentFallbackDescriptorSets();

	// Recreate deferred resources (G-buffer) to match new swapchain dimensions
	destroyDeferredResources();
	if (!createDeferredResources()) {
		std::cerr << "Warning: Failed to recreate deferred resources after swapchain recreation\n";
	}

	// Wait for all command buffers to complete before clearing resources
	for (const auto& fence : inFlightFences)
	{
		vk::Result result = waitForFencesSafe(*fence, VK_TRUE);
		if (result != vk::Result::eSuccess)
		{
			std::cerr << "Error: Failed to wait for fence before clearing resources: " << vk::to_string(result) <<
				std::endl;
		}
	}

	// Clear all entity descriptor sets since they're now invalid (allocated from the old pool)
	{
		// Serialize descriptor frees against any other descriptor operations
		std::lock_guard<std::mutex> lk(descriptorMutex);
		for (auto& kv : entityResources)
		{
			auto& resources = kv.second;
			resources.basicDescriptorSets.clear();
			resources.pbrDescriptorSets.clear();
			// Descriptor initialization flags must be reset because new descriptor sets
			// will be allocated and only the current frame will be initialized at runtime.
			resources.pbrUboBindingWritten.assign(MAX_FRAMES_IN_FLIGHT, false);
			resources.basicUboBindingWritten.assign(MAX_FRAMES_IN_FLIGHT, false);
			resources.pbrImagesWritten.assign(MAX_FRAMES_IN_FLIGHT, false);
			resources.basicImagesWritten.assign(MAX_FRAMES_IN_FLIGHT, false);
			resources.pbrFixedBindingsWritten.assign(MAX_FRAMES_IN_FLIGHT, false);
		}
	}

	// Clear ray query descriptor sets - they reference the old output image which will be destroyed
	// Must clear before recreating to avoid descriptor set corruption
	rayQueryDescriptorSets.clear();
	rayQueryDescriptorsWritten.clear();
	rayQueryDescriptorsDirtyMask.store(0u, std::memory_order_relaxed);

	// Destroy ray query output image resources - they're sized to old swapchain dimensions
	rayQueryOutputImageView = vk::raii::ImageView(nullptr);
	rayQueryOutputImage = vk::raii::Image(nullptr);
	rayQueryOutputImageAllocation = nullptr;

	createGraphicsPipeline();
	createPBRPipeline();
	createLightingPipeline();
	createCompositePipeline();

	// Recreate Forward+ specific pipelines/resources and resize tile buffers for new extent
	if (useForwardPlus)
	{
		createDepthPrepassPipeline();
		uint32_t tilesX = (swapChainExtent.width + forwardPlusTileSizeX - 1) / forwardPlusTileSizeX;
		uint32_t tilesY = (swapChainExtent.height + forwardPlusTileSizeY - 1) / forwardPlusTileSizeY;
		createOrResizeForwardPlusBuffers(tilesX, tilesY, forwardPlusSlicesZ);
		if (!createDepthPyramidResources())
		{
			std::cerr << "Warning: Failed to recreate depth pyramid resources after swapchain recreation\n";
		}
		if (!createSAOResources())
		{
			std::cerr << "Warning: Failed to recreate SAO resources after swapchain recreation\n";
		}
		if (!createVolumetricResources())
		{
			std::cerr << "Warning: Failed to recreate volumetric resources after swapchain recreation\n";
		}
	}

	createTransparentDescriptorSets();

	// Re-create command buffers to ensure fresh recording against new swapchain state
	commandBuffers.clear();
	createCommandBuffers();
	currentFrame = 0;

	// Recreate ray query resources with new swapchain dimensions
	// This must happen after descriptor pool is valid but before marking descriptor sets valid
	if (rayQueryEnabled && accelerationStructureEnabled)
	{
		if (!createRayQueryResources())
		{
			std::cerr << "Warning: Failed to recreate ray query resources after swapchain recreation\n";
		}
	}

	// Recreate descriptor sets for all entities after swapchain/pipeline rebuild
	for (const auto& kv : entityResources)
	{
		const auto& entity = kv.first;
		if (!entity)
			continue;
		auto meshComponent = entity->GetComponent<MeshComponent>();
		if (!meshComponent)
			continue;

		std::string texturePath = meshComponent->GetTexturePath();
		// Fallback for basic pipeline: use baseColor when legacy path is empty
		if (texturePath.empty())
		{
			const std::string& baseColor = meshComponent->GetBaseColorTexturePath();
			if (!baseColor.empty())
			{
				texturePath = baseColor;
			}
		}
		// Recreate basic descriptor sets (ignore failures here to avoid breaking resize)
		createDescriptorSets(entity, texturePath, false);
		// Recreate PBR descriptor sets
		createDescriptorSets(entity, texturePath, true);
	}

	// Descriptor sets are now valid again
	descriptorSetsValid.store(true, std::memory_order_relaxed);

	// Resume background uploads worker now that swapchain and descriptors are recreated
	StartUploadsWorker();
}

void Renderer::updateTAAJitterState()
{
	constexpr uint32_t jitterSequenceLength = 8;

	taaPreviousJitterPixels = taaCurrentJitterPixels;
	taaPreviousJitterNdc = taaCurrentJitterNdc;

	if (swapChainExtent.width == 0 || swapChainExtent.height == 0)
	{
		taaCurrentJitterPixels = glm::vec2(0.0f);
		taaCurrentJitterNdc = glm::vec2(0.0f);
		taaHistoryReadIndex = 0;
		taaHistoryWriteIndex = 0;
		return;
	}

	const uint32_t sampleIndex = (taaFrameIndex % jitterSequenceLength) + 1u;
	const float jitterX = halton(sampleIndex, 2u) - 0.5f;
	const float jitterY = halton(sampleIndex, 3u) - 0.5f;

	taaCurrentJitterPixels = glm::vec2(jitterX, jitterY);
	taaCurrentJitterNdc = glm::vec2(
		(2.0f * jitterX) / static_cast<float>(swapChainExtent.width),
		(2.0f * jitterY) / static_cast<float>(swapChainExtent.height));

	if (MAX_FRAMES_IN_FLIGHT > 0u)
	{
		taaHistoryWriteIndex = currentFrame;
		taaHistoryReadIndex = (currentFrame + MAX_FRAMES_IN_FLIGHT - 1u) % MAX_FRAMES_IN_FLIGHT;
	}
	else
	{
		taaHistoryWriteIndex = 0;
		taaHistoryReadIndex = 0;
	}

	++taaFrameIndex;
}

void Renderer::prepareFrameUboTemplate(CameraComponent* camera)
{
	frameUboTemplate = UniformBufferObject{};
	if (!camera) return;

	frameUboTemplate.view = camera->GetViewMatrix();
	frameUboTemplate.proj = camera->GetProjectionMatrix();
	frameUboTemplate.proj[1][1] *= -1; // Flip Y for Vulkan
	if (taaJitterEnabled)
	{
		frameUboTemplate.proj[2][0] += taaCurrentJitterNdc.x;
		frameUboTemplate.proj[2][1] += taaCurrentJitterNdc.y;
	}
	frameUboTemplate.camPos = glm::vec4(camera->GetPosition(), 1.0f);

	frameUboTemplate.lightCount = static_cast<int>(lastFrameLightCount);
	frameUboTemplate.exposure = std::clamp(this->exposure, 0.2f, 4.0f);
	frameUboTemplate.gamma = this->gamma;
	frameUboTemplate.screenDimensions = glm::vec2(swapChainExtent.width, swapChainExtent.height);
	frameUboTemplate.nearZ = camera->GetNearPlane();
	frameUboTemplate.farZ = camera->GetFarPlane();
	frameUboTemplate.slicesZ = static_cast<float>(forwardPlusSlicesZ);
	for (uint32_t i = 0; i < CSM_CASCADE_COUNT; ++i)
	{
		frameUboTemplate.cascadeLightSpaceMatrices[i] =
			csmDataValid ? csmCascadeLightSpaceMatrices[i] : glm::mat4(1.0f);
	}
	frameUboTemplate.cascadeSplitDepths = csmDataValid
		                                      ? glm::vec4(csmCascadeSplitDepths[0], csmCascadeSplitDepths[1],
		                                                  csmCascadeSplitDepths[2], 1.0f)
		                                      : glm::vec4(frameUboTemplate.farZ, frameUboTemplate.farZ,
		                                                  frameUboTemplate.farZ, 0.0f);
	frameUboTemplate.cascadeTransitionData = csmDataValid ? csmTransitionData : glm::vec4(0.0f);

	int outputIsSRGB = (swapChainImageFormat == vk::Format::eR8G8B8A8Srgb ||
		                   swapChainImageFormat == vk::Format::eB8G8R8A8Srgb)
		                   ? 1
		                   : 0;
	frameUboTemplate.padding0 = outputIsSRGB;
	// Raster PBR shader uses padding1 as the Forward+ enable flag.
	// 0 = disabled (always use global light loop), non-zero = enabled (use culled tile lists).
	frameUboTemplate.padding1 = useForwardPlus ? 1.0f : 0.0f;
	frameUboTemplate.padding2 = enableRasterRayQueryShadows ? 1.0f : 0.0f;

	bool reflReady = false;
	if (enablePlanarReflections && !reflections.empty())
	{
		const uint32_t count = static_cast<uint32_t>(reflections.size());
		const uint32_t prev = (currentFrame + count - 1u) % count;
		auto& rtPrev = reflections[prev];
		reflReady = (!!*rtPrev.colorView) && (!!*rtPrev.colorSampler);
	}
	frameUboTemplate.reflectionEnabled = reflReady ? 1 : 0;
	frameUboTemplate.reflectionVP = sampleReflectionVP;
	frameUboTemplate.clipPlaneWS = currentReflectionPlane;
	frameUboTemplate.reflectionIntensity = std::clamp(reflectionIntensity, 0.0f, 2.0f);
	frameUboTemplate.enableRayQueryReflections = enableRayQueryReflections ? 1 : 0;
	frameUboTemplate.enableRayQueryTransparency = enableRayQueryTransparency ? 1 : 0;

	// Ray-query shared buffers are also used by raster PBR when doing ray-query shadows.
	// Populate counts so shaders can bounds-check even when running in raster mode.
	frameUboTemplate.geometryInfoCount = static_cast<int>(geometryInfoCountCPU);
	frameUboTemplate.materialCount = static_cast<int>(materialCountCPU);
}

// Update uniform buffer
void Renderer::updateUniformBuffer(uint32_t currentImage, Entity* entity, EntityResources* entityRes,
                                   CameraComponent* camera, TransformComponent* tc)
{
	if (!entityRes)
	{
		return;
	}

	// Get transform component
	auto transformComponent = tc ? tc : (entity ? entity->GetComponent<TransformComponent>() : nullptr);
	if (!transformComponent)
	{
		return;
	}

	// Create uniform buffer object
	UniformBufferObject ubo{};
	ubo.model = transformComponent->GetModelMatrix();
	ubo.view = camera->GetViewMatrix();
	ubo.proj = camera->GetProjectionMatrix();
	ubo.proj[1][1] *= -1; // Flip Y for Vulkan

	// Continue with the rest of the uniform buffer setup
	updateUniformBufferInternal(currentImage, entity, entityRes, camera, ubo);
}

// Overloaded version that accepts a custom transform matrix
void Renderer::updateUniformBuffer(uint32_t currentImage, Entity* entity, EntityResources* entityRes,
                                   CameraComponent* camera, const glm::mat4& customTransform)
{
	if (!entityRes) return;
	// Create the uniform buffer object with custom transform
	UniformBufferObject ubo{};
	ubo.model = customTransform;
	ubo.view = camera->GetViewMatrix();
	ubo.proj = camera->GetProjectionMatrix();
	ubo.proj[1][1] *= -1; // Flip Y for Vulkan

	// Continue with the rest of the uniform buffer setup
	updateUniformBufferInternal(currentImage, entity, entityRes, camera, ubo);
}

// Internal helper function to complete uniform buffer setup
void Renderer::updateUniformBufferInternal(uint32_t currentImage, Entity* entity, EntityResources* entityRes,
                                           CameraComponent* camera, UniformBufferObject& ubo)
{
	if (!entityRes)
	{
		return;
	}

	// Use frame template for most fields
	UniformBufferObject finalUbo = frameUboTemplate;
	finalUbo.model = ubo.model;

	// For reflection pass, we must override view/proj/reflection flags
	if (ubo.reflectionPass == 1)
	{
		finalUbo.view = ubo.view;
		finalUbo.proj = ubo.proj;
		finalUbo.reflectionPass = 1;
		finalUbo.reflectionEnabled = 0;
		finalUbo.reflectionVP = ubo.reflectionVP;
		finalUbo.clipPlaneWS = ubo.clipPlaneWS;
		finalUbo.padding2 = ubo.padding2;
	}

	// Copy to uniform buffer (guard against null mapped pointer)
	void* dst = entityRes->uniformBuffersMapped[currentImage];
	if (!dst)
	{
		std::cerr << "Warning: UBO mapped ptr null for entity '" << (entity ? entity->GetName() : "unknown") <<
			"' frame " << currentImage << std::endl;
		return;
	}
	std::memcpy(dst, &finalUbo, sizeof(UniformBufferObject));
}

void Renderer::ensureEntityMaterialCache(Entity* entity, EntityResources& res)
{
	if (!entity)
		return;

	if (res.materialCacheValid)
		return;

	res.materialCacheValid = true;
	res.cachedMaterial = nullptr;
	res.cachedIsBlended = false;
	res.cachedIsGlass = false;
	res.cachedIsLiquid = false;

	// Defaults represent the common case (no explicit material); textures come from descriptor bindings.
	MaterialProperties mp{};
	// Sensible defaults for entities without explicit material
	mp.baseColorFactor = glm::vec4(1.0f);
	mp.metallicFactor = 0.0f;
	mp.roughnessFactor = 1.0f;
	mp.baseColorTextureSet = 0;
	mp.physicalDescriptorTextureSet = 0;
	mp.normalTextureSet = -1;
	mp.occlusionTextureSet = -1;
	mp.emissiveTextureSet = -1;
	mp.alphaMask = 0.0f;
	mp.alphaMaskCutoff = 0.5f;
	mp.emissiveFactor = glm::vec3(0.0f);
	mp.emissiveStrength = 1.0f;
	mp.transmissionFactor = 0.0f;
	mp.useSpecGlossWorkflow = 0;
	mp.glossinessFactor = 0.0f;
	mp.specularFactor = glm::vec3(1.0f);
	mp.ior = 1.5f;
	mp.hasEmissiveStrengthExtension = 0;

	if (modelLoader)
	{
		const std::string& entityName = entity->GetName();
		const size_t tagPos = entityName.find("_Material_");
		if (tagPos != std::string::npos)
		{
			const size_t afterTag = tagPos + std::string("_Material_").size();
			if (afterTag < entityName.length())
			{
				// Entity name format: "modelName_Material_<index>_<materialName>"
				const std::string remainder = entityName.substr(afterTag);
				const size_t nextUnderscore = remainder.find('_');
				if (nextUnderscore != std::string::npos && nextUnderscore + 1 < remainder.length())
				{
					const std::string materialName = remainder.substr(nextUnderscore + 1);
					if (const Material* material = modelLoader->GetMaterial(materialName))
					{
						res.cachedMaterial = material;
						res.cachedIsGlass = material->isGlass;
						res.cachedIsLiquid = material->isLiquid;

						// Base factors
						mp.baseColorFactor = glm::vec4(material->albedo, material->alpha);
						mp.metallicFactor = material->metallic;
						mp.roughnessFactor = material->roughness;

						// Texture set flags (-1 = no texture)
						mp.baseColorTextureSet = material->albedoTexturePath.empty() ? -1 : 0;
						// physical descriptor: MR or SpecGloss
						if (material->useSpecularGlossiness)
						{
							mp.useSpecGlossWorkflow = 1;
							mp.physicalDescriptorTextureSet = material->specGlossTexturePath.empty() ? -1 : 0;
							mp.glossinessFactor = material->glossinessFactor;
							mp.specularFactor = material->specularFactor;
						}
						else
						{
							mp.useSpecGlossWorkflow = 0;
							mp.physicalDescriptorTextureSet = material->metallicRoughnessTexturePath.empty() ? -1 : 0;
						}
						mp.normalTextureSet = material->normalTexturePath.empty() ? -1 : 0;
						mp.occlusionTextureSet = material->occlusionTexturePath.empty() ? -1 : 0;
						mp.emissiveTextureSet = material->emissiveTexturePath.empty() ? -1 : 0;

						// Emissive and transmission/IOR
						mp.emissiveFactor = material->emissive;
						mp.emissiveStrength = material->emissiveStrength;
						// Heuristic: consider emissive strength extension present when strength != 1.0
						mp.hasEmissiveStrengthExtension = (std::abs(material->emissiveStrength - 1.0f) > 1e-6f) ? 1 : 0;
						mp.transmissionFactor = material->transmissionFactor;
						mp.ior = material->ior;

						// Alpha mask handling
						mp.alphaMask = (material->alphaMode == "MASK") ? 1.0f : 0.0f;
						mp.alphaMaskCutoff = material->alphaCutoff;

						// Blended classification (opaque materials stay in the opaque pass)
						const bool alphaBlend = (material->alphaMode == "BLEND");
						const bool highTransmission = (material->transmissionFactor > 0.2f);
						res.cachedIsBlended = alphaBlend || highTransmission || res.cachedIsGlass || res.cachedIsLiquid;
					}
				}
			}
		}
	}

	res.cachedMaterialProps = mp;
}

// Render the scene (unique_ptr container overload)
// Convert to a raw-pointer snapshot so callers can safely release their container locks.
void Renderer::Render(const std::vector<std::unique_ptr<Entity>>& entities, CameraComponent* camera,
                      ImGuiSystem* imguiSystem)
{
	std::vector<Entity*> snapshot;
	snapshot.reserve(entities.size());
	for (const auto& uptr : entities)
	{
		snapshot.push_back(uptr.get());
	}
	Render(snapshot, camera, imguiSystem);
}

// Render the scene (raw pointer snapshot overload)
void Renderer::Render(const std::vector<Entity*>& entities, CameraComponent* camera, ImGuiSystem* imguiSystem)
{
	// Update watchdog timestamp to prove frame is progressing
	lastFrameUpdateTime.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);
	watchdogProgressLabel.store("Render: frame begin", std::memory_order_relaxed);

	if (memoryPool)
		memoryPool->setRenderingActive(true);
	struct RenderingStateGuard
	{
		MemoryPool* pool;

		explicit RenderingStateGuard(MemoryPool* p) : pool(p)
		{
		}

		~RenderingStateGuard()
		{
			if (pool)
				pool->setRenderingActive(false);
		}
	} guard(memoryPool.get());

	// Track if ray query rendered successfully this frame to skip rasterization code path
	bool rayQueryRenderedThisFrame = false;

	// --- Extract lights for the frame ---
	// Build a single light list once per frame (emissive lights only for this scene)
	std::vector<ExtractedLight> lightsSubset;
	if (!staticLights.empty())
	{
		lightsSubset.reserve(std::min(staticLights.size(), static_cast<size_t>(MAX_ACTIVE_LIGHTS)));
		for (const auto& L : staticLights)
		{
			// Include all lights (Directional, Point, Emissive) up to the limit
			lightsSubset.push_back(L);
			if (lightsSubset.size() >= MAX_ACTIVE_LIGHTS)
				break;
		}
	}
	lastFrameLightCount = static_cast<uint32_t>(lightsSubset.size());
	updateCascadedShadowData(lightsSubset, camera);
	if (!lightsSubset.empty())
	{
		updateLightStorageBuffer(currentFrame, lightsSubset, camera);
	}

	updateTAAJitterState();

	// Pre-calculate frame-constant UBO data
	prepareFrameUboTemplate(camera);

	// Wait for the previous frame's work on this frame slot to complete
	// Use a finite timeout loop so we can keep the watchdog alive during long GPU work
	// (e.g., acceleration structure builds/refits can legitimately take seconds on large scenes).
	watchdogProgressLabel.store("Render: wait inFlightFence", std::memory_order_relaxed);
	vk::Result fenceResult = waitForFencesSafe(*inFlightFences[currentFrame], VK_TRUE);
	if (fenceResult != vk::Result::eSuccess)
	{
		std::cerr << "Error: Failed to wait for in-flight fence: " << vk::to_string(fenceResult) << std::endl;
	}
	else
	{
		resolveGpuProfilingFrame(currentFrame);
	}

	// Reset the fence immediately after successful wait, before any new work
	watchdogProgressLabel.store("Render: reset inFlightFence", std::memory_order_relaxed);
	device.resetFences(*inFlightFences[currentFrame]);

	if (currentFrame < frustumCullPerFrame.size() && frustumCullPerFrame[currentFrame].visibleCountMapped)
	{
		const uint32_t* visibleCountPtr = static_cast<const uint32_t*>(frustumCullPerFrame[currentFrame].
			visibleCountMapped);
		lastGpuFrustumVisibleCount = visibleCountPtr ? *visibleCountPtr : 0u;
	}
	else
	{
		lastGpuFrustumVisibleCount = 0;
	}

	// Execute any pending GPU uploads (enqueued by worker/loading threads) on the render thread
	// at this safe point to ensure all Vulkan submits happen on a single thread.
	// This prevents validation/GPU-AV PostSubmit crashes due to cross-thread queue usage.
	watchdogProgressLabel.store("Render: ProcessPendingMeshUploads", std::memory_order_relaxed);
	ProcessPendingMeshUploads();
	// Execute any pending per-entity GPU resource preallocation requested by the scene loader.
	// This prevents background threads from mutating `entityResources`/`meshResources` concurrently
	// with rendering (which can corrupt unordered_map internals and crash).
	watchdogProgressLabel.store("Render: ProcessPendingEntityPreallocations", std::memory_order_relaxed);
	ProcessPendingEntityPreallocations();
	watchdogProgressLabel.store("Render: after ProcessPendingEntityPreallocations", std::memory_order_relaxed);

	// Process deferred AS deletion queue at safe point (after fence wait)
	// Increment frame counters and delete AS structures that are no longer in use
	// Wait for MAX_FRAMES_IN_FLIGHT + 1 frames to ensure GPU has finished all work
	// (The +1 ensures we've waited through a full cycle of all frame slots)
	{
		auto it = pendingASDeletions.begin();
		while (it != pendingASDeletions.end())
		{
			it->framesSinceDestroy++;
			if (it->framesSinceDestroy > MAX_FRAMES_IN_FLIGHT)
			{
				// Safe to delete - all frames have finished using these AS structures
				it = pendingASDeletions.erase(it);
			}
			else
			{
				++it;
			}
		}
	}
	watchdogProgressLabel.store("Render: after pendingASDeletions", std::memory_order_relaxed);

	// Opportunistically request AS rebuild when more meshes become ready than in the last built AS.
	// This makes the TLAS grow as streaming/allocations complete, then settle (no rebuild spam).
	// NOTE: This scan can be relatively heavy and is not needed for the default startup path.
	// Only run it when opportunistic rebuilds are enabled.
	// While loading, allow opportunistic AS rebuild scanning even if the user-facing toggle is off.
	// This prevents nondeterministic “missing outdoor props” across app restarts when the first TLAS
	// build happens before all entities exist.
	if (rayQueryEnabled && accelerationStructureEnabled && (asOpportunisticRebuildEnabled || IsLoading()))
	{
		watchdogProgressLabel.store("Render: AS readiness scan", std::memory_order_relaxed);
		size_t readyRenderableCount = 0;
		size_t readyUniqueMeshCount = 0;
		{
			auto lastKick = std::chrono::steady_clock::now();
			auto kickWatchdog = [&]()
			{
				auto now = std::chrono::steady_clock::now();
				if (now - lastKick > std::chrono::milliseconds(200))
				{
					lastFrameUpdateTime.store(now, std::memory_order_relaxed);
					lastKick = now;
				}
			};
			std::map<MeshComponent*, uint32_t> meshToBLASProbe;
			for (Entity* e : entities)
			{
				kickWatchdog();
				if (!e || !e->IsActive())
					continue;
				// In Ray Query static-only mode, ignore dynamic/animated entities for readiness
				if (IsRayQueryStaticOnly())
				{
					const std::string& nm = e->GetName();
					if (nm.find("_AnimNode_") != std::string::npos)
						continue;
					if (!nm.empty() && nm.rfind("Ball_", 0) == 0)
						continue;
				}
				auto meshComp = e->GetComponent<MeshComponent>();
				if (!meshComp)
					continue;
				try
				{
					auto it = meshResources.find(meshComp);
					if (it == meshResources.end())
						continue;
					const auto& res = it->second;
					// STRICT readiness: uploads must be finished (staging sizes zero)
					if (res.vertexBufferSizeBytes != 0 || res.indexBufferSizeBytes != 0)
						continue;
					if (!*res.vertexBuffer || !*res.indexBuffer)
						continue;
					if (res.indexCount == 0)
						continue;
				}
				catch (...)
				{
					continue;
				}
				readyRenderableCount++;
				if (meshToBLASProbe.find(meshComp) == meshToBLASProbe.end())
				{
					meshToBLASProbe[meshComp] = static_cast<uint32_t>(meshToBLASProbe.size());
				}
			}
			readyUniqueMeshCount = meshToBLASProbe.size();
		}
		// During scene loading/finalization, the TLAS may be built before all entities exist.
		// Allow rebuilds even if AS is "frozen" so the TLAS converges to the full scene across restarts.
		if ((!asFrozen || IsLoading()) && (readyRenderableCount > lastASBuiltInstanceCount || readyUniqueMeshCount >
			lastASBuiltBLASCount) && !asBuildRequested.load(std::memory_order_relaxed))
		{
			std::cout << "AS rebuild requested: counts increased (built instances=" << lastASBuiltInstanceCount
				<< ", ready instances=" << readyRenderableCount
				<< ", built meshes=" << lastASBuiltBLASCount
				<< ", ready meshes=" << readyUniqueMeshCount << ")\n";
			RequestAccelerationStructureBuild("counts increased");
		}

		// Post-load repair: if loading is done and the current TLAS instance count is far below readiness,
		// force a one-time rebuild even when frozen so we include the whole scene.
		if (!IsLoading() && !asBuildRequested.load(std::memory_order_relaxed))
		{
			const size_t targetInstances = readyRenderableCount;
			if (targetInstances > 0 && lastASBuiltInstanceCount < static_cast<size_t>(static_cast<double>(
				targetInstances) * 0.95))
			{
				asDevOverrideAllowRebuild = true; // allow rebuild even if frozen
				std::cout << "AS rebuild requested: post-load full build (built instances=" << lastASBuiltInstanceCount
					<< ", ready instances=" << targetInstances << ")\n";
				RequestAccelerationStructureBuild("post-load full build");
			}
		}
	}

	// If in Ray Query static-only mode and TLAS not yet built post-load, request a one-time build now.
	// (Does not require a readiness scan.)
	if (rayQueryEnabled && accelerationStructureEnabled && currentRenderMode
		==
		RenderMode::RayQuery && IsRayQueryStaticOnly() &&
		!IsLoading() &&
		!*tlasStructure.handle && !asBuildRequested.load(std::memory_order_relaxed)
	)
	{
		RequestAccelerationStructureBuild("static-only initial build");
	}

	// Check if acceleration structure build was requested (e.g., after scene loading or counts grew)
	// Build at this safe frame point to avoid threading issues
	watchdogProgressLabel.store("Render: AS build request check", std::memory_order_relaxed);
	if (asBuildRequested.load(std::memory_order_acquire))
	{
		watchdogProgressLabel.store("Render: AS build request handling", std::memory_order_relaxed);

		// Defer TLAS/BLAS build while the scene loader is still active to avoid partial builds.
		// IMPORTANT: Do NOT use IsLoading() here; IsLoading() also includes the post-load
		// "finalizing" stage, and deferring on that would deadlock the AS build forever.
		if (IsSceneLoaderActive())
		{
			// Keep the request flag set; we'll build once the loader (and critical textures) finish.
		}
		else if (asFrozen && !asDevOverrideAllowRebuild && !IsLoading())
		{
			// Ignore rebuilds while frozen to avoid wiping TLAS during animation playback
			std::cout << "AS rebuild request ignored (frozen). Reason: " << lastASBuildRequestReason << "\n";
			asBuildRequested.store(false, std::memory_order_release);
			asBuildRequestStartNs.store(0, std::memory_order_relaxed);
			watchdogSuppressed.store(false, std::memory_order_relaxed);
		}
		else
		{
			// Gate initial build until readiness is high enough to represent the full scene
			size_t totalRenderableEntities = 0;
			size_t readyRenderableCount = 0;
			size_t readyUniqueMeshCount = 0;
			size_t missingMeshResources = 0;
			size_t pendingUploadsCount = 0;
			size_t nullBuffersCount = 0;
			size_t zeroIndicesCount = 0;
			{
				auto lastKick = std::chrono::steady_clock::now();
				auto kickWatchdog = [&]()
				{
					auto now = std::chrono::steady_clock::now();
					if (now - lastKick > std::chrono::milliseconds(200))
					{
						lastFrameUpdateTime.store(now, std::memory_order_relaxed);
						lastKick = now;
					}
				};
				std::map<MeshComponent*, uint32_t> meshToBLASProbe;
				for (Entity* e : entities)
				{
					kickWatchdog();
					if (!e || !e->IsActive())
						continue;
					// In Ray Query static-only mode, ignore dynamic/animated entities for totals/readiness
					if (IsRayQueryStaticOnly())
					{
						const std::string& nm = e->GetName();
						if (nm.find("_AnimNode_") != std::string::npos)
							continue;
						if (!nm.empty() && nm.rfind("Ball_", 0) == 0)
							continue;
					}
					auto meshComp = e->GetComponent<MeshComponent>();
					if (!meshComp)
						continue;
					totalRenderableEntities++;
					try
					{
						auto it = meshResources.find(meshComp);
						if (it == meshResources.end())
						{
							missingMeshResources++;
							continue;
						}
						const auto& res = it->second;
						// STRICT readiness here too: uploads finished
						if (res.vertexBufferSizeBytes != 0 || res.indexBufferSizeBytes != 0)
						{
							pendingUploadsCount++;
							continue;
						}
						if (!*res.vertexBuffer || !*res.indexBuffer)
						{
							nullBuffersCount++;
							continue;
						}
						if (res.indexCount == 0)
						{
							zeroIndicesCount++;
							continue;
						}
					}
					catch (...)
					{
						continue;
					}
					readyRenderableCount++;
					if (meshToBLASProbe.find(meshComp) == meshToBLASProbe.end())
					{
						meshToBLASProbe[meshComp] = static_cast<uint32_t>(meshToBLASProbe.size());
					}
				}
				readyUniqueMeshCount = meshToBLASProbe.size();
			}
			const double readiness = (totalRenderableEntities > 0)
				                         ? static_cast<double>(readyRenderableCount) / static_cast<double>(
					                         totalRenderableEntities)
				                         : 0.0;
			const double buildThreshold = 0.95; // prefer building when ~full scene is ready

			// Bounded deferral: avoid getting stuck forever waiting for perfect readiness.
			// After a short timeout from the original request, build with the best available data.
			const uint64_t reqNs = asBuildRequestStartNs.load(std::memory_order_relaxed);
			const uint64_t nowNs = std::chrono::steady_clock::now().time_since_epoch().count();
			const double maxDeferralSeconds = 15.0;
			const bool deferralTimedOut = (reqNs != 0) && (nowNs > reqNs) &&
				(static_cast<double>(nowNs - reqNs) / 1'000'000'000.0) >= maxDeferralSeconds;

			if (readiness < buildThreshold && !asDevOverrideAllowRebuild && !deferralTimedOut)
			{
				// Intentionally no stdout spam here (Windows consoles are slow and there's no user-facing benefit).
				// Keep the request flag set; try again next frame
			}
			else
			{
				if (deferralTimedOut && readiness < buildThreshold && !asDevOverrideAllowRebuild)
				{
					std::cout << "AS build forced after " << maxDeferralSeconds
						<< "s deferral (readiness " << readyRenderableCount << "/" << totalRenderableEntities
						<< ", uniqueMeshesReady=" << readyUniqueMeshCount << ")\n";
				}
				struct WatchdogSuppressGuard
				{
					std::atomic<bool>& flag;

					explicit WatchdogSuppressGuard(std::atomic<bool>& f) : flag(f)
					{
						flag.store(true, std::memory_order_relaxed);
					}

					~WatchdogSuppressGuard()
					{
						flag.store(false, std::memory_order_relaxed);
					}
				} watchdogGuard(watchdogSuppressed);

				// Ensure previous GPU work is complete BEFORE building AS.
				//
				// Wait for all *other* frame-in-flight fences to signal using a finite timeout loop
				// and kick the watchdog while we wait.
				// Do NOT include `currentFrame` here because its fence was reset at frame start
				// and will not signal until we submit the current frame.
				{
					std::vector<vk::Fence> fencesToWait;
					if (inFlightFences.size() > 1)
					{
						fencesToWait.reserve(inFlightFences.size() - 1);
					}
					for (uint32_t i = 0; i < static_cast<uint32_t>(inFlightFences.size()); ++i)
					{
						if (i == currentFrame)
							continue;
						if (!!*inFlightFences[i])
						{
							fencesToWait.push_back(*inFlightFences[i]);
						}
					}
					if (!fencesToWait.empty())
					{
						vk::Result result = waitForFencesSafe(fencesToWait, VK_TRUE);
						if (result != vk::Result::eSuccess)
						{
							std::cerr << "Error: Failed to wait for fences before acceleration structure build: " <<
								vk::to_string(result) << std::endl;
						}
					}
				}

				watchdogProgressLabel.store("Render: buildAccelerationStructures", std::memory_order_relaxed);
				if (buildAccelerationStructures(entities))
				{
					watchdogProgressLabel.store("Render: after buildAccelerationStructures", std::memory_order_relaxed);
					asBuildRequested.store(false, std::memory_order_release);
					asBuildRequestStartNs.store(0, std::memory_order_relaxed);
					// AS build request resolved; restore normal watchdog sensitivity.
					watchdogSuppressed.store(false, std::memory_order_relaxed);
					// Transition the loading UI to a finalizing phase (descriptor cold-init, etc.).
					if (IsLoading())
					{
						SetLoadingPhase(LoadingPhase::Finalizing);
						SetLoadingPhaseProgress(0.0f);
					}

					// The TLAS handle can transition from null -> valid (or change on rebuild).
					// Ensure raster PBR descriptor sets (set 0, binding 11 `tlas`) are rewritten after an AS build
					// so subsequent Raster draws never see an unwritten/stale acceleration-structure descriptor.
					for (auto& kv : entityResources)
					{
						kv.second.pbrFixedBindingsWritten.assign(MAX_FRAMES_IN_FLIGHT, false);
					}
					for (Entity* e : entities)
					{
						MarkEntityDescriptorsDirty(e);
					}

					// Freeze only when the built AS covers essentially the full set of renderable entities.
					// NOTE: `lastASBuiltInstanceCount` is an ENTITY count; TLAS instance count (instancing) is tracked separately.
					if (asFreezeAfterFullBuild)
					{
						const double threshold = 0.95;
						if (totalRenderableEntities > 0 &&
							static_cast<double>(lastASBuiltInstanceCount) >= threshold * static_cast<double>(
								totalRenderableEntities))
						{
							asFrozen = true;
						}
					}

					// One concise TLAS summary with consistent units.
					if (!!*tlasStructure.handle)
					{
						if (IsRayQueryStaticOnly())
						{
							std::cout << "TLAS ready (static-only): tlasInstances=" << lastASBuiltTlasInstanceCount
								<< ", entities=" << lastASBuiltInstanceCount
								<< ", BLAS=" << lastASBuiltBLASCount
								<< ", addr=0x" << std::hex << tlasStructure.deviceAddress << std::dec << std::endl;
						}
						else
						{
							std::cout << "TLAS ready: tlasInstances=" << lastASBuiltTlasInstanceCount
								<< ", entities=" << lastASBuiltInstanceCount
								<< ", BLAS=" << lastASBuiltBLASCount
								<< ", addr=0x" << std::hex << tlasStructure.deviceAddress << std::dec << std::endl;
						}
					}
				}
				else
				{
					if (!accelerationStructureEnabled || !rayQueryEnabled)
					{
						// Permanent failure due to lack of support; do not retry.
						asBuildRequested.store(false, std::memory_order_release);
						asBuildRequestStartNs.store(0, std::memory_order_relaxed);
						watchdogSuppressed.store(false, std::memory_order_relaxed);
					}
					else
					{
						// If nothing is ready yet (e.g., mesh uploads still pending), don't spam logs.
						if (readyRenderableCount > 0 || readyUniqueMeshCount > 0)
						{
							std::cout << "Failed to build acceleration structures, will retry next frame" << std::endl;
						}
					}
				}
				// Reset dev override after one use
				asDevOverrideAllowRebuild = false;
			}
		}
	}

	// Safe point: the previous work referencing this frame's descriptor sets is complete.
	// Apply any deferred descriptor set updates for entities whose textures finished streaming.
	watchdogProgressLabel.store("Render: ProcessDirtyDescriptorsForFrame", std::memory_order_relaxed);
	ProcessDirtyDescriptorsForFrame(currentFrame);
	watchdogProgressLabel.store("Render: after ProcessDirtyDescriptorsForFrame", std::memory_order_relaxed);

	// --- 1. PREPARATION PASS ---
	// Gather active entities with mesh resources, perform per-frame descriptor initialization,
	// and execute culling. This single pass replaces multiple redundant scans and reduces map lookups.
	std::vector<RenderJob> opaqueJobs;
	std::vector<RenderJob> transparentJobs;
	std::vector<FrustumCullAABB> gpuFrustumAabbs;
	opaqueJobs.reserve(entities.size());
	gpuFrustumAabbs.reserve(entities.size());

	FrustumPlanes frustum{};
	glm::mat4 frustumCullVP(1.0f);
	const bool doCulling = enableFrustumCulling && camera;
	if (doCulling)
	{
		glm::mat4 proj = camera->GetProjectionMatrix();
		proj[1][1] *= -1.0f;
		frustumCullVP = proj * camera->GetViewMatrix();
		frustum = extractFrustumPlanes(frustumCullVP);
	}
	lastGpuFrustumInputCount = 0;

	{
		watchdogProgressLabel.store("Render: preparation pass", std::memory_order_relaxed);

		lastCullingVisibleCount = 0;
		lastCullingCulledCount = 0;

		uint32_t entityProcessCount = 0;
		for (Entity* entity : entities)
		{
			if (!entity || !entity->IsActive())
				continue;
			auto meshComponent = entity->GetComponent<MeshComponent>();
			if (!meshComponent)
				continue;

			auto entityIt = entityResources.find(entity);
			if (entityIt == entityResources.end())
				continue;

			auto meshIt = meshResources.find(meshComponent);
			if (meshIt == meshResources.end())
				continue;

			EntityResources& entityRes = entityIt->second;
			MeshResources& meshRes = meshIt->second;

			// Ensure material cache is valid once per frame
			ensureEntityMaterialCache(entity, entityRes);

			// --- Per-frame Descriptor Cold-Init (Integrated) ---
			if (entityRes.basicDescriptorSets.empty() || entityRes.pbrDescriptorSets.empty())
			{
				std::string texPath = meshComponent->GetBaseColorTexturePath();
				if (texPath.empty()) texPath = meshComponent->GetTexturePath();
				if (entityRes.basicDescriptorSets.empty()) createDescriptorSets(entity, entityRes, texPath, false);
				if (entityRes.pbrDescriptorSets.empty()) createDescriptorSets(entity, entityRes, texPath, true);
			}

			// Initialize binding 0 (UBO) for the current frame slot if not already done.
			if (!entityRes.pbrUboBindingWritten[currentFrame] || !entityRes.basicUboBindingWritten[currentFrame])
			{
				std::string texPath = meshComponent->GetBaseColorTexturePath();
				if (texPath.empty()) texPath = meshComponent->GetTexturePath();
				if (!entityRes.pbrUboBindingWritten[currentFrame])
				{
					updateDescriptorSetsForFrame(entity, entityRes, texPath, true, currentFrame, false, true);
				}
				if (!entityRes.basicUboBindingWritten[currentFrame])
				{
					updateDescriptorSetsForFrame(entity, entityRes, texPath, false, currentFrame, false, true);
				}
			}

			// Initialize images for the current frame slot if not already done.
			if (!entityRes.pbrImagesWritten[currentFrame] || !entityRes.basicImagesWritten[currentFrame])
			{
				std::string texPath = meshComponent->GetBaseColorTexturePath();
				if (texPath.empty()) texPath = meshComponent->GetTexturePath();
				if (!entityRes.pbrImagesWritten[currentFrame])
				{
					updateDescriptorSetsForFrame(entity, entityRes, texPath, true, currentFrame, true, false);
					entityRes.pbrImagesWritten[currentFrame] = true;
				}
				if (!entityRes.basicImagesWritten[currentFrame])
				{
					updateDescriptorSetsForFrame(entity, entityRes, texPath, false, currentFrame, true, false);
					entityRes.basicImagesWritten[currentFrame] = true;
				}
			}

			// --- Culling & Classification ---
			auto* tc = entity->GetComponent<TransformComponent>();
			bool useBlended = entityRes.cachedIsBlended;

			if (meshComponent->HasLocalAABB())
			{
				const glm::mat4 model = tc ? tc->GetModelMatrix() : glm::mat4(1.0f);

				if (doCulling)
				{
					const glm::vec3 meshAabbMin = meshComponent->GetBaseMeshAABBMin();
					const glm::vec3 meshAabbMax = meshComponent->GetBaseMeshAABBMax();
					const auto& instances = meshComponent->GetInstances();

					if (instances.empty())
					{
						glm::vec3 instanceWorldMin, instanceWorldMax;
						transformAABB(model, meshAabbMin, meshAabbMax, instanceWorldMin, instanceWorldMax);
						gpuFrustumAabbs.push_back(FrustumCullAABB{
							glm::vec4(instanceWorldMin, 0.0f), glm::vec4(instanceWorldMax, 0.0f)
						});
					}
					else
					{
						for (const auto& instance : instances)
						{
							const glm::mat4 worldModel = model * instance.modelMatrix;
							glm::vec3 instanceWorldMin, instanceWorldMax;
							transformAABB(worldModel, meshAabbMin, meshAabbMax, instanceWorldMin, instanceWorldMax);
							gpuFrustumAabbs.push_back(FrustumCullAABB{
								glm::vec4(instanceWorldMin, 0.0f), glm::vec4(instanceWorldMax, 0.0f)
							});
						}
					}
				}

				glm::vec3 wmin, wmax;
				transformAABB(model, meshComponent->GetLocalAABBMin(), meshComponent->GetLocalAABBMax(), wmin, wmax);

				// 1. Frustum Culling
				if (doCulling && !aabbIntersectsFrustum(wmin, wmax, frustum))
				{
					lastCullingCulledCount++;
					continue;
				}

				// 2. Distance-based LOD
				if (enableDistanceLOD && camera)
				{
					glm::vec3 camPos = camera->GetPosition();
					bool cameraInside = (camPos.x >= wmin.x && camPos.x <= wmax.x &&
						camPos.y >= wmin.y && camPos.y <= wmax.y &&
						camPos.z >= wmin.z && camPos.z <= wmax.z);
					if (!cameraInside)
					{
						float dx = std::max({0.0f, wmin.x - camPos.x, camPos.x - wmax.x});
						float dy = std::max({0.0f, wmin.y - camPos.y, camPos.y - wmax.y});
						float dz = std::max({0.0f, wmin.z - camPos.z, camPos.z - wmax.z});
						float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
						float z_eff = std::max(0.1f, dist);
						float fov = glm::radians(camera->GetFieldOfView());
						float radius = glm::length(0.5f * (wmax - wmin));
						float pixelDiameter = (radius * 2.0f * static_cast<float>(swapChainExtent.height)) / (z_eff *
							2.0f * std::tan(fov * 0.5f));
						float threshold = useBlended ? lodPixelThresholdTransparent : lodPixelThresholdOpaque;
						if (pixelDiameter < threshold)
						{
							lastCullingCulledCount++;
							continue;
						}
					}
				}
			}

			lastCullingVisibleCount++;
			bool isAlphaMasked = false;
			if (entityRes.materialCacheValid)
			{
				isAlphaMasked = (entityRes.cachedMaterialProps.alphaMask > 0.5f);
			}

			// Update UBO for visible entity once per frame (shared across all main passes)
			updateUniformBuffer(currentFrame, entity, &entityRes, camera, tc);

			RenderJob job{entity, &entityRes, &meshRes, meshComponent, tc, isAlphaMasked};
			if (useBlended)
			{
				transparentJobs.push_back(job);
			}
			else
			{
				opaqueJobs.push_back(job);
			}

			// Update watchdog periodically
			if (++entityProcessCount % 100 == 0)
			{
				lastFrameUpdateTime.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);
			}
		}
		watchdogProgressLabel.store("Render: after preparation pass", std::memory_order_relaxed);
	}

	uint32_t frustumCullDispatchCount = 0;
	const bool deferredRasterMode = (currentRenderMode == RenderMode::Deferred);

	// If the scene loader has finished and there are no remaining blocking tasks,
	// hide the fullscreen loading overlay.
	if (IsLoading() && GetLoadingPhase() == LoadingPhase::Finalizing)
	{
		const bool loaderDone = !loadingFlag.load(std::memory_order_relaxed);
		const bool criticalDone = (criticalJobsOutstanding.load(std::memory_order_relaxed) == 0u);
		const bool noASPending = !asBuildRequested.load(std::memory_order_relaxed);
		const bool noPreallocPending = !pendingEntityPreallocQueued.load(std::memory_order_relaxed);
		const bool noDirtyEntities = descriptorDirtyEntities.empty();
		const bool noDeferredDescOps = !descriptorRefreshPending.load(std::memory_order_relaxed);
		if (loaderDone && criticalDone && noASPending && noPreallocPending && noDirtyEntities && noDeferredDescOps)
		{
			MarkInitialLoadComplete();
		}
	}

	// Safe point: flush any descriptor updates that were deferred while a command buffer
	// was recording in a prior frame. Only apply ops for the current frame to avoid
	// update-after-bind on pending frames.
	if (descriptorRefreshPending.load(std::memory_order_relaxed))
	{
		watchdogProgressLabel.store("Render: flush deferred descriptor ops", std::memory_order_relaxed);
		std::vector<PendingDescOp> ops;
		{
			std::lock_guard<std::mutex> lk(pendingDescMutex);
			ops.swap(pendingDescOps);
			descriptorRefreshPending.store(false, std::memory_order_relaxed);
		}
		uint32_t opCount = 0;
		for (auto& op : ops)
		{
			// Kick watchdog periodically during potentially heavy descriptor update bursts
			if ((++opCount % 50u) == 0u)
			{
				lastFrameUpdateTime.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);
			}

			if (op.frameIndex == currentFrame)
			{
				// Now not recording; safe to apply updates for this frame
				updateDescriptorSetsForFrame(op.entity, op.texPath, op.usePBR, op.frameIndex, op.imagesOnly);
			}
			else
			{
				// Keep other frame ops queued for next frame’s safe point
				std::lock_guard<std::mutex> lk(pendingDescMutex);
				pendingDescOps.push_back(op);
				descriptorRefreshPending.store(true, std::memory_order_relaxed);
			}
		}
		watchdogProgressLabel.store("Render: after deferred descriptor ops", std::memory_order_relaxed);
	}

	// Safe point: handle any pending reflection resource (re)creation and per-frame descriptor refreshes
	if (reflectionResourcesDirty)
	{
		if (enablePlanarReflections)
		{
			uint32_t rw = std::max(
				1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.width) * reflectionResolutionScale));
			uint32_t rh = std::max(
				1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.height) * reflectionResolutionScale));
			createReflectionResources(rw, rh);
		}
		else
		{
			destroyReflectionResources();
		}
		reflectionResourcesDirty = false;
	}

	// Reflection descriptor binding refresh is handled elsewhere; avoid redundant per-frame mass updates here.
	// Pick the VP associated with the previous frame's reflection texture for sampling in the main pass
	if (enablePlanarReflections && !reflectionVPs.empty())
	{
		uint32_t prev = (currentFrame > 0) ? (currentFrame - 1) : (static_cast<uint32_t>(reflectionVPs.size()) - 1);
		sampleReflectionVP = reflectionVPs[prev];
	}

	// This function updates bindings 6/7/8 (storage buffers) which don't have UPDATE_AFTER_BIND.
	// Updating these every frame causes "updated without UPDATE_AFTER_BIND" errors with MAX_FRAMES_IN_FLIGHT > 1.
	// These bindings are already initialized in createDescriptorSets and updated when buffers change.
	// Binding 10 (reflection map) has UPDATE_AFTER_BIND and can be updated separately if needed.
	// refreshPBRForwardPlusBindingsForFrame(currentFrame);

	// Acquire next swapchain image
	// acquireNextImage returns imageIndex (which swapchain image is available).
	// Use currentFrame to select an imageAvailableSemaphore for acquire.
	// Use imageIndex to select renderFinishedSemaphore for present (ties semaphore to the specific image).
	const uint32_t acquireSemaphoreIndex = currentFrame % static_cast<uint32_t>(imageAvailableSemaphores.size());

	uint32_t imageIndex;
	vk::Result acquireResultCode = vk::Result::eSuccess;
	// Helper overloads to normalize acquireNextImage return across Vulkan-Hpp versions
	auto extractAcquire = [](auto const& ret, vk::Result& code, uint32_t& idx)
	{
		using RetT = std::decay_t<decltype(ret)>;
		if constexpr (std::is_same_v<RetT, vk::ResultValue<uint32_t>>)
		{
			code = ret.result;
			idx = ret.value;
		}
		else
		{
			// Assume older std::pair<vk::Result, uint32_t>
			code = ret.first;
			idx = ret.second;
		}
	};
	try
	{
		watchdogProgressLabel.store("Render: acquireNextImage", std::memory_order_relaxed);
		auto acquireRet = swapChain.acquireNextImage(UINT64_MAX, *imageAvailableSemaphores[acquireSemaphoreIndex]);
		// Vulkan-Hpp changed the return type of acquireNextImage for RAII swapchain across versions.
		// Support both vk::ResultValue<uint32_t> (newer) and std::pair<vk::Result, uint32_t> (older).
		extractAcquire(acquireRet, acquireResultCode, imageIndex);
	}
	catch (const vk::OutOfDateKHRError&)
	{
		watchdogProgressLabel.store("Render: acquireNextImage out-of-date", std::memory_order_relaxed);
		// Swapchain is out of date (e.g., window resized) before we could
		// query the result. Trigger recreation and exit this frame cleanly.
		framebufferResized.store(true, std::memory_order_relaxed);
		if (imguiSystem)
			ImGui::EndFrame();
		// IMPORTANT: We already reset the in-flight fence at the start of the frame.
		// Because we're exiting early (no submit), signal it via an empty submit so
		// swapchain recreation won't hang waiting for an unsignaled fence.
		{
			vk::SubmitInfo2 emptySubmit2{};
			std::lock_guard<std::mutex> lock(queueMutex);
			graphicsQueue.submit2(emptySubmit2, *inFlightFences[currentFrame]);
		}
		recreateSwapChain();
		return;
	}

	// imageIndex already populated above
	watchdogProgressLabel.store("Render: acquired swapchain image", std::memory_order_relaxed);

	if (acquireResultCode == vk::Result::eSuboptimalKHR || framebufferResized.load(std::memory_order_relaxed))
	{
		framebufferResized.store(false, std::memory_order_relaxed);
		if (imguiSystem)
			ImGui::EndFrame();
		// Fence was reset earlier; ensure it is signaled before we bail out
		// to avoid a deadlock in swapchain recreation.
		{
			vk::SubmitInfo2 emptySubmit2{};
			std::lock_guard<std::mutex> lock(queueMutex);
			graphicsQueue.submit2(emptySubmit2, *inFlightFences[currentFrame]);
		}
		recreateSwapChain();
		return;
	}
	if (acquireResultCode != vk::Result::eSuccess)
	{
		throw std::runtime_error("Failed to acquire swap chain image");
	}

	if (framebufferResized.load(std::memory_order_relaxed))
	{
		// Signal the fence via empty submit since no real work will be submitted
		// this frame, preventing a wait on an unsignaled fence during resize.
		{
			vk::SubmitInfo2 emptySubmit2{};
			std::lock_guard<std::mutex> lock(queueMutex);
			graphicsQueue.submit2(emptySubmit2, *inFlightFences[currentFrame]);
		}
		recreateSwapChain();
		return;
	}

	// Perform any descriptor updates that must not happen during command buffer recording
	if (useForwardPlus)
	{
		uint32_t tilesX_pre = (swapChainExtent.width + forwardPlusTileSizeX - 1) / forwardPlusTileSizeX;
		uint32_t tilesY_pre = (swapChainExtent.height + forwardPlusTileSizeY - 1) / forwardPlusTileSizeY;
		// Only update current frame's descriptors to avoid touching in-flight frames
		createOrResizeForwardPlusBuffers(tilesX_pre, tilesY_pre, forwardPlusSlicesZ, /*updateOnlyCurrentFrame=*/true);
		// After (re)creating Forward+ buffers, bindings 7/8 will be refreshed as needed.
	}

	if (deferredRasterMode && doCulling && !gpuFrustumAabbs.empty())
	{
		const uint32_t dispatchCount = static_cast<uint32_t>(gpuFrustumAabbs.size());
		if (createOrResizeFrustumCullBuffers(dispatchCount, /*updateOnlyCurrentFrame=*/true) &&
			currentFrame < frustumCullPerFrame.size())
		{
			auto& frustumFrame = frustumCullPerFrame[currentFrame];
			if (frustumFrame.instanceAabbMapped && frustumFrame.visibleCountMapped)
			{
				std::memcpy(frustumFrame.instanceAabbMapped, gpuFrustumAabbs.data(),
				            gpuFrustumAabbs.size() * sizeof(FrustumCullAABB));
				const uint32_t zero = 0;
				std::memcpy(frustumFrame.visibleCountMapped, &zero, sizeof(uint32_t));
				frustumCullDispatchCount = dispatchCount;
				lastGpuFrustumInputCount = dispatchCount;
			}
		}
	}

	// Ensure light buffers are sufficiently large before recording to avoid resizing while in use
	{
		// Reserve capacity based on emissive lights only (punctual lights disabled for now)
		size_t desiredLightCapacity = 0;
		if (!staticLights.empty())
		{
			size_t emissiveCount = 0;
			for (const auto& L : staticLights)
			{
				if (L.type == ExtractedLight::Type::Emissive)
				{
					++emissiveCount;
					if (emissiveCount >= MAX_ACTIVE_LIGHTS)
						break;
				}
			}
			desiredLightCapacity = emissiveCount;
		}
		if (desiredLightCapacity > 0)
		{
			createOrResizeLightStorageBuffers(desiredLightCapacity);
			// Ensure compute (binding 0) sees the current frame's lights buffer
			refreshForwardPlusComputeLightsBindingForFrame(currentFrame);
			// Bindings 6/7/8 for PBR are refreshed only when buffers change (handled in resize path).
		}
	}

	// Safe point: Update ray query descriptor sets if ray query mode is active
	// This MUST happen before command buffer recording starts to avoid "descriptor updated without UPDATE_AFTER_BIND" errors
	if (currentRenderMode == RenderMode::RayQuery && rayQueryEnabled && accelerationStructureEnabled)
	{
		if (!!*tlasStructure.handle)
		{
			watchdogProgressLabel.store("Render: updateRayQueryDescriptorSets", std::memory_order_relaxed);
			updateRayQueryDescriptorSets(currentFrame, entities);
			watchdogProgressLabel.store("Render: after updateRayQueryDescriptorSets", std::memory_order_relaxed);
		}
	}

	// Refit TLAS if needed (either for Ray Query mode or for Raster shadows)
	const bool needTLAS = (currentRenderMode == RenderMode::RayQuery || enableRasterRayQueryShadows) &&
		accelerationStructureEnabled;
	if (needTLAS && !!*tlasStructure.handle)
	{
		if (!IsRayQueryStaticOnly())
		{
			watchdogProgressLabel.store("Render: refitTopLevelAS", std::memory_order_relaxed);
			refitTopLevelAS(entities, camera);
		}
	}

	commandBuffers[currentFrame].reset();
	// Begin command buffer recording for this frame
	commandBuffers[currentFrame].begin(vk::CommandBufferBeginInfo());
	resetGpuProfilingQueries(commandBuffers[currentFrame], currentFrame);
	isRecordingCmd.store(true, std::memory_order_relaxed);
	if (framebufferResized.load(std::memory_order_relaxed))
	{
		commandBuffers[currentFrame].end();
		recreateSwapChain();
		return;
	}

	if (deferredRasterMode && enableFrustumCulling && frustumCullDispatchCount > 0)
	{
		writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::FrustumCull, true);
		updateFrustumCullParams(currentFrame, frustumCullVP, frustumCullDispatchCount);
		dispatchFrustumCull(commandBuffers[currentFrame], frustumCullDispatchCount);
		writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::FrustumCull, false);
	}

	// Ray query rendering mode dispatch
	if (currentRenderMode == RenderMode::RayQuery && rayQueryEnabled && accelerationStructureEnabled)
	{
		writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::RayQuery, true);
		// Check if TLAS handle is valid (dereference RAII handle)
		if (!*tlasStructure.handle)
		{
			// TLAS not built yet.
			// During loading, allow the raster path (and the progress overlay) to render normally
			// instead of presenting a diagnostic magenta frame.
			if (!IsLoading())
			{
				// Present a diagnostic frame from the ray-query path to avoid accidentally showing
				// rasterized content in RayQuery mode.
				// Transition swapchain image from PRESENT to TRANSFER_DST
				vk::ImageMemoryBarrier2 swapchainBarrier{};
				swapchainBarrier.srcStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
				swapchainBarrier.srcAccessMask = vk::AccessFlagBits2::eNone;
				swapchainBarrier.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
				swapchainBarrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
				swapchainBarrier.oldLayout = (imageIndex < swapChainImageLayouts.size())
					                             ? swapChainImageLayouts[imageIndex]
					                             : vk::ImageLayout::eUndefined;
				swapchainBarrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
				swapchainBarrier.image = swapChainImages[imageIndex];
				swapchainBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
				swapchainBarrier.subresourceRange.levelCount = 1;
				swapchainBarrier.subresourceRange.layerCount = 1;

				vk::DependencyInfo depInfoSwap{};
				depInfoSwap.imageMemoryBarrierCount = 1;
				depInfoSwap.pImageMemoryBarriers = &swapchainBarrier;
				commandBuffers[currentFrame].pipelineBarrier2(depInfoSwap);
				if (imageIndex < swapChainImageLayouts.size())
					swapChainImageLayouts[imageIndex] = swapchainBarrier.newLayout;

				// Clear to a distinct magenta diagnostic color
				vk::ClearColorValue clearColor{std::array<float, 4>{1.0f, 0.0f, 1.0f, 1.0f}};
				vk::ImageSubresourceRange clearRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
				commandBuffers[currentFrame].clearColorImage(swapChainImages[imageIndex],
				                                             vk::ImageLayout::eTransferDstOptimal, clearColor,
				                                             clearRange);

				// Transition back to PRESENT
				swapchainBarrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
				swapchainBarrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
				swapchainBarrier.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
				swapchainBarrier.dstAccessMask = vk::AccessFlagBits2::eNone;
				swapchainBarrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
				swapchainBarrier.newLayout = vk::ImageLayout::ePresentSrcKHR;
				commandBuffers[currentFrame].pipelineBarrier2(depInfoSwap);
				if (imageIndex < swapChainImageLayouts.size())
					swapChainImageLayouts[imageIndex] = swapchainBarrier.newLayout;

				rayQueryRenderedThisFrame = true; // Skip raster; ensure we are looking at RQ path only
			}
		}
		else
		{
			// TLAS is valid and descriptor sets were already updated at safe point
			// Proceed with ray query rendering
			// Bind ray query compute pipeline
			commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, *rayQueryPipeline);

			// Bind descriptor set
			commandBuffers[currentFrame].bindDescriptorSets(
				vk::PipelineBindPoint::eCompute,
				*rayQueryPipelineLayout,
				0,
				*rayQueryDescriptorSets[currentFrame],
				nullptr);

			// This dedicated UBO is separate from entity UBOs and uses a Ray Query-specific layout.
			if (rayQueryUniformBuffersMapped.size() > currentFrame && rayQueryUniformBuffersMapped[currentFrame])
			{
				RayQueryUniformBufferObject ubo{};
				ubo.model = glm::mat4(1.0f); // Identity - not used for ray query

				// Force view matrix update to reflect current camera position
				// (the dirty flag isn't automatically set when camera position changes)
				camera->ForceViewMatrixUpdate();

				// Get camera matrices
				glm::mat4 camView = camera->GetViewMatrix();
				ubo.view = camView;
				ubo.proj = camera->GetProjectionMatrix();
				ubo.proj[1][1] *= -1; // Flip Y for Vulkan
				ubo.camPos = glm::vec4(camera->GetPosition(), 1.0f);
				// Clamp to sane ranges to avoid black output (exposure=0 → 1-exp(0)=0)
				ubo.exposure = std::clamp(exposure, 0.2f, 4.0f);
				ubo.gamma = std::clamp(gamma, 1.6f, 2.6f);
				// Match raster convention: ambient scale factor for simple IBL/ambient term.
				// (Raster defaults to ~1.0 in the main pass; keep Ray Query consistent.)
				ubo.scaleIBLAmbient = 1.0f;
				// Provide the per-frame light count so the ray query shader can iterate lights.
				ubo.lightCount = static_cast<int>(lastFrameLightCount);
				ubo.screenDimensions = glm::vec2(swapChainExtent.width, swapChainExtent.height);
				ubo.enableRayQueryReflections = enableRayQueryReflections ? 1 : 0;
				ubo.enableRayQueryTransparency = enableRayQueryTransparency ? 1 : 0;
				// Max secondary bounces (reflection/refraction). Stored in the padding slot to avoid UBO layout churn.
				// Shader clamps this value.
				ubo._pad0 = rayQueryMaxBounces;
				// Thick-glass toggles and tuning
				ubo.enableThickGlass = enableThickGlass ? 1 : 0;
				ubo.thicknessClamp = thickGlassThicknessClamp;
				ubo.absorptionScale = thickGlassAbsorptionScale;
				// Ray Query hard shadows (see `shaders/ray_query.slang`)
				ubo._pad1 = enableRayQueryShadows ? 1 : 0;
				ubo.shadowSampleCount = std::clamp(rayQueryShadowSampleCount, 1, 32);
				ubo.shadowSoftness = std::clamp(rayQueryShadowSoftness, 0.0f, 1.0f);
				ubo.reflectionIntensity = reflectionIntensity;
				// Provide geometry info count for shader-side bounds checking (per-instance)
				ubo.geometryInfoCount = static_cast<int>(tlasInstanceCount);
				// Provide material buffer count for shader-side bounds checking
				ubo.materialCount = static_cast<int>(materialCountCPU);

				// Copy to mapped memory
				std::memcpy(rayQueryUniformBuffersMapped[currentFrame], &ubo, sizeof(RayQueryUniformBufferObject));
			}
			else
			{
				// Keep concise error for visibility
				std::cerr << "Ray Query UBO not mapped for frame " << currentFrame << "\n";
			}

			// Dispatch compute shader (8x8 workgroups as defined in shader)
			uint32_t workgroupsX = (swapChainExtent.width + 7) / 8;
			uint32_t workgroupsY = (swapChainExtent.height + 7) / 8;
			commandBuffers[currentFrame].dispatch(workgroupsX, workgroupsY, 1);

			// Barrier: wait for compute shader to finish writing to output image,
			// then make it readable by fragment shader for sampling in composite pass
			vk::ImageMemoryBarrier2 rqToSample{};
			rqToSample.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
			rqToSample.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
			rqToSample.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
			rqToSample.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
			rqToSample.oldLayout = vk::ImageLayout::eGeneral;
			rqToSample.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
			rqToSample.image = *rayQueryOutputImage;
			rqToSample.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
			rqToSample.subresourceRange.levelCount = 1;
			rqToSample.subresourceRange.layerCount = 1;

			vk::DependencyInfo depRQToSample{};
			depRQToSample.imageMemoryBarrierCount = 1;
			depRQToSample.pImageMemoryBarriers = &rqToSample;
			commandBuffers[currentFrame].pipelineBarrier2(depRQToSample);

			// Composite fullscreen: sample rayQueryOutputImage to the swapchain using the composite pipeline
			// Transition swapchain image to COLOR_ATTACHMENT_OPTIMAL
			vk::ImageMemoryBarrier2 swapchainToColor{};
			swapchainToColor.srcStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
			swapchainToColor.srcAccessMask = vk::AccessFlagBits2::eNone;
			swapchainToColor.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
			swapchainToColor.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite |
				vk::AccessFlagBits2::eColorAttachmentRead;
			swapchainToColor.oldLayout = (imageIndex < swapChainImageLayouts.size())
				                             ? swapChainImageLayouts[imageIndex]
				                             : vk::ImageLayout::eUndefined;
			swapchainToColor.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
			swapchainToColor.image = swapChainImages[imageIndex];
			swapchainToColor.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
			swapchainToColor.subresourceRange.levelCount = 1;
			swapchainToColor.subresourceRange.layerCount = 1;
			vk::DependencyInfo depSwapToColor{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &swapchainToColor};
			commandBuffers[currentFrame].pipelineBarrier2(depSwapToColor);
			if (imageIndex < swapChainImageLayouts.size())
				swapChainImageLayouts[imageIndex] = swapchainToColor.newLayout;

			// Begin dynamic rendering for composite (no depth)
			colorAttachments[0].imageView = *swapChainImageViews[imageIndex];
			colorAttachments[0].loadOp = vk::AttachmentLoadOp::eClear;
			depthAttachment.loadOp = vk::AttachmentLoadOp::eDontCare;
			renderingInfo.renderArea = vk::Rect2D({0, 0}, swapChainExtent);
			auto savedDepthPtr2 = renderingInfo.pDepthAttachment;
			renderingInfo.pDepthAttachment = nullptr;
			commandBuffers[currentFrame].beginRendering(renderingInfo);

			if (!!*compositePipeline)
			{
				commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *compositePipeline);
			}
			vk::Viewport vp(0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
			                static_cast<float>(swapChainExtent.height), 0.0f, 1.0f);
			vk::Rect2D sc({0, 0}, swapChainExtent);
			commandBuffers[currentFrame].setViewport(0, vp);
			commandBuffers[currentFrame].setScissor(0, sc);

			// Bind the RQ composite descriptor set (samples rayQueryOutputImage)
			if (!rqCompositeDescriptorSets.empty())
			{
				commandBuffers[currentFrame].bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					*compositePipelineLayout,
					0,
					{*rqCompositeDescriptorSets[currentFrame]},
					{});
			}

			CompositePushConstants pc2{};
			pc2.exposure = std::clamp(this->exposure, 0.2f, 4.0f);
			pc2.gamma = this->gamma;
			pc2.outputIsSRGB = (swapChainImageFormat == vk::Format::eR8G8B8A8Srgb || swapChainImageFormat ==
				                   vk::Format::eB8G8R8A8Srgb)
				                   ? 1
				                   : 0;
			pc2.enableTAA = 0;
			pc2.taaHistoryValid = 0;
			pc2.enableSAO = 0;
			pc2.saoValid = 0;
			pc2.enableVolumetric = 0;
			pc2.volumetricValid = 0;
			pc2.taaFeedback = 0.1f;
			commandBuffers[currentFrame].pushConstants<CompositePushConstants>(
				*compositePipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, pc2);

			commandBuffers[currentFrame].draw(3, 1, 0, 0);
			commandBuffers[currentFrame].endRendering();
			renderingInfo.pDepthAttachment = savedDepthPtr2;

			// Transition swapchain back to PRESENT and RQ image back to GENERAL for next frame
			vk::ImageMemoryBarrier2 swapchainToPresent{};
			swapchainToPresent.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
			swapchainToPresent.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
			swapchainToPresent.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
			swapchainToPresent.dstAccessMask = vk::AccessFlagBits2::eNone;
			swapchainToPresent.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
			swapchainToPresent.newLayout = vk::ImageLayout::ePresentSrcKHR;
			swapchainToPresent.image = swapChainImages[imageIndex];
			swapchainToPresent.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
			swapchainToPresent.subresourceRange.levelCount = 1;
			swapchainToPresent.subresourceRange.layerCount = 1;

			vk::ImageMemoryBarrier2 rqBackToGeneral{};
			rqBackToGeneral.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
			rqBackToGeneral.srcAccessMask = vk::AccessFlagBits2::eShaderRead;
			rqBackToGeneral.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
			rqBackToGeneral.dstAccessMask = vk::AccessFlagBits2::eShaderWrite;
			rqBackToGeneral.oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
			rqBackToGeneral.newLayout = vk::ImageLayout::eGeneral;
			rqBackToGeneral.image = *rayQueryOutputImage;
			rqBackToGeneral.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
			rqBackToGeneral.subresourceRange.levelCount = 1;
			rqBackToGeneral.subresourceRange.layerCount = 1;

			std::array<vk::ImageMemoryBarrier2, 2> barriers{swapchainToPresent, rqBackToGeneral};
			vk::DependencyInfo depEnd{
				.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
				.pImageMemoryBarriers = barriers.data()
			};
			commandBuffers[currentFrame].pipelineBarrier2(depEnd);
			if (imageIndex < swapChainImageLayouts.size())
				swapChainImageLayouts[imageIndex] = swapchainToPresent.newLayout;

			// Ray query rendering complete - set flag to skip rasterization code path
			rayQueryRenderedThisFrame = true;
		}
		writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::RayQuery, false);
	}

	// Process texture streaming uploads (see Renderer::ProcessPendingTextureJobs)

	vk::raii::Pipeline* currentPipeline = nullptr;
	vk::raii::PipelineLayout* currentLayout = nullptr;

	// Incrementally process pending texture uploads on the main thread so that
	// all Vulkan submits happen from a single place while worker threads only
	// handle CPU-side decoding. While the loading screen is up, prioritize
	// critical textures so the first rendered frame looks mostly correct.
	if (IsLoading())
	{
		// Larger budget while loading screen is visible so we don't stall
		// streaming of near-field baseColor textures.
		ProcessPendingTextureJobs(/*maxJobs=*/16, /*includeCritical=*/true, /*includeNonCritical=*/false);
	}
	else
	{
		// After loading screen disappears, we want the scene to remain
		// responsive (~20 fps) while textures stream in. Limit the number
		// of non-critical uploads per frame so we don't tank frame time.
		static uint32_t streamingFrameCounter = 0;
		streamingFrameCounter++;
		// Ray Query needs textures visible quickly; process more streaming work when in Ray Query mode.
		if (currentRenderMode == RenderMode::RayQuery)
		{
			// Aggressively drain both critical and non-critical queues each frame for faster bring-up.
			ProcessPendingTextureJobs(/*maxJobs=*/32, /*includeCritical=*/true, /*includeNonCritical=*/true);
		}
		else
		{
			// Raster path: keep previous throttling to avoid stalls.
			if ((streamingFrameCounter % 3) == 0)
			{
				ProcessPendingTextureJobs(/*maxJobs=*/1, /*includeCritical=*/false, /*includeNonCritical=*/true);
			}
		}
	}

	// Renderer UI - available for both ray query and rasterization modes.
	// Hide UI during loading; the progress overlay is handled by ImGuiSystem::NewFrame().
	if (imguiSystem && !imguiSystem->IsFrameRendered() && !IsLoading())
	{
		if (ImGui::Begin("Renderer"))
		{
			// Declare variables that need to persist across conditional blocks
			bool prevFwdPlus = useForwardPlus;

			// === RENDERING MODE SELECTION (TOP) ===
			ImGui::Text("Rendering Mode:");
			if (rayQueryEnabled && accelerationStructureEnabled)
			{
				const char* modeNames[] = {"Rasterization", "Deferred", "Ray Query"};
				int currentMode = 0;
				if (currentRenderMode == RenderMode::Deferred)
				{
					currentMode = 1;
				}
				else if (currentRenderMode == RenderMode::RayQuery)
				{
					currentMode = 2;
				}
				if (ImGui::Combo("Mode", &currentMode, modeNames, 3))
				{
					RenderMode newMode = RenderMode::Rasterization;
					if (currentMode == 1)
					{
						newMode = RenderMode::Deferred;
					}
					else if (currentMode == 2)
					{
						newMode = RenderMode::RayQuery;
					}
					if (newMode != currentRenderMode)
					{
						currentRenderMode = newMode;
						std::cout << "Switched to " << modeNames[currentMode] << " mode\n";

						// Request acceleration structure build when switching to ray query mode
						if (currentRenderMode == RenderMode::RayQuery)
						{
							std::cout << "Requesting acceleration structure build...\n";
							RequestAccelerationStructureBuild();
						}

						// Switching modes can change which pipelines are bound and whether ray-query-dependent
						// descriptor bindings (e.g., PBR binding 11 `tlas`) become statically used.
						// Mark entity descriptor sets dirty so the next safe point refreshes bindings for this frame.
						for (auto& kv : entityResources)
						{
							kv.second.pbrFixedBindingsWritten.assign(MAX_FRAMES_IN_FLIGHT, false);
						}
						for (Entity* e : entities)
						{
							MarkEntityDescriptorsDirty(e);
						}
					}
				}
			}
			else
			{
				const char* modeNames[] = {"Rasterization", "Deferred"};
				int currentMode = (currentRenderMode == RenderMode::Deferred) ? 1 : 0;
				if (ImGui::Combo("Mode", &currentMode, modeNames, 2))
				{
					currentRenderMode = (currentMode == 1) ? RenderMode::Deferred : RenderMode::Rasterization;
					std::cout << "Switched to " << modeNames[currentMode] << " mode\n";
				}
				ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Ray query not supported on this device");
			}

			// === RASTERIZATION-SPECIFIC OPTIONS ===
			if (currentRenderMode == RenderMode::Rasterization)
			{
				ImGui::Separator();
				ImGui::Text("Rasterization Options:");

				// Lighting Controls - BRDF/PBR is now the default lighting model
				bool useBasicLighting = imguiSystem && !imguiSystem->IsPBREnabled();
				if (ImGui::Checkbox("Use Basic Lighting (Phong)", &useBasicLighting))
				{
					imguiSystem->SetPBREnabled(!useBasicLighting);
					std::cout << "Lighting mode: " << (!useBasicLighting ? "BRDF/PBR (default)" : "Basic Phong") <<
						std::endl;
				}

				if (!useBasicLighting)
				{
					ImGui::Text("Status: BRDF/PBR pipeline active (default)");
					ImGui::Text("All models rendered with physically-based lighting");
				}
				else
				{
					ImGui::Text("Status: Basic Phong pipeline active");
					ImGui::Text("All models rendered with basic Phong shading");
				}

				ImGui::Checkbox("Forward+ (tiled light culling)", &useForwardPlus);
				if (useForwardPlus && !prevFwdPlus)
				{
					// Lazily create Forward+ resources if enabled at runtime
					if (!*forwardPlusPipeline || !*forwardPlusDescriptorSetLayout || forwardPlusPerFrame.empty())
					{
						createForwardPlusPipelinesAndResources();
					}
					if (!*depthPrepassPipeline)
					{
						createDepthPrepassPipeline();
					}
				}

				// Raster shadows via ray queries (experimental)
				if (rayQueryEnabled && accelerationStructureEnabled)
				{
					ImGui::Checkbox("RayQuery shadows (raster)", &enableRasterRayQueryShadows);
				}
				else
				{
					ImGui::TextDisabled("RayQuery shadows (raster) (requires ray query + AS)");
				}

				// Planar reflections controls
				ImGui::Spacing();
				/*
				if (ImGui::Checkbox("Planar reflections (experimental)", &enablePlanarReflections)) {
				  // Defer actual (re)creation/destruction to the next safe point at frame start
				  reflectionResourcesDirty = true;
				}
				*/
				enablePlanarReflections = false;
				float scaleBefore = reflectionResolutionScale;
				if (ImGui::SliderFloat("Reflection resolution scale", &reflectionResolutionScale, 0.25f, 1.0f, "%.2f"))
				{
					reflectionResolutionScale = std::clamp(reflectionResolutionScale, 0.25f, 1.0f);
					if (enablePlanarReflections && std::abs(scaleBefore - reflectionResolutionScale)
						>
						1e-3f
					)
					{
						reflectionResourcesDirty = true;
					}
				}
				if (enablePlanarReflections && !reflections.empty())
				{
					auto& rt = reflections[currentFrame];
					if (rt.width > 0)
					{
						ImGui::Text("Reflection RT: %ux%u", rt.width, rt.height);
					}
				}
			}

			// === RAY QUERY-SPECIFIC OPTIONS ===
			if (currentRenderMode == RenderMode::RayQuery && rayQueryEnabled && accelerationStructureEnabled)
			{
				ImGui::Separator();
				ImGui::Text("Ray Query Status:");

				// Show acceleration structure status
				if (!!*tlasStructure.handle)
				{
					ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Acceleration Structures: Built (%zu meshes)",
					                   blasStructures.size());
				}
				else
				{
					ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Acceleration Structures: Not built");
				}

				ImGui::Spacing();
				ImGui::Text("Ray Query Features:");
				ImGui::Checkbox("Enable Hard Shadows", &enableRayQueryShadows);
				if (enableRayQueryShadows)
				{
					ImGui::SliderInt("Shadow samples", &rayQueryShadowSampleCount, 1, 32);
					ImGui::SliderFloat("Shadow softness (fraction of range)", &rayQueryShadowSoftness, 0.0f, 0.2f,
					                   "%.3f");
				}
				ImGui::Checkbox("Enable Reflections", &enableRayQueryReflections);
				ImGui::Checkbox("Enable Transparency/Refraction", &enableRayQueryTransparency);
				ImGui::SliderInt("Max secondary bounces", &rayQueryMaxBounces, 0, 10);
				// Thick-glass realism controls
				ImGui::Separator();
				ImGui::Text("Thick Glass");
				ImGui::Checkbox("Enable Thick Glass", &enableThickGlass);
				ImGui::SliderFloat("Thickness Clamp (m)", &thickGlassThicknessClamp, 0.0f, 0.5f, "%.3f");
				ImGui::SliderFloat("Absorption Scale", &thickGlassAbsorptionScale, 0.0f, 4.0f, "%.2f");
			}

			// === SHARED OPTIONS (BOTH MODES) ===
			ImGui::Separator();
			ImGui::Text("Culling & LOD:");
			if (ImGui::Checkbox("Frustum culling", &enableFrustumCulling))
			{
				// no-op, takes effect immediately
			}
			if (ImGui::Checkbox("Distance LOD (projected-size skip)", &enableDistanceLOD))
			{
			}
			ImGui::SliderFloat("LOD threshold opaque (px)", &lodPixelThresholdOpaque, 0.5f, 8.0f, "%.1f");
			ImGui::SliderFloat("LOD threshold transparent (px)", &lodPixelThresholdTransparent, 0.5f, 12.0f, "%.1f");
			// Anisotropy control (recreate samplers on change)
			{
				float deviceMaxAniso = physicalDevice.getProperties().limits.maxSamplerAnisotropy;
				if (ImGui::SliderFloat("Sampler max anisotropy", &samplerMaxAnisotropy, 1.0f, deviceMaxAniso, "%.1f"))
				{
					// Recreate samplers for all textures to apply new anisotropy
					std::unique_lock<std::shared_mutex> texLock(textureResourcesMutex);
					for (auto& kv : textureResources)
					{
						createTextureSampler(kv.second);
					}
					// Default texture
					createTextureSampler(defaultTextureResources);
				}
			}
			if (lastCullingVisibleCount + lastCullingCulledCount > 0)
			{
				ImGui::Text("Culling: visible=%u, culled=%u", lastCullingVisibleCount, lastCullingCulledCount);
			}

			// Basic tone mapping controls
			ImGui::Separator();
			ImGui::Text("Tone Mapping & Tuning:");
			ImGui::SliderFloat("Reflection intensity", &reflectionIntensity, 0.0f, 2.0f, "%.2f");
			ImGui::SliderFloat("Exposure", &exposure, 0.1f, 4.0f, "%.2f");
			ImGui::SliderFloat("Gamma", &gamma, 1.6f, 2.6f, "%.2f");

			ImGui::Separator();
			ImGui::Text("GPU Profiling:");
			if (!gpuProfilingSupported)
			{
				ImGui::TextDisabled("Timestamp queries unavailable on this GPU/queue.");
			}
			else
			{
				bool profilingEnabled = gpuProfilingEnabled;
				if (ImGui::Checkbox("Enable pass timing", &profilingEnabled))
				{
					gpuProfilingEnabled = profilingEnabled;
					if (!gpuProfilingEnabled)
					{
						gpuProfilingLastCompleted = GpuProfilingFrameStats{};
						gpuProfilingLastFrameValid = false;
						gpuProfilingLastFrameMs = 0.0f;
					}
				}

				ImGui::SliderFloat("GPU frame budget (ms)", &gpuProfilingLastFrameBudgetMs, 8.0f, 40.0f, "%.2f");
				ImGui::Text("Samples: %llu", static_cast<unsigned long long>(gpuProfilingSamplesCollected));
				if (gpuProfilingLastFrameValid)
				{
					const bool overBudget = gpuProfilingLastFrameMs > gpuProfilingLastFrameBudgetMs;
					ImGui::TextColored(overBudget ? ImVec4(1.0f, 0.35f, 0.25f, 1.0f) : ImVec4(0.35f, 1.0f, 0.45f, 1.0f),
					                   "Frame GPU: %.3f ms",
					                   gpuProfilingLastFrameMs);
				}
				else
				{
					ImGui::TextDisabled("Frame GPU: n/a");
				}

				if (gpuProfilingEnabled && ImGui::CollapsingHeader("Pass timings", ImGuiTreeNodeFlags_DefaultOpen))
				{
					for (uint32_t passIndex = 0; passIndex < static_cast<uint32_t>(GPU_PROFILE_PASS_COUNT); ++passIndex)
					{
						if (gpuProfilingLastCompleted.passValid[passIndex] != 0u)
						{
							ImGui::Text("%s: %.3f ms", gpuProfilePassName(passIndex),
							            gpuProfilingLastCompleted.passMs[passIndex]);
						}
					}
				}
			}

			ImGui::Separator();
			ImGui::Text("Integration Checks:");
			auto drawIntegrationCheck = [](const char* label, bool pass, const char* degradedNote)
			{
				if (pass)
				{
					ImGui::TextColored(ImVec4(0.35f, 1.0f, 0.45f, 1.0f), "%s: OK", label);
				}
				else
				{
					ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "%s: Fallback", label);
					ImGui::TextDisabled("%s", degradedNote);
				}
			};
			const bool deferredModeActive = (currentRenderMode == RenderMode::Deferred);
			auto drawDeferredIntegrationCheck = [&](const char* label, bool pass, const char* degradedNote)
			{
				if (deferredModeActive)
				{
					drawIntegrationCheck(label, pass, degradedNote);
				}
				else
				{
					ImGui::TextDisabled("%s: Deferred only", label);
				}
			};

			if (!deferredModeActive)
			{
				ImGui::TextDisabled("Deferred integration checks are inactive in this mode.");
			}

			const bool depthPyramidReady = depthPyramidHistoryValid && !!*depthPyramidImage && !!*depthPyramidFullView
				&& !!*depthPyramidSampler &&
				(depthPyramidMipCount > 0u);
			drawDeferredIntegrationCheck("Depth pyramid", depthPyramidReady,
			                             "Occlusion, SAO, and volumetrics may run in reduced mode.");

			const bool csmReady = !enableCascadedShadowMaps || csmDataValid;
			drawDeferredIntegrationCheck("CSM", csmReady, "Directional shadows use non-CSM fallback.");

			const bool taaReady = !taaJitterEnabled ||
			(taaHistoryValid && taaHistoryReadIndex < taaHistoryImageViews.size() &&
				taaHistoryReadIndex < taaHistoryImageLayouts.size() &&
				taaHistoryImageLayouts[taaHistoryReadIndex] == vk::ImageLayout::eShaderReadOnlyOptimal && !!*
				taaHistorySampler);
			drawDeferredIntegrationCheck("TAA history", taaReady, "Composite pass uses current frame color only.");

			const bool saoReady = !enableSAO ||
			(saoHistoryValid && currentFrame < saoImageViews.size() && currentFrame < saoImageLayouts.size() &&
				saoImageLayouts[currentFrame] == vk::ImageLayout::eShaderReadOnlyOptimal && !!*saoSampler);
			drawDeferredIntegrationCheck("SAO", saoReady, "Ambient occlusion contribution is skipped.");

			const bool volumetricReady = !enableVolumetricScattering ||
			(volumetricHistoryValid && currentFrame < volumetricImageViews.size() &&
				currentFrame < volumetricImageLayouts.size() &&
				volumetricImageLayouts[currentFrame] == vk::ImageLayout::eShaderReadOnlyOptimal && !!*
				volumetricSampler);
			drawDeferredIntegrationCheck("Volumetric", volumetricReady,
			                             "Volumetric scattering contribution is skipped.");

			const bool gpuCullingReady = !enableFrustumCulling ||
				(!!*frustumCullPipeline && !!*frustumCullPipelineLayout && frustumCullDispatchCount > 0u);
			drawDeferredIntegrationCheck("GPU culling", gpuCullingReady,
			                             "Renderer falls back to per-job direct draws.");

			const bool profilingReady = !gpuProfilingEnabled || gpuProfilingSupported;
			drawIntegrationCheck("GPU timing", profilingReady, "Pass timings are unavailable on this device.");
		}
		ImGui::End();
	}

	// Rasterization rendering: only execute if ray query did not render this frame.
	if (!rayQueryRenderedThisFrame)
	{
		// Optional: render planar reflections first
		/*
		if (enablePlanarReflections) {
		  glm::vec4 planeWS(0.0f, 1.0f, 0.0f, 0.0f);
		  renderReflectionPass(commandBuffers[currentFrame], planeWS, camera, opaqueJobs);
		}
		*/

		// Sort transparent entities back-to-front for correct blending of nested glass/liquids
		if (!transparentJobs.empty())
		{
			glm::vec3 camPos = camera ? camera->GetPosition() : glm::vec3(0.0f);
			std::ranges::sort(transparentJobs,
			                  [camPos](const RenderJob& a, const RenderJob& b)
			                  {
				                  glm::vec3 pa = a.transformComp ? a.transformComp->GetPosition() : glm::vec3(0.0f);
				                  glm::vec3 pb = b.transformComp ? b.transformComp->GetPosition() : glm::vec3(0.0f);
				                  float da2 = glm::length2(pa - camPos);
				                  float db2 = glm::length2(pb - camPos);
				                  if (da2 != db2) return da2 > db2;
				                  if (a.entityRes->cachedIsLiquid != b.entityRes->cachedIsLiquid) return a.entityRes->
					                  cachedIsLiquid;
				                  return a.entity < b.entity;
			                  });
		}

		// Track whether we executed a depth pre-pass this frame (used to choose depth load op and pipeline state)
		bool didOpaqueDepthPrepass = false;

		// Optional Forward+ depth pre-pass for opaque geometry
		if (useForwardPlus)
		{
			if (!opaqueJobs.empty())
			{
				writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::DepthPrepass, true);
				// Transition depth image for attachment write (Sync2)
				vk::ImageMemoryBarrier2 depthBarrier2{
					.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
					.srcAccessMask = {},
					.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests,
					.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
					.oldLayout = vk::ImageLayout::eUndefined,
					.newLayout = vk::ImageLayout::eDepthAttachmentOptimal,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.image = *depthImage,
					.subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}
				};
				vk::DependencyInfo depInfoDepth{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &depthBarrier2};
				commandBuffers[currentFrame].pipelineBarrier2(depInfoDepth);

				// Depth-only rendering
				vk::RenderingAttachmentInfo depthOnlyAttachment{
					.imageView = *depthImageView, .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
					.loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore,
					.clearValue = vk::ClearDepthStencilValue{1.0f, 0}
				};
				vk::RenderingInfo depthOnlyInfo{
					.renderArea = vk::Rect2D({0, 0}, swapChainExtent), .layerCount = 1, .colorAttachmentCount = 0,
					.pColorAttachments = nullptr, .pDepthAttachment = &depthOnlyAttachment
				};
				commandBuffers[currentFrame].beginRendering(depthOnlyInfo);
				vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
				                      static_cast<float>(swapChainExtent.height), 0.0f, 1.0f);
				commandBuffers[currentFrame].setViewport(0, viewport);
				vk::Rect2D scissor({0, 0}, swapChainExtent);
				commandBuffers[currentFrame].setScissor(0, scissor);

				// Bind depth pre-pass pipeline
				if (!!*depthPrepassPipeline)
				{
					commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *depthPrepassPipeline);
				}

				for (const auto& job : opaqueJobs)
				{
					if (job.isAlphaMasked) continue;

					// Bind geometry
					std::array<vk::Buffer, 2> buffers = {*job.meshRes->vertexBuffer, *job.entityRes->instanceBuffer};
					std::array<vk::DeviceSize, 2> offsets = {0, 0};
					commandBuffers[currentFrame].bindVertexBuffers(0, buffers, offsets);
					commandBuffers[currentFrame].bindIndexBuffer(*job.meshRes->indexBuffer, 0, vk::IndexType::eUint32);

					// Bind descriptor set (PBR set 0)
					commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
					                                                *pbrPipelineLayout,
					                                                0,
					                                                *job.entityRes->pbrDescriptorSets[currentFrame],
					                                                nullptr);

					// Issue draw
					uint32_t instanceCount = std::max(1u, static_cast<uint32_t>(job.meshComp->GetInstanceCount()));
					commandBuffers[currentFrame].drawIndexed(job.meshRes->indexCount, instanceCount, 0, 0, 0);
				}

				commandBuffers[currentFrame].endRendering();
				writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::DepthPrepass, false);

				// Barrier to ensure depth is visible for subsequent passes (Sync2)
				vk::ImageMemoryBarrier2 depthToRead2{
					.srcStageMask = vk::PipelineStageFlagBits2::eLateFragmentTests,
					.srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
					.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests,
					.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead,
					.oldLayout = vk::ImageLayout::eDepthAttachmentOptimal,
					.newLayout = vk::ImageLayout::eDepthAttachmentOptimal,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.image = *depthImage,
					.subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}
				};
				vk::DependencyInfo depInfoDepthToRead{
					.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &depthToRead2
				};
				commandBuffers[currentFrame].pipelineBarrier2(depInfoDepthToRead);

				didOpaqueDepthPrepass = true;
			}

			if (deferredRasterMode && didOpaqueDepthPrepass)
			{
				writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::DepthPyramid, true);
				dispatchDepthPyramid(commandBuffers[currentFrame]);
				writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::DepthPyramid, false);

				writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::SAO, true);
				dispatchSAO(commandBuffers[currentFrame]);
				writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::SAO, false);

				writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::Volumetric, true);
				dispatchVolumetric(commandBuffers[currentFrame]);
				writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::Volumetric, false);
			}

			if (deferredRasterMode)
			{
				if (!createDeferredResources())
				{
					std::cerr << "Failed to create deferred resources - falling back to Forward+\n";
				}
				else
				{
					writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::Opaque, true);

					std::array<vk::ImageMemoryBarrier2, 4> gbufferBarriers;
					uint32_t gbufferBarrierCount = 0;

					if (currentFrame < gBufferAlbedoImages.size() && !!*gBufferAlbedoImages[currentFrame])
					{
						vk::ImageLayout oldLayout = (currentFrame < gBufferAlbedoImageLayouts.size())
							                            ? gBufferAlbedoImageLayouts[currentFrame]
							                            : vk::ImageLayout::eUndefined;
						gbufferBarriers[gbufferBarrierCount++] = vk::ImageMemoryBarrier2{
							.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
							.srcAccessMask = vk::AccessFlagBits2::eNone,
							.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
							.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
							.oldLayout = oldLayout,
							.newLayout = vk::ImageLayout::eColorAttachmentOptimal,
							.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.image = *gBufferAlbedoImages[currentFrame],
							.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
						};
					}
					if (currentFrame < gBufferNormalImages.size() && !!*gBufferNormalImages[currentFrame])
					{
						vk::ImageLayout oldLayout = (currentFrame < gBufferNormalImageLayouts.size())
							                            ? gBufferNormalImageLayouts[currentFrame]
							                            : vk::ImageLayout::eUndefined;
						gbufferBarriers[gbufferBarrierCount++] = vk::ImageMemoryBarrier2{
							.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
							.srcAccessMask = vk::AccessFlagBits2::eNone,
							.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
							.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
							.oldLayout = oldLayout,
							.newLayout = vk::ImageLayout::eColorAttachmentOptimal,
							.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.image = *gBufferNormalImages[currentFrame],
							.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
						};
					}
					if (currentFrame < gBufferMaterialImages.size() && !!*gBufferMaterialImages[currentFrame])
					{
						vk::ImageLayout oldLayout = (currentFrame < gBufferMaterialImageLayouts.size())
							                            ? gBufferMaterialImageLayouts[currentFrame]
							                            : vk::ImageLayout::eUndefined;
						gbufferBarriers[gbufferBarrierCount++] = vk::ImageMemoryBarrier2{
							.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
							.srcAccessMask = vk::AccessFlagBits2::eNone,
							.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
							.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
							.oldLayout = oldLayout,
							.newLayout = vk::ImageLayout::eColorAttachmentOptimal,
							.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.image = *gBufferMaterialImages[currentFrame],
							.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
						};
					}
					if (currentFrame < gBufferEmissiveImages.size() && !!*gBufferEmissiveImages[currentFrame])
					{
						vk::ImageLayout oldLayout = (currentFrame < gBufferEmissiveImageLayouts.size())
							                            ? gBufferEmissiveImageLayouts[currentFrame]
							                            : vk::ImageLayout::eUndefined;
						gbufferBarriers[gbufferBarrierCount++] = vk::ImageMemoryBarrier2{
							.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
							.srcAccessMask = vk::AccessFlagBits2::eNone,
							.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
							.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
							.oldLayout = oldLayout,
							.newLayout = vk::ImageLayout::eColorAttachmentOptimal,
							.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.image = *gBufferEmissiveImages[currentFrame],
							.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
						};
					}

					if (gbufferBarrierCount > 0)
					{
						vk::DependencyInfo gbufferDep{
							.imageMemoryBarrierCount = gbufferBarrierCount,
							.pImageMemoryBarriers = gbufferBarriers.data()
						};
						commandBuffers[currentFrame].pipelineBarrier2(gbufferDep);
					}

					if (currentFrame < gBufferAlbedoImageLayouts.size()) gBufferAlbedoImageLayouts[currentFrame] =
						vk::ImageLayout::eColorAttachmentOptimal;
					if (currentFrame < gBufferNormalImageLayouts.size()) gBufferNormalImageLayouts[currentFrame] =
						vk::ImageLayout::eColorAttachmentOptimal;
					if (currentFrame < gBufferMaterialImageLayouts.size()) gBufferMaterialImageLayouts[currentFrame] =
						vk::ImageLayout::eColorAttachmentOptimal;
					if (currentFrame < gBufferEmissiveImageLayouts.size()) gBufferEmissiveImageLayouts[currentFrame] =
						vk::ImageLayout::eColorAttachmentOptimal;

					vk::RenderingAttachmentInfo gbufferColorAttachments[4];
					uint32_t gbufferColorCount = 0;

					if (currentFrame < gBufferAlbedoImageViews.size() && !!*gBufferAlbedoImageViews[currentFrame])
					{
						gbufferColorAttachments[gbufferColorCount++] = vk::RenderingAttachmentInfo{
							.imageView = *gBufferAlbedoImageViews[currentFrame],
							.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
							.loadOp = vk::AttachmentLoadOp::eClear,
							.storeOp = vk::AttachmentStoreOp::eStore,
							.clearValue = vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}}
						};
					}

					if (currentFrame < gBufferNormalImageViews.size() && !!*gBufferNormalImageViews[currentFrame])
					{
						gbufferColorAttachments[gbufferColorCount++] = vk::RenderingAttachmentInfo{
							.imageView = *gBufferNormalImageViews[currentFrame],
							.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
							.loadOp = vk::AttachmentLoadOp::eClear,
							.storeOp = vk::AttachmentStoreOp::eStore,
							.clearValue = vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}}
						};
					}

					if (currentFrame < gBufferMaterialImageViews.size() && !!*gBufferMaterialImageViews[currentFrame])
					{
						gbufferColorAttachments[gbufferColorCount++] = vk::RenderingAttachmentInfo{
							.imageView = *gBufferMaterialImageViews[currentFrame],
							.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
							.loadOp = vk::AttachmentLoadOp::eClear,
							.storeOp = vk::AttachmentStoreOp::eStore,
							.clearValue = vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}}
						};
					}

					if (currentFrame < gBufferEmissiveImageViews.size() && !!*gBufferEmissiveImageViews[currentFrame])
					{
						gbufferColorAttachments[gbufferColorCount++] = vk::RenderingAttachmentInfo{
							.imageView = *gBufferEmissiveImageViews[currentFrame],
							.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
							.loadOp = vk::AttachmentLoadOp::eClear,
							.storeOp = vk::AttachmentStoreOp::eStore,
							.clearValue = vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}}
						};
					}

					vk::RenderingAttachmentInfo gbufferDepthAttachment{
						.imageView = *depthImageView,
						.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
						.loadOp = didOpaqueDepthPrepass ? vk::AttachmentLoadOp::eLoad : vk::AttachmentLoadOp::eClear,
						.storeOp = vk::AttachmentStoreOp::eStore,
						.clearValue = vk::ClearDepthStencilValue{1.0f, 0}
					};

					vk::RenderingInfo gbufferRenderingInfo{
						.renderArea = vk::Rect2D({0, 0}, swapChainExtent),
						.layerCount = 1,
						.colorAttachmentCount = gbufferColorCount,
						.pColorAttachments = gbufferColorCount > 0 ? gbufferColorAttachments : nullptr,
						.pDepthAttachment = &gbufferDepthAttachment
					};

					commandBuffers[currentFrame].beginRendering(gbufferRenderingInfo);
					vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
					                      static_cast<float>(swapChainExtent.height), 0.0f, 1.0f);
					commandBuffers[currentFrame].setViewport(0, viewport);
					vk::Rect2D scissor({0, 0}, swapChainExtent);
					commandBuffers[currentFrame].setScissor(0, scissor);

					// Indirect draw state for Deferred path
					FrustumCullPerFrame* indirectFrame = nullptr;
					bool canUseIndirectOpaqueDraws = false;

					for (size_t jobIndex = 0; jobIndex < opaqueJobs.size(); ++jobIndex)
					{
						const auto& job = opaqueJobs[jobIndex];

						vk::raii::Pipeline* selectedPipeline = nullptr;
						if (job.isAlphaMasked)
						{
							selectedPipeline = &pbrGraphicsPipeline;
						}
						else
						{
							selectedPipeline = didOpaqueDepthPrepass && !!*pbrPrepassGraphicsPipeline
								                   ? &pbrPrepassGraphicsPipeline
								                   : &pbrGraphicsPipeline;
						}

						if (currentPipeline != selectedPipeline)
						{
							commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics,
							                                          **selectedPipeline);
							currentPipeline = selectedPipeline;
							currentLayout = &pbrPipelineLayout;
						}

						std::array<vk::Buffer, 2> buffers = {
							*job.meshRes->vertexBuffer, *job.entityRes->instanceBuffer
						};
						std::array<vk::DeviceSize, 2> offsets = {0, 0};
						commandBuffers[currentFrame].bindVertexBuffers(0, buffers, offsets);
						commandBuffers[currentFrame].bindIndexBuffer(*job.meshRes->indexBuffer, 0,
						                                             vk::IndexType::eUint32);

						vk::DescriptorSet set1Opaque = (transparentDescriptorSets.empty() || IsLoading())
							                               ? *transparentFallbackDescriptorSets[currentFrame]
							                               : *transparentDescriptorSets[currentFrame];
						commandBuffers[currentFrame].bindDescriptorSets(
							vk::PipelineBindPoint::eGraphics,
							**currentLayout,
							0,
							{*job.entityRes->pbrDescriptorSets[currentFrame], set1Opaque},
							{});

						commandBuffers[currentFrame].pushConstants<MaterialProperties>(
							**currentLayout, vk::ShaderStageFlagBits::eFragment, 0,
							{job.entityRes->cachedMaterialProps});

						if (canUseIndirectOpaqueDraws && indirectFrame != nullptr)
						{
							const vk::DeviceSize commandOffset = static_cast<vk::DeviceSize>(jobIndex * sizeof(
								vk::DrawIndexedIndirectCommand));
							commandBuffers[currentFrame].drawIndexedIndirect(
								*indirectFrame->indirectCommandsBuffer,
								commandOffset,
								1,
								sizeof(vk::DrawIndexedIndirectCommand));
						}
						else
						{
							uint32_t instanceCount = std::max(
								1u, static_cast<uint32_t>(job.meshComp->GetInstanceCount()));
							commandBuffers[currentFrame].drawIndexed(job.meshRes->indexCount, instanceCount, 0, 0, 0);
						}
					}

					commandBuffers[currentFrame].endRendering();
					writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::Opaque, false);

					std::array<vk::ImageMemoryBarrier2, 4> gbufferToSampleBarriers;
					uint32_t gbufferToSampleCount = 0;

					if (currentFrame < gBufferAlbedoImages.size() && !!*gBufferAlbedoImages[currentFrame])
					{
						gbufferToSampleBarriers[gbufferToSampleCount++] = vk::ImageMemoryBarrier2{
							.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
							.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
							.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
							.dstAccessMask = vk::AccessFlagBits2::eShaderRead,
							.oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
							.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
							.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.image = *gBufferAlbedoImages[currentFrame],
							.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
						};
					}

					if (currentFrame < gBufferNormalImages.size() && !!*gBufferNormalImages[currentFrame])
					{
						gbufferToSampleBarriers[gbufferToSampleCount++] = vk::ImageMemoryBarrier2{
							.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
							.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
							.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
							.dstAccessMask = vk::AccessFlagBits2::eShaderRead,
							.oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
							.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
							.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.image = *gBufferNormalImages[currentFrame],
							.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
						};
					}

					if (currentFrame < gBufferMaterialImages.size() && !!*gBufferMaterialImages[currentFrame])
					{
						gbufferToSampleBarriers[gbufferToSampleCount++] = vk::ImageMemoryBarrier2{
							.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
							.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
							.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
							.dstAccessMask = vk::AccessFlagBits2::eShaderRead,
							.oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
							.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
							.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.image = *gBufferMaterialImages[currentFrame],
							.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
						};
					}

					if (currentFrame < gBufferEmissiveImages.size() && !!*gBufferEmissiveImages[currentFrame])
					{
						gbufferToSampleBarriers[gbufferToSampleCount++] = vk::ImageMemoryBarrier2{
							.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
							.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
							.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
							.dstAccessMask = vk::AccessFlagBits2::eShaderRead,
							.oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
							.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
							.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.image = *gBufferEmissiveImages[currentFrame],
							.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
						};
					}

					if (gbufferToSampleCount > 0)
					{
						vk::DependencyInfo gbufferToSampleDep{
							.imageMemoryBarrierCount = gbufferToSampleCount,
							.pImageMemoryBarriers = gbufferToSampleBarriers.data()
						};
						commandBuffers[currentFrame].pipelineBarrier2(gbufferToSampleDep);
					}

					if (currentFrame < gBufferAlbedoImageLayouts.size()) gBufferAlbedoImageLayouts[currentFrame] =
						vk::ImageLayout::eShaderReadOnlyOptimal;

					if (currentFrame < deferredLightingImages.size() && !!*deferredLightingImages[currentFrame])
					{
						vk::ImageLayout oldLayout = (currentFrame < deferredLightingImageLayouts.size())
							                            ? deferredLightingImageLayouts[currentFrame]
							                            : vk::ImageLayout::eUndefined;
						vk::ImageMemoryBarrier2 lightingToCompute{
							.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
							.srcAccessMask = vk::AccessFlagBits2::eNone,
							.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
							.dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
							.oldLayout = oldLayout,
							.newLayout = vk::ImageLayout::eGeneral,
							.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.image = *deferredLightingImages[currentFrame],
							.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
						};
						vk::DependencyInfo lightingDep{
							.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &lightingToCompute
						};
						commandBuffers[currentFrame].pipelineBarrier2(lightingDep);
						deferredLightingImageLayouts[currentFrame] = vk::ImageLayout::eGeneral;
					}

					if (!!*deferredLightingPipeline && !!*deferredLightingPipelineLayout &&
						currentFrame < deferredLightingDescriptorSets.size() && !!*deferredLightingDescriptorSets[
							currentFrame])
					{
						commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute,
						                                          *deferredLightingPipeline);
						commandBuffers[currentFrame].bindDescriptorSets(
							vk::PipelineBindPoint::eCompute,
							*deferredLightingPipelineLayout,
							0,
							*deferredLightingDescriptorSets[currentFrame],
							nullptr);

						uint32_t workgroupsX = (swapChainExtent.width + 7) / 8;
						uint32_t workgroupsY = (swapChainExtent.height + 7) / 8;
						commandBuffers[currentFrame].dispatch(workgroupsX, workgroupsY, 1);
					}
					else
					{
						vk::ImageLayout albedoLayout = (currentFrame < gBufferAlbedoImageLayouts.size())
							                               ? gBufferAlbedoImageLayouts[currentFrame]
							                               : vk::ImageLayout::eShaderReadOnlyOptimal;
						vk::ImageMemoryBarrier2 albedoToTransfer{
							.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
							.srcAccessMask = vk::AccessFlagBits2::eShaderRead,
							.dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
							.dstAccessMask = vk::AccessFlagBits2::eTransferRead,
							.oldLayout = albedoLayout,
							.newLayout = vk::ImageLayout::eTransferSrcOptimal,
							.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.image = *gBufferAlbedoImages[currentFrame],
							.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
						};
						vk::ImageMemoryBarrier2 lightingToTransfer{
							.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
							.srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
							.dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
							.dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
							.oldLayout = vk::ImageLayout::eGeneral,
							.newLayout = vk::ImageLayout::eTransferDstOptimal,
							.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.image = *deferredLightingImages[currentFrame],
							.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
						};
						std::array<vk::ImageMemoryBarrier2, 2> copyBarriers = {albedoToTransfer, lightingToTransfer};
						vk::DependencyInfo copyDep{
							.imageMemoryBarrierCount = static_cast<uint32_t>(copyBarriers.size()),
							.pImageMemoryBarriers = copyBarriers.data()
						};
						commandBuffers[currentFrame].pipelineBarrier2(copyDep);

						vk::ImageCopy copyRegion{
							.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
							.srcOffset = {0, 0, 0},
							.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
							.dstOffset = {0, 0, 0},
							.extent = {swapChainExtent.width, swapChainExtent.height, 1}
						};
						commandBuffers[currentFrame].copyImage(
							*gBufferAlbedoImages[currentFrame],
							vk::ImageLayout::eTransferSrcOptimal,
							*deferredLightingImages[currentFrame],
							vk::ImageLayout::eTransferDstOptimal,
							copyRegion);

						vk::ImageMemoryBarrier2 albedoBack{
							.srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
							.srcAccessMask = vk::AccessFlagBits2::eTransferRead,
							.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
							.dstAccessMask = vk::AccessFlagBits2::eShaderRead,
							.oldLayout = vk::ImageLayout::eTransferSrcOptimal,
							.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
							.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.image = *gBufferAlbedoImages[currentFrame],
							.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
						};
						vk::ImageMemoryBarrier2 lightingBack{
							.srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
							.srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
							.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
							.dstAccessMask = vk::AccessFlagBits2::eShaderRead,
							.oldLayout = vk::ImageLayout::eTransferDstOptimal,
							.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
							.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.image = *deferredLightingImages[currentFrame],
							.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
						};
						std::array<vk::ImageMemoryBarrier2, 2> backBarriers = {albedoBack, lightingBack};
						vk::DependencyInfo backDep{
							.imageMemoryBarrierCount = static_cast<uint32_t>(backBarriers.size()),
							.pImageMemoryBarriers = backBarriers.data()
						};
						commandBuffers[currentFrame].pipelineBarrier2(backDep);
					}

					if (currentFrame < deferredLightingImages.size() && !!*deferredLightingImages[currentFrame])
					{
						vk::ImageLayout currentLayoutState = (currentFrame < deferredLightingImageLayouts.size())
							                                     ? deferredLightingImageLayouts[currentFrame]
							                                     : vk::ImageLayout::eGeneral;
						vk::ImageMemoryBarrier2 lightingToSample{
							.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
							.srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
							.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
							.dstAccessMask = vk::AccessFlagBits2::eShaderRead,
							.oldLayout = currentLayoutState,
							.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
							.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
							.image = *deferredLightingImages[currentFrame],
							.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
						};
						vk::DependencyInfo lightingSampleDep{
							.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &lightingToSample
						};
						commandBuffers[currentFrame].pipelineBarrier2(lightingSampleDep);
						deferredLightingImageLayouts[currentFrame] = vk::ImageLayout::eShaderReadOnlyOptimal;
					}
				}
			}
			else
			{
				// Forward+ compute culling based on current camera and screen tiles
				uint32_t tilesX = (swapChainExtent.width + forwardPlusTileSizeX - 1) / forwardPlusTileSizeX;
				uint32_t tilesY = (swapChainExtent.height + forwardPlusTileSizeY - 1) / forwardPlusTileSizeY;

				// Lights already extracted at frame start - use lastFrameLightCount for Forward+ params
				glm::mat4 view = camera->GetViewMatrix();
				glm::mat4 proj = camera->GetProjectionMatrix();
				proj[1][1] *= -1.0f;
				float nearZ = camera->GetNearPlane();
				float farZ = camera->GetFarPlane();
				updateForwardPlusParams(currentFrame, view, proj, lastFrameLightCount, tilesX, tilesY,
				                        forwardPlusSlicesZ, nearZ, farZ);
				// As a last guard before dispatch, make sure compute binding 0 is valid for this frame
				refreshForwardPlusComputeLightsBindingForFrame(currentFrame);

				writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::ForwardPlus, true);
				dispatchForwardPlus(commandBuffers[currentFrame], tilesX, tilesY, forwardPlusSlicesZ);
				writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::ForwardPlus, false);
			}

			FrustumCullPerFrame* indirectFrame = nullptr;
			bool indirectDrawAnyVisible = false;
			bool canUseIndirectOpaqueDraws = false;

			if (deferredRasterMode &&
				enableIndirectDrawCommands &&
				enableFrustumCulling &&
				frustumCullDispatchCount > 0 &&
				currentFrame < frustumCullPerFrame.size())
			{
				indirectFrame = &frustumCullPerFrame[currentFrame];

				const bool hasIndirectBuffer = !!*indirectFrame->indirectCommandsBuffer &&
					(indirectFrame->indirectCommandsMapped != nullptr) &&
					(indirectFrame->indirectCommandCapacity >= opaqueJobs.size());
				if (hasIndirectBuffer)
				{
					const bool useOcclusionCount =
						enableOcclusionCulling &&
						depthPyramidHistoryValid &&
						!!*occlusionCullPipeline &&
						!!*occlusionCullPipelineLayout &&
						!!*indirectFrame->occlusionComputeSet &&
						!!*indirectFrame->occlusionVisibleIndicesBuffer &&
						!!*indirectFrame->occlusionVisibleCountBuffer &&
						!!*indirectFrame->occlusionParamsBuffer &&
						!!*depthPyramidImage &&
						!!*depthPyramidFullView &&
						!!*depthPyramidSampler &&
						(depthPyramidMipCount > 0);

					const uint32_t* visibleCountPtr = nullptr;
					if (useOcclusionCount && indirectFrame->occlusionVisibleCountMapped)
					{
						visibleCountPtr = static_cast<const uint32_t*>(indirectFrame->occlusionVisibleCountMapped);
					}
					else if (indirectFrame->visibleCountMapped)
					{
						visibleCountPtr = static_cast<const uint32_t*>(indirectFrame->visibleCountMapped);
					}

					if (visibleCountPtr != nullptr)
					{
						indirectDrawAnyVisible = (*visibleCountPtr > 0u);
						auto* drawCommands = static_cast<vk::DrawIndexedIndirectCommand*>(indirectFrame->
							indirectCommandsMapped);
						for (size_t i = 0; i < opaqueJobs.size(); ++i)
						{
							const auto& job = opaqueJobs[i];
							const uint32_t instanceCount = std::max(
								1u, static_cast<uint32_t>(job.meshComp->GetInstanceCount()));
							drawCommands[i] = vk::DrawIndexedIndirectCommand{
								.indexCount = job.meshRes->indexCount,
								.instanceCount = indirectDrawAnyVisible ? instanceCount : 0u,
								.firstIndex = 0,
								.vertexOffset = 0,
								.firstInstance = 0
							};
						}

						canUseIndirectOpaqueDraws = true;
					}
				}
			}

			if (canUseIndirectOpaqueDraws)
			{
				vk::MemoryBarrier2 hostToIndirect{
					.srcStageMask = vk::PipelineStageFlagBits2::eHost,
					.srcAccessMask = vk::AccessFlagBits2::eHostWrite,
					.dstStageMask = vk::PipelineStageFlagBits2::eDrawIndirect,
					.dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead
				};
				vk::DependencyInfo depHostToIndirect{.memoryBarrierCount = 1, .pMemoryBarriers = &hostToIndirect};
				commandBuffers[currentFrame].pipelineBarrier2(depHostToIndirect);
			}

			// Viewport and scissor used by both Forward+ opaque and transparent passes
			vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
			                      static_cast<float>(swapChainExtent.height), 0.0f, 1.0f);
			vk::Rect2D scissor({0, 0}, swapChainExtent);

			if (!deferredRasterMode)
			{
				// Transition off-screen color to attachment write (Sync2). On first use after creation or after switching
				// from a mode that never produced this image, the layout may still be UNDEFINED.
				vk::ImageLayout oscOldLayout = vk::ImageLayout::eUndefined;
				vk::PipelineStageFlags2 oscSrcStage = vk::PipelineStageFlagBits2::eTopOfPipe;
				vk::AccessFlags2 oscSrcAccess = vk::AccessFlagBits2::eNone;
				if (currentFrame < opaqueSceneColorImageLayouts.size())
				{
					oscOldLayout = opaqueSceneColorImageLayouts[currentFrame];
					if (oscOldLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
					{
						oscSrcStage = vk::PipelineStageFlagBits2::eFragmentShader;
						oscSrcAccess = vk::AccessFlagBits2::eShaderRead;
					}
					else if (oscOldLayout == vk::ImageLayout::eColorAttachmentOptimal)
					{
						oscSrcStage = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
						oscSrcAccess = vk::AccessFlagBits2::eColorAttachmentWrite;
					}
					else
					{
						oscOldLayout = vk::ImageLayout::eUndefined;
						oscSrcStage = vk::PipelineStageFlagBits2::eTopOfPipe;
						oscSrcAccess = vk::AccessFlagBits2::eNone;
					}
				}
				vk::ImageMemoryBarrier2 oscToColor2{
					.srcStageMask = oscSrcStage,
					.srcAccessMask = oscSrcAccess,
					.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
					.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite |
					vk::AccessFlagBits2::eColorAttachmentRead,
					.oldLayout = oscOldLayout,
					.newLayout = vk::ImageLayout::eColorAttachmentOptimal,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.image = *opaqueSceneColorImages[currentFrame],
					.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
				};
				vk::DependencyInfo depOscToColor{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &oscToColor2};
				commandBuffers[currentFrame].pipelineBarrier2(depOscToColor);
				if (currentFrame < opaqueSceneColorImageLayouts.size())
				{
					opaqueSceneColorImageLayouts[currentFrame] = vk::ImageLayout::eColorAttachmentOptimal;
				}
				// PASS 1: OFF-SCREEN COLOR (Opaque)
				// Clear the off-screen target at the start of opaque rendering to a neutral black background
				vk::RenderingAttachmentInfo colorAttachment{
					.imageView = *opaqueSceneColorImageViews[currentFrame],
					.imageLayout = vk::ImageLayout::eColorAttachmentOptimal, .loadOp = vk::AttachmentLoadOp::eClear,
					.storeOp = vk::AttachmentStoreOp::eStore,
					.clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})
				};
				depthAttachment.imageView = *depthImageView;
				depthAttachment.loadOp = (didOpaqueDepthPrepass)
					                         ? vk::AttachmentLoadOp::eLoad
					                         : vk::AttachmentLoadOp::eClear;
				vk::RenderingInfo passInfo{
					.renderArea = vk::Rect2D({0, 0}, swapChainExtent), .layerCount = 1, .colorAttachmentCount = 1,
					.pColorAttachments = &colorAttachment, .pDepthAttachment = &depthAttachment
				};
				writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::Opaque, true);
				commandBuffers[currentFrame].beginRendering(passInfo);
				commandBuffers[currentFrame].setViewport(0, viewport);
				commandBuffers[currentFrame].setScissor(0, scissor);

				{
					uint32_t opaqueDrawsThisPass = 0;
					for (size_t jobIndex = 0; jobIndex < opaqueJobs.size(); ++jobIndex)
					{
						const auto& job = opaqueJobs[jobIndex];
						bool useBasic = (imguiSystem && !imguiSystem->IsPBREnabled());
						vk::raii::Pipeline* selectedPipeline = nullptr;
						vk::raii::PipelineLayout* selectedLayout = nullptr;
						if (useBasic)
						{
							selectedPipeline = &graphicsPipeline;
							selectedLayout = &pipelineLayout;
						}
						else
						{
							// If masked, we need depth writes with alpha test; otherwise, after-prepass read-only is fine.
							if (job.isAlphaMasked)
							{
								selectedPipeline = &pbrGraphicsPipeline; // writes depth, compare Less
							}
							else
							{
								selectedPipeline = didOpaqueDepthPrepass && !!*pbrPrepassGraphicsPipeline
									                   ? &pbrPrepassGraphicsPipeline
									                   : &pbrGraphicsPipeline;
							}
							selectedLayout = &pbrPipelineLayout;
						}
						if (currentPipeline != selectedPipeline)
						{
							commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics,
							                                          **selectedPipeline);
							currentPipeline = selectedPipeline;
							currentLayout = selectedLayout;
						}

						std::array<vk::Buffer, 2> buffers = {
							*job.meshRes->vertexBuffer, *job.entityRes->instanceBuffer
						};
						std::array<vk::DeviceSize, 2> offsets = {0, 0};
						commandBuffers[currentFrame].bindVertexBuffers(0, buffers, offsets);
						commandBuffers[currentFrame].bindIndexBuffer(*job.meshRes->indexBuffer, 0,
						                                             vk::IndexType::eUint32);

						auto* descSetsPtr = useBasic
							                    ? &job.entityRes->basicDescriptorSets
							                    : &job.entityRes->pbrDescriptorSets;
						if (descSetsPtr->empty() || currentFrame >= descSetsPtr->size())
						{
							continue;
						}

						if (useBasic)
						{
							commandBuffers[currentFrame].bindDescriptorSets(
								vk::PipelineBindPoint::eGraphics,
								**selectedLayout,
								0,
								{*(*descSetsPtr)[currentFrame]},
								{});
						}
						else
						{
							vk::DescriptorSet set1Opaque = (transparentDescriptorSets.empty() || IsLoading())
								                               ? *transparentFallbackDescriptorSets[currentFrame]
								                               : *transparentDescriptorSets[currentFrame];
							commandBuffers[currentFrame].bindDescriptorSets(
								vk::PipelineBindPoint::eGraphics,
								**selectedLayout,
								0,
								{*(*descSetsPtr)[currentFrame], set1Opaque},
								{});

							commandBuffers[currentFrame].pushConstants<MaterialProperties>(
								**selectedLayout, vk::ShaderStageFlagBits::eFragment, 0,
								{job.entityRes->cachedMaterialProps});
						}

						if (canUseIndirectOpaqueDraws && indirectFrame != nullptr)
						{
							const vk::DeviceSize commandOffset = static_cast<vk::DeviceSize>(jobIndex * sizeof(
								vk::DrawIndexedIndirectCommand));
							commandBuffers[currentFrame].drawIndexedIndirect(
								*indirectFrame->indirectCommandsBuffer,
								commandOffset,
								1,
								sizeof(vk::DrawIndexedIndirectCommand));
						}
						else
						{
							uint32_t instanceCount = std::max(
								1u, static_cast<uint32_t>(job.meshComp->GetInstanceCount()));
							commandBuffers[currentFrame].drawIndexed(job.meshRes->indexCount, instanceCount, 0, 0, 0);
						}

						++opaqueDrawsThisPass;
					}
				}
				commandBuffers[currentFrame].endRendering();
				writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::Opaque, false);
			}
			// PASS 1b: PRESENT – composite path
			{
				bool canWriteTAAHistory = false;
				if (deferredRasterMode &&
					taaJitterEnabled &&
					!!*taaHistorySampler &&
					taaHistoryWriteIndex < taaHistoryImages.size() &&
					taaHistoryWriteIndex < taaHistoryImageLayouts.size())
				{
					canWriteTAAHistory = true;
				}

				if (canWriteTAAHistory)
				{
					writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::TAAHistory, true);
					vk::ImageMemoryBarrier2 opaqueToTransferSrc2{
						.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
						.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
						.dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
						.dstAccessMask = vk::AccessFlagBits2::eTransferRead,
						.oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
						.newLayout = vk::ImageLayout::eTransferSrcOptimal,
						.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.image = *opaqueSceneColorImages[currentFrame],
						.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
					};

					vk::ImageLayout oldHistoryLayout = taaHistoryImageLayouts[taaHistoryWriteIndex];
					vk::PipelineStageFlags2 historySrcStage = vk::PipelineStageFlagBits2::eTopOfPipe;
					vk::AccessFlags2 historySrcAccess = vk::AccessFlagBits2::eNone;
					if (oldHistoryLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
					{
						historySrcStage = vk::PipelineStageFlagBits2::eFragmentShader;
						historySrcAccess = vk::AccessFlagBits2::eShaderRead;
					}
					else if (oldHistoryLayout == vk::ImageLayout::eTransferDstOptimal)
					{
						historySrcStage = vk::PipelineStageFlagBits2::eTransfer;
						historySrcAccess = vk::AccessFlagBits2::eTransferWrite;
					}

					vk::ImageMemoryBarrier2 historyToTransferDst{
						.srcStageMask = historySrcStage,
						.srcAccessMask = historySrcAccess,
						.dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
						.dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
						.oldLayout = oldHistoryLayout,
						.newLayout = vk::ImageLayout::eTransferDstOptimal,
						.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.image = *taaHistoryImages[taaHistoryWriteIndex],
						.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
					};

					std::array<vk::ImageMemoryBarrier2, 2> preCopyBarriers = {
						opaqueToTransferSrc2, historyToTransferDst
					};
					vk::DependencyInfo depPreCopy{
						.imageMemoryBarrierCount = static_cast<uint32_t>(preCopyBarriers.size()),
						.pImageMemoryBarriers = preCopyBarriers.data()
					};
					commandBuffers[currentFrame].pipelineBarrier2(depPreCopy);

					vk::ImageCopy copyRegion{
						.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
						.srcOffset = {0, 0, 0},
						.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
						.dstOffset = {0, 0, 0},
						.extent = {swapChainExtent.width, swapChainExtent.height, 1}
					};
					commandBuffers[currentFrame].copyImage(
						*opaqueSceneColorImages[currentFrame],
						vk::ImageLayout::eTransferSrcOptimal,
						*taaHistoryImages[taaHistoryWriteIndex],
						vk::ImageLayout::eTransferDstOptimal,
						copyRegion);

					vk::ImageMemoryBarrier2 historyToSample{
						.srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
						.srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
						.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
						.dstAccessMask = vk::AccessFlagBits2::eShaderRead,
						.oldLayout = vk::ImageLayout::eTransferDstOptimal,
						.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
						.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.image = *taaHistoryImages[taaHistoryWriteIndex],
						.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
					};
					vk::ImageMemoryBarrier2 opaqueToSample2{
						.srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
						.srcAccessMask = vk::AccessFlagBits2::eTransferRead,
						.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
						.dstAccessMask = vk::AccessFlagBits2::eShaderRead,
						.oldLayout = vk::ImageLayout::eTransferSrcOptimal,
						.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
						.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.image = *opaqueSceneColorImages[currentFrame],
						.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
					};
					std::array<vk::ImageMemoryBarrier2, 2> postCopyBarriers = {historyToSample, opaqueToSample2};
					vk::DependencyInfo depPostCopy{
						.imageMemoryBarrierCount = static_cast<uint32_t>(postCopyBarriers.size()),
						.pImageMemoryBarriers = postCopyBarriers.data()
					};
					commandBuffers[currentFrame].pipelineBarrier2(depPostCopy);
					taaHistoryImageLayouts[taaHistoryWriteIndex] = vk::ImageLayout::eShaderReadOnlyOptimal;
					taaHistoryValid = true;
					writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::TAAHistory, false);
				}
				else
				{
					vk::ImageMemoryBarrier2 opaqueToSample2{
						.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
						.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
						.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
						.dstAccessMask = vk::AccessFlagBits2::eShaderRead,
						.oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
						.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
						.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.image = *opaqueSceneColorImages[currentFrame],
						.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
					};
					vk::DependencyInfo depOpaqueToSample{
						.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &opaqueToSample2
					};
					commandBuffers[currentFrame].pipelineBarrier2(depOpaqueToSample);
					taaHistoryValid = false;
				}

				if (currentFrame < opaqueSceneColorImageLayouts.size())
				{
					opaqueSceneColorImageLayouts[currentFrame] = vk::ImageLayout::eShaderReadOnlyOptimal;
				}

				// Make the swapchain image ready for color attachment output and clear it (Sync2)
				vk::ImageMemoryBarrier2 swapchainToColor2{
					.srcStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe,
					.srcAccessMask = vk::AccessFlagBits2::eNone,
					.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
					.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite |
					vk::AccessFlagBits2::eColorAttachmentRead,
					.oldLayout = vk::ImageLayout::eUndefined,
					.newLayout = vk::ImageLayout::eColorAttachmentOptimal,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.image = swapChainImages[imageIndex],
					.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
				};
				vk::DependencyInfo depSwapchainToColor{
					.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &swapchainToColor2
				};
				commandBuffers[currentFrame].pipelineBarrier2(depSwapchainToColor);

				// Begin rendering to swapchain for composite
				colorAttachments[0].imageView = *swapChainImageViews[imageIndex];
				colorAttachments[0].loadOp = vk::AttachmentLoadOp::eClear;
				// clear before composing base layer (full-screen composite overwrites all pixels)
				depthAttachment.loadOp = vk::AttachmentLoadOp::eDontCare; // no depth for composite
				renderingInfo.renderArea = vk::Rect2D({0, 0}, swapChainExtent);
				// IMPORTANT: Composite pass does not use a depth attachment. Avoid binding it to satisfy dynamic rendering VUIDs.
				auto savedDepthPtr = renderingInfo.pDepthAttachment; // save to restore later
				renderingInfo.pDepthAttachment = nullptr;
				writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::Composite, true);
				commandBuffers[currentFrame].beginRendering(renderingInfo);

				// Bind composite pipeline
				if (!!*compositePipeline)
				{
					commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *compositePipeline);
				}
				vk::Viewport vp(0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
				                static_cast<float>(swapChainExtent.height), 0.0f, 1.0f);
				commandBuffers[currentFrame].setViewport(0, vp);
				vk::Rect2D sc({0, 0}, swapChainExtent);
				commandBuffers[currentFrame].setScissor(0, sc);

				// Bind descriptor set 0 for the composite. During loading, force fallback to avoid sampling uninitialized off-screen color.
				vk::DescriptorSet setComposite = (transparentDescriptorSets.empty() || IsLoading())
					                                 ? *transparentFallbackDescriptorSets[currentFrame]
					                                 : *transparentDescriptorSets[currentFrame];
				commandBuffers[currentFrame].bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					*compositePipelineLayout,
					0,
					{setComposite},
					{});

				CompositePushConstants pc{};
				pc.exposure = std::clamp(this->exposure, 0.2f, 4.0f);
				pc.gamma = this->gamma;
				pc.outputIsSRGB = (swapChainImageFormat == vk::Format::eR8G8B8A8Srgb || swapChainImageFormat ==
					                  vk::Format::eB8G8R8A8Srgb)
					                  ? 1
					                  : 0;
				pc.enableTAA = (deferredRasterMode && taaJitterEnabled) ? 1 : 0;
				const bool taaReadable = pc.enableTAA != 0 && taaHistoryValid &&
					taaHistoryReadIndex < taaHistoryImageViews.size() &&
					taaHistoryReadIndex < taaHistoryImageLayouts.size() &&
					taaHistoryImageLayouts[taaHistoryReadIndex] == vk::ImageLayout::eShaderReadOnlyOptimal &&
					!!*taaHistorySampler;
				pc.taaHistoryValid = taaReadable ? 1 : 0;

				pc.enableSAO = (deferredRasterMode && enableSAO) ? 1 : 0;
				const bool saoReadable = pc.enableSAO != 0 && saoHistoryValid &&
					currentFrame < saoImageViews.size() &&
					currentFrame < saoImageLayouts.size() &&
					saoImageLayouts[currentFrame] == vk::ImageLayout::eShaderReadOnlyOptimal &&
					!!*saoSampler;
				pc.saoValid = saoReadable ? 1 : 0;

				pc.enableVolumetric = (deferredRasterMode && enableVolumetricScattering) ? 1 : 0;
				const bool volumetricReadable = pc.enableVolumetric != 0 && volumetricHistoryValid &&
					currentFrame < volumetricImageViews.size() &&
					currentFrame < volumetricImageLayouts.size() &&
					volumetricImageLayouts[currentFrame] == vk::ImageLayout::eShaderReadOnlyOptimal &&
					!!*volumetricSampler;
				pc.volumetricValid = volumetricReadable ? 1 : 0;
				pc.taaFeedback = 0.1f;
				commandBuffers[currentFrame].pushConstants<CompositePushConstants>(
					*compositePipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, pc);

				// Draw fullscreen triangle
				commandBuffers[currentFrame].draw(3, 1, 0, 0);

				commandBuffers[currentFrame].endRendering();
				writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::Composite, false);
				// Restore depth attachment pointer for subsequent passes
				renderingInfo.pDepthAttachment = savedDepthPtr;
			}
			// PASS 2: RENDER TRANSPARENT OBJECTS TO THE SWAPCHAIN
			{
				// Ensure depth attachment is bound again for the transparent pass
				renderingInfo.pDepthAttachment = &depthAttachment;
				colorAttachments[0].imageView = *swapChainImageViews[imageIndex];
				colorAttachments[0].loadOp = vk::AttachmentLoadOp::eLoad;
				depthAttachment.loadOp = vk::AttachmentLoadOp::eLoad;
				renderingInfo.renderArea = vk::Rect2D({0, 0}, swapChainExtent);
				writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::Transparent, true);
				commandBuffers[currentFrame].beginRendering(renderingInfo);
				commandBuffers[currentFrame].setViewport(0, viewport);
				commandBuffers[currentFrame].setScissor(0, scissor);

				if (!transparentJobs.empty())
				{
					currentLayout = &pbrTransparentPipelineLayout;
					vk::raii::Pipeline* activeTransparentPipeline = nullptr;

					for (const auto& job : transparentJobs)
					{
						vk::raii::Pipeline* desiredPipeline = job.entityRes->cachedIsGlass
							                                      ? &glassGraphicsPipeline
							                                      : &pbrBlendGraphicsPipeline;
						if (desiredPipeline != activeTransparentPipeline)
						{
							commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics,
							                                          **desiredPipeline);
							activeTransparentPipeline = desiredPipeline;
						}

						std::array<vk::Buffer, 2> buffers = {
							*job.meshRes->vertexBuffer, *job.entityRes->instanceBuffer
						};
						std::array<vk::DeviceSize, 2> offsets = {0, 0};
						commandBuffers[currentFrame].bindVertexBuffers(0, buffers, offsets);
						commandBuffers[currentFrame].bindIndexBuffer(*job.meshRes->indexBuffer, 0,
						                                             vk::IndexType::eUint32);

						vk::DescriptorSet set1 = (transparentDescriptorSets.empty() || IsLoading())
							                         ? *transparentFallbackDescriptorSets[currentFrame]
							                         : *transparentDescriptorSets[currentFrame];
						commandBuffers[currentFrame].bindDescriptorSets(
							vk::PipelineBindPoint::eGraphics,
							**currentLayout,
							0,
							{*job.entityRes->pbrDescriptorSets[currentFrame], set1},
							{});

						MaterialProperties pushConstants = job.entityRes->cachedMaterialProps;
						if (job.entityRes->cachedIsLiquid)
						{
							pushConstants.transmissionFactor = 0.0f;
						}
						commandBuffers[currentFrame].pushConstants<MaterialProperties>(
							**currentLayout, vk::ShaderStageFlagBits::eFragment, 0, {
								pushConstants
							}
						);
						uint32_t instanceCountT = std::max(1u, static_cast<uint32_t>(job.meshComp->GetInstanceCount()));
						commandBuffers[currentFrame].drawIndexed(job.meshRes->indexCount, instanceCountT, 0, 0, 0);
					}
				}
				// End transparent rendering pass before any layout transitions (even if no transparent draws)
				commandBuffers[currentFrame].endRendering();
				writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::Transparent, false);
			}
			{
				// Screenshot and final present transition are handled in rasterization path only
				// Ray query path handles these separately

				// Final layout transition for present (rasterization path only)
				{
					vk::ImageMemoryBarrier2 presentBarrier2{
						.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
						.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
						.dstStageMask = vk::PipelineStageFlagBits2::eNone,
						.dstAccessMask = {},
						.oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
						.newLayout = vk::ImageLayout::ePresentSrcKHR,
						.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
						.image = swapChainImages[imageIndex],
						.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
					};
					vk::DependencyInfo depToPresentFinal{
						.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &presentBarrier2
					};
					commandBuffers[currentFrame].pipelineBarrier2(depToPresentFinal);
					if (imageIndex < swapChainImageLayouts.size())
						swapChainImageLayouts[imageIndex] = presentBarrier2.newLayout;
				}
			}
		} // skip rasterization when ray query has rendered

		// Render ImGui UI overlay AFTER rasterization/ray query (must always execute regardless of render mode)
		// ImGui expects Render() to be called every frame after NewFrame() - skipping it causes hangs
		if (imguiSystem && !imguiSystem->IsFrameRendered())
		{
			// When ray query renders, swapchain is in PRESENT layout with valid content.
			// When rasterization renders, swapchain is also in PRESENT layout with valid content.
			// Transition to COLOR_ATTACHMENT with loadOp=eLoad to preserve existing pixels for ImGui overlay.
			vk::ImageMemoryBarrier2 presentToColor{
				.srcStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe,
				.srcAccessMask = vk::AccessFlagBits2::eNone,
				.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
				.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
				.oldLayout = (imageIndex < swapChainImageLayouts.size())
					             ? swapChainImageLayouts[imageIndex]
					             : vk::ImageLayout::eUndefined,
				.newLayout = vk::ImageLayout::eColorAttachmentOptimal,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = swapChainImages[imageIndex],
				.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
			};
			vk::DependencyInfo depInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &presentToColor};
			commandBuffers[currentFrame].pipelineBarrier2(depInfo);
			if (imageIndex < swapChainImageLayouts.size())
				swapChainImageLayouts[imageIndex] = presentToColor.newLayout;

			// Begin a dedicated render pass for ImGui (UI overlay)
			vk::RenderingAttachmentInfo imguiColorAttachment{
				.imageView = *swapChainImageViews[imageIndex],
				.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
				.loadOp = vk::AttachmentLoadOp::eLoad, // Load existing content
				.storeOp = vk::AttachmentStoreOp::eStore
			};
			vk::RenderingInfo imguiRenderingInfo{
				.renderArea = vk::Rect2D({0, 0}, swapChainExtent),
				.layerCount = 1,
				.colorAttachmentCount = 1,
				.pColorAttachments = &imguiColorAttachment,
				.pDepthAttachment = nullptr
			};
			commandBuffers[currentFrame].beginRendering(imguiRenderingInfo);
			writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::ImGui, true);

			imguiSystem->Render(commandBuffers[currentFrame], currentFrame);

			commandBuffers[currentFrame].endRendering();
			writeGpuProfileTimestamp(commandBuffers[currentFrame], GpuProfilePass::ImGui, false);

			// Transition swapchain back to PRESENT layout after ImGui renders
			vk::ImageMemoryBarrier2 colorToPresent{
				.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
				.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
				.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe,
				.dstAccessMask = vk::AccessFlagBits2::eNone,
				.oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
				.newLayout = vk::ImageLayout::ePresentSrcKHR,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = swapChainImages[imageIndex],
				.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
			};
			vk::DependencyInfo depInfoBack{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &colorToPresent};
			commandBuffers[currentFrame].pipelineBarrier2(depInfoBack);
			if (imageIndex < swapChainImageLayouts.size())
				swapChainImageLayouts[imageIndex] = colorToPresent.newLayout;
		}

		commandBuffers[currentFrame].end();
		isRecordingCmd.store(false, std::memory_order_relaxed);

		// Submit and present (Synchronization 2)
		uint64_t uploadsValueToWait = uploadTimelineLastSubmitted.load(std::memory_order_relaxed);

		// Use acquireSemaphoreIndex for imageAvailable semaphore (same as we used in acquireNextImage)
		// Use imageIndex for renderFinished semaphore (matches the image being presented)

		std::array<vk::SemaphoreSubmitInfo, 2> waitInfos = {
			vk::SemaphoreSubmitInfo{
				.semaphore = *imageAvailableSemaphores[acquireSemaphoreIndex],
				.value = 0,
				.stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
				.deviceIndex = 0
			},
			vk::SemaphoreSubmitInfo{
				.semaphore = *uploadsTimeline,
				.value = uploadsValueToWait,
				.stageMask = vk::PipelineStageFlagBits2::eFragmentShader,
				.deviceIndex = 0
			}
		};

		vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *commandBuffers[currentFrame], .deviceMask = 0};
		vk::SemaphoreSubmitInfo signalInfo{
			.semaphore = *renderFinishedSemaphores[imageIndex], .value = 0,
			.stageMask = vk::PipelineStageFlagBits2::eAllGraphics, .deviceIndex = 0
		};
		vk::SubmitInfo2 submit2{
			.waitSemaphoreInfoCount = static_cast<uint32_t>(waitInfos.size()),
			.pWaitSemaphoreInfos = waitInfos.data(),
			.commandBufferInfoCount = 1,
			.pCommandBufferInfos = &cmdInfo,
			.signalSemaphoreInfoCount = 1,
			.pSignalSemaphoreInfos = &signalInfo
		};

		if (framebufferResized.load(std::memory_order_relaxed))
		{
			vk::SubmitInfo2 emptySubmit2{};
			{
				std::lock_guard<std::mutex> lock(queueMutex);
				graphicsQueue.submit2(emptySubmit2, *inFlightFences[currentFrame]);
			}
			recreateSwapChain();
			return;
		}

		// Update watchdog BEFORE queue submit because submit can block waiting for GPU
		// This proves frame CPU work is complete even if GPU queue is busy
		lastFrameUpdateTime.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);
		{
			std::lock_guard<std::mutex> lock(queueMutex);
			graphicsQueue.submit2(submit2, *inFlightFences[currentFrame]);
		}
		if (isGpuProfilingActive() && currentFrame < gpuProfilingQueriesPending.size())
		{
			gpuProfilingQueriesPending[currentFrame] = 1u;
		}

		vk::PresentInfoKHR presentInfo{
			.waitSemaphoreCount = 1, .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex], .swapchainCount = 1,
			.pSwapchains = &*swapChain, .pImageIndices = &imageIndex
		};
		vk::Result presentResult = vk::Result::eSuccess;
		try
		{
			std::lock_guard<std::mutex> lock(queueMutex);
			presentResult = presentQueue.presentKHR(presentInfo);
		}
		catch (const vk::OutOfDateKHRError&)
		{
			framebufferResized.store(true, std::memory_order_relaxed);
		}
		if (presentResult == vk::Result::eSuboptimalKHR || framebufferResized.load(std::memory_order_relaxed))
		{
			framebufferResized.store(false, std::memory_order_relaxed);
			recreateSwapChain();
		}
		else if (presentResult != vk::Result::eSuccess)
		{
			throw std::runtime_error("Failed to present swap chain image");
		}

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	// Public toggle APIs for planar reflections (keyboard/UI)

}
