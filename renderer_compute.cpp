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
#include "renderer.h"
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>

// This file contains compute-related methods from the Renderer class

bool Renderer::createComputeResources()
{
	try
	{
		std::array<vk::DescriptorPoolSize, 4> poolSizes = {
		    vk::DescriptorPoolSize{
		        .type            = vk::DescriptorType::eStorageBuffer,
		        .descriptorCount = 16u * MAX_FRAMES_IN_FLIGHT
		    },
		    vk::DescriptorPoolSize{
		        .type            = vk::DescriptorType::eUniformBuffer,
		        .descriptorCount = 4u * MAX_FRAMES_IN_FLIGHT},
		    vk::DescriptorPoolSize{
		        .type            = vk::DescriptorType::eCombinedImageSampler,
		        .descriptorCount = 64u},
		    vk::DescriptorPoolSize{
		        .type            = vk::DescriptorType::eStorageImage,
		        .descriptorCount = 64u}};

		vk::DescriptorPoolCreateInfo poolInfo{
			.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
			.maxSets       = 10u * MAX_FRAMES_IN_FLIGHT + 48u,
		    .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
		    .pPoolSizes    = poolSizes.data()};

		computeDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);

		return true;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Failed to create compute resources: " << e.what() << std::endl;
		return false;
	}
}

// Forward+ compute (tiled light culling)
bool Renderer::createForwardPlusPipelinesAndResources()
{
	try
	{
		// Load compute shader
		auto                   cullSpv    = readFile("shaders/forward_plus_cull.spv");
		vk::raii::ShaderModule cullModule = createShaderModule(cullSpv);

		// Descriptor set layout: 0=lights SSBO (RO), 1=tile headers SSBO (RW), 2=tile indices SSBO (RW), 3=params UBO (RO)
		std::array<vk::DescriptorSetLayoutBinding, 4> bindings = {
		    vk::DescriptorSetLayoutBinding{.binding = 0, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
		    vk::DescriptorSetLayoutBinding{.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
		    vk::DescriptorSetLayoutBinding{.binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
		    vk::DescriptorSetLayoutBinding{.binding = 3, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}};

		vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};
		forwardPlusDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

		// Pipeline layout
		vk::PipelineLayoutCreateInfo plInfo{.setLayoutCount = 1, .pSetLayouts = &*forwardPlusDescriptorSetLayout};
		forwardPlusPipelineLayout = vk::raii::PipelineLayout(device, plInfo);

		// Pipeline
		vk::PipelineShaderStageCreateInfo stage{.stage = vk::ShaderStageFlagBits::eCompute, .module = *cullModule, .pName = "main"};
		vk::ComputePipelineCreateInfo     cpInfo{.stage = stage, .layout = *forwardPlusPipelineLayout};
		forwardPlusPipeline = vk::raii::Pipeline(device, nullptr, cpInfo);

		// Allocate per-frame structs
		forwardPlusPerFrame.resize(MAX_FRAMES_IN_FLIGHT);

		// Allocate compute descriptor sets (reuse computeDescriptorPool)
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *forwardPlusDescriptorSetLayout);
		vk::DescriptorSetAllocateInfo        allocInfo{.descriptorPool = *computeDescriptorPool, .descriptorSetCount = MAX_FRAMES_IN_FLIGHT, .pSetLayouts = layouts.data()};
		auto                                 sets = vk::raii::DescriptorSets(device, allocInfo);
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			forwardPlusPerFrame[i].computeSet = std::move(sets[i]);
		}

		// Initial buffer allocation based on current swapchain extent (also updates descriptors)
		uint32_t tilesX = (swapChainExtent.width + forwardPlusTileSizeX - 1) / forwardPlusTileSizeX;
		uint32_t tilesY = (swapChainExtent.height + forwardPlusTileSizeY - 1) / forwardPlusTileSizeY;
		if (!createOrResizeForwardPlusBuffers(tilesX, tilesY, forwardPlusSlicesZ))
		{
			return false;
		}

		if (!createDepthPyramidPipeline())
		{
			return false;
		}

		if (!createSAOPipeline())
		{
			return false;
		}

		if (!createVolumetricPipeline())
		{
			return false;
		}

		if (!createFrustumCullPipeline())
		{
			return false;
		}

		if (!createOcclusionCullPipeline())
		{
			return false;
		}

		if (!createOrResizeFrustumCullBuffers(1u))
		{
			return false;
		}

		return true;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Failed to create Forward+ compute resources: " << e.what() << std::endl;
		return false;
	}
}

bool Renderer::createDepthPyramidPipeline()
{
	if (!!*depthPyramidPipeline && !!*depthPyramidPipelineLayout && !!*depthPyramidDescriptorSetLayout)
	{
		return true;
	}

	try
	{
		auto                   depthPyramidSpv    = readFile("shaders/depth_pyramid.spv");
		vk::raii::ShaderModule depthPyramidModule = createShaderModule(depthPyramidSpv);

		std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {
		    vk::DescriptorSetLayoutBinding{.binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
		    vk::DescriptorSetLayoutBinding{.binding = 1, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}};

		vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};
		depthPyramidDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

		vk::PushConstantRange       pushRange{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(uint32_t) * 4};
		vk::PipelineLayoutCreateInfo plInfo{.setLayoutCount = 1, .pSetLayouts = &*depthPyramidDescriptorSetLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushRange};
		depthPyramidPipelineLayout = vk::raii::PipelineLayout(device, plInfo);

		vk::PipelineShaderStageCreateInfo stage{.stage = vk::ShaderStageFlagBits::eCompute, .module = *depthPyramidModule, .pName = "main"};
		vk::ComputePipelineCreateInfo     cpInfo{.stage = stage, .layout = *depthPyramidPipelineLayout};
		depthPyramidPipeline = vk::raii::Pipeline(device, nullptr, cpInfo);

		return true;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Failed to create depth pyramid compute resources: " << e.what() << std::endl;
		return false;
	}
}

bool Renderer::createSAOPipeline()
{
	if (!!*saoPipeline && !!*saoPipelineLayout && !!*saoDescriptorSetLayout)
	{
		return true;
	}

	try
	{
		auto                   saoSpv    = readFile("shaders/sao.spv");
		vk::raii::ShaderModule saoModule = createShaderModule(saoSpv);

		std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {
		    vk::DescriptorSetLayoutBinding{.binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
		    vk::DescriptorSetLayoutBinding{.binding = 1, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}};

		vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};
		saoDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

		vk::PushConstantRange pushRange{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(SAOPushConstants)};
		vk::PipelineLayoutCreateInfo plInfo{.setLayoutCount = 1, .pSetLayouts = &*saoDescriptorSetLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushRange};
		saoPipelineLayout = vk::raii::PipelineLayout(device, plInfo);

		vk::PipelineShaderStageCreateInfo stage{.stage = vk::ShaderStageFlagBits::eCompute, .module = *saoModule, .pName = "main"};
		vk::ComputePipelineCreateInfo     cpInfo{.stage = stage, .layout = *saoPipelineLayout};
		saoPipeline = vk::raii::Pipeline(device, nullptr, cpInfo);
		return true;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Failed to create SAO compute pipeline: " << e.what() << std::endl;
		return false;
	}
}

bool Renderer::createVolumetricPipeline()
{
	if (!!*volumetricPipeline && !!*volumetricPipelineLayout && !!*volumetricDescriptorSetLayout)
	{
		return true;
	}

	try
	{
		auto                   volumetricSpv    = readFile("shaders/volumetric.spv");
		vk::raii::ShaderModule volumetricModule = createShaderModule(volumetricSpv);

		std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {
		    vk::DescriptorSetLayoutBinding{.binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
		    vk::DescriptorSetLayoutBinding{.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
		    vk::DescriptorSetLayoutBinding{.binding = 2, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}};

		vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};
		volumetricDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

		vk::PushConstantRange pushRange{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(VolumetricPushConstants)};
		vk::PipelineLayoutCreateInfo plInfo{.setLayoutCount = 1, .pSetLayouts = &*volumetricDescriptorSetLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushRange};
		volumetricPipelineLayout = vk::raii::PipelineLayout(device, plInfo);

		vk::PipelineShaderStageCreateInfo stage{.stage = vk::ShaderStageFlagBits::eCompute, .module = *volumetricModule, .pName = "main"};
		vk::ComputePipelineCreateInfo     cpInfo{.stage = stage, .layout = *volumetricPipelineLayout};
		volumetricPipeline = vk::raii::Pipeline(device, nullptr, cpInfo);
		return true;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Failed to create volumetric compute pipeline: " << e.what() << std::endl;
		return false;
	}
}

bool Renderer::createFrustumCullPipeline()
{
	try
	{
		if (!*computeDescriptorPool)
		{
			return false;
		}

		if (!*frustumCullPipeline || !*frustumCullPipelineLayout || !*frustumCullDescriptorSetLayout)
		{
			auto                   frustumCullSpv    = readFile("shaders/frustum_cull.spv");
			vk::raii::ShaderModule frustumCullModule = createShaderModule(frustumCullSpv);

			std::array<vk::DescriptorSetLayoutBinding, 4> bindings = {
			    vk::DescriptorSetLayoutBinding{.binding = 0, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
			    vk::DescriptorSetLayoutBinding{.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
			    vk::DescriptorSetLayoutBinding{.binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
			    vk::DescriptorSetLayoutBinding{.binding = 3, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}};

			vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};
			frustumCullDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

			vk::PipelineLayoutCreateInfo plInfo{.setLayoutCount = 1, .pSetLayouts = &*frustumCullDescriptorSetLayout};
			frustumCullPipelineLayout = vk::raii::PipelineLayout(device, plInfo);

			vk::PipelineShaderStageCreateInfo stage{.stage = vk::ShaderStageFlagBits::eCompute, .module = *frustumCullModule, .pName = "main"};
			vk::ComputePipelineCreateInfo     cpInfo{.stage = stage, .layout = *frustumCullPipelineLayout};
			frustumCullPipeline = vk::raii::Pipeline(device, nullptr, cpInfo);
		}

		bool needSetAllocation = (frustumCullPerFrame.size() != MAX_FRAMES_IN_FLIGHT);
		if (!needSetAllocation)
		{
			for (const auto &f : frustumCullPerFrame)
			{
				if (!*f.computeSet)
				{
					needSetAllocation = true;
					break;
				}
			}
		}

		if (needSetAllocation)
		{
			frustumCullPerFrame.clear();
			frustumCullPerFrame.resize(MAX_FRAMES_IN_FLIGHT);

			std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *frustumCullDescriptorSetLayout);
			vk::DescriptorSetAllocateInfo        allocInfo{.descriptorPool = *computeDescriptorPool, .descriptorSetCount = MAX_FRAMES_IN_FLIGHT, .pSetLayouts = layouts.data()};
			auto                                 sets = vk::raii::DescriptorSets(device, allocInfo);
			for (size_t i = 0; i < frustumCullPerFrame.size(); ++i)
			{
				frustumCullPerFrame[i].computeSet = std::move(sets[i]);
			}
		}

		return true;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Failed to create frustum culling compute resources: " << e.what() << std::endl;
		return false;
	}
}

bool Renderer::createOcclusionCullPipeline()
{
	try
	{
		if (!*computeDescriptorPool)
		{
			return false;
		}

		if (!*occlusionCullPipeline || !*occlusionCullPipelineLayout || !*occlusionCullDescriptorSetLayout)
		{
			auto                   occlusionCullSpv    = readFile("shaders/occlusion_cull.spv");
			vk::raii::ShaderModule occlusionCullModule = createShaderModule(occlusionCullSpv);

			std::array<vk::DescriptorSetLayoutBinding, 7> bindings = {
			    vk::DescriptorSetLayoutBinding{.binding = 0, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
			    vk::DescriptorSetLayoutBinding{.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
			    vk::DescriptorSetLayoutBinding{.binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
			    vk::DescriptorSetLayoutBinding{.binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
			    vk::DescriptorSetLayoutBinding{.binding = 4, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
			    vk::DescriptorSetLayoutBinding{.binding = 5, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
			    vk::DescriptorSetLayoutBinding{.binding = 6, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}};

			vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};
			occlusionCullDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

			vk::PipelineLayoutCreateInfo plInfo{.setLayoutCount = 1, .pSetLayouts = &*occlusionCullDescriptorSetLayout};
			occlusionCullPipelineLayout = vk::raii::PipelineLayout(device, plInfo);

			vk::PipelineShaderStageCreateInfo stage{.stage = vk::ShaderStageFlagBits::eCompute, .module = *occlusionCullModule, .pName = "main"};
			vk::ComputePipelineCreateInfo     cpInfo{.stage = stage, .layout = *occlusionCullPipelineLayout};
			occlusionCullPipeline = vk::raii::Pipeline(device, nullptr, cpInfo);
		}

		if (frustumCullPerFrame.size() != MAX_FRAMES_IN_FLIGHT)
		{
			frustumCullPerFrame.resize(MAX_FRAMES_IN_FLIGHT);
		}

		bool needSetAllocation = false;
		for (const auto &f : frustumCullPerFrame)
		{
			if (!*f.occlusionComputeSet)
			{
				needSetAllocation = true;
				break;
			}
		}

		if (needSetAllocation)
		{
			std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *occlusionCullDescriptorSetLayout);
			vk::DescriptorSetAllocateInfo        allocInfo{.descriptorPool = *computeDescriptorPool, .descriptorSetCount = MAX_FRAMES_IN_FLIGHT, .pSetLayouts = layouts.data()};
			auto                                 sets = vk::raii::DescriptorSets(device, allocInfo);
			for (size_t i = 0; i < frustumCullPerFrame.size(); ++i)
			{
				frustumCullPerFrame[i].occlusionComputeSet = std::move(sets[i]);
			}
		}

		return true;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Failed to create occlusion culling compute resources: " << e.what() << std::endl;
		return false;
	}
}

bool Renderer::createOrResizeFrustumCullBuffers(uint32_t instanceCapacity, bool updateOnlyCurrentFrame)
{
	try
	{
		if (!createFrustumCullPipeline())
		{
			return false;
		}

		if (!createOcclusionCullPipeline())
		{
			return false;
		}

		const size_t requestedCapacity = std::max<size_t>(1u, static_cast<size_t>(instanceCapacity));

		size_t beginFrame = 0;
		size_t endFrame   = MAX_FRAMES_IN_FLIGHT;
		if (updateOnlyCurrentFrame)
		{
			beginFrame = static_cast<size_t>(currentFrame);
			endFrame   = beginFrame + 1;
		}

		for (size_t i = beginFrame; i < endFrame; ++i)
		{
			auto &f = frustumCullPerFrame[i];

			bool needAabbBuffer = (f.instanceCapacity < requestedCapacity) || !*f.instanceAabbBuffer;
			if (needAabbBuffer)
			{
				if (!!*f.instanceAabbBuffer)
				{
					f.instanceAabbBuffer = vk::raii::Buffer(nullptr);
					f.instanceAabbBufferAllocation.reset();
				}
				auto [buf, alloc] = createBufferPooled(
				    requestedCapacity * sizeof(FrustumCullAABB),
				    vk::BufferUsageFlagBits::eStorageBuffer,
				    vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
				f.instanceAabbBuffer           = std::move(buf);
				f.instanceAabbBufferAllocation = std::move(alloc);
				f.instanceAabbMapped           = f.instanceAabbBufferAllocation ? f.instanceAabbBufferAllocation->mappedPtr : nullptr;
				f.instanceCapacity             = requestedCapacity;
			}

			bool needVisibleIndicesBuffer = (f.visibleIndicesCapacity < requestedCapacity) || !*f.visibleIndicesBuffer;
			if (needVisibleIndicesBuffer)
			{
				if (!!*f.visibleIndicesBuffer)
				{
					f.visibleIndicesBuffer = vk::raii::Buffer(nullptr);
					f.visibleIndicesBufferAllocation.reset();
				}
				auto [buf, alloc] = createBufferPooled(
				    requestedCapacity * sizeof(uint32_t),
				    vk::BufferUsageFlagBits::eStorageBuffer,
				    vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
				f.visibleIndicesBuffer           = std::move(buf);
				f.visibleIndicesBufferAllocation = std::move(alloc);
				f.visibleIndicesMapped           = f.visibleIndicesBufferAllocation ? f.visibleIndicesBufferAllocation->mappedPtr : nullptr;
				f.visibleIndicesCapacity         = requestedCapacity;
			}

			bool needVisibleCountBuffer = !*f.visibleCountBuffer;
			if (needVisibleCountBuffer)
			{
				auto [buf, alloc] = createBufferPooled(
				    sizeof(uint32_t),
				    vk::BufferUsageFlagBits::eStorageBuffer,
				    vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
				f.visibleCountBuffer           = std::move(buf);
				f.visibleCountBufferAllocation = std::move(alloc);
				f.visibleCountMapped           = f.visibleCountBufferAllocation ? f.visibleCountBufferAllocation->mappedPtr : nullptr;
			}

			bool needOcclusionVisibleIndicesBuffer = (f.occlusionVisibleIndicesCapacity < requestedCapacity) || !*f.occlusionVisibleIndicesBuffer;
			if (needOcclusionVisibleIndicesBuffer)
			{
				if (!!*f.occlusionVisibleIndicesBuffer)
				{
					f.occlusionVisibleIndicesBuffer = vk::raii::Buffer(nullptr);
					f.occlusionVisibleIndicesBufferAllocation.reset();
				}
				auto [buf, alloc] = createBufferPooled(
				    requestedCapacity * sizeof(uint32_t),
				    vk::BufferUsageFlagBits::eStorageBuffer,
				    vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
				f.occlusionVisibleIndicesBuffer           = std::move(buf);
				f.occlusionVisibleIndicesBufferAllocation = std::move(alloc);
				f.occlusionVisibleIndicesMapped           = f.occlusionVisibleIndicesBufferAllocation ? f.occlusionVisibleIndicesBufferAllocation->mappedPtr : nullptr;
				f.occlusionVisibleIndicesCapacity         = requestedCapacity;
			}

			bool needOcclusionVisibleCountBuffer = !*f.occlusionVisibleCountBuffer;
			if (needOcclusionVisibleCountBuffer)
			{
				auto [buf, alloc] = createBufferPooled(
				    sizeof(uint32_t),
				    vk::BufferUsageFlagBits::eStorageBuffer,
				    vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
				f.occlusionVisibleCountBuffer           = std::move(buf);
				f.occlusionVisibleCountBufferAllocation = std::move(alloc);
				f.occlusionVisibleCountMapped           = f.occlusionVisibleCountBufferAllocation ? f.occlusionVisibleCountBufferAllocation->mappedPtr : nullptr;
			}

			bool needParamsBuffer = !*f.paramsBuffer;
			if (needParamsBuffer)
			{
				auto [buf, alloc] = createBufferPooled(
				    sizeof(FrustumCullParams),
				    vk::BufferUsageFlagBits::eUniformBuffer,
				    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
				f.paramsBuffer           = std::move(buf);
				f.paramsBufferAllocation = std::move(alloc);
				f.paramsMapped           = f.paramsBufferAllocation ? f.paramsBufferAllocation->mappedPtr : nullptr;
			}

			bool needOcclusionParamsBuffer = !*f.occlusionParamsBuffer;
			if (needOcclusionParamsBuffer)
			{
				auto [buf, alloc] = createBufferPooled(
				    sizeof(OcclusionCullParams),
				    vk::BufferUsageFlagBits::eUniformBuffer,
				    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
				f.occlusionParamsBuffer           = std::move(buf);
				f.occlusionParamsBufferAllocation = std::move(alloc);
				f.occlusionParamsMapped           = f.occlusionParamsBufferAllocation ? f.occlusionParamsBufferAllocation->mappedPtr : nullptr;
			}

			bool needIndirectCommandsBuffer = (f.indirectCommandCapacity < requestedCapacity) || !*f.indirectCommandsBuffer;
			if (needIndirectCommandsBuffer)
			{
				if (!!*f.indirectCommandsBuffer)
				{
					f.indirectCommandsBuffer = vk::raii::Buffer(nullptr);
					f.indirectCommandsBufferAllocation.reset();
				}

				auto [buf, alloc] = createBufferPooled(
				    requestedCapacity * sizeof(vk::DrawIndexedIndirectCommand),
				    vk::BufferUsageFlagBits::eIndirectBuffer,
				    vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
				f.indirectCommandsBuffer           = std::move(buf);
				f.indirectCommandsBufferAllocation = std::move(alloc);
				f.indirectCommandsMapped           = f.indirectCommandsBufferAllocation ? f.indirectCommandsBufferAllocation->mappedPtr : nullptr;
				f.indirectCommandCapacity          = requestedCapacity;

				if (f.indirectCommandsMapped)
				{
					std::memset(f.indirectCommandsMapped, 0, requestedCapacity * sizeof(vk::DrawIndexedIndirectCommand));
				}
			}

			if (!descriptorSetsValid.load(std::memory_order_relaxed) || isRecordingCmd.load(std::memory_order_relaxed))
			{
				continue;
			}

			if (*f.computeSet && *f.instanceAabbBuffer && *f.visibleIndicesBuffer && *f.visibleCountBuffer && *f.paramsBuffer)
			{
				vk::DescriptorBufferInfo aabbInfo{.buffer = *f.instanceAabbBuffer, .offset = 0, .range = VK_WHOLE_SIZE};
				vk::DescriptorBufferInfo visibleIndicesInfo{.buffer = *f.visibleIndicesBuffer, .offset = 0, .range = VK_WHOLE_SIZE};
				vk::DescriptorBufferInfo visibleCountInfo{.buffer = *f.visibleCountBuffer, .offset = 0, .range = VK_WHOLE_SIZE};
				vk::DescriptorBufferInfo paramsInfo{.buffer = *f.paramsBuffer, .offset = 0, .range = VK_WHOLE_SIZE};

				std::array<vk::WriteDescriptorSet, 4> writes = {
				    vk::WriteDescriptorSet{.dstSet = *f.computeSet, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &aabbInfo},
				    vk::WriteDescriptorSet{.dstSet = *f.computeSet, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &visibleIndicesInfo},
				    vk::WriteDescriptorSet{.dstSet = *f.computeSet, .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &visibleCountInfo},
				    vk::WriteDescriptorSet{.dstSet = *f.computeSet, .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &paramsInfo}};

				std::lock_guard<std::mutex> lk(descriptorMutex);
				device.updateDescriptorSets(writes, {});
			}

			if (*f.occlusionComputeSet && *f.instanceAabbBuffer && *f.visibleIndicesBuffer && *f.visibleCountBuffer && *f.occlusionVisibleIndicesBuffer && *f.occlusionVisibleCountBuffer && *f.occlusionParamsBuffer && *depthPyramidSampler && *depthPyramidFullView)
			{
				vk::DescriptorBufferInfo aabbInfo{.buffer = *f.instanceAabbBuffer, .offset = 0, .range = VK_WHOLE_SIZE};
				vk::DescriptorBufferInfo frustumVisibleIndicesInfo{.buffer = *f.visibleIndicesBuffer, .offset = 0, .range = VK_WHOLE_SIZE};
				vk::DescriptorBufferInfo frustumVisibleCountInfo{.buffer = *f.visibleCountBuffer, .offset = 0, .range = VK_WHOLE_SIZE};
				vk::DescriptorImageInfo  depthPyramidInfo{.sampler = *depthPyramidSampler, .imageView = *depthPyramidFullView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
				vk::DescriptorBufferInfo occlusionVisibleIndicesInfo{.buffer = *f.occlusionVisibleIndicesBuffer, .offset = 0, .range = VK_WHOLE_SIZE};
				vk::DescriptorBufferInfo occlusionVisibleCountInfo{.buffer = *f.occlusionVisibleCountBuffer, .offset = 0, .range = VK_WHOLE_SIZE};
				vk::DescriptorBufferInfo occlusionParamsInfo{.buffer = *f.occlusionParamsBuffer, .offset = 0, .range = VK_WHOLE_SIZE};

				std::array<vk::WriteDescriptorSet, 7> writes = {
				    vk::WriteDescriptorSet{.dstSet = *f.occlusionComputeSet, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &aabbInfo},
				    vk::WriteDescriptorSet{.dstSet = *f.occlusionComputeSet, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &frustumVisibleIndicesInfo},
				    vk::WriteDescriptorSet{.dstSet = *f.occlusionComputeSet, .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &frustumVisibleCountInfo},
				    vk::WriteDescriptorSet{.dstSet = *f.occlusionComputeSet, .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &depthPyramidInfo},
				    vk::WriteDescriptorSet{.dstSet = *f.occlusionComputeSet, .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &occlusionVisibleIndicesInfo},
				    vk::WriteDescriptorSet{.dstSet = *f.occlusionComputeSet, .dstBinding = 5, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &occlusionVisibleCountInfo},
				    vk::WriteDescriptorSet{.dstSet = *f.occlusionComputeSet, .dstBinding = 6, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &occlusionParamsInfo}};

				std::lock_guard<std::mutex> lk(descriptorMutex);
				device.updateDescriptorSets(writes, {});
			}
		}

		return true;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Failed to create/resize frustum culling buffers: " << e.what() << std::endl;
		return false;
	}
}

void Renderer::updateFrustumCullParams(uint32_t frameIndex, const glm::mat4 &vp, uint32_t instanceCount)
{
	if (frameIndex >= frustumCullPerFrame.size())
		return;

	auto &f = frustumCullPerFrame[frameIndex];
	const FrustumPlanes planes = extractFrustumPlanes(vp);

	if (f.paramsMapped)
	{
		FrustumCullParams params{};
		for (uint32_t i = 0; i < 6; ++i)
		{
			params.frustumPlanes[i] = planes.planes[i];
		}
		params.instanceCount = instanceCount;

		std::memcpy(f.paramsMapped, &params, sizeof(FrustumCullParams));
	}

	if (f.occlusionParamsMapped)
	{
		OcclusionCullParams params{};
		params.clipFromWorld = vp;
		params.depthPyramidInfo = glm::vec4(
		    static_cast<float>(std::max(1u, swapChainExtent.width)),
		    static_cast<float>(std::max(1u, swapChainExtent.height)),
		    static_cast<float>(std::max(1u, depthPyramidMipCount)),
		    0.0035f);
		params.safetyInfo = glm::vec4(2.0f, 0.35f, 0.0015f, 0.0f);

		std::memcpy(f.occlusionParamsMapped, &params, sizeof(OcclusionCullParams));
	}
}

void Renderer::dispatchFrustumCull(vk::raii::CommandBuffer &cmd, uint32_t instanceCount)
{
	if (!*frustumCullPipeline || !*frustumCullPipelineLayout)
		return;
	if (currentFrame >= frustumCullPerFrame.size())
		return;
	if (instanceCount == 0)
		return;

	auto &f = frustumCullPerFrame[currentFrame];
	if (!*f.computeSet || !*f.instanceAabbBuffer || !*f.visibleIndicesBuffer || !*f.visibleCountBuffer || !*f.paramsBuffer)
		return;

	if (f.occlusionVisibleCountMapped)
	{
		const uint32_t zero = 0;
		std::memcpy(f.occlusionVisibleCountMapped, &zero, sizeof(uint32_t));
	}

	vk::MemoryBarrier2 hostToCompute{
	    .srcStageMask  = vk::PipelineStageFlagBits2::eHost,
	    .srcAccessMask = vk::AccessFlagBits2::eHostWrite,
	    .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
	    .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite};
	vk::DependencyInfo depHostToCompute{.memoryBarrierCount = 1, .pMemoryBarriers = &hostToCompute};
	cmd.pipelineBarrier2(depHostToCompute);

	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *frustumCullPipeline);
	vk::DescriptorSet set = *f.computeSet;
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *frustumCullPipelineLayout, 0, set, {});

	const uint32_t groupsX = (instanceCount + 63u) / 64u;
	cmd.dispatch(std::max(1u, groupsX), 1, 1);

	vk::MemoryBarrier2 frustumToConsumers{
	    .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
	    .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
	    .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eVertexShader | vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eDrawIndirect,
	    .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eIndirectCommandRead};
	vk::DependencyInfo depFrustum{.memoryBarrierCount = 1, .pMemoryBarriers = &frustumToConsumers};

	const bool canRunOcclusion = enableOcclusionCulling && depthPyramidHistoryValid &&
	                             !!*occlusionCullPipeline && !!*occlusionCullPipelineLayout &&
	                             !!*f.occlusionComputeSet &&
	                             !!*f.occlusionVisibleIndicesBuffer && !!*f.occlusionVisibleCountBuffer && !!*f.occlusionParamsBuffer &&
	                             !!*depthPyramidImage && !!*depthPyramidFullView && !!*depthPyramidSampler &&
	                             (depthPyramidMipCount > 0);

	if (!canRunOcclusion)
	{
		cmd.pipelineBarrier2(depFrustum);
		return;
	}

	cmd.pipelineBarrier2(depFrustum);

	vk::ImageMemoryBarrier2 depthPyramidReadBarrier{
	    .srcStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
	    .srcAccessMask       = vk::AccessFlagBits2::eShaderWrite,
	    .dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
	    .dstAccessMask       = vk::AccessFlagBits2::eShaderSampledRead,
	    .oldLayout           = vk::ImageLayout::eShaderReadOnlyOptimal,
	    .newLayout           = vk::ImageLayout::eShaderReadOnlyOptimal,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .image               = *depthPyramidImage,
	    .subresourceRange    = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = depthPyramidMipCount, .baseArrayLayer = 0, .layerCount = 1}};
	vk::DependencyInfo depDepthRead{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &depthPyramidReadBarrier};
	cmd.pipelineBarrier2(depDepthRead);

	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *occlusionCullPipeline);
	vk::DescriptorSet occlusionSet = *f.occlusionComputeSet;
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *occlusionCullPipelineLayout, 0, occlusionSet, {});
	cmd.dispatch(std::max(1u, groupsX), 1, 1);

	vk::MemoryBarrier2 occlusionToConsumers{
	    .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
	    .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
	    .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eVertexShader | vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eDrawIndirect,
	    .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eIndirectCommandRead};
	vk::DependencyInfo depOcclusion{.memoryBarrierCount = 1, .pMemoryBarriers = &occlusionToConsumers};
	cmd.pipelineBarrier2(depOcclusion);
}

void Renderer::destroyDepthPyramidResources()
{
	depthPyramidDescriptorSets.clear();
	depthPyramidDispatchSteps.clear();
	depthPyramidMipLayouts.clear();
	depthPyramidMipViews.clear();
	depthPyramidFullView = vk::raii::ImageView(nullptr);
	depthPyramidImageAllocation.reset();
	depthPyramidImage    = vk::raii::Image(nullptr);
	depthPyramidMipCount = 0;
	depthPyramidHistoryValid = false;
}

void Renderer::destroySAOResources()
{
	saoSampler = vk::raii::Sampler(nullptr);
	saoDescriptorSets.clear();
	saoImages.clear();
	saoImageAllocations.clear();
	saoImageViews.clear();
	saoImageLayouts.clear();
	saoHistoryValid = false;
}

void Renderer::destroyVolumetricResources()
{
	volumetricSampler = vk::raii::Sampler(nullptr);
	volumetricDescriptorSets.clear();
	volumetricImages.clear();
	volumetricImageAllocations.clear();
	volumetricImageViews.clear();
	volumetricImageLayouts.clear();
	volumetricHistoryValid = false;
}

bool Renderer::createSAOResources()
{
	try
	{
		destroySAOResources();

		if (!*computeDescriptorPool || !*depthPyramidFullView || !*depthPyramidSampler || depthPyramidMipCount == 0)
		{
			return false;
		}

		if (!createSAOPipeline())
		{
			return false;
		}

		saoImages.reserve(MAX_FRAMES_IN_FLIGHT);
		saoImageAllocations.reserve(MAX_FRAMES_IN_FLIGHT);
		saoImageViews.reserve(MAX_FRAMES_IN_FLIGHT);
		saoImageLayouts.reserve(MAX_FRAMES_IN_FLIGHT);
		saoDescriptorSets.reserve(MAX_FRAMES_IN_FLIGHT);

		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			auto [image, allocation] = createImagePooled(
			    swapChainExtent.width,
			    swapChainExtent.height,
			    vk::Format::eR8G8B8A8Unorm,
			    vk::ImageTiling::eOptimal,
			    vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
			    vk::MemoryPropertyFlagBits::eDeviceLocal);

			saoImages.push_back(std::move(image));
			saoImageAllocations.push_back(std::move(allocation));
			saoImageViews.push_back(createImageView(saoImages.back(), vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor));
			transitionImageLayout(*saoImages.back(), vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);
			saoImageLayouts.push_back(vk::ImageLayout::eShaderReadOnlyOptimal);
		}

		vk::SamplerCreateInfo samplerInfo{
		    .magFilter               = vk::Filter::eLinear,
		    .minFilter               = vk::Filter::eLinear,
		    .mipmapMode              = vk::SamplerMipmapMode::eLinear,
		    .addressModeU            = vk::SamplerAddressMode::eClampToEdge,
		    .addressModeV            = vk::SamplerAddressMode::eClampToEdge,
		    .addressModeW            = vk::SamplerAddressMode::eClampToEdge,
		    .minLod                  = 0.0f,
		    .maxLod                  = 0.0f,
		    .borderColor             = vk::BorderColor::eFloatOpaqueWhite,
		    .unnormalizedCoordinates = VK_FALSE};
		saoSampler = vk::raii::Sampler(device, samplerInfo);

		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *saoDescriptorSetLayout);
		vk::DescriptorSetAllocateInfo        allocInfo{.descriptorPool = *computeDescriptorPool, .descriptorSetCount = MAX_FRAMES_IN_FLIGHT, .pSetLayouts = layouts.data()};
		auto                                 sets = vk::raii::DescriptorSets(device, allocInfo);

		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			saoDescriptorSets.emplace_back(std::move(sets[i]));

			vk::DescriptorImageInfo depthInfo{.sampler = *depthPyramidSampler, .imageView = *depthPyramidFullView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
			vk::DescriptorImageInfo outInfo{.imageView = *saoImageViews[i], .imageLayout = vk::ImageLayout::eGeneral};

			std::array<vk::WriteDescriptorSet, 2> writes = {
			    vk::WriteDescriptorSet{.dstSet = *saoDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &depthInfo},
			    vk::WriteDescriptorSet{.dstSet = *saoDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &outInfo}};

			std::lock_guard<std::mutex> lk(descriptorMutex);
			device.updateDescriptorSets(writes, {});
		}

		saoHistoryValid = false;
		return true;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Failed to create SAO resources: " << e.what() << std::endl;
		destroySAOResources();
		return false;
	}
}

bool Renderer::createVolumetricResources()
{
	try
	{
		destroyVolumetricResources();

		if (!*computeDescriptorPool || !*depthPyramidFullView || !*depthPyramidSampler || depthPyramidMipCount == 0)
		{
			return false;
		}

		if (!createVolumetricPipeline())
		{
			return false;
		}

		if (lightStorageBuffers.size() < MAX_FRAMES_IN_FLIGHT)
		{
			return false;
		}

		volumetricImages.reserve(MAX_FRAMES_IN_FLIGHT);
		volumetricImageAllocations.reserve(MAX_FRAMES_IN_FLIGHT);
		volumetricImageViews.reserve(MAX_FRAMES_IN_FLIGHT);
		volumetricImageLayouts.reserve(MAX_FRAMES_IN_FLIGHT);
		volumetricDescriptorSets.reserve(MAX_FRAMES_IN_FLIGHT);

		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			auto [image, allocation] = createImagePooled(
			    swapChainExtent.width,
			    swapChainExtent.height,
			    vk::Format::eR16G16B16A16Sfloat,
			    vk::ImageTiling::eOptimal,
			    vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
			    vk::MemoryPropertyFlagBits::eDeviceLocal);

			volumetricImages.push_back(std::move(image));
			volumetricImageAllocations.push_back(std::move(allocation));
			volumetricImageViews.push_back(createImageView(volumetricImages.back(), vk::Format::eR16G16B16A16Sfloat, vk::ImageAspectFlagBits::eColor));
			transitionImageLayout(*volumetricImages.back(), vk::Format::eR16G16B16A16Sfloat, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);
			volumetricImageLayouts.push_back(vk::ImageLayout::eShaderReadOnlyOptimal);
		}

		vk::SamplerCreateInfo samplerInfo{
		    .magFilter               = vk::Filter::eLinear,
		    .minFilter               = vk::Filter::eLinear,
		    .mipmapMode              = vk::SamplerMipmapMode::eLinear,
		    .addressModeU            = vk::SamplerAddressMode::eClampToEdge,
		    .addressModeV            = vk::SamplerAddressMode::eClampToEdge,
		    .addressModeW            = vk::SamplerAddressMode::eClampToEdge,
		    .minLod                  = 0.0f,
		    .maxLod                  = 0.0f,
		    .borderColor             = vk::BorderColor::eFloatOpaqueBlack,
		    .unnormalizedCoordinates = VK_FALSE};
		volumetricSampler = vk::raii::Sampler(device, samplerInfo);

		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *volumetricDescriptorSetLayout);
		vk::DescriptorSetAllocateInfo        allocInfo{.descriptorPool = *computeDescriptorPool, .descriptorSetCount = MAX_FRAMES_IN_FLIGHT, .pSetLayouts = layouts.data()};
		auto                                 sets = vk::raii::DescriptorSets(device, allocInfo);

		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			volumetricDescriptorSets.emplace_back(std::move(sets[i]));

			vk::DescriptorImageInfo depthInfo{.sampler = *depthPyramidSampler, .imageView = *depthPyramidFullView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
			vk::DescriptorBufferInfo lightInfo{.buffer = *lightStorageBuffers[i].buffer, .offset = 0, .range = VK_WHOLE_SIZE};
			vk::DescriptorImageInfo outInfo{.imageView = *volumetricImageViews[i], .imageLayout = vk::ImageLayout::eGeneral};

			std::array<vk::WriteDescriptorSet, 3> writes = {
			    vk::WriteDescriptorSet{.dstSet = *volumetricDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &depthInfo},
			    vk::WriteDescriptorSet{.dstSet = *volumetricDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &lightInfo},
			    vk::WriteDescriptorSet{.dstSet = *volumetricDescriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &outInfo}};

			std::lock_guard<std::mutex> lk(descriptorMutex);
			device.updateDescriptorSets(writes, {});
		}

		volumetricHistoryValid = false;
		return true;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Failed to create volumetric resources: " << e.what() << std::endl;
		destroyVolumetricResources();
		return false;
	}
}

bool Renderer::createDepthPyramidResources()
{
	try
	{
		if (!*depthImage || !*depthImageView || swapChainExtent.width == 0 || swapChainExtent.height == 0)
		{
			return false;
		}

		if (!createDepthPyramidPipeline())
		{
			return false;
		}

		destroyDepthPyramidResources();

		uint32_t maxDim = std::max(swapChainExtent.width, swapChainExtent.height);
		depthPyramidMipCount = 1;
		while (maxDim > 1)
		{
			maxDim = (maxDim + 1) / 2;
			++depthPyramidMipCount;
		}

		std::tie(depthPyramidImage, depthPyramidImageAllocation) = createImagePooled(
		    swapChainExtent.width,
		    swapChainExtent.height,
		    depthPyramidFormat,
		    vk::ImageTiling::eOptimal,
		    vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
		    vk::MemoryPropertyFlagBits::eDeviceLocal,
		    depthPyramidMipCount);

		depthPyramidMipViews.reserve(depthPyramidMipCount);
		for (uint32_t mip = 0; mip < depthPyramidMipCount; ++mip)
		{
			vk::ImageViewCreateInfo viewInfo{
			    .image            = *depthPyramidImage,
			    .viewType         = vk::ImageViewType::e2D,
			    .format           = depthPyramidFormat,
			    .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = mip, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
			depthPyramidMipViews.emplace_back(device, viewInfo);
		}

		vk::ImageViewCreateInfo fullViewInfo{
		    .image            = *depthPyramidImage,
		    .viewType         = vk::ImageViewType::e2D,
		    .format           = depthPyramidFormat,
		    .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = depthPyramidMipCount, .baseArrayLayer = 0, .layerCount = 1}};
		depthPyramidFullView = vk::raii::ImageView(device, fullViewInfo);

		transitionImageLayout(*depthPyramidImage, depthPyramidFormat, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, depthPyramidMipCount);
		depthPyramidMipLayouts.assign(depthPyramidMipCount, vk::ImageLayout::eGeneral);

		depthPyramidSampler = vk::raii::Sampler(nullptr);
		vk::SamplerCreateInfo samplerInfo{
		    .magFilter               = vk::Filter::eNearest,
		    .minFilter               = vk::Filter::eNearest,
		    .mipmapMode              = vk::SamplerMipmapMode::eNearest,
		    .addressModeU            = vk::SamplerAddressMode::eClampToEdge,
		    .addressModeV            = vk::SamplerAddressMode::eClampToEdge,
		    .addressModeW            = vk::SamplerAddressMode::eClampToEdge,
		    .minLod                  = 0.0f,
		    .maxLod                  = static_cast<float>(depthPyramidMipCount - 1),
		    .borderColor             = vk::BorderColor::eFloatOpaqueWhite,
		    .unnormalizedCoordinates = VK_FALSE};
		depthPyramidSampler = vk::raii::Sampler(device, samplerInfo);

		std::vector<vk::DescriptorSetLayout> layouts(depthPyramidMipCount, *depthPyramidDescriptorSetLayout);
		vk::DescriptorSetAllocateInfo        allocInfo{.descriptorPool = *computeDescriptorPool, .descriptorSetCount = depthPyramidMipCount, .pSetLayouts = layouts.data()};
		auto                                 sets = vk::raii::DescriptorSets(device, allocInfo);
		depthPyramidDescriptorSets.clear();
		depthPyramidDescriptorSets.reserve(sets.size());
		for (auto &set : sets)
		{
			depthPyramidDescriptorSets.emplace_back(std::move(set));
		}

		depthPyramidDispatchSteps.resize(depthPyramidMipCount);
		for (uint32_t mip = 0; mip < depthPyramidMipCount; ++mip)
		{
			auto mipExtent = [&](uint32_t level) {
				return vk::Extent2D{
				    std::max(1u, swapChainExtent.width >> level),
				    std::max(1u, swapChainExtent.height >> level)};
			};

			const vk::Extent2D srcExtent = (mip == 0) ? swapChainExtent : mipExtent(mip - 1);
			const vk::Extent2D dstExtent = mipExtent(mip);

			depthPyramidDispatchSteps[mip] = DepthPyramidDispatchStep{.srcExtent = srcExtent, .dstExtent = dstExtent};

			vk::ImageView srcView = (mip == 0) ? *depthImageView : *depthPyramidMipViews[mip - 1];
			vk::ImageView dstView = *depthPyramidMipViews[mip];

			vk::DescriptorImageInfo srcInfo{.sampler = *depthPyramidSampler, .imageView = srcView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
			vk::DescriptorImageInfo dstInfo{.imageView = dstView, .imageLayout = vk::ImageLayout::eGeneral};

			std::array<vk::WriteDescriptorSet, 2> writes = {
			    vk::WriteDescriptorSet{.dstSet = *depthPyramidDescriptorSets[mip], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &srcInfo},
			    vk::WriteDescriptorSet{.dstSet = *depthPyramidDescriptorSets[mip], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &dstInfo}};

			std::lock_guard<std::mutex> lk(descriptorMutex);
			device.updateDescriptorSets(writes, {});
		}

		return true;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Failed to create depth pyramid resources: " << e.what() << std::endl;
		destroyDepthPyramidResources();
		return false;
	}
}

void Renderer::dispatchDepthPyramid(vk::raii::CommandBuffer &cmd)
{
	if (!*depthPyramidPipeline || !*depthPyramidPipelineLayout || !*depthPyramidImage)
		return;
	if (depthPyramidDescriptorSets.empty() || depthPyramidDispatchSteps.empty())
		return;
	if (!*depthImage)
		return;

	auto transitionDepthPyramidMip = [&](uint32_t mipLevel, vk::ImageLayout newLayout, vk::PipelineStageFlags2 dstStageMask, vk::AccessFlags2 dstAccessMask) {
		if (mipLevel >= depthPyramidMipLayouts.size())
			return;

		const vk::ImageLayout oldLayout = depthPyramidMipLayouts[mipLevel];
		if (oldLayout == newLayout)
			return;

		vk::PipelineStageFlags2 srcStageMask  = vk::PipelineStageFlagBits2::eTopOfPipe;
		vk::AccessFlags2        srcAccessMask = vk::AccessFlagBits2::eNone;
		if (oldLayout == vk::ImageLayout::eGeneral)
		{
			srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
			srcAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
		}
		else if (oldLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
		{
			srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
			srcAccessMask = vk::AccessFlagBits2::eShaderRead;
		}

		vk::ImageMemoryBarrier2 barrier{
		    .srcStageMask        = srcStageMask,
		    .srcAccessMask       = srcAccessMask,
		    .dstStageMask        = dstStageMask,
		    .dstAccessMask       = dstAccessMask,
		    .oldLayout           = oldLayout,
		    .newLayout           = newLayout,
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .image               = *depthPyramidImage,
		    .subresourceRange    = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = mipLevel, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};

		vk::DependencyInfo depInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier};
		cmd.pipelineBarrier2(depInfo);
		depthPyramidMipLayouts[mipLevel] = newLayout;
	};

	vk::ImageMemoryBarrier2 depthToSample{
	    .srcStageMask        = vk::PipelineStageFlagBits2::eLateFragmentTests,
	    .srcAccessMask       = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
	    .dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
	    .dstAccessMask       = vk::AccessFlagBits2::eShaderSampledRead,
	    .oldLayout           = vk::ImageLayout::eDepthAttachmentOptimal,
	    .newLayout           = vk::ImageLayout::eShaderReadOnlyOptimal,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .image               = *depthImage,
	    .subresourceRange    = {.aspectMask = vk::ImageAspectFlagBits::eDepth, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
	vk::DependencyInfo depDepthToSample{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &depthToSample};
	cmd.pipelineBarrier2(depDepthToSample);

	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *depthPyramidPipeline);

	struct DepthPyramidPushConstantsCPU
	{
		uint32_t srcWidth;
		uint32_t srcHeight;
		uint32_t dstWidth;
		uint32_t dstHeight;
	};

	for (uint32_t mip = 0; mip < depthPyramidMipCount; ++mip)
	{
		if (mip >= depthPyramidDescriptorSets.size() || mip >= depthPyramidDispatchSteps.size())
			break;

		transitionDepthPyramidMip(mip, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite);
		if (mip > 0)
		{
			transitionDepthPyramidMip(mip - 1, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderSampledRead);
		}

		vk::DescriptorSet set = *depthPyramidDescriptorSets[mip];
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *depthPyramidPipelineLayout, 0, set, {});

		const auto &step = depthPyramidDispatchSteps[mip];
		DepthPyramidPushConstantsCPU pushConstants{
		    .srcWidth  = std::max(1u, step.srcExtent.width),
		    .srcHeight = std::max(1u, step.srcExtent.height),
		    .dstWidth  = std::max(1u, step.dstExtent.width),
		    .dstHeight = std::max(1u, step.dstExtent.height)};
		cmd.pushConstants<DepthPyramidPushConstantsCPU>(*depthPyramidPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pushConstants);

		const uint32_t groupsX = (pushConstants.dstWidth + 7) / 8;
		const uint32_t groupsY = (pushConstants.dstHeight + 7) / 8;
		cmd.dispatch(std::max(1u, groupsX), std::max(1u, groupsY), 1);

		transitionDepthPyramidMip(mip, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderSampledRead);
	}

	vk::ImageMemoryBarrier2 depthBackToAttachment{
	    .srcStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
	    .srcAccessMask       = vk::AccessFlagBits2::eShaderSampledRead,
	    .dstStageMask        = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
	    .dstAccessMask       = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
	    .oldLayout           = vk::ImageLayout::eShaderReadOnlyOptimal,
	    .newLayout           = vk::ImageLayout::eDepthAttachmentOptimal,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .image               = *depthImage,
	    .subresourceRange    = {.aspectMask = vk::ImageAspectFlagBits::eDepth, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
	vk::DependencyInfo depDepthBack{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &depthBackToAttachment};
	cmd.pipelineBarrier2(depDepthBack);
	depthPyramidHistoryValid = true;
}

void Renderer::dispatchSAO(vk::raii::CommandBuffer &cmd)
{
	if (!enableSAO)
		return;
	if (!*saoPipeline || !*saoPipelineLayout)
		return;
	if (!depthPyramidHistoryValid || !*depthPyramidImage || !*depthPyramidFullView || !*depthPyramidSampler)
		return;
	if (currentFrame >= saoDescriptorSets.size() || currentFrame >= saoImages.size() || currentFrame >= saoImageLayouts.size())
		return;

	vk::ImageLayout oldSaoLayout = saoImageLayouts[currentFrame];
	if (oldSaoLayout != vk::ImageLayout::eGeneral)
	{
		vk::PipelineStageFlags2 srcStageMask  = vk::PipelineStageFlagBits2::eTopOfPipe;
		vk::AccessFlags2        srcAccessMask = vk::AccessFlagBits2::eNone;
		if (oldSaoLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
		{
			srcStageMask  = vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eComputeShader;
			srcAccessMask = vk::AccessFlagBits2::eShaderRead;
		}

		vk::ImageMemoryBarrier2 saoToGeneral{
		    .srcStageMask        = srcStageMask,
		    .srcAccessMask       = srcAccessMask,
		    .dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
		    .dstAccessMask       = vk::AccessFlagBits2::eShaderWrite,
		    .oldLayout           = oldSaoLayout,
		    .newLayout           = vk::ImageLayout::eGeneral,
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .image               = *saoImages[currentFrame],
		    .subresourceRange    = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
		vk::DependencyInfo depToGeneral{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &saoToGeneral};
		cmd.pipelineBarrier2(depToGeneral);
		saoImageLayouts[currentFrame] = vk::ImageLayout::eGeneral;
	}

	vk::ImageMemoryBarrier2 depthReadBarrier{
	    .srcStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
	    .srcAccessMask       = vk::AccessFlagBits2::eShaderWrite,
	    .dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
	    .dstAccessMask       = vk::AccessFlagBits2::eShaderSampledRead,
	    .oldLayout           = vk::ImageLayout::eShaderReadOnlyOptimal,
	    .newLayout           = vk::ImageLayout::eShaderReadOnlyOptimal,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .image               = *depthPyramidImage,
	    .subresourceRange    = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = depthPyramidMipCount, .baseArrayLayer = 0, .layerCount = 1}};
	vk::DependencyInfo depDepthRead{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &depthReadBarrier};
	cmd.pipelineBarrier2(depDepthRead);

	SAOPushConstants push = saoSettings;
	push.invResolution    = glm::vec2(
	    1.0f / static_cast<float>(std::max(1u, swapChainExtent.width)),
	    1.0f / static_cast<float>(std::max(1u, swapChainExtent.height)));

	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *saoPipeline);
	vk::DescriptorSet set = *saoDescriptorSets[currentFrame];
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *saoPipelineLayout, 0, set, {});
	cmd.pushConstants<SAOPushConstants>(*saoPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, push);

	const uint32_t groupsX = (std::max(1u, swapChainExtent.width) + 7u) / 8u;
	const uint32_t groupsY = (std::max(1u, swapChainExtent.height) + 7u) / 8u;
	cmd.dispatch(groupsX, groupsY, 1);

	vk::ImageMemoryBarrier2 saoToSample{
	    .srcStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
	    .srcAccessMask       = vk::AccessFlagBits2::eShaderWrite,
	    .dstStageMask        = vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eComputeShader,
	    .dstAccessMask       = vk::AccessFlagBits2::eShaderRead,
	    .oldLayout           = vk::ImageLayout::eGeneral,
	    .newLayout           = vk::ImageLayout::eShaderReadOnlyOptimal,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .image               = *saoImages[currentFrame],
	    .subresourceRange    = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
	vk::DependencyInfo depSaoToSample{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &saoToSample};
	cmd.pipelineBarrier2(depSaoToSample);
	saoImageLayouts[currentFrame] = vk::ImageLayout::eShaderReadOnlyOptimal;
	saoHistoryValid               = true;
}

void Renderer::dispatchVolumetric(vk::raii::CommandBuffer &cmd)
{
	if (!enableVolumetricScattering)
	{
		volumetricHistoryValid = false;
		return;
	}
	if (!*volumetricPipeline || !*volumetricPipelineLayout)
	{
		volumetricHistoryValid = false;
		return;
	}
	if (!depthPyramidHistoryValid || !*depthPyramidImage || !*depthPyramidFullView || !*depthPyramidSampler)
	{
		volumetricHistoryValid = false;
		return;
	}
	if (currentFrame >= volumetricDescriptorSets.size() || currentFrame >= volumetricImages.size() || currentFrame >= volumetricImageLayouts.size())
	{
		volumetricHistoryValid = false;
		return;
	}

	vk::ImageLayout oldVolumetricLayout = volumetricImageLayouts[currentFrame];
	if (oldVolumetricLayout != vk::ImageLayout::eGeneral)
	{
		vk::PipelineStageFlags2 srcStageMask  = vk::PipelineStageFlagBits2::eTopOfPipe;
		vk::AccessFlags2        srcAccessMask = vk::AccessFlagBits2::eNone;
		if (oldVolumetricLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
		{
			srcStageMask  = vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eComputeShader;
			srcAccessMask = vk::AccessFlagBits2::eShaderRead;
		}

		vk::ImageMemoryBarrier2 toGeneral{
		    .srcStageMask        = srcStageMask,
		    .srcAccessMask       = srcAccessMask,
		    .dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
		    .dstAccessMask       = vk::AccessFlagBits2::eShaderWrite,
		    .oldLayout           = oldVolumetricLayout,
		    .newLayout           = vk::ImageLayout::eGeneral,
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .image               = *volumetricImages[currentFrame],
		    .subresourceRange    = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
		vk::DependencyInfo depToGeneral{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toGeneral};
		cmd.pipelineBarrier2(depToGeneral);
		volumetricImageLayouts[currentFrame] = vk::ImageLayout::eGeneral;
	}

	vk::ImageMemoryBarrier2 depthReadBarrier{
	    .srcStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
	    .srcAccessMask       = vk::AccessFlagBits2::eShaderWrite,
	    .dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
	    .dstAccessMask       = vk::AccessFlagBits2::eShaderSampledRead,
	    .oldLayout           = vk::ImageLayout::eShaderReadOnlyOptimal,
	    .newLayout           = vk::ImageLayout::eShaderReadOnlyOptimal,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .image               = *depthPyramidImage,
	    .subresourceRange    = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = depthPyramidMipCount, .baseArrayLayer = 0, .layerCount = 1}};
	vk::DependencyInfo depDepthRead{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &depthReadBarrier};
	cmd.pipelineBarrier2(depDepthRead);

	VolumetricPushConstants push = volumetricSettings;
	push.invResolution           = glm::vec2(
	    1.0f / static_cast<float>(std::max(1u, swapChainExtent.width)),
	    1.0f / static_cast<float>(std::max(1u, swapChainExtent.height)));
	push.lightCount = lastFrameLightCount;
	push.stepCount = std::max(1u, push.stepCount);

	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *volumetricPipeline);
	vk::DescriptorSet set = *volumetricDescriptorSets[currentFrame];
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *volumetricPipelineLayout, 0, set, {});
	cmd.pushConstants<VolumetricPushConstants>(*volumetricPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, push);

	const uint32_t groupsX = (std::max(1u, swapChainExtent.width) + 7u) / 8u;
	const uint32_t groupsY = (std::max(1u, swapChainExtent.height) + 7u) / 8u;
	cmd.dispatch(groupsX, groupsY, 1);

	vk::ImageMemoryBarrier2 toSample{
	    .srcStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
	    .srcAccessMask       = vk::AccessFlagBits2::eShaderWrite,
	    .dstStageMask        = vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eComputeShader,
	    .dstAccessMask       = vk::AccessFlagBits2::eShaderRead,
	    .oldLayout           = vk::ImageLayout::eGeneral,
	    .newLayout           = vk::ImageLayout::eShaderReadOnlyOptimal,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .image               = *volumetricImages[currentFrame],
	    .subresourceRange    = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
	vk::DependencyInfo depToSample{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toSample};
	cmd.pipelineBarrier2(depToSample);
	volumetricImageLayouts[currentFrame] = vk::ImageLayout::eShaderReadOnlyOptimal;
	volumetricHistoryValid               = true;
}

bool Renderer::createOrResizeForwardPlusBuffers(uint32_t tilesX, uint32_t tilesY, uint32_t slicesZ, bool updateOnlyCurrentFrame)
{
	try
	{
		size_t clusters = static_cast<size_t>(tilesX) * static_cast<size_t>(tilesY) * static_cast<size_t>(slicesZ);
		size_t indices  = clusters * static_cast<size_t>(MAX_LIGHTS_PER_TILE);

		// Range of frames to touch this call
		size_t beginFrame = 0;
		size_t endFrame   = MAX_FRAMES_IN_FLIGHT;
		if (updateOnlyCurrentFrame)
		{
			beginFrame = static_cast<size_t>(currentFrame);
			endFrame   = beginFrame + 1;
		}

		for (size_t i = beginFrame; i < endFrame; ++i)
		{
			auto &f         = forwardPlusPerFrame[i];
			bool  needTiles = (f.tilesCapacity < clusters) || (!*f.tileHeaders);
			bool  needIdx   = (f.indicesCapacity < indices) || (!*f.tileLightIndices);

			if (needTiles)
			{
				if (!!*f.tileHeaders)
				{
					f.tileHeaders = vk::raii::Buffer(nullptr);
					f.tileHeadersAlloc.reset();
				}
				auto [buf, alloc]  = createBufferPooled(clusters * sizeof(TileHeader), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
				f.tileHeaders      = std::move(buf);
				f.tileHeadersAlloc = std::move(alloc);
				f.tilesCapacity    = clusters;
				// Initialize headers to zero so that count==0 when Forward+ is disabled or before first dispatch
				if (!!f.tileHeadersAlloc && f.tileHeadersAlloc->mappedPtr)
				{
					std::memset(f.tileHeadersAlloc->mappedPtr, 0, clusters * sizeof(TileHeader));
				}
			}
			if (needIdx)
			{
				if (!!*f.tileLightIndices)
				{
					f.tileLightIndices = vk::raii::Buffer(nullptr);
					f.tileLightIndicesAlloc.reset();
				}
				auto [buf, alloc]       = createBufferPooled(indices * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
				f.tileLightIndices      = std::move(buf);
				f.tileLightIndicesAlloc = std::move(alloc);
				f.indicesCapacity       = indices;
				// Initialize indices to zero to avoid stray reads
				if (!!f.tileLightIndicesAlloc && f.tileLightIndicesAlloc->mappedPtr)
				{
					std::memset(f.tileLightIndicesAlloc->mappedPtr, 0, indices * sizeof(uint32_t));
				}
			}
			if (!*f.params)
			{
				auto [pbuf, palloc] = createBufferPooled(sizeof(glm::mat4) * 2 + sizeof(glm::vec4) * 3, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
				f.params            = std::move(pbuf);
				f.paramsAlloc       = std::move(palloc);
				f.paramsMapped      = f.paramsAlloc->mappedPtr;
			}

			// Update compute descriptor set writes for this frame (only if buffers changed or first time)
			if (!!*forwardPlusPerFrame[i].computeSet)
			{
				if (!descriptorSetsValid.load(std::memory_order_relaxed))
				{
					// Descriptor sets are being recreated; skip writes this iteration
					continue;
				}
				if (isRecordingCmd.load(std::memory_order_relaxed))
				{
					// Avoid update-after-bind while a command buffer is recording
					continue;
				}
				// Only update descriptors if we resized or created any buffer this iteration
				if (needTiles || needIdx || !!*f.params)
				{
					// Build writes conditionally to avoid dereferencing uninitialized light buffers
					std::vector<vk::WriteDescriptorSet> writes;

					// Binding 0: lights SSBO (only if available)
					bool                     haveLightBuffer = (i < lightStorageBuffers.size()) && !!*lightStorageBuffers[i].buffer;
					vk::DescriptorBufferInfo lightsInfo{};
					if (haveLightBuffer)
					{
						lightsInfo = vk::DescriptorBufferInfo{.buffer = *lightStorageBuffers[i].buffer, .offset = 0, .range = VK_WHOLE_SIZE};
						writes.push_back(vk::WriteDescriptorSet{.dstSet = *forwardPlusPerFrame[i].computeSet, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &lightsInfo});
					}

					// Binding 1: tile headers
					vk::DescriptorBufferInfo headersInfo{.buffer = *f.tileHeaders, .offset = 0, .range = VK_WHOLE_SIZE};
					writes.push_back(vk::WriteDescriptorSet{.dstSet = *forwardPlusPerFrame[i].computeSet, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &headersInfo});

					// Binding 2: tile indices
					vk::DescriptorBufferInfo indicesInfo{.buffer = *f.tileLightIndices, .offset = 0, .range = VK_WHOLE_SIZE};
					writes.push_back(vk::WriteDescriptorSet{.dstSet = *forwardPlusPerFrame[i].computeSet, .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &indicesInfo});

					// Binding 3: params UBO
					vk::DescriptorBufferInfo paramsInfo{.buffer = *f.params, .offset = 0, .range = VK_WHOLE_SIZE};
					writes.push_back(vk::WriteDescriptorSet{.dstSet = *forwardPlusPerFrame[i].computeSet, .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &paramsInfo});

					if (!writes.empty())
					{
						std::lock_guard<std::mutex> lk(descriptorMutex);
						device.updateDescriptorSets(writes, {});
					}
				}
			}
		}

		// Update PBR descriptor sets to bind new tile buffers for forward shading.
		// Avoid updating sets that may be in use by in-flight command buffers.
		// If updateOnlyCurrentFrame=true, only update the current frame's sets (safe point after fence wait).
		try
		{
			// Only update PBR descriptor sets for bindings 7/8 in two situations:
			//  - When called in initialization/device-idle paths (updateOnlyCurrentFrame=false), or
			//  - When this call resulted in (re)creating the buffers for the current frame
			size_t beginFrameSets = 0;
			size_t endFrameSets   = forwardPlusPerFrame.size();
			if (updateOnlyCurrentFrame)
			{
				beginFrameSets = static_cast<size_t>(currentFrame);
				endFrameSets   = beginFrameSets + 1;
			}

			for (auto &kv : entityResources)
			{
				auto &resources = kv.second;
				if (resources.pbrDescriptorSets.empty())
					continue;
				for (size_t i = beginFrameSets; i < endFrameSets && i < resources.pbrDescriptorSets.size() && i < forwardPlusPerFrame.size(); ++i)
				{
					if (!descriptorSetsValid.load(std::memory_order_relaxed))
						continue;
					if (isRecordingCmd.load(std::memory_order_relaxed))
						continue;
					if (!(*resources.pbrDescriptorSets[i]))
						continue;
					auto &f = forwardPlusPerFrame[i];
					if (!*f.tileHeaders || !*f.tileLightIndices)
						continue;
					vk::DescriptorBufferInfo              headersInfo{.buffer = *f.tileHeaders, .offset = 0, .range = VK_WHOLE_SIZE};
					vk::DescriptorBufferInfo              indicesInfo{.buffer = *f.tileLightIndices, .offset = 0, .range = VK_WHOLE_SIZE};
					std::array<vk::WriteDescriptorSet, 2> writes = {
					    vk::WriteDescriptorSet{.dstSet = *resources.pbrDescriptorSets[i], .dstBinding = 7, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &headersInfo},
					    vk::WriteDescriptorSet{.dstSet = *resources.pbrDescriptorSets[i], .dstBinding = 8, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &indicesInfo}};
					{
						std::lock_guard<std::mutex> lk(descriptorMutex);
						device.updateDescriptorSets(writes, {});
					}
				}
			}
		}
		catch (...)
		{
		}

		return true;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Failed to create/resize Forward+ buffers: " << e.what() << std::endl;
		return false;
	}
}

void Renderer::updateForwardPlusParams(uint32_t frameIndex, const glm::mat4 &view, const glm::mat4 &proj, uint32_t lightCount, uint32_t tilesX, uint32_t tilesY, uint32_t slicesZ, float nearZ, float farZ)
{
	if (frameIndex >= forwardPlusPerFrame.size())
		return;
	auto &f = forwardPlusPerFrame[frameIndex];
	if (!f.paramsMapped)
		return;

	// Pack: [view][proj][screen xy, tile xy][lightCount, maxPerTile, tilesX, tilesY][near, far, slicesZ, 0]
	struct ParamsCPU
	{
		glm::mat4  view;
		glm::mat4  proj;
		glm::vec4  screenTile;        // x=width,y=height,z=tileX,w=tileY
		glm::uvec4 counts;            // x=lightCount,y=maxPerTile,z=tilesX,w=tilesY
		glm::vec4  zParams;           // x=nearZ,y=farZ,z=slicesZ,w=0
	};

	ParamsCPU p{};
	p.view       = view;
	p.proj       = proj;
	p.screenTile = glm::vec4(static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), static_cast<float>(forwardPlusTileSizeX), static_cast<float>(forwardPlusTileSizeY));
	p.counts     = glm::uvec4(lightCount, MAX_LIGHTS_PER_TILE, tilesX, tilesY);
	p.zParams    = glm::vec4(nearZ, farZ, static_cast<float>(slicesZ), 0.0f);

	std::memcpy(f.paramsAlloc->mappedPtr, &p, sizeof(ParamsCPU));
}

void Renderer::dispatchForwardPlus(vk::raii::CommandBuffer &cmd, uint32_t tilesX, uint32_t tilesY, uint32_t slicesZ)
{
	if (!*forwardPlusPipeline)
		return;
	if (currentFrame >= forwardPlusPerFrame.size())
		return;
	auto &f = forwardPlusPerFrame[currentFrame];
	if (!*f.computeSet)
		return;

	// Ensure a valid lights buffer is bound; otherwise skip compute this frame
	bool haveLightBuffer = (currentFrame < lightStorageBuffers.size()) && !!*lightStorageBuffers[currentFrame].buffer;
	if (!haveLightBuffer)
		return;

	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *forwardPlusPipeline);
	vk::DescriptorSet set = *f.computeSet;
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *forwardPlusPipelineLayout, 0, set, {});
	// One invocation per cluster (X,Y by workgroup grid, Z as third dimension)
	cmd.dispatch(tilesX, tilesY, slicesZ);
	// Make tilelist writes visible to fragment shader (Sync2)
	vk::MemoryBarrier2 memBarrier2{
	    .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
	    .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
	    .dstStageMask  = vk::PipelineStageFlagBits2::eFragmentShader,
	    .dstAccessMask = vk::AccessFlagBits2::eShaderRead};
	vk::DependencyInfo depInfoComputeToFrag{.memoryBarrierCount = 1, .pMemoryBarriers = &memBarrier2};
	cmd.pipelineBarrier2(depInfoComputeToFrag);
}

// Ensure compute descriptor binding 0 (lights SSBO) is bound for the given frame.
void Renderer::refreshForwardPlusComputeLightsBindingForFrame(uint32_t frameIndex)
{
	try
	{
		if (frameIndex >= forwardPlusPerFrame.size())
			return;
		if (!*forwardPlusPerFrame[frameIndex].computeSet)
			return;
		if (frameIndex >= lightStorageBuffers.size())
			return;
		if (!*lightStorageBuffers[frameIndex].buffer)
			return;

		// Updating descriptor sets during recording causes validation errors:
		// "commandBuffer must be in the recording state" and invalidates the command buffer.
		// These descriptor sets are already initialized earlier at the safe point (line 1059),
		// so this redundant update during recording is unnecessary and harmful.
		if (isRecordingCmd.load(std::memory_order_relaxed))
		{
			return;        // Skip update, descriptor is already valid from earlier initialization
		}

		vk::DescriptorBufferInfo lightsInfo{.buffer = *lightStorageBuffers[frameIndex].buffer, .offset = 0, .range = VK_WHOLE_SIZE};
		vk::WriteDescriptorSet   write{.dstSet = *forwardPlusPerFrame[frameIndex].computeSet, .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &lightsInfo};
		{
			std::lock_guard<std::mutex> lk(descriptorMutex);
			device.updateDescriptorSets(write, {});
		}
	}
	catch (const std::exception &e)
	{
		std::cerr << "Failed to refresh Forward+ compute lights binding for frame " << frameIndex << ": " << e.what() << std::endl;
	}
}
