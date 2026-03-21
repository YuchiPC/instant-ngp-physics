/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   testbed_volume.cu
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/random_val.cuh> // helpers to generate random values, directions
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/trainer.h>

#include <nanovdb/NanoVDB.h>

#include <filesystem/path.h>

#include <cstring>
#include <fstream>

namespace ngp {

Testbed::NetworkDims Testbed::network_dims_volume() const {
	NetworkDims dims;
	dims.n_input = 3;
	dims.n_output = 4;
	dims.n_pos = 3;
	return dims;
}

__device__ vec4 proc_envmap(const vec3& dir, const vec3& up_dir, const vec3& sun_dir, const vec3& skycol) {
	float skyam = dot(up_dir, dir) * 0.5f + 0.5f;
	float sunam = std::max(0.f, dot(sun_dir, dir));
	sunam *= sunam;
	sunam *= sunam;
	sunam *= sunam;
	sunam *= sunam;
	sunam *= sunam;
	sunam *= sunam;

	vec4 result;
	result.rgb() = skycol * skyam + vec3{255.f / 255.0f, 215.f / 255.0f, 195.f / 255.0f} * (20.f * sunam);
	result.a = 1.0f;
	return result;
}

__device__ vec4 proc_envmap_render(const vec3& dir, const vec3& up_dir, const vec3& sun_dir, const vec3& skycol) {
	vec4 result = vec4(0.0f);
	result = proc_envmap(dir, up_dir, sun_dir, skycol);
	return result;
}

// ============================================================================
// Shadow ray: march toward sun accumulating optical depth, return transmittance
// Used for direct illumination at each scattering event.
// ============================================================================
__device__ float march_to_sun(
	vec3 pos,
	const vec3& sun_dir,
	const BoundingBox& aabb,
	const nanovdb::FloatGrid* grid,
	const vec3& world2index_offset,
	float world2index_scale,
	float sigma_t_scale,
	int max_steps
) {
	float tau = 0.0f;
	float diag = length(aabb.diag());
	float dt = diag / (float)max_steps;

	auto acc = grid->tree().getAccessor();
	for (int i = 0; i < max_steps; i++) {
		pos += sun_dir * dt;
		if (!aabb.contains(pos)) break;
		vec3 nanovdbpos = pos * world2index_scale + world2index_offset;
		float density = acc.getValue({(int)nanovdbpos.x, (int)nanovdbpos.y, (int)nanovdbpos.z});
		tau += fmaxf(density, 0.0f) * dt * sigma_t_scale;
	}
	return expf(-tau);
}

// ============================================================================
// Compute in-scattered radiance at a point using multi-scattering octave method
// (Wrenninge / Frostbite SIGGRAPH 2016)
//
// For each octave n:
//   sigma_t is attenuated by a^n (medium appears more transparent)
//   phase g is attenuated by c^n (scattering becomes more isotropic)
// This approximates how multiply-scattered light sees a "softer" medium.
// ============================================================================
__device__ vec3 compute_inscattered_radiance(
	const vec3& pos,
	const vec3& view_dir,
	const vec3& sun_dir,
	const vec3& sun_color,
	float sun_intensity,
	float density,
	float sigma_t_scale,
	const HydrometeorProps& props,
	bool use_dual_lobe,
	bool enable_beer_powder,
	int ms_octaves,
	float ms_attenuation,
	const BoundingBox& aabb,
	const nanovdb::FloatGrid* grid,
	const vec3& world2index_offset,
	float world2index_scale,
	int shadow_steps,
	const vec3& sky_col,
	const vec3& up_dir
) {
	float cos_theta = dot(view_dir, sun_dir);

	// Compute transmittance toward sun (shared across octaves)
	float T_sun = march_to_sun(pos, sun_dir, aabb, grid, world2index_offset,
	                           world2index_scale, sigma_t_scale, shadow_steps);

	// Ambient sky approximation: fraction of light from sky hemisphere
	// Use a cheap vertical transmittance estimate
	float ambient_factor = 0.15f; // ambient-to-direct ratio

	vec3 L_total = vec3(0.0f);
	float sigma_t_ms = density * sigma_t_scale;
	float g1_ms = props.g1;
	float g2_ms = props.g2;

	for (int octave = 0; octave < ms_octaves; octave++) {
		// Phase function at this octave's effective g
		float phase;
		if (use_dual_lobe) {
			phase = dual_lobe_hg(cos_theta, g1_ms, g2_ms, props.w_g1);
		} else {
			phase = henyey_greenstein(cos_theta, g1_ms);
		}

		// Sun transmittance at this octave's effective extinction
		float att_n = (octave == 0) ? 1.0f : powf(ms_attenuation, (float)octave);
		float T_sun_ms = powf(T_sun, att_n); // equivalent to exp(-sigma_t_ms * sun_optical_depth)

		// Beer-Powder: modulate to darken thin cloud edges
		float energy = 1.0f;
		if (enable_beer_powder && sigma_t_ms > 0.0f) {
			energy = beer_powder(sigma_t_ms * 0.1f); // 0.1 is a tunable local-thickness proxy
		}

		// Direct sun contribution at this octave
		vec3 L_sun = sun_color * sun_intensity * phase * T_sun_ms * props.albedo * energy;

		// Ambient sky contribution (isotropic phase = 1/4pi)
		float ambient_phase = 1.0f / (4.0f * PI());
		vec3 L_ambient = sky_col * ambient_phase * ambient_factor * att_n;

		L_total += L_sun + L_ambient;

		// Attenuate for next octave
		sigma_t_ms *= ms_attenuation;
		g1_ms *= ms_attenuation;
		g2_ms *= ms_attenuation;
	}

	return L_total;
}

__device__ inline bool
	walk_to_next_event(default_rng_t& rng, const BoundingBox& aabb, vec3& pos, const vec3& dir, const uint8_t* bitgrid, float scale) {
	while (1) {
		float zeta1 = random_val(rng); // sample a free flight distance and go there!
		float dt = -std::log(1.0f - zeta1) *
			scale; // todo - for spatially varying majorant, we must check dt against the range over which the majorant is defined. we can
				   // turn this into an optical thickness accumulating loop...
		pos += dir * dt;
		if (!aabb.contains(pos)) {
			return false; // escape to the mooon!
		}
		uint32_t bitidx = morton3D(int(pos.x * 128.f + 0.5f), int(pos.y * 128.f + 0.5f), int(pos.z * 128.f + 0.5f));
		if (bitidx < 128 * 128 * 128 && bitgrid[bitidx >> 3] & (1 << (bitidx & 7))) {
			break;
		}
		// loop around and try again as we are in density=0 region!
	}
	return true;
}

static constexpr uint32_t MAX_TRAIN_VERTICES =
	4; // record the first few real interactions and use as training data. uses a local array so cant be big.

__global__ void volume_generate_training_data_kernel(
	uint32_t n_elements,
	vec3* pos_out,
	vec4* target_out,
	const void* nanovdb,
	const uint8_t* bitgrid,
	vec3 world2index_offset,
	float world2index_scale,
	BoundingBox aabb,
	default_rng_t rng,
	float albedo,
	float scattering,
	float distance_scale,
	float global_majorant,
	vec3 up_dir,
	vec3 sun_dir,
	vec3 sky_col,
	// Physics parameters
	float sun_intensity,
	int shadow_steps,
	int ms_octaves,
	float ms_attenuation,
	bool enable_direct_light,
	bool enable_beer_powder,
	bool use_dual_lobe,
	HydrometeorProps hydro_props,
	bool enable_physics // master toggle
) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n_elements) {
		return;
	}
	rng.advance(idx * 256);
	uint32_t numout = 0;
	vec3 outpos[MAX_TRAIN_VERTICES];
	float outdensity[MAX_TRAIN_VERTICES];
	vec3 outradiance[MAX_TRAIN_VERTICES];
	float scale = distance_scale / global_majorant;
	const nanovdb::FloatGrid* grid = reinterpret_cast<const nanovdb::FloatGrid*>(nanovdb);
	auto acc = grid->tree().getAccessor();

	vec3 sun_color = vec3{1.0f, 0.95f, 0.85f};
	float effective_albedo = enable_physics ? hydro_props.albedo : albedo;

	while (numout < MAX_TRAIN_VERTICES) {
		uint32_t prev_numout = numout;
		vec3 pos = random_dir(rng) * 2.0f + 0.5f;
		vec3 target = random_val_3d(rng) * aabb.diag() + aabb.min;
		vec3 dir = normalize(target - pos);
		auto box_intersection = aabb.ray_intersect(pos, dir);
		float t = max(box_intersection.x, 0.0f);
		pos = pos + (t + 1e-6f) * dir;
		float throughput = 1.f;
		for (int iter = 0; iter < 128; ++iter) {
			if (!walk_to_next_event(rng, aabb, pos, dir, bitgrid, scale)) {
				break;
			}
			vec3 nanovdbpos = pos * world2index_scale + world2index_offset;
			float density = acc.getValue(
				{int(nanovdbpos.x + random_val(rng)), int(nanovdbpos.y + random_val(rng)), int(nanovdbpos.z + random_val(rng))}
			);

			if (numout < MAX_TRAIN_VERTICES) {
				outdensity[numout] = density;
				outpos[numout] = pos;
				if (enable_physics && enable_direct_light && density > 0.001f) {
					outradiance[numout] = compute_inscattered_radiance(
						pos, dir, sun_dir, sun_color, sun_intensity,
						density, global_majorant, hydro_props, use_dual_lobe,
						enable_beer_powder, ms_octaves, ms_attenuation,
						aabb, grid, world2index_offset, world2index_scale,
						shadow_steps, sky_col, up_dir
					);
				} else {
					outradiance[numout] = vec3(0.0f);
				}
				numout++;
			}

			float extinction_prob = density / global_majorant;
			float scatter_prob = extinction_prob * effective_albedo;
			float zeta2 = random_val(rng);
			if (zeta2 >= extinction_prob) {
				continue; // null collision
			}
			if (zeta2 < scatter_prob) {
				if (enable_physics) {
					dir = sample_dual_lobe_hg(dir, hydro_props, rng);
				} else {
					dir = normalize(random_dir(rng)); // original isotropic
				}
			} else {
				throughput = 0.f; // absorb
				break;
			}
		}
		vec4 envcolor = proc_envmap(dir, up_dir, sun_dir, sky_col) * throughput;
		uint32_t oidx = idx * MAX_TRAIN_VERTICES;
		for (uint32_t i = prev_numout; i < numout; ++i) {
			pos_out[oidx + i] = outpos[i];
			if (enable_physics) {
				vec3 combined_rgb = envcolor.rgb() + outradiance[i];
				target_out[oidx + i] = vec4(combined_rgb, outdensity[i]);
			} else {
				target_out[oidx + i] = envcolor;
				target_out[oidx + i].w = outdensity[i];
			}
		}
	}
}

void Testbed::train_volume(size_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
	const uint32_t n_output_dims = 4;
	const uint32_t n_input_dims = 3;

	// Auxiliary matrices for training
	const uint32_t batch_size = (uint32_t)target_batch_size;

	// Permute all training records to de-correlate training data

	const uint32_t n_elements = batch_size;
	m_volume.training.positions.enlarge(n_elements);
	m_volume.training.targets.enlarge(n_elements);

	float distance_scale = 1.f / std::max(m_volume.inv_distance_scale, 0.01f);
	auto sky_col = m_background_color.rgb();

	// Compute blended hydrometeor properties from phase fractions
	HydrometeorProps hydro_props = blend_hydrometeor_props(m_volume.phase_fractions);
	if (m_volume.albedo_override > 0.0f) hydro_props.albedo = m_volume.albedo_override;
	if (m_volume.g_override != 0.0f) { hydro_props.g1 = m_volume.g_override; hydro_props.g2 = -m_volume.g_override * 0.35f; }

	// When physics disabled, use original defaults for training
	float train_albedo = m_volume.enable_physics ? hydro_props.albedo : 0.95f;

	linear_kernel(
		volume_generate_training_data_kernel,
		0,
		stream,
		n_elements / MAX_TRAIN_VERTICES,
		m_volume.training.positions.data(),
		m_volume.training.targets.data(),
		m_volume.nanovdb_grid.data(),
		m_volume.bitgrid.data(),
		m_volume.world2index_offset,
		m_volume.world2index_scale,
		m_render_aabb,
		m_rng,
		train_albedo,
		hydro_props.g1,
		distance_scale,
		m_volume.global_majorant,
		m_up_dir,
		m_sun_dir,
		sky_col,
		m_volume.sun_intensity,
		m_volume.shadow_steps,
		m_volume.ms_octaves,
		m_volume.ms_attenuation,
		m_volume.enable_direct_light,
		m_volume.enable_beer_powder,
		m_volume.use_dual_lobe,
		hydro_props,
		m_volume.enable_physics
	);
	m_rng.advance(n_elements * 256);

	GPUMatrix<float> training_batch_matrix((float*)(m_volume.training.positions.data()), n_input_dims, batch_size);
	GPUMatrix<float> training_target_matrix((float*)(m_volume.training.targets.data()), n_output_dims, batch_size);

	auto ctx = m_trainer->training_step(stream, training_batch_matrix, training_target_matrix);

	m_training_step++;

	if (get_loss_scalar) {
		m_loss_scalar.update(m_trainer->loss(stream, *ctx));
	}
}

__global__ void init_rays_volume(
	uint32_t sample_index,
	vec3* __restrict__ positions,
	Testbed::VolPayload* __restrict__ payloads,
	uint32_t* pixel_counter,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix,
	vec2 screen_center,
	Lens lens,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	BoundingBox aabb,
	float near_distance,
	float plane_z,
	float aperture_size,
	Foveation foveation,
	Buffer2DView<const vec4> envmap,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	Buffer2DView<const uint8_t> hidden_area_mask,
	default_rng_t rng,
	const uint8_t* bitgrid,
	float distance_scale,
	float global_majorant,
	vec3 up_dir,
	vec3 sun_dir,
	vec3 sky_col
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= resolution.x || y >= resolution.y) {
		return;
	}
	uint32_t idx = x + resolution.x * y;
	rng.advance(idx << 8);
	if (plane_z < 0) {
		aperture_size = 0.0;
	}

	Ray ray = pixel_to_ray(
		sample_index,
		{(int)x, (int)y},
		resolution,
		focal_length,
		camera_matrix,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		hidden_area_mask,
		lens
	);

	if (!ray.is_valid()) {
		depth_buffer[idx] = MAX_DEPTH();
		return;
	}

	ray.d = normalize(ray.d);
	auto box_intersection = aabb.ray_intersect(ray.o, ray.d);
	float t = max(box_intersection.x, 0.0f);
	ray.advance(t + 1e-6f);
	float scale = distance_scale / global_majorant;

	if (t >= box_intersection.y || !walk_to_next_event(rng, aabb, ray.o, ray.d, bitgrid, scale)) {
		frame_buffer[idx] = proc_envmap_render(ray.d, up_dir, sun_dir, sky_col);
		depth_buffer[idx] = MAX_DEPTH();
	} else {
		uint32_t dstidx = atomicAdd(pixel_counter, 1);
		positions[dstidx] = ray.o;
		payloads[dstidx] = {ray.d, vec4(0.f), idx};
		depth_buffer[idx] = dot(camera_matrix[2], ray.o - camera_matrix[3]);
	}
}

__global__ void volume_render_kernel_gt(
	uint32_t n_pixels,
	ivec2 resolution,
	default_rng_t rng,
	BoundingBox aabb,
	const vec3* __restrict__ positions_in,
	const Testbed::VolPayload* __restrict__ payloads_in,
	const uint32_t* pixel_counter_in,
	const vec3 up_dir,
	const vec3 sun_dir,
	const vec3 sky_col,
	const void* nanovdb,
	const uint8_t* bitgrid,
	float global_majorant,
	vec3 world2index_offset,
	float world2index_scale,
	float distance_scale,
	float albedo,
	float scattering, // HG g when physics enabled; blend factor when disabled
	vec4* __restrict__ frame_buffer,
	// Physics parameters
	float sun_intensity,
	int shadow_steps,
	int ms_octaves,
	float ms_attenuation,
	bool enable_direct_light,
	bool enable_beer_powder,
	bool use_dual_lobe,
	HydrometeorProps hydro_props,
	bool enable_physics // master toggle
) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n_pixels || idx >= pixel_counter_in[0]) {
		return;
	}
	uint32_t pixidx = payloads_in[idx].pixidx;

	uint32_t y = pixidx / resolution.x;
	if (y >= resolution.y) {
		return;
	}

	vec3 pos = positions_in[idx];
	vec3 dir = payloads_in[idx].dir;
	rng.advance(pixidx << 8);
	const nanovdb::FloatGrid* grid = reinterpret_cast<const nanovdb::FloatGrid*>(nanovdb);
	auto acc = grid->tree().getAccessor();

	float scale = distance_scale / global_majorant;

	bool absorbed = false;
	bool scattered = false;

	if (!enable_physics) {
		// ===== ORIGINAL RENDERING (no physics) =====
		for (int iter = 0; iter < 128; ++iter) {
			vec3 nanovdbpos = pos * world2index_scale + world2index_offset;
			float density =
				acc.getValue({int(nanovdbpos.x + random_val(rng)), int(nanovdbpos.y + random_val(rng)), int(nanovdbpos.z + random_val(rng))});
			float extinction_prob = density / global_majorant;
			float scatter_prob = extinction_prob * albedo;
			float zeta2 = random_val(rng);
			if (zeta2 < scatter_prob) {
				dir = normalize(random_dir(rng)); // isotropic scattering (original)
				scattered = true;
			} else if (zeta2 < extinction_prob) {
				absorbed = true;
				break;
			}
			if (!walk_to_next_event(rng, aabb, pos, dir, bitgrid, scale)) {
				break;
			}
		}
		vec4 col;
		if (absorbed) {
			col = vec4(0.0f, 0.0f, 0.0f, 1.0f);
		} else if (scattered) {
			col = proc_envmap(dir, up_dir, sun_dir, sky_col);
		} else {
			col = proc_envmap_render(dir, up_dir, sun_dir, sky_col);
		}
		frame_buffer[pixidx] = col;
	} else {
		// ===== PHYSICS-BASED RENDERING =====
		vec3 sun_color = vec3{1.0f, 0.95f, 0.85f};
		vec3 accumulated_radiance = vec3(0.0f);
		float transmittance = 1.0f;

		for (int iter = 0; iter < 128; ++iter) {
			vec3 nanovdbpos = pos * world2index_scale + world2index_offset;
			float density =
				acc.getValue({int(nanovdbpos.x + random_val(rng)), int(nanovdbpos.y + random_val(rng)), int(nanovdbpos.z + random_val(rng))});
			float extinction_prob = density / global_majorant;
			float scatter_prob = extinction_prob * hydro_props.albedo;
			float zeta2 = random_val(rng);
			if (zeta2 < scatter_prob) {
				// Compute direct illumination before changing direction
				if (enable_direct_light && density > 0.001f) {
					vec3 L_inscatter = compute_inscattered_radiance(
						pos, dir, sun_dir, sun_color, sun_intensity,
						density, global_majorant, hydro_props, use_dual_lobe,
						enable_beer_powder, ms_octaves, ms_attenuation,
						aabb, grid, world2index_offset, world2index_scale,
						shadow_steps, sky_col, up_dir
					);
					accumulated_radiance += transmittance * L_inscatter;
				}
				transmittance *= hydro_props.albedo;

				// HG phase function sampling
				dir = sample_dual_lobe_hg(dir, hydro_props, rng);
				scattered = true;
			} else if (zeta2 < extinction_prob) {
				absorbed = true;
				break;
			}
			if (!walk_to_next_event(rng, aabb, pos, dir, bitgrid, scale)) {
				break;
			}
		}

		vec4 col;
		if (absorbed) {
			col = vec4(accumulated_radiance, 1.0f);
		} else {
			vec4 env = scattered ? proc_envmap(dir, up_dir, sun_dir, sky_col) : proc_envmap_render(dir, up_dir, sun_dir, sky_col);
			col.rgb() = accumulated_radiance + transmittance * env.rgb();
			col.a = 1.0f;
		}
		frame_buffer[pixidx] = col;
	}
}

__global__ void volume_render_kernel_step(
	uint32_t n_pixels,
	ivec2 resolution,
	default_rng_t rng,
	BoundingBox aabb,
	const vec3* __restrict__ positions_in,
	const Testbed::VolPayload* __restrict__ payloads_in,
	const uint32_t* pixel_counter_in,
	vec3* __restrict__ positions_out,
	Testbed::VolPayload* __restrict__ payloads_out,
	uint32_t* pixel_counter_out,
	const vec4* network_outputs_in,
	const vec3 up_dir,
	const vec3 sun_dir,
	const vec3 sky_col,
	const void* nanovdb,
	const uint8_t* bitgrid,
	float global_majorant,
	vec3 world2index_offset,
	float world2index_scale,
	float distance_scale,
	float albedo,
	float scattering,
	vec4* __restrict__ frame_buffer,
	bool force_finish_ray,
	// Physics parameters
	float sun_intensity,
	int shadow_steps,
	int ms_octaves,
	float ms_attenuation,
	bool enable_direct_light,
	bool enable_beer_powder,
	bool use_dual_lobe,
	HydrometeorProps hydro_props,
	bool enable_physics // master toggle
) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n_pixels || idx >= pixel_counter_in[0]) {
		return;
	}
	Testbed::VolPayload payload = payloads_in[idx];
	uint32_t pixidx = payload.pixidx;
	uint32_t y = pixidx / resolution.x;
	if (y >= resolution.y) {
		return;
	}
	vec3 pos = positions_in[idx];
	vec3 dir = payload.dir;
	rng.advance(pixidx << 8);
	const nanovdb::FloatGrid* grid = reinterpret_cast<const nanovdb::FloatGrid*>(nanovdb);
	auto acc = grid->tree().getAccessor();

	vec4 local_output = network_outputs_in[idx];
	float scale = distance_scale / global_majorant;
	float density = local_output.w;
	float extinction_prob = density / global_majorant;
	if (extinction_prob > 1.f) {
		extinction_prob = 1.f;
	}
	float T = 1.f - payload.col.a;
	float alpha = extinction_prob * T;

	vec3 final_rgb = local_output.rgb();

	if (enable_physics && enable_direct_light && density > 0.001f) {
		// Add physics-based direct illumination on top of network output
		vec3 sun_color = vec3{1.0f, 0.95f, 0.85f};
		vec3 L_physics = compute_inscattered_radiance(
			pos, dir, sun_dir, sun_color, sun_intensity,
			density, global_majorant, hydro_props, use_dual_lobe,
			enable_beer_powder, ms_octaves, ms_attenuation,
			aabb, grid, world2index_offset, world2index_scale,
			shadow_steps, sky_col, up_dir
		);
		// Blend: network provides learned ambient/indirect, physics adds direct lighting
		final_rgb = final_rgb + L_physics * 0.5f;
	}

	payload.col.rgb() += final_rgb * alpha;
	payload.col.a += alpha;
	if (payload.col.a > 0.99f || !walk_to_next_event(rng, aabb, pos, dir, bitgrid, scale) || force_finish_ray) {
		payload.col += (1.f - payload.col.a) * proc_envmap_render(dir, up_dir, sun_dir, sky_col);
		frame_buffer[pixidx] = payload.col;
		return;
	}
	uint32_t dstidx = atomicAdd(pixel_counter_out, 1);
	positions_out[dstidx] = pos;
	payloads_out[dstidx] = payload;
}

void Testbed::render_volume(
	cudaStream_t stream,
	const CudaRenderBufferView& render_buffer,
	const vec2& focal_length,
	const mat4x3& camera_matrix,
	const vec2& screen_center,
	const Foveation& foveation,
	const Lens& lens
) {
	auto jit_guard = m_network->jit_guard(stream, true);

	float plane_z = m_slice_plane_z + m_scale;
	float distance_scale = 1.f / std::max(m_volume.inv_distance_scale, 0.01f);
	auto res = render_buffer.resolution;

	size_t n_pixels = (size_t)res.x * res.y;
	for (uint32_t i = 0; i < 2; ++i) {
		m_volume.pos[i].enlarge(n_pixels);
		m_volume.payload[i].enlarge(n_pixels);
	}
	m_volume.hit_counter.enlarge(2);
	m_volume.hit_counter.memset(0);

	vec3 sky_col = m_background_color.rgb();

	const dim3 threads = {16, 8, 1};
	const dim3 blocks = {div_round_up((uint32_t)res.x, threads.x), div_round_up((uint32_t)res.y, threads.y), 1};
	init_rays_volume<<<blocks, threads, 0, stream>>>(
		render_buffer.spp,
		m_volume.pos[0].data(),
		m_volume.payload[0].data(),
		m_volume.hit_counter.data(),
		res,
		focal_length,
		camera_matrix,
		screen_center,
		lens,
		m_parallax_shift,
		m_snap_to_pixel_centers,
		m_render_aabb,
		m_render_near_distance,
		plane_z,
		m_aperture_size,
		foveation,
		m_envmap.inference_view(),
		render_buffer.frame_buffer,
		render_buffer.depth_buffer,
		render_buffer.hidden_area_mask ? render_buffer.hidden_area_mask->const_view() : Buffer2DView<const uint8_t>{},
		m_rng,
		m_volume.bitgrid.data(),
		distance_scale,
		m_volume.global_majorant,
		m_up_dir,
		m_sun_dir,
		sky_col
	);
	m_rng.advance(n_pixels * 256);

	uint32_t n = n_pixels;
	CUDA_CHECK_THROW(cudaMemcpyAsync(&n, m_volume.hit_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

	// Compute blended hydrometeor properties for rendering
	HydrometeorProps render_hydro_props = blend_hydrometeor_props(m_volume.phase_fractions);
	if (m_volume.albedo_override > 0.0f) render_hydro_props.albedo = m_volume.albedo_override;
	if (m_volume.g_override != 0.0f) { render_hydro_props.g1 = m_volume.g_override; render_hydro_props.g2 = -m_volume.g_override * 0.35f; }

	// When physics disabled, use original albedo for rendering
	float render_albedo = m_volume.enable_physics ? std::min(render_hydro_props.albedo, 0.9999f) : std::min(0.95f, 0.995f);

	if (m_render_ground_truth) {
		linear_kernel(
			volume_render_kernel_gt,
			0,
			stream,
			n,
			res,
			m_rng,
			m_render_aabb,
			m_volume.pos[0].data(),
			m_volume.payload[0].data(),
			m_volume.hit_counter.data(),

			m_up_dir,
			m_sun_dir,
			sky_col,
			m_volume.nanovdb_grid.data(),
			m_volume.bitgrid.data(),
			m_volume.global_majorant,
			m_volume.world2index_offset,
			m_volume.world2index_scale,
			distance_scale,
			render_albedo,
			render_hydro_props.g1,
			render_buffer.frame_buffer,
			m_volume.sun_intensity,
			m_volume.shadow_steps,
			m_volume.ms_octaves,
			m_volume.ms_attenuation,
			m_volume.enable_direct_light,
			m_volume.enable_beer_powder,
			m_volume.use_dual_lobe,
			render_hydro_props,
			m_volume.enable_physics
		);
		m_rng.advance(n_pixels * 256);
	} else {
		m_volume.radiance_and_density.enlarge(n);

		int max_iter = 64;
		for (int iter = 0; iter < max_iter && n > 0; ++iter) {
			uint32_t srcbuf = (iter & 1);
			uint32_t dstbuf = 1 - srcbuf;

			uint32_t n_elements = next_multiple(n, BATCH_SIZE_GRANULARITY);
			GPUMatrix<float> positions_matrix((float*)m_volume.pos[srcbuf].data(), 3, n_elements);
			GPUMatrix<float> densities_matrix((float*)m_volume.radiance_and_density.data(), 4, n_elements);
			m_network->inference(stream, positions_matrix, densities_matrix);

			CUDA_CHECK_THROW(cudaMemsetAsync(m_volume.hit_counter.data() + dstbuf, 0, sizeof(uint32_t), stream));

			linear_kernel(
				volume_render_kernel_step,
				0,
				stream,
				n,
				res,
				m_rng,
				m_render_aabb,
				m_volume.pos[srcbuf].data(),
				m_volume.payload[srcbuf].data(),
				m_volume.hit_counter.data() + srcbuf,
				m_volume.pos[dstbuf].data(),
				m_volume.payload[dstbuf].data(),
				m_volume.hit_counter.data() + dstbuf,
				m_volume.radiance_and_density.data(),
				m_up_dir,
				m_sun_dir,
				sky_col,
				m_volume.nanovdb_grid.data(),
				m_volume.bitgrid.data(),
				m_volume.global_majorant,
				m_volume.world2index_offset,
				m_volume.world2index_scale,
				distance_scale,
				render_albedo,
				render_hydro_props.g1,
				render_buffer.frame_buffer,
				(iter >= max_iter - 1),
				m_volume.sun_intensity,
				m_volume.shadow_steps,
				m_volume.ms_octaves,
				m_volume.ms_attenuation,
				m_volume.enable_direct_light,
				m_volume.enable_beer_powder,
				m_volume.use_dual_lobe,
				render_hydro_props,
				m_volume.enable_physics
			);

			m_rng.advance(n_pixels * 256);
			if (((iter + 1) % 4) == 0) {
				CUDA_CHECK_THROW(cudaMemcpyAsync(&n, m_volume.hit_counter.data() + dstbuf, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
				CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			}
		}
	}
}

#define NANOVDB_MAGIC_NUMBER 0x304244566f6e614eUL // "NanoVDB0" in hex - little endian (uint64_t)
struct NanoVDBFileHeader {
	uint64_t magic;     // 8 bytes
	uint32_t version;   // 4 bytes version numbers
	uint16_t gridCount; // 2 bytes
	uint16_t codec;     // 2 bytes - must be 0
};
static_assert(sizeof(NanoVDBFileHeader) == 16, "nanovdb padding error");

struct NanoVDBMetaData {
	uint64_t gridSize, fileSize, nameKey, voxelCount; // 4 * 8 = 32B.
	uint32_t gridType;                                // 4B.
	uint32_t gridClass;                               // 4B.
	double worldBBox[2][3];                           // 2 * 3 * 8 = 48B.
	int indexBBox[2][3];                              // 2 * 3 * 4 = 24B.
	double voxelSize[3];                              // 24B.
	uint32_t nameSize;                                // 4B.
	uint32_t nodeCount[4];                            // 4 x 4 = 16B
	uint32_t tileCount[3];                            // 3 x 4 = 12B
	uint16_t codec;                                   // 2B
	uint16_t padding;                                 // 2B, due to 8B alignment from uint64_t
	uint32_t version;                                 // 4B
};
static_assert(sizeof(NanoVDBMetaData) == 176, "nanovdb padding error");

// Detect hydrometeor species from NanoVDB grid name
// Matches common HRRR/WRF variable names and descriptive names
static int detect_species_from_name(const char* name) {
	// Convert to lowercase for matching
	char lower[256] = {};
	for (int i = 0; i < 255 && name[i]; i++) {
		lower[i] = (name[i] >= 'A' && name[i] <= 'Z') ? (name[i] + 32) : name[i];
	}
	// Ice crystals: QICE, cloud_ice, ice, qi
	if (strstr(lower, "ice") || strstr(lower, "qi")) return (int)EHydrometeorType::Ice;
	// Snow: QSNOW, snow, qs
	if (strstr(lower, "snow") || strstr(lower, "qs")) return (int)EHydrometeorType::Snow;
	// Graupel: QGRAUP, graupel, hail, qg
	if (strstr(lower, "graup") || strstr(lower, "hail") || strstr(lower, "qg")) return (int)EHydrometeorType::Graupel;
	// Water: QCLOUD, cloud_water, water, lwc, qc, QRAIN, rain, qr
	if (strstr(lower, "water") || strstr(lower, "qcloud") || strstr(lower, "lwc") ||
	    strstr(lower, "qc") || strstr(lower, "rain") || strstr(lower, "qr")) return (int)EHydrometeorType::Water;
	// Default: treat as water (most common single-grid case)
	return (int)EHydrometeorType::Water;
}

static const char* species_name(int species_id) {
	switch (species_id) {
		case 0: return "Water";
		case 1: return "Ice";
		case 2: return "Snow";
		case 3: return "Graupel";
		default: return "Unknown";
	}
}

void Testbed::load_volume(const fs::path& data_path) {
	if (!data_path.exists()) {
		throw std::runtime_error{data_path.str() + " does not exist."};
	}
	tlog::info() << "Loading NanoVDB file from " << data_path;
	std::ifstream f{native_string(data_path), std::ios::in | std::ios::binary};
	NanoVDBFileHeader header;
	NanoVDBMetaData metadata;
	f.read(reinterpret_cast<char*>(&header), sizeof(header));
	f.read(reinterpret_cast<char*>(&metadata), sizeof(metadata));

	if (header.magic != NANOVDB_MAGIC_NUMBER) {
		throw std::runtime_error{"not a nanovdb file"};
	}
	if (header.gridCount == 0) {
		throw std::runtime_error{"no grids in file"};
	}
	if (header.gridCount > 1) {
		tlog::warning() << "Only loading first grid in file";
	}
	if (metadata.codec != 0) {
		throw std::runtime_error{"cannot use compressed nvdb files"};
	}
	char name[256] = {};
	if (metadata.nameSize > 256) {
		throw std::runtime_error{"nanovdb name too long"};
	}
	f.read(name, metadata.nameSize);
	tlog::info() << name << ": gridSize=" << metadata.gridSize << " filesize=" << metadata.fileSize << " voxelCount=" << metadata.voxelCount
				 << " gridType=" << metadata.gridType << " gridClass=" << metadata.gridClass << " indexBBox=[min=["
				 << metadata.indexBBox[0][0] << "," << metadata.indexBBox[0][1] << "," << metadata.indexBBox[0][2] << "],max]["
				 << metadata.indexBBox[1][0] << "," << metadata.indexBBox[1][1] << "," << metadata.indexBBox[1][2] << "]]";

	// Auto-detect hydrometeor species from grid name and set initial fractions
	int species_id = detect_species_from_name(name);
	m_volume.detected_species_name = species_name(species_id);
	for (int s = 0; s < 4; s++) m_volume.phase_fractions[s] = 0.0f;
	m_volume.phase_fractions[species_id] = 1.0f;
	m_volume.fractions_from_file = true;
	tlog::info() << "Auto-detected species: " << m_volume.detected_species_name
	             << " (adjust manually in UI for mixed-phase clouds)";

	std::vector<char> cpugrid;
	cpugrid.resize(metadata.gridSize);
	f.read(cpugrid.data(), metadata.gridSize);
	m_volume.nanovdb_grid.enlarge(metadata.gridSize);
	m_volume.nanovdb_grid.copy_from_host(cpugrid);
	const nanovdb::FloatGrid* grid = reinterpret_cast<const nanovdb::FloatGrid*>(cpugrid.data());

	float mn = 10000.0f, mx = -10000.0f;
	bool hmm = grid->hasMinMax();
	int xsize = std::max(1, metadata.indexBBox[1][0] - metadata.indexBBox[0][0]);
	int ysize = std::max(1, metadata.indexBBox[1][1] - metadata.indexBBox[0][1]);
	int zsize = std::max(1, metadata.indexBBox[1][2] - metadata.indexBBox[0][2]);
	float maxsize = std::max(std::max(xsize, ysize), zsize);
	float scale = 1.0f / maxsize;
	m_aabb = m_render_aabb = BoundingBox{
		vec3{0.5f - xsize * scale * 0.5f, 0.5f - ysize * scale * 0.5f, 0.5f - zsize * scale * 0.5f},
		vec3{0.5f + xsize * scale * 0.5f, 0.5f + ysize * scale * 0.5f, 0.5f + zsize * scale * 0.5f},
	};
	m_render_aabb_to_local = mat3::identity();

	m_volume.world2index_scale = maxsize;
	m_volume.world2index_offset = vec3{
		(metadata.indexBBox[0][0] + metadata.indexBBox[1][0]) * 0.5f - 0.5f * maxsize,
		(metadata.indexBBox[0][1] + metadata.indexBBox[1][1]) * 0.5f - 0.5f * maxsize,
		(metadata.indexBBox[0][2] + metadata.indexBBox[1][2]) * 0.5f - 0.5f * maxsize,
	};

	auto acc = grid->tree().getAccessor();
	std::vector<uint8_t> bitgrid;
	bitgrid.resize(128 * 128 * 128 / 8);
	for (int i = metadata.indexBBox[0][0]; i < metadata.indexBBox[1][0]; ++i) {
		for (int j = metadata.indexBBox[0][1]; j < metadata.indexBBox[1][1]; ++j) {
			for (int k = metadata.indexBBox[0][2]; k < metadata.indexBBox[1][2]; ++k) {
				float d = acc.getValue({i, j, k});
				if (d > mx) {
					mx = d;
				}
				if (d < mn) {
					mn = d;
				}
				if (d > 0.001f) {
					float fx = ((i + 0.5f) - m_volume.world2index_offset.x) / m_volume.world2index_scale;
					float fy = ((j + 0.5f) - m_volume.world2index_offset.y) / m_volume.world2index_scale;
					float fz = ((k + 0.5f) - m_volume.world2index_offset.z) / m_volume.world2index_scale;
					uint32_t bitidx = morton3D(int(fx * 128.0f + 0.5f), int(fy * 128.0f + 0.5f), int(fz * 128.0f + 0.5f));
					if (bitidx < 128 * 128 * 128) {
						bitgrid[bitidx / 8] |= 1 << (bitidx & 7);
					}
				}
			}
		}
	}
	m_volume.bitgrid.enlarge(bitgrid.size());
	m_volume.bitgrid.copy_from_host(bitgrid);
	tlog::info() << "nanovdb extrema: " << mn << " " << mx << " (" << hmm << ")";
	m_volume.global_majorant = mx;
}

} // namespace ngp
