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

// ============================================================================
// Analytical gradient of the Henyey-Greenstein phase function w.r.t. g
// HG(cos_θ, g) = (1-g²) / (4π·D^(3/2))  where D = 1 + g² - 2g·cos_θ
// dHG/dg = (1/(4π)) · (g³ + g²·cos_θ - 5g + 3·cos_θ) / D^(5/2)
// ============================================================================
__device__ float henyey_greenstein_grad_g(float cos_theta, float g) {
	float g2 = g * g;
	float D = 1.0f + g2 - 2.0f * g * cos_theta;
	float D_safe = fmaxf(D, 1e-4f);
	float D52 = D_safe * D_safe * sqrtf(D_safe);
	float num = g * g2 + g2 * cos_theta - 5.0f * g + 3.0f * cos_theta;
	float result = (1.0f / (4.0f * PI())) * num / D52;
	return fminf(fmaxf(result, -10.0f), 10.0f);
}

// ============================================================================
// Physics-in-the-loop: differentiable forward model + gradient computation.
//
// This kernel implements the core "physics in the loop" from the literature
// (NeRFactor, TensoIR, PhySG, PBR-NeRF pattern):
//   Network → material params → [differentiable physics] → radiance → loss
//
// The network predicts material parameters (albedo, g, w_g1, density).
// The physics forward model converts these to radiance:
//   L = albedo × phase(cos_θ, g1, g2, w_g1) × T_sun × I_sun × sun_color + ambient
//
// The loss is L2 in radiance space. Gradients are analytically backpropagated
// through the physics to the network outputs. This gives physics-weighted
// training: errors that matter visually (forward scattering peak, brightly
// lit regions) receive stronger gradients.
//
// The GT radiance uses the same simplified physics model, ensuring the loss
// can reach zero when predictions match ground truth.
// ============================================================================

// Huber loss kernel for cloud physics mode (standard Reinhard targets).
// Replaces built-in L2 with Huber for better convergence near the optimum.
template <typename T>
__global__ void cloud_physics_huber_kernel(
	uint32_t n_elements,
	uint32_t padded_output_width,
	uint32_t n_output_dims,
	const T* __restrict__ network_output,
	const float* __restrict__ target,
	T* __restrict__ dL_doutput,
	float* __restrict__ loss_values,
	float loss_scale
) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n_elements) return;

	const float n_total = (float)(n_elements * n_output_dims);
	constexpr float HUBER_DELTA = 0.001f;

	uint32_t out_offset = idx * padded_output_width;
	uint32_t tgt_offset = idx * n_output_dims;

	for (uint32_t c = 0; c < padded_output_width; c++) {
		if (c < n_output_dims) {
			float pred = (float)network_output[out_offset + c];
			float gt = target[tgt_offset + c];
			float diff = pred - gt;
			float ad = fabsf(diff);

			float loss_val = (ad <= HUBER_DELTA) ? 0.5f * diff * diff : HUBER_DELTA * (ad - 0.5f * HUBER_DELTA);
			loss_values[out_offset + c] = loss_val / n_total;

			float grad = (ad <= HUBER_DELTA) ? diff : copysignf(HUBER_DELTA, diff);
			grad = loss_scale * grad / n_total;
			if (!isfinite(grad)) grad = 0.0f;
			dL_doutput[out_offset + c] = (T)grad;
		} else {
			loss_values[out_offset + c] = 0.0f;
			dL_doutput[out_offset + c] = (T)0.0f;
		}
	}
}

template <typename T>
__global__ void physics_forward_backward_kernel(
	uint32_t n_elements,
	uint32_t padded_output_width,
	uint32_t n_output_dims,                 // actual output dims (4), for normalization
	const T* __restrict__ network_output,
	const float* __restrict__ target,
	const float* __restrict__ physics_aux,
	T* __restrict__ dL_doutput,
	float* __restrict__ loss_values,
	float loss_scale,
	vec3 sun_color,
	float sun_intensity,
	vec3 sky_col,
	float ambient_factor,
	int ms_octaves,
	float ms_attenuation,
	bool enable_beer_powder,
	float sigma_t_scale,
	float radiance_loss_weight,
	float density_loss_weight
) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n_elements) return;

	const float n_total = (float)(n_elements * n_output_dims);

	uint32_t out_offset = idx * padded_output_width;
	float raw0 = (float)network_output[out_offset + 0];
	float raw1 = (float)network_output[out_offset + 1];
	float raw2 = (float)network_output[out_offset + 2];
	float raw3 = (float)network_output[out_offset + 3];

	// Map to physical parameter space
	float pred_albedo  = fminf(fmaxf(raw0, 0.0f), 1.0f);
	float pred_g_mapped = fminf(fmaxf(raw1, 0.0f), 1.0f);
	float pred_w_g1    = fminf(fmaxf(raw2, 0.0f), 1.0f);
	float pred_density = raw3; // already ≥ 0 from ReLU

	float pred_g1 = pred_g_mapped * 2.0f - 1.0f;  // [0,1] → [-1,1]
	float pred_g2 = -pred_g1 * 0.35f;              // backward lobe

	// ---- Read GT target ----
	uint32_t tgt_offset = idx * 4;
	float gt_rad_r = target[tgt_offset + 0];
	float gt_rad_g = target[tgt_offset + 1];
	float gt_rad_b = target[tgt_offset + 2];
	float gt_density = target[tgt_offset + 3];

	// ---- Read physics auxiliary data ----
	uint32_t aux_offset = idx * 4;
	float T_sun     = physics_aux[aux_offset + 0];
	float cos_theta = physics_aux[aux_offset + 1];

	// ---- Multi-scatter forward model (Wrenninge/Frostbite octave method) ----
	// Matches compute_inscattered_radiance used for GT and rendering.
	float isotropic_phase = 1.0f / (4.0f * PI());
	// Use gt_density for beer_powder (volume-level effect, not differentiable w.r.t. network)
	float sigma_t_ms = gt_density * sigma_t_scale;
	float g1_ms = pred_g1;
	float g2_ms = pred_g2;

	vec3 pred_rad = vec3(0.0f);
	float drad_dalbedo_acc = 0.0f;  // accumulate d(pred_rad)/d(albedo) across octaves
	float drad_dg1_acc = 0.0f;      // accumulate d(pred_rad)/d(g1)
	float drad_dw_acc = 0.0f;       // accumulate d(pred_rad)/d(w_g1)

	for (int octave = 0; octave < ms_octaves; octave++) {
		float phase = dual_lobe_hg(cos_theta, g1_ms, g2_ms, pred_w_g1);
		float att_n = (octave == 0) ? 1.0f : powf(ms_attenuation, (float)octave);
		float T_sun_ms = powf(T_sun, att_n);

		float energy = 1.0f;
		if (enable_beer_powder && sigma_t_ms > 0.0f) {
			energy = beer_powder(sigma_t_ms * 0.1f);
		}

		float sun_phase_T = sun_intensity * phase * T_sun_ms * energy;
		pred_rad.x += pred_albedo * sun_phase_T * sun_color.x + sky_col.x * isotropic_phase * ambient_factor * att_n;
		pred_rad.y += pred_albedo * sun_phase_T * sun_color.y + sky_col.y * isotropic_phase * ambient_factor * att_n;
		pred_rad.z += pred_albedo * sun_phase_T * sun_color.z + sky_col.z * isotropic_phase * ambient_factor * att_n;

		// Accumulate gradient components for this octave
		float sun_T_energy = sun_intensity * T_sun_ms * energy;
		drad_dalbedo_acc += phase * sun_T_energy;

		float dphase_dg1_oct = pred_w_g1 * henyey_greenstein_grad_g(cos_theta, g1_ms) * ms_attenuation
		                     + (1.0f - pred_w_g1) * henyey_greenstein_grad_g(cos_theta, g2_ms) * (-0.35f) * ms_attenuation;
		if (octave == 0) {
			dphase_dg1_oct = pred_w_g1 * henyey_greenstein_grad_g(cos_theta, g1_ms)
			               + (1.0f - pred_w_g1) * henyey_greenstein_grad_g(cos_theta, g2_ms) * (-0.35f);
		}
		drad_dg1_acc += pred_albedo * dphase_dg1_oct * sun_T_energy;

		float dphase_dw_oct = henyey_greenstein(cos_theta, g1_ms) - henyey_greenstein(cos_theta, g2_ms);
		drad_dw_acc += pred_albedo * dphase_dw_oct * sun_T_energy;

		sigma_t_ms *= ms_attenuation;
		g1_ms *= ms_attenuation;
		g2_ms *= ms_attenuation;
	}

	// ---- Reinhard tone-map loss ----
	float tm_pred_r = pred_rad.x / (1.0f + pred_rad.x);
	float tm_pred_g = pred_rad.y / (1.0f + pred_rad.y);
	float tm_pred_b = pred_rad.z / (1.0f + pred_rad.z);
	float tm_gt_r = gt_rad_r / (1.0f + gt_rad_r);
	float tm_gt_g = gt_rad_g / (1.0f + gt_rad_g);
	float tm_gt_b = gt_rad_b / (1.0f + gt_rad_b);

	float diff_r = tm_pred_r - tm_gt_r;
	float diff_g = tm_pred_g - tm_gt_g;
	float diff_b = tm_pred_b - tm_gt_b;
	float diff_d = pred_density - gt_density;

	// Use Huber loss (L1 for large errors, L2 for small) — combines stability with precision.
	// Huber δ=0.01: below 0.01 diff uses L2 (smooth gradient), above uses L1 (constant drive).
	constexpr float HUBER_DELTA = 0.0001f;
	auto huber = [](float d, float delta) -> float {
		float ad = fabsf(d);
		return (ad <= delta) ? 0.5f * d * d : delta * (ad - 0.5f * delta);
	};
	auto huber_grad = [](float d, float delta) -> float {
		return (fabsf(d) <= delta) ? d : copysignf(delta, d);
	};

	float loss_rad = radiance_loss_weight * (huber(diff_r, HUBER_DELTA) + huber(diff_g, HUBER_DELTA) + huber(diff_b, HUBER_DELTA));
	float loss_den = density_loss_weight * diff_d * diff_d;

	for (uint32_t c = 0; c < padded_output_width; c++) {
		float val = 0.0f;
		if (c < 3) val = loss_rad / 3.0f / n_total;
		else if (c == 3) val = loss_den / n_total;
		loss_values[idx * padded_output_width + c] = val;
	}

	// ---- Gradients through Huber, Reinhard, and physics ----
	float inv_sq_r = 1.0f / ((1.0f + pred_rad.x) * (1.0f + pred_rad.x));
	float inv_sq_g = 1.0f / ((1.0f + pred_rad.y) * (1.0f + pred_rad.y));
	float inv_sq_b = 1.0f / ((1.0f + pred_rad.z) * (1.0f + pred_rad.z));

	float dLrad_r = radiance_loss_weight * huber_grad(diff_r, HUBER_DELTA) * inv_sq_r / n_total;
	float dLrad_g = radiance_loss_weight * huber_grad(diff_g, HUBER_DELTA) * inv_sq_g / n_total;
	float dLrad_b = radiance_loss_weight * huber_grad(diff_b, HUBER_DELTA) * inv_sq_b / n_total;

	// Multi-scatter accumulated gradients → per-color-channel
	float drad_dalbedo_r = drad_dalbedo_acc * sun_color.x;
	float drad_dalbedo_g = drad_dalbedo_acc * sun_color.y;
	float drad_dalbedo_b = drad_dalbedo_acc * sun_color.z;

	float dL_dalbedo = dLrad_r * drad_dalbedo_r + dLrad_g * drad_dalbedo_g + dLrad_b * drad_dalbedo_b;
	float dL_dg1     = dLrad_r * drad_dg1_acc * sun_color.x + dLrad_g * drad_dg1_acc * sun_color.y + dLrad_b * drad_dg1_acc * sun_color.z;
	float dL_dw      = dLrad_r * drad_dw_acc * sun_color.x + dLrad_g * drad_dw_acc * sun_color.y + dLrad_b * drad_dw_acc * sun_color.z;

	// Chain rule through parameter mappings:
	// albedo = clamp(raw0, 0, 1)  → d/draw0 = 1 inside [0,1], 0 outside
	// g_mapped = clamp(raw1, 0, 1) → g1 = g_mapped×2-1 → d(g1)/d(raw1) = 2 inside [0,1]
	// w_g1 = clamp(raw2, 0, 1)    → d/draw2 = 1 inside [0,1]
	float clamp0 = (raw0 > 0.0f && raw0 < 1.0f) ? 1.0f : 0.0f;
	float clamp1 = (raw1 > 0.0f && raw1 < 1.0f) ? 1.0f : 0.0f;
	float clamp2 = (raw2 > 0.0f && raw2 < 1.0f) ? 1.0f : 0.0f;

	float grad0 = loss_scale * dL_dalbedo * clamp0;
	float grad1 = loss_scale * dL_dg1 * 2.0f * clamp1;
	float grad2 = loss_scale * dL_dw * clamp2;
	float grad3 = loss_scale * density_loss_weight * 2.0f * diff_d / n_total;

	// NaN guard (no hard clip — normalization by n_total keeps scale reasonable)
	if (!isfinite(grad0)) grad0 = 0.0f;
	if (!isfinite(grad1)) grad1 = 0.0f;
	if (!isfinite(grad2)) grad2 = 0.0f;
	if (!isfinite(grad3)) grad3 = 0.0f;

	dL_doutput[out_offset + 0] = (T)grad0;
	dL_doutput[out_offset + 1] = (T)grad1;
	dL_doutput[out_offset + 2] = (T)grad2;
	dL_doutput[out_offset + 3] = (T)grad3;
	for (uint32_t c = 4; c < padded_output_width; c++) {
		dL_doutput[out_offset + c] = (T)0.0f;
	}
}

__global__ void volume_generate_training_data_kernel(
	uint32_t n_elements,
	vec3* pos_out,
	vec4* target_out,
	vec4* physics_aux_out, // PITL: per-vertex (T_sun, cos_theta, 0, 0); NULL when not PITL
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
	bool enable_physics, // master toggle
	bool physics_in_the_loop // true = physics-in-the-loop training
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
	float outcos[MAX_TRAIN_VERTICES]; // PITL: cos(θ) = dot(dir, sun_dir) at each vertex
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
				outcos[numout] = dot(dir, sun_dir); // capture viewing angle BEFORE scattering changes dir
				if (physics_in_the_loop && enable_physics && enable_direct_light && density > 0.001f) {
					// PITL: compute GT radiance inline (using correct pre-scatter dir)
					outradiance[numout] = compute_inscattered_radiance(
						pos, dir, sun_dir, sun_color, sun_intensity,
						density, global_majorant, hydro_props, use_dual_lobe,
						enable_beer_powder, ms_octaves, ms_attenuation,
						aabb, grid, world2index_offset, world2index_scale,
						shadow_steps, sky_col, up_dir
					);
				} else if (physics_in_the_loop) {
					outradiance[numout] = vec3(0.0f);
				} else if (enable_physics && enable_direct_light && density > 0.001f) {
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
		if (physics_in_the_loop) {
			float gt_albedo = hydro_props.albedo;
			float gt_g_mapped = (hydro_props.g1 + 1.0f) * 0.5f;
			float gt_w_g1 = hydro_props.w_g1;
			for (uint32_t i = prev_numout; i < numout; ++i) {
				pos_out[oidx + i] = outpos[i];
				float cos_th = outcos[i];
				float T_s = 0.0f;
				if (outdensity[i] > 0.001f) {
					T_s = march_to_sun(outpos[i], sun_dir, aabb, grid,
					                   world2index_offset, world2index_scale,
					                   global_majorant, shadow_steps);
				}
				// Targets: both material params (Phase 1) and radiance (Phase 2)
				// target = (gt_radiance.rgb, gt_density)
				// physics_aux = (T_sun, cos_theta, gt_albedo, gt_g_mapped | gt_w_g1)
				target_out[oidx + i] = vec4{outradiance[i].x, outradiance[i].y, outradiance[i].z, outdensity[i]};
				if (physics_aux_out) {
					physics_aux_out[oidx + i] = vec4{T_s, cos_th, 0.0f, 0.0f};
				}
			}
		} else if (enable_physics) {
			// Cloud physics: standard Reinhard tone-mapped inscattered radiance.
			for (uint32_t i = prev_numout; i < numout; ++i) {
				vec3 rad = outradiance[i];
				vec3 mapped = vec3{
					rad.x / (1.0f + rad.x),
					rad.y / (1.0f + rad.y),
					rad.z / (1.0f + rad.z)
				};
				pos_out[oidx + i] = outpos[i];
				target_out[oidx + i] = vec4(mapped, outdensity[i]);
			}
		} else {
			for (uint32_t i = prev_numout; i < numout; ++i) {
				pos_out[oidx + i] = outpos[i];
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
	if (m_volume.physics_in_the_loop) {
		m_volume.training.physics_aux.enlarge(n_elements);
	}

	float distance_scale = 1.f / std::max(m_volume.inv_distance_scale, 0.01f);
	auto sky_col = m_background_color.rgb();

	// Compute blended hydrometeor properties from phase fractions
	HydrometeorProps hydro_props = blend_hydrometeor_props(m_volume.phase_fractions);

	// In physics-in-the-loop, training targets must use TRUE physical params (no overrides).
	// Overrides only affect rendering (step kernel), not training, so the network learns
	// the actual material and the user can edit at render time without corrupting weights.
	if (!m_volume.physics_in_the_loop) {
		if (m_volume.albedo_override > 0.0f) hydro_props.albedo = m_volume.albedo_override;
		if (m_volume.g_override != 0.0f) { hydro_props.g1 = m_volume.g_override; hydro_props.g2 = -m_volume.g_override * 0.35f; }
	}

	// When physics disabled, use original defaults for training
	float train_albedo = m_volume.enable_physics ? hydro_props.albedo : 0.95f;

	linear_kernel(
		volume_generate_training_data_kernel,
		0,
		stream,
		n_elements / MAX_TRAIN_VERTICES,
		m_volume.training.positions.data(),
		m_volume.training.targets.data(),
		m_volume.physics_in_the_loop ? m_volume.training.physics_aux.data() : (vec4*)nullptr,
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
		m_volume.enable_physics,
		m_volume.physics_in_the_loop
	);
	m_rng.advance(n_elements * 256);

	GPUMatrix<float> training_batch_matrix((float*)(m_volume.training.positions.data()), n_input_dims, batch_size);
	GPUMatrix<float> training_target_matrix((float*)(m_volume.training.targets.data()), n_output_dims, batch_size);

	if (m_volume.physics_in_the_loop) {
		// ---- Physics-in-the-loop: two-phase training ----
		// Phase 1 (steps 0-10K): direct material param supervision via standard L2.
		//   Targets = (gt_albedo, gt_g_mapped, gt_w_g1, gt_density) from training data.
		//   Converges quickly to accurate material params + density.
		// Phase 2 (steps 10K+): differentiable physics gradient fine-tuning.
		//   Physics forward model (multi-scatter RTE) → tone-mapped radiance → L2 loss.
		//   Gradients backpropagate through physics, refining the learned params.
		{
			// Pure physics-in-the-loop: differentiable multi-scatter RTE
			const float loss_scale = LOSS_SCALE();
			const uint32_t padded_width = m_network->padded_output_width();

			GPUMatrix<network_precision_t> ext_dL_dy(padded_width, batch_size, stream);

			auto ctx = m_trainer->forward(stream, loss_scale, training_batch_matrix, training_target_matrix,
			                               nullptr, false, false, &ext_dL_dy);

			vec3 sun_color = vec3{1.0f, 0.95f, 0.85f};
			linear_kernel(physics_forward_backward_kernel<network_precision_t>, 0, stream,
				batch_size,
				padded_width,
				(uint32_t)n_output_dims,
				ctx->output.data(),
				(const float*)m_volume.training.targets.data(),
				(const float*)m_volume.training.physics_aux.data(),
				ext_dL_dy.data(),
				ctx->L.data(),
				loss_scale,
				sun_color,
				m_volume.sun_intensity,
				sky_col,
				0.15f,
				m_volume.ms_octaves,
				m_volume.ms_attenuation,
				m_volume.enable_beer_powder,
				m_volume.global_majorant,
				1.0f,
				1.0f
			);

			m_trainer->backward(stream, *ctx, training_batch_matrix);
			m_trainer->optimizer_step(stream, loss_scale);

			m_training_step++;

			if (get_loss_scalar) {
				m_loss_scalar.update(m_trainer->loss(stream, *ctx));
			}
		}
	} else if (m_volume.enable_physics) {
		// ---- Cloud physics with Huber loss ----
		const float loss_scale = LOSS_SCALE();
		const uint32_t padded_width = m_network->padded_output_width();

		GPUMatrix<network_precision_t> ext_dL_dy(padded_width, batch_size, stream);

		auto ctx = m_trainer->forward(stream, loss_scale, training_batch_matrix, training_target_matrix,
		                               nullptr, false, false, &ext_dL_dy);

		linear_kernel(cloud_physics_huber_kernel<network_precision_t>, 0, stream,
			batch_size,
			padded_width,
			(uint32_t)n_output_dims,
			ctx->output.data(),
			(const float*)m_volume.training.targets.data(),
			ext_dL_dy.data(),
			ctx->L.data(),
			loss_scale
		);

		m_trainer->backward(stream, *ctx, training_batch_matrix);
		m_trainer->optimizer_step(stream, loss_scale);

		m_training_step++;

		if (get_loss_scalar) {
			m_loss_scalar.update(m_trainer->loss(stream, *ctx));
		}
	} else {
		// ---- Standard training (no physics) ----
		auto ctx = m_trainer->training_step(stream, training_batch_matrix, training_target_matrix);

		m_training_step++;

		if (get_loss_scalar) {
			m_loss_scalar.update(m_trainer->loss(stream, *ctx));
		}
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
	bool enable_physics, // master toggle
	bool physics_in_the_loop // PITL: source = inscattered only (no env per sample)
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
		// ===== PHYSICS-BASED RENDERING (alpha-compositing ray march) =====
		// Fixed direction (no scattering bounces) — matches step kernel behavior.
		//
		// Two modes:
		//   PITL:   source = inscattered only.  pixel = Σ L_i * α_i + (1-A) * env
		//   Legacy: source = env + inscattered.  pixel = env + Σ L_i * α_i
		//           (legacy matches baked radiance training targets)
		vec3 sun_color = vec3{1.0f, 0.95f, 0.85f};
		vec4 env = proc_envmap_render(dir, up_dir, sun_dir, sky_col);
		vec4 accum = vec4(0.0f);

		for (int iter = 0; iter < 128; ++iter) {
			vec3 nanovdbpos = pos * world2index_scale + world2index_offset;
			float density =
				acc.getValue({int(nanovdbpos.x + random_val(rng)), int(nanovdbpos.y + random_val(rng)), int(nanovdbpos.z + random_val(rng))});

			float extinction_prob = density / global_majorant;
			if (extinction_prob > 1.0f) extinction_prob = 1.0f;

			float T = 1.0f - accum.a;
			float alpha = extinction_prob * T;

			// PITL: source = inscattered only (cloud properly occludes background)
			// Legacy: source = env + inscattered (background always visible)
			vec3 local_rgb = physics_in_the_loop ? vec3(0.0f) : env.rgb();
			if (enable_direct_light && density > 0.001f) {
				local_rgb = local_rgb + compute_inscattered_radiance(
					pos, dir, sun_dir, sun_color, sun_intensity,
					density, global_majorant, hydro_props, use_dual_lobe,
					enable_beer_powder, ms_octaves, ms_attenuation,
					aabb, grid, world2index_offset, world2index_scale,
					shadow_steps, sky_col, up_dir
				);
			}

			accum.rgb() += local_rgb * alpha;
			accum.a += alpha;

			if (accum.a > 0.99f) break;
			if (!walk_to_next_event(rng, aabb, pos, dir, bitgrid, scale)) break;
		}

		// Remaining transmittance → environment
		vec4 col;
		col.rgb() = accum.rgb() + (1.0f - accum.a) * env.rgb();
		col.a = 1.0f;
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
	bool enable_physics, // master toggle
	bool physics_in_the_loop, // network predicts material params; renderer evaluates RTE
	float albedo_override, // >0: override network-predicted albedo for material editing
	float g_override       // !=0: override network-predicted g for material editing
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

	vec3 final_rgb;

	if (physics_in_the_loop && enable_physics) {
		// ====================================================================
		// PHYSICS-IN-THE-LOOP: network predicted material parameters, not RGB.
		// Channel layout: (albedo, g1_mapped, w_g1, density)
		// Evaluate RTE at render time → enables relighting & material editing.
		// ====================================================================
		// Source = inscattered radiance only. Background comes through (1-A)*env
		// at ray termination, so clouds properly occlude the sky.
		final_rgb = vec3(0.0f);

		float safe_density = fmaxf(density, 0.0f);
		if (enable_direct_light && safe_density > 0.001f) {
			// Network-predicted material parameters
			float pred_albedo = fmaxf(fminf(local_output.x, 0.9999f), 0.0f);
			float pred_g1     = fmaxf(fminf(local_output.y, 1.0f), 0.0f) * 2.0f - 1.0f; // [0,1]→[-1,1]
			float pred_w_g1   = fmaxf(fminf(local_output.z, 1.0f), 0.0f);

			// Apply user overrides for interactive material editing
			if (albedo_override > 0.0f) pred_albedo = albedo_override;
			if (g_override != 0.0f) {
				pred_g1 = g_override;
				pred_w_g1 = hydro_props.w_g1; // use blended w_g1 when g is overridden
			}

			HydrometeorProps pred_props;
			pred_props.albedo = pred_albedo;
			pred_props.g1     = pred_g1;
			pred_props.g2     = -pred_g1 * 0.35f; // backward lobe derived from forward
			pred_props.w_g1   = pred_w_g1;

			vec3 sun_color = vec3{1.0f, 0.95f, 0.85f};
			vec3 inscattered = compute_inscattered_radiance(
				pos, dir, sun_dir, sun_color, sun_intensity,
				safe_density, global_majorant, pred_props, use_dual_lobe,
				enable_beer_powder, ms_octaves, ms_attenuation,
				aabb, grid, world2index_offset, world2index_scale,
				shadow_steps, sky_col, up_dir
			);
			if (isfinite(inscattered.x) && isfinite(inscattered.y) && isfinite(inscattered.z)) {
				final_rgb = final_rgb + inscattered;
			}
		}
	} else if (enable_physics) {
		// Cloud physics: standard inverse Reinhard.
		final_rgb = local_output.rgb();
		final_rgb = vec3{
			fmaxf(fminf(final_rgb.x, 0.999f), 0.0f),
			fmaxf(fminf(final_rgb.y, 0.999f), 0.0f),
			fmaxf(fminf(final_rgb.z, 0.999f), 0.0f)
		};
		final_rgb = vec3{
			final_rgb.x / (1.0f - final_rgb.x),
			final_rgb.y / (1.0f - final_rgb.y),
			final_rgb.z / (1.0f - final_rgb.z)
		};
		if (!isfinite(final_rgb.x)) final_rgb.x = 0.0f;
		if (!isfinite(final_rgb.y)) final_rgb.y = 0.0f;
		if (!isfinite(final_rgb.z)) final_rgb.z = 0.0f;
	} else {
		final_rgb = local_output.rgb();
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
			m_volume.enable_physics,
			m_volume.physics_in_the_loop
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
				m_volume.enable_physics,
				m_volume.physics_in_the_loop,
				m_volume.albedo_override,
				m_volume.g_override
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

	// Auto-detect hydrometeor species from grid name
	int species_id = detect_species_from_name(name);
	bool is_generic_name = (strcmp(name, "density") == 0 || strcmp(name, "") == 0 || strcmp(name, "float") == 0);
	if (is_generic_name) {
		// Generic grid name — no species info in the nvdb file.
		// The combined density field doesn't encode which hydrometeor types contributed.
		// Default to Water; user should adjust fractions manually in UI.
		m_volume.detected_species_name = "Water (generic grid — adjust in UI)";
		for (int s = 0; s < 4; s++) m_volume.phase_fractions[s] = 0.0f;
		m_volume.phase_fractions[0] = 1.0f;
		m_volume.fractions_from_file = false;
		tlog::info() << "Grid name '" << name << "' has no species info — defaulting to Water. "
		             << "Adjust phase fractions in UI for mixed-phase clouds.";
	} else {
		m_volume.detected_species_name = species_name(species_id);
		for (int s = 0; s < 4; s++) m_volume.phase_fractions[s] = 0.0f;
		m_volume.phase_fractions[species_id] = 1.0f;
		m_volume.fractions_from_file = true;
		tlog::info() << "Auto-detected species: " << m_volume.detected_species_name;
	}

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
