import math
import os
import re
import time

from . import nodes as core


mx = core.mx
np = core.np

SPECTRUM_USER_POLICIES = ["fast", "standard"]


def _spectrum_preset(
    mode,
    manual_forecast_steps="",
    weight=1.0,
    degree=3,
    ridge=0.1,
    window_size=2.0,
    flex_window=0.0,
    warmup_steps=5,
    final_real_steps=3,
    min_points=0,
    time_coord_mode="step",
):
    return {
        "spectrum_mode": mode,
        "manual_forecast_steps": manual_forecast_steps,
        "spectrum_weight": float(weight),
        "spectrum_degree": int(degree),
        "spectrum_ridge": float(ridge),
        "spectrum_window_size": float(window_size),
        "spectrum_flex_window": float(flex_window),
        "spectrum_warmup_steps": int(warmup_steps),
        "stop_caching_step": 100,
        "_final_real_steps": int(final_real_steps),
        "spectrum_min_points": int(min_points),
        "spectrum_time_coord_mode": str(time_coord_mode or "step"),
    }


def _with_final_real_steps(preset, steps, final_real_steps=None):
    result = dict(preset)
    if final_real_steps is None:
        final_real_steps = int(result.pop("_final_real_steps", 3))
    else:
        result.pop("_final_real_steps", None)
    final_real_steps = int(final_real_steps)
    if final_real_steps <= 0:
        result["stop_caching_step"] = 100
    else:
        result["stop_caching_step"] = max(0, int(steps) - final_real_steps)
    return result


def _spectrum_user_policy_preset(policy, steps):
    policy = str(policy or "standard").lower().replace(" ", "_")
    steps = int(steps)
    if policy == "standard" and steps < 35:
        preset = _with_final_real_steps(
            _spectrum_preset(
                "repo",
                "",
                0.5,
                4,
                0.1,
                2.0,
                0.75,
                5,
                3,
                min_points=5,
                time_coord_mode="sigma",
            ),
            steps,
            3,
        )
        preset.update({
            "spectrum_limiter_max_intervention": 0.0,
            "spectrum_limiter_mode": "fast",
            "spectrum_schedule_mode": "repo",
            "spectrum_time_base": max(1, steps),
            "spectrum_dynamic_max_w": 0.0,
            "spectrum_dynamic_w_bias": 0.0,
        })
        return preset
    if policy in ("fast", "standard"):
        warmup_steps = 3 if policy == "fast" else 5
        preset = _with_final_real_steps(
            _spectrum_preset(
                "repo",
                "",
                0.5,
                4,
                0.1,
                2.0,
                0.75,
                warmup_steps,
                3,
            ),
            steps,
            3,
        )
        preset.update({
            "spectrum_limiter_max_intervention": 0.0,
            "spectrum_limiter_mode": "fast",
            "spectrum_schedule_mode": "repo",
            "spectrum_time_base": max(1, steps),
            "spectrum_dynamic_max_w": 0.8,
            "spectrum_dynamic_w_bias": 0.0,
        })
        return preset
    return _spectrum_user_policy_preset("standard", steps)


SPECTRUM_FORCE_REAL_PRE_RATIO = 1.016
SPECTRUM_FORCE_REAL_BLOCK_RATIO = 0.50


def resolve_spectrum_auto(speed_patch, steps, sampler_name, mode="standard"):
    policy = str(mode or "standard").lower().replace(" ", "_")
    if policy not in SPECTRUM_USER_POLICIES:
        policy = "standard"
    return resolve_spectrum_config(
        {"profile": policy, "final_real_steps": 0},
        speed_patch,
        steps,
        sampler_name,
    )


def terminal_profile_label(resolved_label, requested_profile=None):
    label = str(resolved_label or "off").strip()
    if label.startswith("sdxl "):
        return label[5:]
    return label


def resolve_spectrum_config(spectrum, speed_patch, steps, sampler_name):
    if not spectrum or not isinstance(spectrum, dict):
        return None, "off", "disabled"
    profile = str(spectrum.get("profile", spectrum.get("mode", "default")) or "default")
    advanced_profile = (
        bool(spectrum.get("advanced", False))
        or bool(spectrum.get("manual", False))
        or profile in ("advanced", "manual")
    )
    user_policy = profile.lower().replace(" ", "_")
    if user_policy not in SPECTRUM_USER_POLICIES:
        user_policy = "standard"
    auto_profile = profile in ("default", "default (25 steps)", "fast", "standard")
    if auto_profile:
        profile = user_policy

    patch_name = core.normalized_speed_patch_name(speed_patch)
    if auto_profile and patch_name:
        return None, "off", "disabled with speed patches"
    if sampler_name != "euler":
        return None, "off", "requires Euler sampler"

    final_real_steps_value = int(spectrum.get("final_real_steps", 0))
    final_real_steps_override = final_real_steps_value if final_real_steps_value > 0 else None
    if advanced_profile:
        preset = _spectrum_preset(
            "auto",
            "",
            float(spectrum.get("weight", 1.0)),
            int(spectrum.get("degree", 3)),
            float(spectrum.get("ridge", 0.1)),
            float(spectrum.get("window_size", 2.0)),
            float(spectrum.get("flex_window", 0.0)),
            int(spectrum.get("warmup_steps", 5)),
            final_real_steps_value,
        )
        preset = _with_final_real_steps(preset, steps, final_real_steps_value)
        return preset, "advanced", "active"

    if auto_profile:
        if not patch_name and int(steps) >= 20:
            preset = _spectrum_user_policy_preset(profile, steps)
            if final_real_steps_override:
                preset = _with_final_real_steps(preset, steps, final_real_steps_override)
            return preset, f"sdxl {profile}", "active"
        return None, "off", "requires at least 20 steps"

    return None, "off", "unknown profile"


def _parse_forecast_steps(value, total_steps):
    result = set()
    for chunk in str(value or "").replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            step = int(chunk)
        except ValueError:
            continue
        if 1 < step < total_steps:
            result.add(step - 1)
    return result


def _feature_rms(value):
    return mx.sqrt(mx.mean(mx.square(value.astype(mx.float32))))


def _feature_relative_rms(delta, reference):
    return _feature_rms(delta) / (_feature_rms(reference) + mx.array(1e-6, dtype=mx.float32))


def _limit_forecast_feature_delta(predicted, values, soft_ratio=1.35, hard_ratio=2.25, knee_strength=0.65):
    if len(values) < 2:
        return predicted, mx.array(1.0, dtype=mx.float32), mx.array(0.0, dtype=mx.float32)
    last = values[-1].reshape(predicted.shape)
    prev = values[-2].reshape(predicted.shape)
    delta = predicted.astype(mx.float32) - last.astype(mx.float32)
    last_delta = last.astype(mx.float32) - prev.astype(mx.float32)
    delta_rms = _feature_rms(delta)
    last_delta_rms = _feature_rms(last_delta)
    safe = mx.maximum(last_delta_rms * float(soft_ratio), mx.array(1e-6, dtype=mx.float32))
    hard = mx.maximum(last_delta_rms * float(hard_ratio), safe)
    ratio = delta_rms / safe
    clipped_scale = mx.minimum(mx.array(1.0, dtype=mx.float32), hard / (delta_rms + mx.array(1e-6, dtype=mx.float32)))
    softened_scale = mx.where(
        ratio > 1.0,
        1.0 / (1.0 + (ratio - 1.0) * float(knee_strength)),
        mx.array(1.0, dtype=mx.float32),
    )
    scale = mx.minimum(clipped_scale, softened_scale)
    return (last + delta * scale).astype(predicted.dtype), scale, ratio


def _forecast_delta_ratio(predicted, values, soft_ratio=1.35):
    if len(values) < 2:
        return mx.array(0.0, dtype=mx.float32)
    last = values[-1].reshape(predicted.shape)
    prev = values[-2].reshape(predicted.shape)
    delta = predicted.astype(mx.float32) - last.astype(mx.float32)
    last_delta = last.astype(mx.float32) - prev.astype(mx.float32)
    delta_rms = _feature_rms(delta)
    last_delta_rms = _feature_rms(last_delta)
    safe = mx.maximum(last_delta_rms * float(soft_ratio), mx.array(1e-6, dtype=mx.float32))
    return delta_rms / safe


def _adaptive_limiter_parameter_push(ratio, base_weight, base_ridge, max_intervention=1.0):
    # Ratio is measured against the soft delta limit. The curve intentionally
    # rises fast: once a forecast starts to overshoot, move toward the visually
    # useful w=1/r=0.4 zone found in the 35-step grid test.
    start = 0.75
    full = 1.10
    if ratio <= start:
        return float(base_weight), float(base_ridge), 0.0
    t = min(1.0, max(0.0, (float(ratio) - start) / (full - start)))
    max_intervention = max(0.0, min(1.0, float(max_intervention)))
    intensity = (1.0 - (1.0 - t) ** 3) * max_intervention
    target_weight = max(float(base_weight), 1.0)
    target_ridge = max(float(base_ridge), 0.4)
    weight = float(base_weight) + (target_weight - float(base_weight)) * intensity
    ridge = float(base_ridge) + (target_ridge - float(base_ridge)) * intensity
    return min(1.0, weight), min(0.65, ridge), intensity


def _adaptive_limiter_profile(total_steps, max_intervention=1.0):
    max_intervention = max(0.0, min(1.0, float(max_intervention)))
    if max_intervention <= 0.0:
        return 9999.0, 9999.0, 0.0
    ratio_scale = 1.0 / (0.6 + 0.4 * max_intervention)
    soft_ratio = 1.35 * ratio_scale
    hard_ratio = max(soft_ratio + 0.05, 2.25 * ratio_scale)
    knee_strength = 0.65 * max_intervention
    return soft_ratio, hard_ratio, knee_strength


def _repo_dynamic_weight(current_window, progress_steps, step_index, max_w=0.8, min_ws=1.0, max_ws=None, bias=0.0):
    if max_ws is None:
        max_ws = 8.0
    if float(max_ws) <= float(min_ws):
        base = max(0.0, min(float(max_w), float(max_w)))
        return base
    remaining = min(float(current_window), int(progress_steps) - max(int(step_index), 1))
    remaining = int(max(0.0, remaining))
    ratio = (remaining - float(min_ws)) / (float(max_ws) - float(min_ws))
    base = max(0.0, min(float(max_w), ratio * float(max_w)))
    bias = max(-1.0, min(1.0, float(bias)))
    if bias < 0.0:
        return base * (1.0 + bias)
    if bias > 0.0:
        return base + (1.0 - base) * bias
    return base


def _normalized_spectrum_coords(values):
    values = [float(value) for value in values]
    if not values:
        return []
    start = values[0]
    end = values[-1]
    denom = end - start
    if abs(denom) < 1e-12:
        return [0.0 for _ in values]
    return [((value - start) / denom) * 2.0 - 1.0 for value in values]


def _spectrum_time_coords_from_step_plan(step_plan, mode):
    mode = str(mode or "step").lower().strip()
    if mode == "sigma":
        sigmas = [core.mx_scalar_float(row[6]) for row in step_plan]
        return _normalized_spectrum_coords(sigmas), True
    if mode == "log_sigma":
        sigmas = [max(core.mx_scalar_float(row[6]), 1e-6) for row in step_plan]
        return _normalized_spectrum_coords([math.log(value) for value in sigmas]), True
    return [float(index) for index in range(len(step_plan))], False


def _chebyshev_basis(taus, degree):
    taus = taus.reshape(-1, 1)
    cols = [mx.ones_like(taus)]
    if degree >= 1:
        cols.append(taus)
    for _ in range(2, degree + 1):
        cols.append(2.0 * taus * cols[-1] - cols[-2])
    return mx.concatenate(cols[: degree + 1], axis=1)


class _ChebyshevForecaster:
    def __init__(self, degree=3, max_points=100, ridge=0.1, total_steps=50, normalized_time=False):
        self.degree = int(max(1, degree))
        self.max_points = int(max(self.degree + 2, max_points))
        self.ridge = float(ridge)
        self.total_steps = float(total_steps)
        self.normalized_time = bool(normalized_time)
        self.times = []
        self.values = []
        self.shape = None
        self.dtype = None

    def ready(self, min_points=None):
        required = min(self.max_points, self.degree + 2) if min_points is None else int(min_points)
        return len(self.values) >= max(1, required)

    def _tau(self, values):
        values = mx.array(values, dtype=mx.float32)
        if self.normalized_time:
            return values
        return (values - (self.total_steps * 0.5)) * (2.0 / self.total_steps)

    def update(self, step_index, value):
        value = value.astype(mx.float32)
        self.shape = value.shape
        self.dtype = value.dtype
        self.times.append(float(step_index))
        self.values.append(value.reshape(-1))
        if len(self.values) > self.max_points:
            self.times = self.times[-self.max_points:]
            self.values = self.values[-self.max_points:]

    def predict_chebyshev(self, step_index):
        features = mx.stack(self.values, axis=0).astype(mx.float32)
        design = _chebyshev_basis(self._tau(self.times), self.degree).astype(mx.float32)
        xt = design.T
        gram = xt @ design
        gram = gram + self.ridge * mx.eye(gram.shape[0], dtype=mx.float32)
        with mx.stream(mx.cpu):
            try:
                chol = mx.linalg.cholesky(gram)
                coef = mx.linalg.solve(chol.T, mx.linalg.solve(chol, xt @ features))
            except Exception:
                jitter = mx.array(1e-6, dtype=mx.float32) * mx.mean(mx.diag(gram))
                gram = gram + jitter * mx.eye(gram.shape[0], dtype=mx.float32)
                chol = mx.linalg.cholesky(gram)
                coef = mx.linalg.solve(chol.T, mx.linalg.solve(chol, xt @ features))
        x_star = _chebyshev_basis(self._tau([float(step_index)]), self.degree).astype(mx.float32)
        return (x_star @ coef).reshape(self.shape)

    def predict_taylor(self, step_index, order=2):
        if len(self.values) < 2:
            return self.values[-1].reshape(self.shape)
        h_i = self.values[-1]
        h_im1 = self.values[-2]
        t_i = self.times[-1]
        t_im1 = self.times[-2]
        dt = max(t_i - t_im1, 1e-8)
        k = (float(step_index) - t_i) / dt
        out = h_i + k * (h_i - h_im1)
        if order >= 2 and len(self.values) >= 3:
            h_im2 = self.values[-3]
            d2 = h_i - 2.0 * h_im1 + h_im2
            out = out + 0.5 * k * (k - 1.0) * d2
        if order >= 3 and len(self.values) >= 4:
            h_im2 = self.values[-3]
            h_im3 = self.values[-4]
            d3 = h_i - 3.0 * h_im1 + 3.0 * h_im2 - h_im3
            out = out + (k * (k - 1.0) * (k - 2.0) / 6.0) * d3
        return out.reshape(self.shape)


class _SpectrumForecaster:
    def __init__(
        self,
        degree=3,
        max_points=100,
        ridge=0.1,
        total_steps=50,
        weight=0.3,
        taylor_order=1,
        normalized_time=False,
    ):
        self.cheb = _ChebyshevForecaster(
            degree,
            max_points,
            ridge,
            total_steps,
            normalized_time=normalized_time,
        )
        self.weight = float(weight)
        self.taylor_order = int(max(1, min(3, taylor_order)))

    def ready(self, min_points=None):
        return self.cheb.ready(min_points=min_points)

    def update(self, step_index, value):
        self.cheb.update(step_index, value)

    def predict(self, step_index, weight=None, ridge=None):
        taylor = self.cheb.predict_taylor(step_index, self.taylor_order)
        if not self.ready():
            return taylor
        old_ridge = self.cheb.ridge
        if ridge is not None:
            self.cheb.ridge = float(ridge)
        try:
            cheb = self.cheb.predict_chebyshev(step_index)
        finally:
            self.cheb.ridge = old_ridge
        blend_weight = self.weight if weight is None else float(weight)
        return ((1.0 - blend_weight) * taylor + blend_weight * cheb).astype(self.cheb.dtype)



def sample_latents_spectrum(
    mlx_model,
    positive,
    negative,
    width,
    height,
    seed,
    steps,
    cfg,
    scheduler,
    sampler_name,
    force_no_cfg,
    preview=False,
    compute_dtype="float16",
    speed_patch=core.SPEED_PATCH_NONE,
    speed_patch_strength=1.0,
    initial_latents=None,
    noise_mask=None,
    denoise=1.0,
    sdxl_time_ids=None,
    differential_mask=False,
    differential_mask_strength=1.0,
    preview_mode="crop",
    preview_crop_info=None,
    spectrum_mode="off",
    manual_forecast_steps="",
    spectrum_weight=0.3,
    spectrum_degree=3,
    spectrum_ridge=0.1,
    spectrum_window_size=2,
    spectrum_flex_window=0.25,
    spectrum_warmup_steps=6,
    spectrum_limiter_max_intervention=1.0,
    spectrum_limiter_mode="fast",
    spectrum_schedule_mode="sdmlx",
    spectrum_time_base=50,
    spectrum_dynamic_max_w=0.0,
    spectrum_dynamic_w_bias=0.0,
    spectrum_min_points=0,
    spectrum_time_coord_mode="step",
    stop_caching_step=-1,
    spectrum_metrics=None,
    spectrum_verbose=True,
    spectrum_progress=True,
):
    from .mlx_sd.config import DiffusionConfig
    from .mlx_sd.sampler import SimpleEulerAncestralSampler, SimpleEulerSampler

    core.configure_mlx_memory_limits()
    if width % 64 != 0 or height % 64 != 0:
        raise ValueError(
            f"SDMLX requires width and height to be divisible by 64, got {width}x{height}. "
            "Use a preset size or choose Custom dimensions such as 768, 1024, or 1344."
        )

    start_time = time.perf_counter()
    loras = mlx_model.get("loras", [])
    static_loras, scheduled_loras = core.split_loras_by_schedule(loras)
    ip_adapters = list(mlx_model.get("ip_adapters", []))
    controlnets = core.collect_conditioning_controlnets(mlx_model, positive, negative)
    speed_patch_name = core.normalized_speed_patch_name(speed_patch)
    fast_mode = True
    effective_fast_ffn = fast_mode
    effective_fast_attention = fast_mode and not core.SDMLX_DISABLE_FAST_ATTENTION
    if fast_mode and not effective_fast_attention and spectrum_verbose:
        print("SDMLX Spectrum: Fast Attention disabled by environment.")
    if scheduled_loras and spectrum_verbose:
        print(
            "SDMLX Spectrum: Scheduled LoRA active: Fast FFN/Fast Attention are disabled "
            "so dynamic LoRA modules remain controllable."
        )
        effective_fast_ffn = False
        effective_fast_attention = False

    unet = core.get_unet_model(
        mlx_model["cache_key"],
        mlx_model["weights"],
        False,
        8,
        64,
        fast_mode,
        effective_fast_ffn,
        effective_fast_attention,
        compute_dtype,
        speed_patch,
        speed_patch_strength,
        static_loras,
    )
    scheduled_lora_stats = core.prepare_scheduled_loras_for_unet(
        unet,
        scheduled_loras,
        core.precision_dtype(compute_dtype),
    )
    speed_patch_strength_key = round(float(speed_patch_strength), 6) if speed_patch_name else 0.0
    mx.random.seed(seed)
    sampler_cls = SimpleEulerAncestralSampler if sampler_name == "euler_ancestral" else SimpleEulerSampler
    sampler = sampler_cls(
        DiffusionConfig(
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012,
            num_train_steps=1000,
        )
    )

    latent_height = height // 8
    latent_width = width // 8
    dtype = core.precision_dtype(compute_dtype)
    initial_latents = initial_latents.astype(mx.float32) if initial_latents is not None else None
    if initial_latents is not None and tuple(initial_latents.shape[1:3]) != (latent_height, latent_width):
        raise ValueError(
            "SDMLX: initial_latents do not match the target size: "
            f"{tuple(initial_latents.shape)} vs {(latent_height, latent_width)}."
        )

    if sdxl_time_ids is None:
        time_id_values = [float(height), float(width), 0.0, 0.0, float(height), float(width)]
        time_id_source = "default"
    else:
        time_id_values = [float(value) for value in sdxl_time_ids]
        if len(time_id_values) != 6:
            raise ValueError(f"SDMLX: sdxl_time_ids must contain 6 values, got {time_id_values}.")
        time_id_source = "crop"
    time_id = mx.array([time_id_values])
    use_cfg = cfg > 1.0 and not force_no_cfg
    if use_cfg:
        context = mx.concatenate([positive["cond"], negative["cond"]], axis=0)
        pooled = mx.concatenate([positive["pooled"], negative["pooled"]], axis=0)
        t_ids = mx.concatenate([time_id] * 2, axis=0)
    else:
        context = positive["cond"]
        pooled = positive["pooled"]
        t_ids = time_id

    context = context.astype(dtype)
    pooled = pooled.astype(dtype)
    t_ids = t_ids.astype(dtype)
    cfg_value = mx.array(cfg, dtype=mx.float32)
    denoise = max(0.0, min(1.0, float(denoise)))
    step_plan = core.scheduler_step_plan_for_denoise(sampler, steps, scheduler, sampler_name, denoise)
    progress_steps = len(step_plan)
    if spectrum_mode == "manual":
        forecast_steps = _parse_forecast_steps(manual_forecast_steps, progress_steps)
    else:
        forecast_steps = set()
    forecast_label = ",".join(str(step + 1) for step in sorted(forecast_steps)) or "none"
    stop_caching_step = int(stop_caching_step)
    if stop_caching_step == -1:
        stop_at_step = int(progress_steps * 0.8)
    elif stop_caching_step > 0:
        stop_at_step = stop_caching_step
    else:
        stop_at_step = progress_steps + 1000
    spectrum_window_size = max(1.0, float(spectrum_window_size))
    spectrum_flex_window = max(0.0, float(spectrum_flex_window))
    spectrum_warmup_steps = max(0, int(spectrum_warmup_steps))
    spectrum_limiter_max_intervention = max(0.0, min(1.0, float(spectrum_limiter_max_intervention)))
    spectrum_limiter_mode = str(spectrum_limiter_mode or "fast").lower().strip()
    spectrum_schedule_mode = str(spectrum_schedule_mode or "sdmlx").lower().strip()
    spectrum_time_base = max(1.0, float(spectrum_time_base))
    spectrum_dynamic_max_w = max(0.0, min(1.0, float(spectrum_dynamic_max_w)))
    spectrum_dynamic_w_bias = max(-1.0, min(1.0, float(spectrum_dynamic_w_bias)))
    spectrum_min_points = max(0, int(spectrum_min_points))
    spectrum_time_coord_mode = str(spectrum_time_coord_mode or "step").lower().strip()
    if spectrum_time_coord_mode not in ("step", "sigma", "log_sigma"):
        spectrum_time_coord_mode = "step"

    dynamic_step_context = bool(ip_adapters or scheduled_loras)
    control_active_by_step = [
        core.controlnets_active_at_percent(controlnets, i / max(progress_steps - 1, 1))
        for i in range(progress_steps)
    ] if controlnets else []
    if controlnets:
        controlnets = core.prepare_controlnets_for_sampling(
            controlnets,
            width,
            height,
            2 if use_cfg else 1,
            dtype,
        )
    mx.eval(context, pooled, t_ids)
    ip_adapter_kv_layers = 0
    if ip_adapters:
        patched = core.ensure_unet_ipadapter_wrapped(unet)
        if patched and spectrum_verbose:
            core.log_timing(f"SDMLX Spectrum: IP-Adapter cross-attention enabled ({patched} Transformer-Blocks).")
        ip_adapters, ip_adapter_kv_layers = core.prepare_ipadapter_kv_cache(
            ip_adapters,
            2 if use_cfg else 1,
            dtype,
            use_cfg,
        )
    if controlnets:
        for control in controlnets:
            core.get_controlnet_union_model(
                control["controlnet"],
                fast_transformer=fast_mode,
                fast_ffn=fast_mode,
                fast_attention=effective_fast_attention,
            )

    if spectrum_verbose:
        print(
            "SDMLX Spectrum: Sampling "
            f"({steps} Steps, manual_plan={forecast_label}, mode={spectrum_mode}, "
            f"w={float(spectrum_weight):g}, m={int(spectrum_degree)}, lam={float(spectrum_ridge):g}, "
            f"window={spectrum_window_size:g}, flex={spectrum_flex_window:g}, "
            f"limiter_max={spectrum_limiter_max_intervention:g}, "
            f"schedule={spectrum_schedule_mode}, time_base={spectrum_time_base:g}, "
            f"dynamic_w={spectrum_dynamic_max_w:g}, "
            f"warmup={spectrum_warmup_steps}, stop_caching={stop_caching_step}, stop_at={stop_at_step}, "
            f"{width}x{height}, CFG {'on' if use_cfg else 'off'} ({float(cfg):g}), "
            f"Scheduler {scheduler}, Sampler {sampler_name}, DType {compute_dtype})..."
        )

    if denoise == 0.0 and initial_latents is not None:
        if spectrum_verbose:
            print("SDMLX Spectrum: denoise=0, initial latents are returned unchanged.")
        mx.eval(initial_latents)
        return initial_latents

    if initial_latents is not None:
        init_noise = mx.random.normal(initial_latents.shape).astype(mx.float32)
        start_sigma = step_plan[0][6]
        latents = core.noise_latents_at_sigma(initial_latents, init_noise, start_sigma)
    else:
        latents = sampler.sample_prior((1, latent_height, latent_width, 4), dtype=mx.float32)

    mask = None
    if noise_mask is not None:
        mask = noise_mask.astype(mx.float32)
        if len(mask.shape) == 3:
            mask = mask[..., None]
        if mask.shape[1] != latent_height or mask.shape[2] != latent_width:
            raise ValueError(
                "SDMLX: noise_mask does not match the latent size: "
                f"{tuple(mask.shape)} vs {(latent_height, latent_width)}."
            )
        if initial_latents is None:
            raise ValueError("SDMLX: noise_mask requires initial_latents.")
        differential_mask_strength = max(0.0, min(1.0, float(differential_mask_strength)))
        start_t = step_plan[0][0]
        end_t = mx.array(0.0, dtype=mx.float32)
        differential_denom = mx.maximum(start_t - end_t, mx.array(1e-6, dtype=mx.float32))

        def active_mask_for_t(timestep):
            if not differential_mask:
                return mask
            threshold = mx.clip((timestep - end_t) / differential_denom, 0.0, 1.0).astype(mx.float32)
            threshold = mx.maximum(threshold, mx.array(1e-6, dtype=mx.float32))
            binary_mask = (mask >= threshold).astype(mx.float32)
            if differential_mask_strength < 1.0:
                strength = mx.array(differential_mask_strength, dtype=mx.float32)
                return strength * binary_mask + (1.0 - strength) * mask
            return binary_mask

        initial_mask = active_mask_for_t(step_plan[0][0])
        preserved = core.noise_latents_at_sigma(initial_latents, init_noise, step_plan[0][6])
        latents = latents * initial_mask + preserved * (1.0 - initial_mask)

    pbar = core.make_comfy_progress_bar(progress_steps) if spectrum_progress else None
    terminal_pbar = (
        core.make_terminal_progress_bar(
            progress_steps,
            description="SDMLX Spectrum Sampling" if spectrum_verbose else "SDMLX Sampling",
        )
        if spectrum_progress
        else None
    )
    previewer, preview_device = core.get_sdxl_system_previewer() if preview else (None, None)

    def run_unet_feature(latents_in, timestep, step_percent, step_has_control):
        if dynamic_step_context:
            if ip_adapters:
                core.SDMLX_IPADAPTER_CONTEXT["adapters"] = ip_adapters
                core.SDMLX_IPADAPTER_CONTEXT["step_percent"] = step_percent
                core.SDMLX_IPADAPTER_CONTEXT["use_cfg"] = use_cfg
            if scheduled_loras:
                core.SDMLX_LORA_CONTEXT["step_percent"] = step_percent

        x_in = mx.concatenate([latents_in] * 2) if use_cfg else latents_in
        x_model = x_in.astype(dtype) if compute_dtype == "float16" else x_in
        t_in = mx.broadcast_to(timestep, [len(x_in)])
        control_down, control_mid = (
            core.controlnet_residuals_for_step(
                controlnets,
                x_model,
                t_in,
                context,
                pooled,
                t_ids,
                width,
                height,
                step_percent,
                dtype,
                fast_mode,
                fast_mode,
                fast_mode,
            )
            if step_has_control
            else (None, None)
        )

        temb = unet.timesteps(t_in).astype(x_model.dtype)
        temb = unet.time_embedding(temb)
        time_emb = unet.add_time_proj(t_ids).flatten(1).astype(x_model.dtype)
        time_emb = mx.concatenate([pooled, time_emb], axis=-1)
        time_emb = unet.add_embedding(time_emb)
        temb = temb + time_emb

        x = unet.conv_in(x_model)
        residuals = [x]
        for block in unet.down_blocks:
            x, res = block(
                x,
                encoder_x=context,
                temb=temb,
                attn_mask=None,
                encoder_attn_mask=None,
            )
            residuals.extend(res)

        if control_down is not None:
            residuals = [
                residual + control.astype(residual.dtype)
                for residual, control in zip(residuals, control_down)
            ]

        x = unet.mid_blocks[0](x, temb)
        x = unet.mid_blocks[1](x, context, None, None)
        x = unet.mid_blocks[2](x, temb)
        if control_mid is not None:
            x = x + control_mid.astype(x.dtype)

        for block in unet.up_blocks:
            x, _ = block(
                x,
                encoder_x=context,
                temb=temb,
                attn_mask=None,
                encoder_attn_mask=None,
                residual_hidden_states=residuals,
            )
        return x.astype(mx.float32)

    def finish_unet_feature(feature):
        x = feature.astype(dtype) if compute_dtype == "float16" else feature
        x = unet.conv_norm_out(x)
        x = core.nn.silu(x)
        return unet.conv_out(x).astype(mx.float32)

    def run_unet_raw(latents_in, timestep, step_percent, step_has_control):
        return finish_unet_feature(run_unet_feature(latents_in, timestep, step_percent, step_has_control))

    def raw_to_noise_pred(raw):
        if use_cfg:
            eps_pos, eps_neg = mx.split(raw, 2)
            return eps_neg + cfg_value * (eps_pos - eps_neg)
        return raw

    old_dpmpp_denoised = None
    old_dpmpp_sigma = None
    forecaster = None
    num_cached = 0
    current_window = float(spectrum_window_size)
    forecast_count = 0
    force_real_count = 0
    forecast_block_candidates = 0
    forecast_block_force_real = 0
    adaptive_risks = []
    adaptive_rescue_suggestions = 0
    adaptive_limiter_enabled = spectrum_mode in ("adaptive_limiter", "adaptive_limiter_window", "repo") and spectrum_limiter_max_intervention > 0.0
    adaptive_force_real_enabled = spectrum_mode == "adaptive_limiter_window"
    adaptive_window_enabled = spectrum_mode == "adaptive_window"
    adaptive_shadow_enabled = (
        spectrum_mode in ("adaptive_shadow", "adaptive_limiter", "adaptive_window", "adaptive_limiter_window", "repo")
        and (spectrum_verbose or spectrum_metrics is not None or adaptive_window_enabled)
    )
    adaptive_limiter_full = (
        spectrum_limiter_mode == "full"
        or adaptive_force_real_enabled
        or spectrum_metrics is not None
    )
    adaptive_limiter_profile = (
        _adaptive_limiter_profile(progress_steps, spectrum_limiter_max_intervention)
        if adaptive_limiter_enabled
        else (9999.0, 9999.0, 0.0)
    )
    adaptive_window_cooldown = 0
    adaptive_window_scale = 1.0
    adaptive_window_cap = None
    spectrum_time_coords, spectrum_normalized_time = _spectrum_time_coords_from_step_plan(
        step_plan,
        spectrum_time_coord_mode,
    )
    try:
        for i, (t, next_t, step_scale, step_dt, step_noise_scale, step_out_scale, step_sigma, next_sigma) in enumerate(step_plan):
            current_time_coord = spectrum_time_coords[i] if i < len(spectrum_time_coords) else float(i)
            min_points_for_ready = spectrum_min_points if spectrum_min_points > 0 else None
            step_percent = i / max(progress_steps - 1, 1)
            step_has_control = control_active_by_step[i] if control_active_by_step else False
            if mask is not None:
                step_mask = active_mask_for_t(t)
                preserved = core.noise_latents_at_sigma(initial_latents, init_noise, step_sigma)
                latents = latents * step_mask + preserved * (1.0 - step_mask)

            if forecaster is None:
                forecaster = _SpectrumForecaster(
                    degree=spectrum_degree,
                    max_points=100,
                    ridge=spectrum_ridge,
                    total_steps=spectrum_time_base,
                    weight=spectrum_weight,
                    normalized_time=spectrum_normalized_time,
                )

            final_guard_active = i >= stop_at_step
            if spectrum_mode == "manual":
                ready = forecaster.ready(min_points=min_points_for_ready or 2)
                can_forecast = bool(i in forecast_steps and ready and not step_has_control and not final_guard_active)
            else:
                ready = (
                    forecaster.ready(min_points=min_points_for_ready or 2)
                    if spectrum_schedule_mode == "repo"
                    else forecaster.ready(min_points=min_points_for_ready)
                )
                interval = max(1, math.floor(current_window))
                can_forecast = bool(
                    i >= spectrum_warmup_steps
                    and ready
                    and not step_has_control
                    and not final_guard_active
                    and ((num_cached + 1) % interval) != 0
                )
            if can_forecast:
                forecast_block_candidates += 1
                dynamic_weight = (
                    _repo_dynamic_weight(
                        current_window,
                        progress_steps,
                        i,
                        max_w=spectrum_dynamic_max_w,
                        max_ws=float(spectrum_window_size) + float(spectrum_flex_window) * 20.0,
                        bias=spectrum_dynamic_w_bias,
                    )
                    if spectrum_dynamic_max_w > 0.0
                    else None
                )
                feature = forecaster.predict(current_time_coord, weight=dynamic_weight)
                limiter_scale = None
                limiter_ratio = None
                limiter_pre_ratio_value = 0.0
                limiter_push_value = 0.0
                limiter_dynamic_weight = float(spectrum_weight)
                limiter_dynamic_ridge = float(spectrum_ridge)
                force_real_forecast_gate = False
                if adaptive_limiter_enabled:
                    limiter_soft_ratio, limiter_hard_ratio, limiter_knee_strength = adaptive_limiter_profile
                    if adaptive_limiter_full:
                        pre_ratio = _forecast_delta_ratio(
                            feature,
                            forecaster.cheb.values,
                            soft_ratio=limiter_soft_ratio,
                        )
                        mx.eval(pre_ratio)
                        limiter_pre_ratio_value = core.mx_scalar_float(pre_ratio)
                        block_force_real_allowed = (
                            (forecast_block_force_real + 1)
                            / max(1.0, float(forecast_block_candidates + 1))
                        ) <= SPECTRUM_FORCE_REAL_BLOCK_RATIO
                        force_real_forecast_gate = (
                            adaptive_force_real_enabled
                            and block_force_real_allowed
                            and limiter_pre_ratio_value >= SPECTRUM_FORCE_REAL_PRE_RATIO
                        )
                        (
                            limiter_dynamic_weight,
                            limiter_dynamic_ridge,
                            limiter_push_value,
                        ) = _adaptive_limiter_parameter_push(
                            limiter_pre_ratio_value,
                            float(spectrum_weight),
                            float(spectrum_ridge),
                            spectrum_limiter_max_intervention,
                        )
                        if force_real_forecast_gate:
                            if spectrum_metrics is not None:
                                spectrum_metrics.setdefault("limiter_rows", []).append({
                                    "step": i + 1,
                                    "total": progress_steps,
                                    "scale": 0.0,
                                    "ratio": limiter_pre_ratio_value,
                                    "soft_ratio": limiter_soft_ratio,
                                    "hard_ratio": limiter_hard_ratio,
                                    "knee_strength": limiter_knee_strength,
                                    "pre_ratio": limiter_pre_ratio_value,
                                    "push": limiter_push_value,
                                    "dynamic_weight": limiter_dynamic_weight,
                                    "dynamic_ridge": limiter_dynamic_ridge,
                                    "force_real": True,
                                    "force_real_count": force_real_count + 1,
                                    "forecast_block_candidates": forecast_block_candidates,
                                    "forecast_block_force_real": forecast_block_force_real + 1,
                                    "force_real_block_ratio": SPECTRUM_FORCE_REAL_BLOCK_RATIO,
                                })
                        elif limiter_push_value > 0.0:
                            feature = forecaster.predict(
                                current_time_coord,
                                weight=limiter_dynamic_weight,
                                ridge=limiter_dynamic_ridge,
                            )
                    if not force_real_forecast_gate:
                        feature, limiter_scale, limiter_ratio = _limit_forecast_feature_delta(
                            feature,
                            forecaster.cheb.values,
                            soft_ratio=limiter_soft_ratio,
                            hard_ratio=limiter_hard_ratio,
                            knee_strength=limiter_knee_strength,
                        )
                        if spectrum_metrics is not None or spectrum_verbose:
                            mx.eval(limiter_scale, limiter_ratio)
                            scale_value = core.mx_scalar_float(limiter_scale)
                            ratio_value = core.mx_scalar_float(limiter_ratio)
                        if spectrum_metrics is not None:
                            spectrum_metrics.setdefault("limiter_rows", []).append({
                                "step": i + 1,
                                "total": progress_steps,
                                "scale": scale_value,
                                "ratio": ratio_value,
                                "soft_ratio": limiter_soft_ratio,
                                "hard_ratio": limiter_hard_ratio,
                                "knee_strength": limiter_knee_strength,
                                "pre_ratio": limiter_pre_ratio_value,
                                "push": limiter_push_value,
                                "dynamic_weight": limiter_dynamic_weight,
                                "dynamic_ridge": limiter_dynamic_ridge,
                                "force_real": False,
                            })
                        if spectrum_verbose and scale_value < 0.999:
                            print(
                                "SDMLX Adaptive Limiter: "
                                f"step {i + 1}/{progress_steps} forecast delta scale={scale_value:.4f}, "
                                f"ratio={ratio_value:.4f}, pre_ratio={limiter_pre_ratio_value:.4f}, "
                                f"push={limiter_push_value:.3f}, "
                                f"w={limiter_dynamic_weight:.3f}, ridge={limiter_dynamic_ridge:.3f}, "
                                f"soft={limiter_soft_ratio:.2f}, hard={limiter_hard_ratio:.2f}, "
                                f"knee={limiter_knee_strength:.2f}."
                            )
                            print(
                                "SDMLX Adaptive Limiter CSV: "
                                f"step={i + 1},total={progress_steps},"
                                f"scale={scale_value:.6f},ratio={ratio_value:.6f},"
                                f"pre_ratio={limiter_pre_ratio_value:.6f},"
                                f"push={limiter_push_value:.6f},"
                                f"dynamic_weight={limiter_dynamic_weight:.6f},"
                                f"dynamic_ridge={limiter_dynamic_ridge:.6f},"
                                f"soft_ratio={limiter_soft_ratio:.6f},hard_ratio={limiter_hard_ratio:.6f},"
                                f"knee_strength={limiter_knee_strength:.6f},"
                                f"ridge={float(spectrum_ridge):g},weight={float(spectrum_weight):g},"
                                f"flex={float(spectrum_flex_window):g}"
                            )
                if force_real_forecast_gate:
                    force_real_count += 1
                    forecast_block_force_real += 1
                    feature = run_unet_feature(latents, t, step_percent, step_has_control)
                    mx.eval(feature)
                    forecaster.update(current_time_coord, feature)
                    num_cached = 0
                    can_forecast = False
                    raw = finish_unet_feature(feature)
                    mx.eval(raw)
                    noise_pred = raw_to_noise_pred(raw)
                    mx.eval(noise_pred)
                else:
                    raw = finish_unet_feature(feature)
                    mx.eval(raw)
                    noise_pred = raw_to_noise_pred(raw)
                    num_cached += 1
                    forecast_count += 1
            else:
                shadow_prediction = None
                if adaptive_shadow_enabled and forecaster.ready(min_points=min_points_for_ready):
                    shadow_prediction = forecaster.predict(current_time_coord)
                feature = run_unet_feature(latents, t, step_percent, step_has_control)
                mx.eval(feature)
                if shadow_prediction is not None:
                    rel_error = _feature_relative_rms(shadow_prediction - feature, feature)
                    velocity = mx.array(0.0, dtype=mx.float32)
                    curvature = mx.array(0.0, dtype=mx.float32)
                    if forecaster.cheb.values:
                        prev = forecaster.cheb.values[-1].reshape(feature.shape)
                        velocity = _feature_relative_rms(feature - prev, feature)
                    if len(forecaster.cheb.values) >= 2:
                        prev = forecaster.cheb.values[-1].reshape(feature.shape)
                        prev2 = forecaster.cheb.values[-2].reshape(feature.shape)
                        curvature = _feature_relative_rms(feature - 2.0 * prev + prev2, feature)
                    cfg_factor = 1.0 + (max(float(cfg), 1.0) - 1.0) / 6.0
                    risk = rel_error * cfg_factor + 0.5 * curvature
                    mx.eval(rel_error, velocity, curvature, risk)
                    risk_value = core.mx_scalar_float(risk)
                    adaptive_risks.append(risk_value)
                    err_value = core.mx_scalar_float(rel_error)
                    vel_value = core.mx_scalar_float(velocity)
                    curve_value = core.mx_scalar_float(curvature)
                    action = "forecast-ok"
                    suggested_ridge = float(spectrum_ridge)
                    suggested_flex = float(spectrum_flex_window)
                    suggested_weight = float(spectrum_weight)
                    window_before = float(current_window)
                    window_after = float(current_window)
                    window_scale = 1.0
                    window_cooldown = 0
                    if risk_value >= 1.50:
                        action = "limit"
                        adaptive_rescue_suggestions += 1
                        suggested_ridge = min(0.65, max(suggested_ridge, float(spectrum_ridge) + 0.25))
                        suggested_flex = max(0.08, float(spectrum_flex_window) * 0.75)
                        window_scale = 0.08
                        window_cooldown = 16
                        window_after = max(1.0, window_before * 0.24)
                    elif risk_value >= 0.80:
                        action = "tighten"
                        suggested_ridge = min(0.55, max(suggested_ridge, float(spectrum_ridge) + 0.12))
                        suggested_flex = max(0.10, float(spectrum_flex_window) * 0.88)
                        window_scale = 0.18
                        window_cooldown = 12
                        window_after = max(1.0, window_before * 0.34)
                    elif risk_value >= 0.45:
                        action = "monitor"
                        suggested_ridge = min(0.35, max(suggested_ridge, float(spectrum_ridge) + 0.05))
                        if adaptive_window_enabled and step_percent >= 0.56 and risk_value >= 0.50:
                            action = "window-brake"
                            window_scale = 0.25
                            window_cooldown = 10
                            window_after = max(1.0, window_before * 0.45)
                        elif adaptive_window_enabled and step_percent >= 0.62 and risk_value >= 0.35:
                            action = "window-trim"
                            window_scale = 0.42
                            window_cooldown = 8
                            window_after = max(1.0, window_before * 0.62)
                    if adaptive_window_enabled and window_cooldown > 0:
                        current_window = min(current_window, window_after)
                        adaptive_window_cap = current_window
                        adaptive_window_scale = window_scale
                        adaptive_window_cooldown = max(adaptive_window_cooldown, window_cooldown)
                    row = {
                        "step": i + 1,
                        "total": progress_steps,
                        "cfg": float(cfg),
                        "degree": int(spectrum_degree),
                        "ridge": float(spectrum_ridge),
                        "weight": float(spectrum_weight),
                        "flex": float(spectrum_flex_window),
                        "risk": risk_value,
                        "err": err_value,
                        "vel": vel_value,
                        "curve": curve_value,
                        "cfg_factor": cfg_factor,
                        "action": action,
                        "suggested_ridge": suggested_ridge,
                        "suggested_weight": suggested_weight,
                        "suggested_flex": suggested_flex,
                        "window_before": window_before,
                        "window_after": float(current_window),
                        "window_scale": window_scale,
                        "window_cooldown": window_cooldown,
                    }
                    if spectrum_metrics is not None:
                        spectrum_metrics.setdefault("adaptive_rows", []).append(row)
                    if spectrum_verbose:
                        print(
                            "SDMLX Adaptive Spectrum: "
                            f"step {i + 1}/{progress_steps} shadow risk={risk_value:.5f}, "
                            f"err={err_value:.5f}, "
                            f"vel={vel_value:.5f}, "
                            f"curve={curve_value:.5f}, "
                            f"cfg_factor={cfg_factor:.3f}, action={action}, "
                            f"window={window_before:.2f}->{float(current_window):.2f}."
                        )
                        print(
                            "SDMLX Adaptive Spectrum CSV: "
                            f"step={i + 1},total={progress_steps},cfg={float(cfg):g},"
                            f"degree={int(spectrum_degree)},ridge={float(spectrum_ridge):g},"
                            f"weight={float(spectrum_weight):g},flex={float(spectrum_flex_window):g},"
                            f"risk={risk_value:.6f},err={err_value:.6f},vel={vel_value:.6f},"
                            f"curve={curve_value:.6f},cfg_factor={cfg_factor:.6f},action={action},"
                            f"suggested_ridge={suggested_ridge:.6f},"
                            f"suggested_weight={suggested_weight:.6f},"
                            f"suggested_flex={suggested_flex:.6f},"
                            f"window_before={window_before:.6f},"
                            f"window_after={float(current_window):.6f},"
                            f"window_scale={window_scale:.6f},"
                            f"window_cooldown={window_cooldown}"
                        )
                forecaster.update(current_time_coord, feature)
                num_cached = 0
                forecast_block_candidates = 0
                forecast_block_force_real = 0
                raw = finish_unet_feature(feature)
                mx.eval(raw)
                noise_pred = raw_to_noise_pred(raw)
                mx.eval(noise_pred)

            denoised_latents = None
            if mask is not None or preview or sampler_name == "dpmpp_2m":
                denoised_latents = core.denoised_latents_estimate(latents, noise_pred, step_scale, step_sigma)
            if mask is not None:
                denoised_latents = denoised_latents * step_mask + initial_latents * (1.0 - step_mask)
                noise_pred = core.noise_pred_from_denoised(latents, denoised_latents, step_scale, step_sigma)

            preview_latents = denoised_latents if preview else None

            if sampler_name == "heun":
                def denoise_next(trial_latents):
                    next_noise_pred = raw_to_noise_pred(run_unet_raw(trial_latents, next_t, step_percent, step_has_control))
                    if mask is not None:
                        next_scale = mx.sqrt(next_sigma * next_sigma + 1.0)
                        next_denoised = core.denoised_latents_estimate(trial_latents, next_noise_pred, next_scale, next_sigma)
                        next_denoised = next_denoised * step_mask + initial_latents * (1.0 - step_mask)
                        next_noise_pred = core.noise_pred_from_denoised(trial_latents, next_denoised, next_scale, next_sigma)
                    return next_noise_pred

                latents = core.heun_sampler_step(latents, noise_pred, step_sigma, next_sigma, denoise_next)
            elif sampler_name == "dpmpp_2m":
                latents = core.dpmpp_2m_sampler_step(
                    latents,
                    denoised_latents,
                    old_dpmpp_denoised,
                    step_sigma,
                    next_sigma,
                    old_dpmpp_sigma,
                )
                old_dpmpp_denoised = denoised_latents
                old_dpmpp_sigma = core.mx_scalar_float(step_sigma)
            else:
                latents = core.apply_sampler_step(noise_pred, latents, step_scale, step_dt, step_noise_scale, step_out_scale)

            if mask is not None:
                step_mask = active_mask_for_t(next_t)
                preserved = core.noise_latents_at_sigma(initial_latents, init_noise, next_sigma)
                latents = latents * step_mask + preserved * (1.0 - step_mask)

            mx.eval(latents)
            preview_bytes = (
                core.decode_system_preview_bytes(
                    preview_latents,
                    previewer,
                    preview_device,
                    preview_mode,
                    preview_crop_info,
                )
                if preview
                else None
            )
            if pbar is not None:
                pbar.update_absolute(i + 1, progress_steps, preview_bytes)
            if terminal_pbar is not None:
                terminal_pbar.update(1)
            marker = "forecast" if can_forecast else "real"
            if spectrum_verbose:
                core.log_timing(f"SDMLX Spectrum: step {i + 1}/{progress_steps} {marker}.")
            if i >= spectrum_warmup_steps:
                flex_increment = spectrum_flex_window
                if adaptive_window_enabled and adaptive_window_cooldown > 0:
                    flex_increment *= adaptive_window_scale
                    adaptive_window_cooldown -= 1
                if spectrum_schedule_mode == "repo":
                    if ready and not can_forecast and not step_has_control and not final_guard_active:
                        current_window += flex_increment
                else:
                    current_window += flex_increment
                if adaptive_window_enabled and adaptive_window_cap is not None:
                    current_window = min(current_window, adaptive_window_cap)
                    if adaptive_window_cooldown <= 0:
                        adaptive_window_cap = None
                        adaptive_window_scale = 1.0
    finally:
        if dynamic_step_context:
            if ip_adapters:
                core.SDMLX_IPADAPTER_CONTEXT["adapters"] = []
                core.SDMLX_IPADAPTER_CONTEXT["step_percent"] = 0.0
                core.SDMLX_IPADAPTER_CONTEXT["use_cfg"] = False
            if scheduled_loras:
                core.SDMLX_LORA_CONTEXT["step_percent"] = 0.0
        if terminal_pbar is not None:
            terminal_pbar.close()

    mx.eval(latents)
    core.release_mlx_cache_memory_after_sampling()
    elapsed = time.perf_counter() - start_time
    if spectrum_metrics is not None:
        rows = spectrum_metrics.get("adaptive_rows", [])
        spectrum_metrics.update({
            "elapsed": elapsed,
            "forecast_count": forecast_count,
            "force_real_count": force_real_count,
            "force_real_block_ratio": SPECTRUM_FORCE_REAL_BLOCK_RATIO if adaptive_force_real_enabled else 0.0,
            "forecast_label": forecast_label,
            "progress_steps": progress_steps,
            "max_risk": max((row["risk"] for row in rows), default=0.0),
            "avg_risk": sum(row["risk"] for row in rows) / len(rows) if rows else 0.0,
            "force_real_suggestions": adaptive_rescue_suggestions,
            "adaptive_row_count": len(rows),
        })
    if spectrum_verbose:
        print(
            f"SDMLX Spectrum: Sampling finished in {elapsed:.2f}s "
            f"(forecasted_steps={forecast_count}, plan={forecast_label}, "
            f"scheduled_lora_modules={scheduled_lora_stats.get('modules', 0)}, "
            f"ip_adapter_kv_layers={ip_adapter_kv_layers}, time_ids={time_id_source})."
        )
    else:
        print(
            f"SDMLX Spectrum: Sampling finished in {elapsed:.2f}s "
            f"(forecasted_steps={forecast_count})."
        )
    if adaptive_risks:
        if spectrum_verbose:
            print(
                "SDMLX Adaptive Spectrum: summary "
                f"max_risk={max(adaptive_risks):.5f}, "
                f"avg_risk={sum(adaptive_risks) / len(adaptive_risks):.5f}, "
                f"force_real_suggestions={adaptive_rescue_suggestions}/{len(adaptive_risks)}."
            )
    return latents
