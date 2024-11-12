from . import smea_sampling
from .smea_sampling import sample_euler_dy, sample_euler_smea_dy, sample_euler_negative, sample_euler_dy_negative, sample_Kohaku_LoNyu_Yog

if smea_sampling.BACKEND == "ComfyUI":
    if not smea_sampling.INITIALIZED:
        from comfy.k_diffusion import sampling as k_diffusion_sampling
        from comfy.samplers import SAMPLER_NAMES

        setattr(k_diffusion_sampling, "sample_euler_dy", sample_euler_dy)
        setattr(k_diffusion_sampling, "sample_euler_smea_dy", sample_euler_smea_dy)
        setattr(k_diffusion_sampling, "sample_euler_negative", sample_euler_negative)
        setattr(k_diffusion_sampling, "sample_euler_dy_negative", sample_euler_dy_negative)
        setattr(k_diffusion_sampling, "sample_Kohaku_LoNyu_Yog", sample_Kohaku_LoNyu_Yog)

        SAMPLER_NAMES.append("euler_dy")
        SAMPLER_NAMES.append("euler_smea_dy")
        SAMPLER_NAMES.append("euler_negative")
        SAMPLER_NAMES.append("euler_dy_negative")
        SAMPLER_NAMES.append("Kohaku_LoNyu_Yog")

        smea_sampling.INITIALIZED = True

NODE_CLASS_MAPPINGS = {}
