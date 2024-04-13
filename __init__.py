from . import smea_sampling
from .smea_sampling import sample_euler_dy, sample_euler_smea_dy

if smea_sampling.BACKEND == "ComfyUI":
    if not smea_sampling.INITIALIZED:
        from comfy.k_diffusion import sampling as k_diffusion_sampling
        from comfy.samplers import SAMPLER_NAMES

        setattr(k_diffusion_sampling, "sample_euler_dy", sample_euler_dy)
        setattr(k_diffusion_sampling, "sample_euler_smea_dy", sample_euler_smea_dy)

        SAMPLER_NAMES.append("euler_dy")
        SAMPLER_NAMES.append("euler_smea_dy")

        smea_sampling.INITIALIZED = True

NODE_CLASS_MAPPINGS = {}
