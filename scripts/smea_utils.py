from importlib import import_module
import torch

sampling = None
BACKEND = None
INITIALIZED = False

if not BACKEND:
    try:
        _ = import_module("modules.sd_samplers_kdiffusion")
        sampling = import_module("k_diffusion.sampling")
        BACKEND = "WebUI"
    except ImportError:
        pass

if not BACKEND:
    try:
        sampling = import_module("comfy.k_diffusion.sampling")
        BACKEND = "ComfyUI"
    except ImportError:
        pass


class _Rescaler:
    def __init__(self, model, x, mode, **extra_args):
        self.model = model
        self.x = x
        self.mode = mode
        self.extra_args = extra_args
        if BACKEND == "WebUI":
            self.init_latent, self.mask, self.nmask = model.init_latent, model.mask, model.nmask
        if BACKEND == "ComfyUI":
            self.latent_image, self.noise = model.latent_image, model.noise
            self.denoise_mask = self.extra_args.get("denoise_mask", None)

    def __enter__(self):
        if BACKEND == "WebUI":
            if self.init_latent is not None:
                self.model.init_latent = torch.nn.functional.interpolate(input=self.init_latent, size=self.x.shape[2:4], mode=self.mode)
            if self.mask is not None:
                self.model.mask = torch.nn.functional.interpolate(input=self.mask.unsqueeze(0), size=self.x.shape[2:4], mode=self.mode).squeeze(0)
            if self.nmask is not None:
                self.model.nmask = torch.nn.functional.interpolate(input=self.nmask.unsqueeze(0), size=self.x.shape[2:4], mode=self.mode).squeeze(0)
        if BACKEND == "ComfyUI":
            if self.latent_image is not None:
                self.model.latent_image = torch.nn.functional.interpolate(input=self.latent_image, size=self.x.shape[2:4], mode=self.mode)
            if self.noise is not None:
                self.model.noise = torch.nn.functional.interpolate(input=self.latent_image, size=self.x.shape[2:4], mode=self.mode)
            if self.denoise_mask is not None:
                self.extra_args["denoise_mask"] = torch.nn.functional.interpolate(input=self.denoise_mask, size=self.x.shape[2:4], mode=self.mode)

        return self

    def __exit__(self, type, value, traceback):
        if BACKEND == "WebUI":
            del self.model.init_latent, self.model.mask, self.model.nmask
            self.model.init_latent, self.model.mask, self.model.nmask = self.init_latent, self.mask, self.nmask
        if BACKEND == "ComfyUI":
            del self.model.latent_image, self.model.noise
            self.model.latent_image, self.model.noise = self.latent_image, self.noise
