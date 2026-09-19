"""ResNet-18 inference workload (vision).

Inputs are synthetic random tensors, generated once on the GPU and reused for every
configuration, so data loading and host-to-device copies never enter the
measurement. Precision is toggled for real: FP16 casts both the model and the inputs
to half precision (the FP32 model is kept untouched and deep-copied for FP16).
"""

import copy

import torch
import torchvision


class ResNet18Workload:
    name = "resnet18"

    def __init__(self, device: str = "cuda", pool_size: int = 2048, image_size: int = 224,
                 seed: int = 0, pretrained: bool = True):
        self.device = device
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.base = torchvision.models.resnet18(weights=weights).eval().to(device)
        gen = torch.Generator(device=device).manual_seed(seed)
        self.pool = torch.randn(pool_size, 3, image_size, image_size, device=device, generator=gen)
        torch.backends.cudnn.benchmark = True
        self.model = None
        self.inputs = None

    def prepare(self, precision: str, batch_size: int):
        if batch_size > len(self.pool):
            raise ValueError(f"batch_size {batch_size} exceeds input pool size {len(self.pool)}")
        if precision == "fp16":
            self.model = copy.deepcopy(self.base).half()
            self.inputs = self.pool.half()
        elif precision == "fp32":
            self.model = self.base
            self.inputs = self.pool
        else:
            raise ValueError(f"unsupported precision {precision!r}")

        model, inputs = self.model, self.inputs
        n_windows = len(inputs) // batch_size

        @torch.inference_mode()
        def step(i: int):
            s = (i % n_windows) * batch_size
            model(inputs[s:s + batch_size])

        return step

    def synchronize(self):
        torch.cuda.synchronize()

    def release(self):
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()
