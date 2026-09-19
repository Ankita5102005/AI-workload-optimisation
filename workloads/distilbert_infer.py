"""DistilBERT inference workload (NLP).

A small slice of SST-2 is tokenised once (fixed padding length, so every batch has the
same shape) and moved to the GPU; the same tokens are reused for every configuration so
the input is never a confounding variable. FP16 casts the model to half precision (the
FP32 model is kept untouched and deep-copied for FP16); token ids stay integer tensors.
"""

import copy

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


class DistilBertWorkload:
    name = "distilbert"

    def __init__(self, device: str = "cuda", model_name: str = MODEL_NAME,
                 max_length: int = 128, pool_size: int = 2048):
        self.device = device
        ds = load_dataset("glue", "sst2", split=f"train[:{pool_size}]")
        tok = AutoTokenizer.from_pretrained(model_name)
        enc = tok(list(ds["sentence"]), padding="max_length", truncation=True,
                  max_length=max_length, return_tensors="pt")
        self.input_ids = enc["input_ids"].to(device)
        self.attention_mask = enc["attention_mask"].to(device)
        self.base = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(device)
        self.model = None

    def prepare(self, precision: str, batch_size: int):
        n = len(self.input_ids)
        if batch_size > n:
            raise ValueError(f"batch_size {batch_size} exceeds input pool size {n}")
        if precision == "fp16":
            self.model = copy.deepcopy(self.base).half()
        elif precision == "fp32":
            self.model = self.base
        else:
            raise ValueError(f"unsupported precision {precision!r}")

        model, ids, mask = self.model, self.input_ids, self.attention_mask
        n_windows = n // batch_size

        @torch.inference_mode()
        def step(i: int):
            s = (i % n_windows) * batch_size
            model(input_ids=ids[s:s + batch_size], attention_mask=mask[s:s + batch_size])

        return step

    def synchronize(self):
        torch.cuda.synchronize()

    def release(self):
        self.model = None
        torch.cuda.empty_cache()
