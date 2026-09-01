<!--
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Deploying Hugging Face Qwen2.5-VL Model in Triton

This guide shows how to serve a multimodal (vision-language) model on Triton
Inference Server using the
[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) PyTorch backend through
the [LLM API](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llm-api/README.md),
exposed by Triton's `llmapi` backend.

> [!IMPORTANT]
> **This workflow depends on an unmerged TensorRT-LLM change.**
> Image support in the Triton `llmapi` backend (the optional `image_url` input
> and the `triton_config.multimodal` opt-in used below) is added by
> [NVIDIA/TensorRT-LLM#18381](https://github.com/NVIDIA/TensorRT-LLM/pull/18381),
> which has not been merged and is not present in any released TensorRT-LLM
> version or container image. Until that PR lands you must build the
> `llmapi` backend files from that branch; a stock container will not accept an
> `image_url` input.

> [!NOTE]
> This guide replaces
> [the Llava1.5 TensorRT-LLM guide](../Llava1.5/llava_trtllm_guide.md), which
> uses the prebuilt-TensorRT-engine multimodal path that TensorRT-LLM has
> declared end-of-life as of TensorRT-LLM v1.2. See
> [triton-inference-server/server#8945](https://github.com/triton-inference-server/server/issues/8945).

## Why the PyTorch backend

The deprecated multimodal path (`tensorrtllm_backend`'s `all_models/multimodal`)
required two ahead-of-time compilation steps before you could serve anything: a
`trtllm-build` invocation to produce the LLM engine, and a separate visual
engine build for the vision encoder. Both artifacts had to be rebuilt whenever
the model, precision, or maximum sequence length changed.

The PyTorch backend needs **no compilation and no engine build at all**. The
model repository is four plain Python/text files, TensorRT-LLM is a
pip-installed wheel inside the container, and the weights are loaded directly
from a Hugging Face snapshot at startup. This is the single biggest practical
difference between the two workflows.

LLaVA-1.5 itself is not a drop-in replacement target here. TensorRT-LLM's
[supported models matrix](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md)
lists `LlavaNextForConditionalGeneration` and `LlavaLlamaModel` (VILA) among the
supported multimodal architectures, but not `LlavaForConditionalGeneration`,
which is the architecture of `llava-hf/llava-1.5-7b-hf`. This guide therefore
uses [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).

## What was validated

| Item | Value |
| ---- | ----- |
| Container | `nvcr.io/nvidia/tritonserver:26.07-trtllm-python-py3` |
| Triton | 2.71.0 |
| TensorRT-LLM | 1.2.1 |
| CUDA | 13.1 |
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Hardware | 1x NVIDIA B200 |

## Prerequisites

### Container

```bash
docker run --rm -it --gpus all --network host \
  -v ${PWD}:/workspace -w /workspace \
  nvcr.io/nvidia/tritonserver:26.07-trtllm-python-py3
```

### Known issue: the container's `openai` package is too old

The 26.07 image ships `openai 1.107.3`, which is older than what
`tensorrt_llm/serve/responses_utils.py` requires. Loading a model fails with:

```
ImportError: cannot import name 'PartReasoningText'
```

Because `tensorrt_llm/_torch/pyexecutor/py_executor.py` imports
`tensorrt_llm.serve`, this breaks loading of **any** model on the `llmapi`
backend, not just multimodal ones. Work around it by installing a newer `openai`
into an overlay directory and putting that directory on `PYTHONPATH`, which
avoids modifying the container's site-packages:

```bash
pip install --target=/workspace/pylibs -U openai
export PYTHONPATH=/workspace/pylibs
```

### Model weights

Provide either a local Hugging Face snapshot directory or the Hugging Face model
id `Qwen/Qwen2.5-VL-3B-Instruct`. If you use the model id, the container needs
network access to huggingface.co at startup.

## Preparing the model repository

Copy the four `llmapi` backend files from TensorRT-LLM's
`triton_backend/all_models/llmapi/tensorrt_llm/` into a model repository:

```
model_repo/
└── tensorrt_llm/
    ├── config.pbtxt
    └── 1/
        ├── model.py
        ├── helpers.py
        └── model.yaml
```

Note that the TensorRT-LLM Triton backend sources now live in the
[NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) repository under
`triton_backend/`; the standalone `tensorrtllm_backend` repository has been
superseded.

Only `1/model.yaml` needs editing:

```yaml
model: /path/to/Qwen2.5-VL-3B-Instruct        # HF snapshot dir or HF model id
backend: "pytorch"
tensor_parallel_size: 1
kv_cache_config:
  free_gpu_memory_fraction: 0.5

triton_config:
  max_batch_size: 0
  decoupled: False
  multimodal: True      # opt-in; default False
```

`triton_config.multimodal` defaults to `False`. This is deliberate: existing
deployments that already declare their own `image_url` input keep their current
behavior when they upgrade. The flip side is that if you forget to set it, any
`image_url` values you send are **silently ignored** and you get a text-only
answer, so set it explicitly for multimodal models.

## Starting the server

In Slurm/MPI environments, launch through `trtllm-llmapi-launch`:

```bash
trtllm-llmapi-launch tritonserver --model-repository=/path/to/model_repo \
  --http-port=8000 --grpc-port=8001 --metrics-port=8002
```

Running plain `tritonserver` fails at engine start with:

```
mpi4py.MPI.Exception: MPI_ERR_SPAWN: could not spawn processes
```

The LLM API uses `MpiPoolSession` to spawn its workers, and
`trtllm-llmapi-launch` (which sets `TLLM_SPAWN_PROXY_PROCESS=1`) is the
supported wrapper for that.

Startup takes roughly 70 seconds. Wait for `Started HTTPService` in the log. A
successful multimodal start also logs:

```
[trtllm] multimodal input enabled for model_type 'qwen2_5_vl'
```

You can poll readiness with:

```bash
curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/v2/health/ready
```

which returns `200` once the server is up.

## Sending an inference request

Requests go to the standard Triton HTTP inference endpoint,
`POST /v2/models/tensorrt_llm/infer`. Inputs are Triton tensors, not OpenAI-style
chat JSON:

| Input | Datatype | Shape | Description |
| ----- | -------- | ----- | ----------- |
| `text_input` | `BYTES` | `[1]` | The plain question. The backend applies the chat template and inserts the per-architecture image placeholders, so do **not** add `<\|vision_start\|>` or similar tokens yourself. |
| `image_url` | `BYTES` | `[N]` | One entry per image. Accepts an `http(s)` URL, a local filesystem path readable by the server, or a `data:image/...;base64,...` URI. |
| `sampling_param_max_tokens` | `INT32` | `[1]` | Maximum number of tokens to generate. |
| `sampling_param_exclude_input_from_output` | `BOOL` | `[1]` | Set to `true`; otherwise the rendered prompt is echoed back in `text_output`. |

The only output is `text_output`.

### Python client

This client uses only the standard library:

```python
import json
import urllib.request

URL = "http://localhost:8000/v2/models/tensorrt_llm/infer"


def ask(prompt, images, max_tokens=64):
    """Send a prompt plus one or more images and return the generated text."""
    body = {
        "inputs": [
            {"name": "text_input", "shape": [1], "datatype": "BYTES",
             "data": [prompt]},
            {"name": "image_url", "shape": [len(images)], "datatype": "BYTES",
             "data": images},
            {"name": "sampling_param_max_tokens", "shape": [1],
             "datatype": "INT32", "data": [max_tokens]},
            {"name": "sampling_param_exclude_input_from_output", "shape": [1],
             "datatype": "BOOL", "data": [True]},
        ],
        "outputs": [{"name": "text_output"}],
    }
    request = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    return result["outputs"][0]["data"][0].strip()


if __name__ == "__main__":
    print(ask(
        "What color is the bus and what does the sign say?",
        ["http://images.cocodataset.org/test2017/000000155781.jpg"],
    ))
```

Expected output:

```
The bus is yellow and white, and the sign on the bus says "Out of Service."
```

### Multiple images

Pass more than one entry in `image_url`; the shape must match the number of
entries:

```python
ask(
    "Describe each image.",
    [
        "http://images.cocodataset.org/test2017/000000155781.jpg",
        "/workspace/images/second.jpg",
    ],
)
```

The model enumerates both images and describes each one in the order they were
sent.

### Image source equivalence

A `data:image/...;base64,...` URI and a local file path produce the same answer
as the `http` URL for the same image, so you can pick whichever form fits your
deployment. Local paths must be readable by the server process, not the client.

### Error behavior

An unreachable image URL surfaces as a Triton error rather than silently
degrading to a text-only answer, for example:

```
[trtllm] Error generating request: Cannot connect to host example.invalid:443
```

## Troubleshooting

| Symptom | Cause and fix |
| ------- | ------------- |
| `mpi4py.MPI.Exception: MPI_ERR_SPAWN: could not spawn processes` | `tritonserver` was started directly. The LLM API spawns workers via `MpiPoolSession`; start it with `trtllm-llmapi-launch` instead. |
| `ImportError: cannot import name 'PartReasoningText'` | The container's `openai` package is too old for `tensorrt_llm.serve`, which is imported unconditionally by the PyTorch executor. Install a newer `openai` into an overlay directory and export it on `PYTHONPATH` (see [Prerequisites](#known-issue-the-containers-openai-package-is-too-old)). |
| `ConnectionRefusedError` from the client | The server is not up yet. Startup takes roughly 70 seconds; wait for `Started HTTPService` in the log, or poll until `curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/v2/health/ready` returns `200`. |
| Images appear to be ignored and answers are text-only | `triton_config.multimodal` is not set to `True` in `1/model.yaml`. It defaults to `False` and image inputs are silently dropped. |

## References

- [TensorRT-LLM LLM API](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llm-api/README.md)
- [TensorRT-LLM supported models](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md)
- [NVIDIA/TensorRT-LLM#18381](https://github.com/NVIDIA/TensorRT-LLM/pull/18381) - adds multimodal input to the Triton `llmapi` backend
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
