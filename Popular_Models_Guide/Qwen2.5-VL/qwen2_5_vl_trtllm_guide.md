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
> **This workflow depends on an unmerged TensorRT-LLM change, plus a small
> patch to run it on today's container.**
>
> Image support in the Triton `llmapi` backend (the optional `image_url` input
> and the `triton_config.multimodal` opt-in used below) is added by
> [NVIDIA/TensorRT-LLM#18381](https://github.com/NVIDIA/TensorRT-LLM/pull/18381),
> which has not been merged and is not present in any released TensorRT-LLM
> version or container image. Until that PR lands you must take the `llmapi`
> backend files from that branch; a stock container will not accept an
> `image_url` input.
>
> Those files call `async_build_multimodal_prompt`, which the same PR adds to
> the `tensorrt_llm` **wheel**. Every published `-trtllm-python-py3` image still
> ships TensorRT-LLM 1.2.1, whose wheel does not have it, so
> [a one-command patch](#patching-modelpy-for-tensorrt-llm-121) is required as
> well. This guide is written for that combination and is verified end to end on
> it; both steps go away only when a container ships a TensorRT-LLM that already
> contains #18381.

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
| TensorRT-LLM | 1.2.1 (with the [`model.py` patch](#patching-modelpy-for-tensorrt-llm-121)) |
| torch | 2.10.0a0+b4e4ee81d3.nv25.12 |
| CUDA | 13.1 |
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Hardware | 1x NVIDIA B200 |

Every command and every response below was run on that configuration. `26.07`
is the newest `-trtllm-python-py3` tag; on it, the multimodal path does not work
without the patch.

## Prerequisites

### Container

Start from a clone of this repository, so that the
[`trtllm_121_compat.py`](trtllm_121_compat.py) used below is mounted into the
container along with it:

```bash
git clone https://github.com/triton-inference-server/tutorials.git
cd tutorials

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

The Triton backend sources live in the
[NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) repository under
`triton_backend/`; the standalone `tensorrtllm_backend` repository has been
superseded. They are plain Python files and are not shipped in the
`tensorrt_llm` wheel, so fetch them from a checkout.

`triton_backend/all_models/llmapi/` contains exactly one model directory
(`tensorrt_llm/`), so it doubles as a Triton model repository and needs no
copying:

```
all_models/llmapi/          <- point --model-repository here
└── tensorrt_llm/
    ├── config.pbtxt
    └── 1/
        ├── model.py
        ├── helpers.py
        └── model.yaml
```

> [!NOTE]
> Until [NVIDIA/TensorRT-LLM#18381](https://github.com/NVIDIA/TensorRT-LLM/pull/18381)
> merges, the `image_url` input and the `triton_config.multimodal` option below
> exist only on that pull request's branch. Clone the fork shown here for now;
> once it lands, clone `https://github.com/NVIDIA/TensorRT-LLM.git` instead.

Only four files are needed, so skip the repository's Git LFS payload and check
out the one directory — a few seconds and about 9 MB, rather than the ~900 MB a
full clone pulls:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --sparse \
    --branch feat/triton-llmapi-multimodal-image \
    https://github.com/faradawn/TensorRT-LLM.git /workspace/trtllm-pr
git -C /workspace/trtllm-pr sparse-checkout set triton_backend/all_models/llmapi
```

Then point `1/model.yaml` at the model and turn on the multimodal opt-in. This
edits the file in place inside the checkout, which leaves that clone's
`git status` dirty — fine for a throwaway container:

```bash
cat > /workspace/trtllm-pr/triton_backend/all_models/llmapi/tensorrt_llm/1/model.yaml <<'EOF'
model: Qwen/Qwen2.5-VL-3B-Instruct
backend: "pytorch"
tensor_parallel_size: 1
kv_cache_config:
  free_gpu_memory_fraction: 0.5

triton_config:
  max_batch_size: 0
  decoupled: False
  multimodal: True
EOF
```

`model` accepts a Hugging Face model id (downloaded to `HF_HOME`) or a local
snapshot directory.

`triton_config.multimodal` defaults to `False`. This is deliberate: existing
deployments that already declare their own `image_url` input keep their current
behavior when they upgrade. The flip side is that if you forget to set it, any
`image_url` values you send are **silently ignored** and you get a text-only
answer, so set it explicitly for multimodal models.

## Patching `model.py` for TensorRT-LLM 1.2.1

The backend files you just cloned build their prompt by calling
`async_build_multimodal_prompt`, which
[#18381](https://github.com/NVIDIA/TensorRT-LLM/pull/18381) adds to
`tensorrt_llm/inputs/utils.py`. That module ships **inside the `tensorrt_llm`
wheel**, not in the `triton_backend/` tree you cloned, and the container's wheel
is 1.2.1 — so you have the caller but never the callee.

This is easy to miss, because nothing fails at startup. The server comes up, logs
`multimodal input enabled`, and answers text-only prompts correctly. Only
requests that actually carry an image fail:

```json
{"error":"Error generating request: cannot import name 'async_build_multimodal_prompt' from 'tensorrt_llm.inputs' (/opt/venv-tritonserver/lib/python3.12/site-packages/tensorrt_llm/inputs/__init__.py)"}
```

Copying the new `utils.py` across does not help either: 1.2.1 lacks everything
that helper is built on — `MEDIA_IO_REGISTRY`, `ContentFormat`,
`MultimodalDataTracker.item_order()`, `interleave_mm_placeholders` and
`async_apply_chat_template`. What does work is replacing that one call with an
equivalent written against the 1.2.1 API. Run the script shipped next to this
guide:

```bash
python3 /workspace/Popular_Models_Guide/Qwen2.5-VL/trtllm_121_compat.py \
    /workspace/trtllm-pr/triton_backend/all_models/llmapi/tensorrt_llm/1/model.py
```

```
Patched .../llmapi/tensorrt_llm/1/model.py for TensorRT-LLM 1.2.1.
```

It edits nothing but that one file, refuses to write source that does not parse,
and is safe to re-run — a second invocation reports `already patched; nothing to
do`. If the call it looks for is gone, it says so and tells you how to check
whether your container already has the function, rather than corrupting the
model repository.

### What the script changes

It adds one method, `_build_multimodal_prompt_121`, and points the call site at
it:

```diff
             image_url = get_input_tensor_by_name(request, 'image_url')
             if image_url is not None and image_url.size > 0:
-                from tensorrt_llm.inputs import async_build_multimodal_prompt
-
                 media = [
                     url.decode("utf-8") if isinstance(url, bytes) else str(url)
                     for url in image_url.reshape(-1)
                 ]
                 validate_media_urls(media)
-                prompt = await async_build_multimodal_prompt(
-                    model_type=self._mm_model_type,
-                    tokenizer=self._mm_tokenizer,
-                    processor=self._mm_processor,
-                    prompt=prompt,
-                    media=media,
-                    modality="image",
-                )
+                prompt = await self._build_multimodal_prompt_121(prompt, media)
```

The new method does what the 1.3 helper does, in 1.2.1's vocabulary:

| Step | 1.3 helper | 1.2.1 equivalent used here |
| ---- | ---------- | -------------------------- |
| download images | `MEDIA_IO_REGISTRY` | `async_load_image` per URL, gathered |
| insert placeholders | `interleave_mm_placeholders`, `item_order()` | `add_multimodal_placeholders`, three-argument form |
| render chat template | `async_apply_chat_template` | `apply_chat_template` via `asyncio.to_thread` |
| build the prompt | returns `PromptInputs` | `prompt_inputs(...)` plus `multi_modal_data` |

`apply_chat_template` is synchronous and does real tokenizer work, so it goes
through `asyncio.to_thread` rather than blocking the engine's event loop while
the images are still downloading. Read
[`trtllm_121_compat.py`](trtllm_121_compat.py) for the full method.

Nothing else in the backend needs touching: `validate_media_urls` and the rest
of the request path run unmodified on 1.2.1.

> [!NOTE]
> Delete this step once a `-trtllm-python-py3` container ships a TensorRT-LLM
> that already contains #18381. Note that #18381 merging is **not** enough on its
> own — the 26.07 image's wheel stays at 1.2.1 no matter what lands upstream, so
> the patch is needed until a *new image* ships.

## Starting the server

In Slurm/MPI environments, launch through `trtllm-llmapi-launch`:

```bash
trtllm-llmapi-launch tritonserver \
  --model-repository=/workspace/trtllm-pr/triton_backend/all_models/llmapi \
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
| `image_url` | `BYTES` | `[N]` | One entry per image. Accepts `http(s)` URLs the server can reach; local paths and other schemes are rejected. |
| `sampling_param_max_tokens` | `INT32` | `[1]` | Maximum number of tokens to generate. |
| `sampling_param_exclude_input_from_output` | `BOOL` | `[1]` | Set to `true`; otherwise the rendered prompt is echoed back in `text_output`. |

The only output is `text_output`.

### Quick check with `curl`

```bash
curl -s http://localhost:8000/v2/models/tensorrt_llm/infer -H 'Content-Type: application/json' -d '{
  "inputs": [
    {"name":"text_input","shape":[1],"datatype":"BYTES","data":["What color is the bus and what does the sign say?"]},
    {"name":"image_url","shape":[1],"datatype":"BYTES","data":["http://images.cocodataset.org/test2017/000000155781.jpg"]},
    {"name":"sampling_param_max_tokens","shape":[1],"datatype":"INT32","data":[64]},
    {"name":"sampling_param_exclude_input_from_output","shape":[1],"datatype":"BOOL","data":[true]}
  ],
  "outputs": [{"name":"text_output"}]
}'
```

```json
{"model_name":"tensorrt_llm","model_version":"1","outputs":[{"name":"text_output","datatype":"BYTES","shape":[1],"data":["The bus is yellow and white, and the sign on the bus says \"Out of Service.\""]}]}
```

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
entries. Every entry must be an `http(s)` URL — see
[Allowed scope of access](#allowed-scope-of-access):

```python
ask(
    "Describe each image.",
    [
        "http://images.cocodataset.org/test2017/000000155781.jpg",
        "http://images.cocodataset.org/val2017/000000039769.jpg",
    ],
    max_tokens=96,
)
```

The model enumerates both images and describes each one in the order they were
sent:

```
The first image depicts a bus on a foggy street at night. The bus has a sign on
its front that reads "OUT OF SERVICE." ... The second image shows two cats lying
on a pink couch.
```

### Allowed scope of access

`image_url` is client-controlled, so only `http(s)` URLs are accepted. Local
filesystem paths, `file://` and other schemes are rejected, because accepting
them would let a caller make the server read image files its process can open.
Host images the model should see on a reachable web URL.

A rejected entry fails the whole request:

```json
{"error":"Error generating request: Unsupported image_url '/workspace/images/second.jpg': only http, https URLs are accepted."}
```

### Error behavior

An unreachable image URL surfaces as a Triton error rather than silently
degrading to a text-only answer:

```json
{"error":"Error generating request: Cannot connect to host example.invalid:443 ssl:default [Name or service not known]"}
```

## Troubleshooting

| Symptom | Cause and fix |
| ------- | ------------- |
| `mpi4py.MPI.Exception: MPI_ERR_SPAWN: could not spawn processes` | `tritonserver` was started directly. The LLM API spawns workers via `MpiPoolSession`; start it with `trtllm-llmapi-launch` instead. |
| `ImportError: cannot import name 'PartReasoningText'` | The container's `openai` package is too old for `tensorrt_llm.serve`, which is imported unconditionally by the PyTorch executor. Install a newer `openai` into an overlay directory and export it on `PYTHONPATH` (see [Prerequisites](#known-issue-the-containers-openai-package-is-too-old)). |
| `ConnectionRefusedError` from the client | The server is not up yet. Startup takes roughly 70 seconds; wait for `Started HTTPService` in the log, or poll until `curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/v2/health/ready` returns `200`. |
| Images appear to be ignored and answers are text-only | `triton_config.multimodal` is not set to `True` in `1/model.yaml`. It defaults to `False` and image inputs are silently dropped. |
| `cannot import name 'async_build_multimodal_prompt' from 'tensorrt_llm.inputs'`, only on requests carrying an image | The container's TensorRT-LLM wheel predates [#18381](https://github.com/NVIDIA/TensorRT-LLM/pull/18381). The server starts and text-only requests still work, which makes this easy to miss. Apply [the 1.2.1 patch](#patching-modelpy-for-tensorrt-llm-121). |
| `Unsupported image_url '...': only http, https URLs are accepted.` | A local path, `file://` or other scheme was passed. Only `http(s)` is accepted; see [Allowed scope of access](#allowed-scope-of-access). |

## References

- [TensorRT-LLM LLM API](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llm-api/README.md)
- [TensorRT-LLM supported models](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md)
- [NVIDIA/TensorRT-LLM#18381](https://github.com/NVIDIA/TensorRT-LLM/pull/18381) - adds multimodal input to the Triton `llmapi` backend
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
