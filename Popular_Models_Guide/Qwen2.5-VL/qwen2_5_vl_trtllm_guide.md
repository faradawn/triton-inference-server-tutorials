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

This guide walks through serving a multimodal (vision-language) model on Triton
Inference Server using the
[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) PyTorch backend through
the [LLM API](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llm-api/README.md),
exposed by Triton's `llmapi` backend.

Unlike the deprecated multimodal path, there is **no engine build**: no
`trtllm-build`, no separate visual engine. The model repository is four small
files and the weights load straight from a Hugging Face snapshot at startup.

> [!NOTE]
> This guide replaces
> [the Llava1.5 TensorRT-LLM guide](../Llava1.5/llava_trtllm_guide.md), which
> uses the prebuilt-TensorRT-engine multimodal path that TensorRT-LLM declared
> end-of-life in v1.2. See
> [triton-inference-server/server#8945](https://github.com/triton-inference-server/server/issues/8945).

This guide was tested with `Qwen/Qwen2.5-VL-3B-Instruct` on 1x NVIDIA B200,
using `nvcr.io/nvidia/tritonserver:26.07-trtllm-python-py3`
(Triton 2.71.0, TensorRT-LLM 1.2.1).

## Files provided with this guide

The `llmapi` backend in TensorRT-LLM v1.2.1 does not accept image input yet;
that is proposed in
[NVIDIA/TensorRT-LLM#18381](https://github.com/NVIDIA/TensorRT-LLM/pull/18381).
Two files here add it on top of v1.2.1, which is the exact TensorRT-LLM version
installed in the container:

1. [model.py](./model.py) - v1.2.1's backend plus the optional `image_url` input.
2. [config.pbtxt](./config.pbtxt) - v1.2.1's config declaring `image_url`.

The other two files in the model repository come from v1.2.1 unchanged.

## Launch Triton TensorRT-LLM container

Start from a clone of this repository, so the two files above are available
inside the container:

```bash
git clone https://github.com/triton-inference-server/tutorials.git
cd tutorials

docker run --rm -it --net host --shm-size=2g \
  --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
  -v ${PWD}:/tutorials \
  -w /workspace \
  nvcr.io/nvidia/tritonserver:26.07-trtllm-python-py3
```

## Update the `openai` package

The container ships `openai 1.107.3`, which is too old for
`tensorrt_llm.serve`. Because the PyTorch executor imports that module
unconditionally, **no** model loads on the `llmapi` backend until this is fixed:

```bash
pip install --target=/workspace/pylibs -U openai
export PYTHONPATH=/workspace/pylibs
```

## Build the model repository

Fetch the backend files from the TensorRT-LLM v1.2.1 tag. Only one directory is
needed, so skip the Git LFS payload — this takes a few seconds:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --sparse \
    --branch v1.2.1 https://github.com/NVIDIA/TensorRT-LLM.git /workspace/trtllm
git -C /workspace/trtllm sparse-checkout set triton_backend/all_models/llmapi

cp -r /workspace/trtllm/triton_backend/all_models/llmapi /workspace/model_repository
```

Copy in the two files provided with this guide:

```bash
QWEN_DIR=/tutorials/Popular_Models_Guide/Qwen2.5-VL
cp ${QWEN_DIR}/model.py     /workspace/model_repository/tensorrt_llm/1/model.py
cp ${QWEN_DIR}/config.pbtxt /workspace/model_repository/tensorrt_llm/config.pbtxt
```

Then write `model.yaml`, which selects the model and turns on image input:

```bash
cat > /workspace/model_repository/tensorrt_llm/1/model.yaml <<'EOF'
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

The repository now looks like this:

```
model_repository/
└── tensorrt_llm/
    ├── config.pbtxt        <- provided here
    └── 1/
        ├── model.py        <- provided here
        ├── helpers.py      <- from v1.2.1
        └── model.yaml      <- written above
```

`model` takes a Hugging Face model id (downloaded to `HF_HOME`) or a local
snapshot directory.

`triton_config.multimodal` defaults to `False`. When it is not set, `image_url`
values are ignored and you get a text-only answer, so set it for multimodal
models.

## Serving with Triton

```bash
trtllm-llmapi-launch tritonserver \
  --model-repository=/workspace/model_repository \
  --http-port=8000 --grpc-port=8001 --metrics-port=8002
```

`trtllm-llmapi-launch` is required: the LLM API spawns its workers with
`MpiPoolSession`, and plain `tritonserver` fails with `MPI_ERR_SPAWN`.

Startup takes about a minute. The server is ready when the log shows:

```
[trtllm] multimodal input enabled for model_type 'qwen2_5_vl'
Started HTTPService at 0.0.0.0:8000
```

You can also poll for readiness:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/v2/health/ready
```

## Send an inference request

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

To send more than one image, pass more entries and match the shape:

```bash
{"name":"image_url","shape":[2],"datatype":"BYTES","data":["http://images.cocodataset.org/test2017/000000155781.jpg","http://images.cocodataset.org/val2017/000000039769.jpg"]}
```

The model describes the images in the order they were sent.

## Request inputs

| Input | Datatype | Shape | Description |
| ----- | -------- | ----- | ----------- |
| `text_input` | `BYTES` | `[1]` | The question. The backend applies the chat template and inserts the image placeholders, so do not add `<\|vision_start\|>` or similar tokens yourself. |
| `image_url` | `BYTES` | `[N]` | One entry per image. Only `http(s)` URLs are accepted. |
| `sampling_param_max_tokens` | `INT32` | `[1]` | Maximum tokens to generate. |
| `sampling_param_exclude_input_from_output` | `BOOL` | `[1]` | Set `true`, otherwise the rendered prompt is echoed back. |

The only output is `text_output`.

`image_url` is client-controlled, so only `http(s)` URLs are accepted. Local
paths and `file://` are rejected, because accepting them would let a caller make
the server read image files its process can open. Host images the model should
see on a reachable web URL.

## Troubleshooting

| Symptom | Cause and fix |
| ------- | ------------- |
| `ImportError: cannot import name 'PartReasoningText'` | The container's `openai` is too old. See [Update the `openai` package](#update-the-openai-package). |
| `mpi4py.MPI.Exception: MPI_ERR_SPAWN` | `tritonserver` was started directly; use `trtllm-llmapi-launch`. |
| `cannot import name 'async_build_multimodal_prompt'` | `model.py` was taken from the #18381 branch rather than the copy provided here. That branch targets TensorRT-LLM 1.3 and calls a function the 1.2.1 wheel does not have. |
| Answers ignore the image | `triton_config.multimodal` is not `True` in `model.yaml`. |
| `Unsupported image_url '...'` | A local path or non-`http(s)` scheme was passed. |
| `ConnectionRefusedError` from the client | The server is still starting; wait for `Started HTTPService`. |

## References

- [TensorRT-LLM LLM API](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llm-api/README.md)
- [TensorRT-LLM supported models](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md)
- [NVIDIA/TensorRT-LLM#18381](https://github.com/NVIDIA/TensorRT-LLM/pull/18381) - adds multimodal input to the Triton `llmapi` backend
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
