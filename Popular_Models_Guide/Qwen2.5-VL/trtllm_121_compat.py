# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Make the Triton ``llmapi`` backend's image path run on TensorRT-LLM 1.2.1.

Used by ``qwen2_5_vl_trtllm_guide.md``. See the guide for the full explanation;
the short version is that the backend's ``model.py`` builds its prompt with
``tensorrt_llm.inputs.async_build_multimodal_prompt``, which is added by
NVIDIA/TensorRT-LLM#18381. That function lives in the ``tensorrt_llm`` *wheel*,
not in the ``triton_backend/`` tree you clone, so on a container whose wheel is
1.2.1 you have the caller but never the callee, and every request carrying an
``image_url`` fails with::

    cannot import name 'async_build_multimodal_prompt' from 'tensorrt_llm.inputs'

1.2.1 also lacks everything that helper is built on -- ``MEDIA_IO_REGISTRY``,
``ContentFormat``, ``MultimodalDataTracker.item_order()``,
``interleave_mm_placeholders`` and ``async_apply_chat_template`` -- so copying
the new ``utils.py`` across is not an option either. This script instead swaps
the single call for an equivalent written against the 1.2.1 API surface.

Usage::

    python3 trtllm_121_compat.py <clone>/triton_backend/all_models/llmapi/tensorrt_llm/1/model.py

Safe to re-run: it exits cleanly if the file is already patched. Delete this
step once a ``-trtllm-python-py3`` container ships a TensorRT-LLM that already
contains #18381.
"""

import argparse
import ast
import pathlib
import sys

# The call this replaces, exactly as it appears in model.py.
OLD_CALL = """                from tensorrt_llm.inputs import async_build_multimodal_prompt

                media = [
                    url.decode("utf-8") if isinstance(url, bytes) else str(url)
                    for url in image_url.reshape(-1)
                ]
                validate_media_urls(media)
                prompt = await async_build_multimodal_prompt(
                    model_type=self._mm_model_type,
                    tokenizer=self._mm_tokenizer,
                    processor=self._mm_processor,
                    prompt=prompt,
                    media=media,
                    modality="image",
                )"""

NEW_CALL = """                media = [
                    url.decode("utf-8") if isinstance(url, bytes) else str(url)
                    for url in image_url.reshape(-1)
                ]
                validate_media_urls(media)
                prompt = await self._build_multimodal_prompt_121(prompt, media)"""

# Inserted immediately above `async def _convert_request`. `asyncio` is already
# imported at module scope in model.py, so this needs no new top-level imports.
NEW_METHOD = '''    async def _build_multimodal_prompt_121(self, text, media):
        """Stand-in for `inputs.async_build_multimodal_prompt` on TRT-LLM 1.2.1.

        1.2.1 has no `async_apply_chat_template` and no
        `MultimodalDataTracker.item_order()`, and its
        `add_multimodal_placeholders` takes three arguments rather than four.
        """
        from tensorrt_llm.inputs import prompt_inputs
        from tensorrt_llm.inputs.utils import (ConversationMessage,
                                               MultimodalDataTracker,
                                               add_multimodal_placeholders,
                                               apply_chat_template,
                                               async_load_image)

        mm_data_tracker = MultimodalDataTracker(self._mm_model_type)
        for url in media:
            mm_data_tracker.add_data("image", async_load_image(url))
        mm_placeholder_counts = mm_data_tracker.placeholder_counts()

        content = add_multimodal_placeholders(self._mm_model_type, text,
                                              mm_placeholder_counts)
        conversation = [
            ConversationMessage(role="user", content=content, media=[])
        ]
        # `apply_chat_template` is synchronous and does real tokenizer work, so
        # keep it off the engine's event loop while the images download.
        prompt_task = asyncio.to_thread(
            apply_chat_template,
            model_type=self._mm_model_type,
            tokenizer=self._mm_tokenizer,
            processor=self._mm_processor,
            conversation=conversation,
            add_generation_prompt=True,
            mm_placeholder_counts=[mm_placeholder_counts],
        )
        prompt, (mm_data, _) = await asyncio.gather(
            prompt_task, mm_data_tracker.retrieve_all_async())

        prompt = prompt_inputs(prompt)
        if mm_data:
            prompt["multi_modal_data"] = mm_data
        return prompt

'''

ANCHOR = "    async def _convert_request(self, request):"

MOVED_ON = """{path} does not contain the call this script replaces.

That usually means the backend has moved on -- most likely #18381 merged, in
which case check whether your container's TensorRT-LLM already provides
`async_build_multimodal_prompt` and skip this step entirely:

    python3 -c "from tensorrt_llm.inputs import async_build_multimodal_prompt"

If that import succeeds, no patch is needed."""


def main():
    parser = argparse.ArgumentParser(
        description="Patch the Triton llmapi backend's model.py for "
        "TensorRT-LLM 1.2.1.")
    parser.add_argument(
        "model_py",
        type=pathlib.Path,
        help="path to all_models/llmapi/tensorrt_llm/1/model.py")
    args = parser.parse_args()

    path = args.model_py
    if not path.is_file():
        sys.exit(f"{path} is not a file")

    source = path.read_text()

    if "_build_multimodal_prompt_121" in source:
        print(f"{path} is already patched; nothing to do.")
        return

    if source.count(OLD_CALL) != 1:
        sys.exit(MOVED_ON.format(path=path))
    if source.count(ANCHOR) != 1:
        sys.exit(f"could not locate `{ANCHOR.strip()}` in {path}")

    source = source.replace(OLD_CALL, NEW_CALL)
    source = source.replace(ANCHOR, NEW_METHOD + ANCHOR, 1)

    # Fail before writing rather than leave a half-broken model repository.
    try:
        ast.parse(source)
    except SyntaxError as exc:
        sys.exit(f"patched source does not parse ({exc}); model.py left alone")

    path.write_text(source)
    print(f"Patched {path} for TensorRT-LLM 1.2.1.")


if __name__ == "__main__":
    main()
