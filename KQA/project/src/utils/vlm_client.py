from __future__ import annotations

import os
import sys
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class QwenVLMClient:
    def __init__(self, checkpoint: Optional[str] = None):
        self.checkpoint = checkpoint or os.getenv("VLM_CHECKPOINT") or "/root/models/Qwen3-VL-8B-Instruct"
        self.rpc_socket = str(os.getenv("VLM_RPC_SOCKET") or "").strip()
        self.http_url = str(os.getenv("VLM_HTTP_URL") or "").strip()
        self._llm = None
        self._processor = None
        self._ready = False

    def _http_available(self) -> bool:
        return bool(self.http_url)

    def _rpc_available(self) -> bool:
        return bool(self.rpc_socket)

    def _ensure_imports(self) -> bool:
        if self._ready:
            return True
        # Make the source checkout self-contained instead of assuming a private
        # /root/code/MMM installation path.
        repo_root = Path(__file__).resolve().parents[4]
        if str(repo_root) not in sys.path and repo_root.exists():
            sys.path.append(str(repo_root))
        try:
            from qwen_vl_utils import process_vision_info  # noqa: F401
            from transformers import AutoProcessor  # noqa: F401
            from vllm import LLM  # noqa: F401
        except Exception:
            return False
        self._ready = True
        return True

    def available(self) -> bool:
        if self._http_available():
            return True
        if self._rpc_available():
            return True
        return self._ensure_imports()

    def _ask_video_via_rpc(
        self,
        video_path: str,
        prompt: str,
        time_range: Optional[Tuple[float, float]] = None,
        fps: float = 1.0,
    ) -> str:
        if not self.rpc_socket:
            return ""
        conn = Client(self.rpc_socket, family="AF_UNIX")
        try:
            conn.send(
                {
                    "cmd": "ask_video",
                    "video_path": video_path,
                    "prompt": prompt,
                    "time_range": list(time_range) if time_range is not None else None,
                    "fps": float(fps),
                }
            )
            resp = conn.recv()
            if isinstance(resp, dict) and bool(resp.get("ok")):
                return str(resp.get("text") or "").strip()
            return ""
        except Exception:
            return ""
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _build_engine(self):
        from transformers import AutoProcessor
        from vllm import LLM

        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.checkpoint, local_files_only=True)
        if self._llm is None:
            self._llm = LLM(
                model=self.checkpoint,
                mm_encoder_tp_mode="data",
                enable_expert_parallel=False,
                tensor_parallel_size=max(1, int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))),
                gpu_memory_utilization=0.6,
                max_model_len=32768,
                seed=123,
            )

    def _ask_video_via_http(
        self,
        video_path: str,
        prompt: str,
        time_range: Optional[Tuple[float, float]] = None,
        fps: float = 1.0,
    ) -> str:
        import base64
        import json
        import urllib.request

        url = self.http_url.rstrip("/") + "/v1/chat/completions"
        # Build video content: encode as base64 data URI for vllm OpenAI-compatible API
        video_content: Dict[str, Any] = {
            "type": "video_url",
            "video_url": {"url": f"file://{video_path}"},
        }
        messages = [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        body = json.dumps({
            "model": os.getenv("VLM_MODEL_NAME", "qwen-vl"),
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 512,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return str(data["choices"][0]["message"]["content"]).strip()
        except Exception:
            return ""

    def ask_video(
        self,
        video_path: str,
        prompt: str,
        time_range: Optional[Tuple[float, float]] = None,
        fps: float = 1.0,
    ) -> str:
        if self._http_available():
            return self._ask_video_via_http(video_path=video_path, prompt=prompt, time_range=time_range, fps=fps)
        if self._rpc_available():
            return self._ask_video_via_rpc(video_path=video_path, prompt=prompt, time_range=time_range, fps=fps)

        if not self._ensure_imports():
            return ""

        from qwen_vl_utils import process_vision_info
        from vllm import SamplingParams

        self._build_engine()
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "fps": fps},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if time_range is not None:
            messages[0]["content"][0]["time_range"] = [float(time_range[0]), float(time_range[1])]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self._processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        mm_data: Dict[str, Any] = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        inputs = [{"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}]
        sp = SamplingParams(temperature=0.2, max_tokens=512, top_k=-1, stop_token_ids=[])
        outs = self._llm.generate(inputs, sampling_params=sp)
        return outs[0].outputs[0].text.strip()
