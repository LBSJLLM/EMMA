from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_TOKEN_LOG_LOCK = threading.Lock()


def _record_token_usage(resp: Any, *, model: str, call_name: str) -> None:
    if "deepseek" not in str(model or "").lower():
        return
    log_path = os.getenv("EMMA_TOKEN_LOG", "").strip()
    if not log_path:
        return
    usage = getattr(resp, "usage", None)
    if usage is None:
        return

    def _usage_value(name: str) -> int:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        try:
            return int(value or 0)
        except Exception:
            return 0

    row = {
        "ts": time.time(),
        "phase": os.getenv("EMMA_TOKEN_PHASE", "qa"),
        "component": "kqa",
        "call_name": call_name,
        "model": model,
        "input_tokens": _usage_value("prompt_tokens"),
        "output_tokens": _usage_value("completion_tokens"),
        "total_tokens": _usage_value("total_tokens"),
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _TOKEN_LOG_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _infer_call_name(system_prompt: str) -> str:
    text = str(system_prompt or "").lower()
    if "single-agent" in text or "single-agent multi-round" in text:
        return "single_agent"
    if "query agent" in text:
        return "query_agent"
    if "validation agent" in text:
        return "validation_agent"
    if "answer agent" in text:
        return "answer_agent"
    if "question" in text and "type" in text:
        return "question_type_classifier"
    return "chat"


class APILLMClient:
    _shared_lock = threading.Lock()
    _rate_locks: Dict[str, threading.Lock] = {}
    _last_call_ts: Dict[str, float] = {}
    _semaphores: Dict[str, threading.BoundedSemaphore] = {}

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        rate_limit_qps: float = 0.0,
        max_concurrency: int = 0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        request_timeout_seconds: float = 0.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("QA_OPENAI_KEY") or ""
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("QA_OPENAI_BASE_URL")
        self.temperature = temperature
        self.rate_limit_qps = float(rate_limit_qps or 0.0)
        self.max_concurrency = int(max_concurrency or 0)
        self.max_retries = max(0, int(max_retries or 0))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds or 0.0))
        self.request_timeout_seconds = max(0.0, float(request_timeout_seconds or 0.0))
        self.client = None
        if self.api_key:
            from openai import OpenAI

            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = OpenAI(**kwargs)
        self._scope_key = f"{self.base_url or 'default'}::{self.model}"

    def _get_rate_lock(self) -> threading.Lock:
        with self._shared_lock:
            lk = self._rate_locks.get(self._scope_key)
            if lk is None:
                lk = threading.Lock()
                self._rate_locks[self._scope_key] = lk
            return lk

    def _get_semaphore(self) -> Optional[threading.BoundedSemaphore]:
        if self.max_concurrency <= 0:
            return None
        with self._shared_lock:
            sem = self._semaphores.get(self._scope_key)
            if sem is None:
                sem = threading.BoundedSemaphore(self.max_concurrency)
                self._semaphores[self._scope_key] = sem
            return sem

    def _throttle(self) -> None:
        if self.rate_limit_qps <= 0:
            return
        min_interval = 1.0 / self.rate_limit_qps
        lock = self._get_rate_lock()
        with lock:
            now = time.monotonic()
            last = self._last_call_ts.get(self._scope_key, 0.0)
            wait_s = (last + min_interval) - now
            if wait_s > 0:
                time.sleep(wait_s)
            self._last_call_ts[self._scope_key] = time.monotonic()

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float:
        # Best-effort parse for Retry-After style hints from SDK exception objects.
        try:
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", None)
            if headers and "Retry-After" in headers:
                return max(0.0, float(headers.get("Retry-After")))
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and status in {408, 409, 429, 500, 502, 503, 504}:
            return True
        msg = str(exc).lower()
        hints = [
            "rate limit",
            "too many requests",
            "429",
            "timeout",
            "timed out",
            "connection",
            "temporarily unavailable",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "overloaded",
        ]
        return any(h in msg for h in hints)

    def available(self) -> bool:
        return self.client is not None

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        if self.client is None:
            return ""
        sem = self._get_semaphore()
        if sem is not None:
            sem.acquire()
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    self._throttle()
                    kwargs: Dict[str, Any] = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "max_tokens": max_tokens,
                    }
                    if self.temperature is not None:
                        kwargs["temperature"] = self.temperature
                    if self.request_timeout_seconds > 0:
                        kwargs["timeout"] = self.request_timeout_seconds
                    resp = self.client.chat.completions.create(**kwargs)
                    _record_token_usage(resp, model=self.model, call_name=_infer_call_name(system_prompt))
                    return (resp.choices[0].message.content or "").strip()
                except Exception as exc:
                    if attempt >= self.max_retries or not self._is_retryable(exc):
                        raise
                    retry_after = self._retry_after_seconds(exc)
                    base = self.retry_backoff_seconds if self.retry_backoff_seconds > 0 else 1.0
                    backoff = retry_after if retry_after > 0 else base * (2 ** attempt)
                    jitter = 1.0 + random.random() * 0.15
                    time.sleep(backoff * jitter)
            return ""
        finally:
            if sem is not None:
                sem.release()

    @staticmethod
    def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        for m in re.finditer(r"\{[\s\S]*\}", text):
            frag = m.group(0)
            try:
                obj = json.loads(frag)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return None
