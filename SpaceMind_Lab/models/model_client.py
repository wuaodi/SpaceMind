#!/usr/bin/env python3
# 统一模型接入层：OpenAI 兼容 API，异步、图像、工具、API key 轮换。用法见 __main__。

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional
from zoneinfo import ZoneInfo

from openai import AsyncOpenAI

from config.paths import RUNTIME_TRACE_DIR

logger = logging.getLogger(__name__)
CHINA_TZ = ZoneInfo("Asia/Shanghai")


def format_exception_details(exc: Exception) -> str:
    """Format SDK exceptions into a readable one-line diagnostic string."""
    parts = [f"{type(exc).__name__}: {exc}"]

    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        parts.append(f"status_code={status_code}")

    request_id = getattr(exc, "request_id", None)
    if request_id:
        parts.append(f"request_id={request_id}")

    body = getattr(exc, "body", None)
    if body:
        parts.append(f"body={body}")

    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers:
            header_request_id = headers.get("x-request-id") or headers.get("request-id")
            if header_request_id and not request_id:
                parts.append(f"response_request_id={header_request_id}")

    return " | ".join(parts)


class ModelClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        api_key_labels: Optional[List[str]] = None,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o",
        max_tokens: int = 512,
        temperature: float = 0.2,
        tool_choice: Optional[str] = None,
        top_p: float = 0.2,
        default_extra_body: Optional[dict[str, Any]] = None,
    ):
        self._api_keys = api_keys if api_keys else ([api_key] if api_key else [])
        self._api_key_labels = api_key_labels if api_key_labels else []
        self._key_index = 0
        self.base_url = base_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.top_p = top_p
        self.default_extra_body = default_extra_body or {}
        self.last_error: Optional[str] = None
        self.trace_dir = RUNTIME_TRACE_DIR
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        self.trace_file = self.trace_dir / f"model_trace_{self.trace_run_id}.jsonl"
        self.trace_js_file = self.trace_dir / "latest_trace.js"
        self.trace_index_file = self.trace_dir / "latest_trace_path.txt"
        self._trace_entries: list[dict[str, Any]] = []
        self._usage_day = ""
        self.key_usage_json_file = self.trace_dir / "api_key_usage_latest.json"
        self.key_usage_text_file = self.trace_dir / "api_key_usage_latest.txt"
        self._key_usage_entries: list[dict[str, Any]] = []
        self._refresh_key_usage_storage()
        self._write_trace_bundle()

    def _current_key(self) -> str:
        if not self._api_keys:
            raise ValueError(
                "No API key configured. Set the model API key env var (for example METACHAT_API_KEY, "
                "or MODELSCOPE_API_KEY)."
            )
        return self._api_keys[self._key_index % len(self._api_keys)]

    def _now_cn(self) -> datetime:
        return datetime.now(CHINA_TZ)

    def _mask_key(self, key: str) -> str:
        if len(key) <= 10:
            return key
        return f"{key[:5]}...{key[-4:]}"

    def _key_label(self, index: int) -> str:
        if index < len(self._api_key_labels):
            label = str(self._api_key_labels[index]).strip()
            if label:
                return label
        return f"key{index + 1}"

    def _load_existing_usage_entries(self) -> list[dict[str, Any]]:
        if not self.key_usage_json_file.exists():
            return []
        try:
            payload = json.loads(self.key_usage_json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if str(payload.get("date")) != self._usage_day:
            return []
        entries = payload.get("entries")
        return entries if isinstance(entries, list) else []

    def _refresh_key_usage_storage(self):
        current_day = self._now_cn().strftime("%Y%m%d")
        if self._usage_day == current_day and self._key_usage_entries:
            return

        self._usage_day = current_day
        self.key_usage_json_file = self.trace_dir / f"api_key_usage_{current_day}.json"
        self.key_usage_text_file = self.trace_dir / f"api_key_usage_{current_day}.txt"
        existing_entries = self._load_existing_usage_entries()
        self._key_usage_entries = []

        for idx, key in enumerate(self._api_keys):
            existing = existing_entries[idx] if idx < len(existing_entries) and isinstance(existing_entries[idx], dict) else {}
            entry = {
                "index": idx,
                "label": self._key_label(idx),
                "masked_key": self._mask_key(key),
                "status": existing.get("status", "unused"),
                "attempts": int(existing.get("attempts", 0)),
                "successes": int(existing.get("successes", 0)),
                "rate_limit_errors": int(existing.get("rate_limit_errors", 0)),
                "quota_errors": int(existing.get("quota_errors", 0)),
                "permission_denied_errors": int(existing.get("permission_denied_errors", 0)),
                "auth_invalid_errors": int(existing.get("auth_invalid_errors", 0)),
                "other_errors": int(existing.get("other_errors", 0)),
                "first_used_at": existing.get("first_used_at"),
                "last_used_at": existing.get("last_used_at"),
                "last_success_at": existing.get("last_success_at"),
                "last_error": existing.get("last_error", ""),
            }
            self._key_usage_entries.append(entry)

        self._write_key_usage_files()

    def _write_key_usage_files(self):
        payload = {
            "date": self._usage_day,
            "timezone": "Asia/Shanghai",
            "model": self.model_name,
            "base_url": self.base_url,
            "entries": self._key_usage_entries,
        }
        self.key_usage_json_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = [
            f"date={self._usage_day} timezone=Asia/Shanghai model={self.model_name}",
            "",
            "label\tstatus\tattempts\tsuccesses\trate_limit\tquota\tpermission_denied\tauth_invalid\tother_errors\tmasked_key\tlast_used_at",
        ]
        for entry in self._key_usage_entries:
            lines.append(
                "\t".join(
                    [
                        str(entry["label"]),
                        str(entry["status"]),
                        str(entry["attempts"]),
                        str(entry["successes"]),
                        str(entry["rate_limit_errors"]),
                        str(entry["quota_errors"]),
                        str(entry["permission_denied_errors"]),
                        str(entry["auth_invalid_errors"]),
                        str(entry["other_errors"]),
                        str(entry["masked_key"]),
                        str(entry["last_used_at"] or ""),
                    ]
                )
            )
        self.key_usage_text_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _mark_key_attempt(self, key_index: int):
        self._refresh_key_usage_storage()
        if not self._key_usage_entries:
            return
        entry = self._key_usage_entries[key_index % len(self._key_usage_entries)]
        timestamp = self._now_cn().isoformat(timespec="seconds")
        entry["attempts"] += 1
        entry["last_used_at"] = timestamp
        if not entry["first_used_at"]:
            entry["first_used_at"] = timestamp
        if entry["status"] == "unused":
            entry["status"] = "used"
        self._write_key_usage_files()

    def _mark_key_success(self, key_index: int):
        self._refresh_key_usage_storage()
        if not self._key_usage_entries:
            return
        entry = self._key_usage_entries[key_index % len(self._key_usage_entries)]
        timestamp = self._now_cn().isoformat(timespec="seconds")
        entry["status"] = "used"
        entry["successes"] += 1
        entry["last_success_at"] = timestamp
        entry["last_used_at"] = timestamp
        self._write_key_usage_files()

    def _is_quota_exhausted_error(self, exc: Exception) -> bool:
        text = format_exception_details(exc).lower()
        markers = ["exceeded today's quota", "quota", "try again tomorrow"]
        return any(marker in text for marker in markers)

    def _is_permission_denied_key_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code != 403:
            return False
        text = format_exception_details(exc).lower()
        markers = [
            "real-name verified",
            "real name verified",
            "associated aliyun account",
            "accountsettings",
            "permissiondeniederror",
            "status_code=403",
        ]
        return any(marker in text for marker in markers)

    def _is_auth_invalid_key_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code != 401:
            return False
        text = format_exception_details(exc).lower()
        markers = [
            "please bind your alibaba cloud account before use",
            "bind your alibaba cloud account",
            "authenticationerror",
            "status_code=401",
        ]
        return any(marker in text for marker in markers)

    def _mark_key_error(
        self,
        key_index: int,
        detail: str,
        *,
        rate_limit: bool = False,
        quota: bool = False,
        permission_denied: bool = False,
        auth_invalid: bool = False,
    ):
        self._refresh_key_usage_storage()
        if not self._key_usage_entries:
            return
        entry = self._key_usage_entries[key_index % len(self._key_usage_entries)]
        entry["last_error"] = detail
        if quota:
            entry["quota_errors"] += 1
            entry["status"] = "quota_exhausted_today"
        elif rate_limit:
            entry["rate_limit_errors"] += 1
            if entry["status"] != "quota_exhausted_today":
                entry["status"] = "rate_limited_today"
        elif permission_denied:
            entry["permission_denied_errors"] += 1
            entry["status"] = "permission_denied"
        elif auth_invalid:
            entry["auth_invalid_errors"] += 1
            entry["status"] = "auth_invalid"
        else:
            entry["other_errors"] += 1
            if entry["status"] == "unused":
                entry["status"] = "used"
        self._write_key_usage_files()

    def _rotate_key(self) -> bool:
        if len(self._api_keys) <= 1:
            return False
        self._key_index = (self._key_index + 1) % len(self._api_keys)
        return True

    def _current_key_status(self) -> str:
        self._refresh_key_usage_storage()
        if not self._key_usage_entries:
            return "unused"
        return str(self._key_usage_entries[self._key_index % len(self._key_usage_entries)].get("status", "unused"))

    def _advance_past_known_bad_keys(self) -> None:
        skip_statuses = {"quota_exhausted_today", "permission_denied", "auth_invalid"}
        if len(self._api_keys) <= 1:
            return
        checked = 0
        while checked < len(self._api_keys) - 1 and self._current_key_status() in skip_statuses:
            if not self._rotate_key():
                break
            checked += 1

    def _client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self._current_key(), base_url=self.base_url)

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        text = format_exception_details(exc).lower()
        markers = ["rate limit", "exceeded today's quota", "quota", "status_code=429"]
        return any(marker in text for marker in markers)

    def _json_clone(self, value: Any) -> Any:
        return json.loads(json.dumps(value, ensure_ascii=False))

    def _serialize_tool_calls(self, tool_calls: Any) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for tc in tool_calls or []:
            function = getattr(tc, "function", None)
            serialized.append({
                "id": getattr(tc, "id", ""),
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": getattr(function, "name", ""),
                    "arguments": getattr(function, "arguments", ""),
                },
            })
        return serialized

    def _serialize_response_message(self, message: Any) -> Optional[dict[str, Any]]:
        if message is None:
            return None
        return {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None),
            "reasoning_content": getattr(message, "reasoning_content", None),
            "tool_calls": self._serialize_tool_calls(getattr(message, "tool_calls", None)),
        }

    def _write_trace_bundle(self):
        bundle = {
            "run_id": self.trace_run_id,
            "trace_file": str(self.trace_file),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "entries": self._trace_entries,
        }
        self.trace_js_file.write_text(
            "window.__TRACE_DATA__ = " + json.dumps(bundle, ensure_ascii=False) + ";\n",
            encoding="utf-8",
        )
        self.trace_index_file.write_text(str(self.trace_file), encoding="utf-8")

    def _record_trace(
        self,
        call_type: str,
        messages: list[dict[str, Any]],
        tools: Optional[List[dict]],
        response_message: Any = None,
        error: Optional[str] = None,
        trace_metadata: Optional[dict[str, Any]] = None,
    ):
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "call_type": call_type,
            "model": self.model_name,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "tool_choice": self.tool_choice,
            "request": {
                "messages": self._json_clone(messages),
                "tools": self._json_clone(tools or []),
                "extra_body": self._json_clone(self.default_extra_body),
            },
            "response": self._serialize_response_message(response_message),
            "error": error,
        }
        if trace_metadata:
            entry.update(self._json_clone(trace_metadata))
        self._trace_entries.append(entry)
        with self.trace_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._write_trace_bundle()

    def _response_summary(self, resp: Any) -> dict[str, Any]:
        summary = {"response_type": type(resp).__name__}
        for attr in ("id", "model", "object", "created"):
            value = getattr(resp, attr, None)
            if value is not None:
                summary[attr] = value

        choices = getattr(resp, "choices", None)
        summary["choices_type"] = type(choices).__name__
        try:
            summary["choices_len"] = len(choices) if choices is not None else None
        except TypeError:
            summary["choices_len"] = None

        usage = getattr(resp, "usage", None)
        if usage is not None:
            try:
                summary["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else str(usage)
            except Exception:
                summary["usage"] = str(usage)

        try:
            if hasattr(resp, "model_dump"):
                dumped = resp.model_dump()
                if isinstance(dumped, dict):
                    summary["dump_keys"] = sorted(dumped.keys())
                    if dumped.get("error") is not None:
                        summary["error"] = dumped.get("error")
        except Exception:
            pass
        return summary

    def _extract_message_or_raise(self, resp: Any) -> Any:
        if resp is None:
            raise ValueError("Chat completion returned None response")

        choices = getattr(resp, "choices", None)
        if not choices:
            raise ValueError(
                "Malformed chat completion response: missing choices. "
                + json.dumps(self._response_summary(resp), ensure_ascii=False)
            )

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None:
            raise ValueError(
                "Malformed chat completion response: missing message in first choice. "
                + json.dumps(self._response_summary(resp), ensure_ascii=False)
            )

        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            text = content if isinstance(content, str) else ""
            if len(text.strip()) <= 2:
                raise ValueError(
                    "Malformed chat completion response: trivial assistant content without tool calls. "
                    + json.dumps(
                        {
                            **self._response_summary(resp),
                            "content_preview": text,
                            "content_len": len(text),
                        },
                        ensure_ascii=False,
                    )
                )
        return message

    async def _request_message(self, call_type: str, kwargs: dict) -> Any:
        last_error = ""
        max_attempts = max(3, len(self._api_keys) if self._api_keys else 1)
        for attempt in range(1, max_attempts + 1):
            self._advance_past_known_bad_keys()
            key_index = self._key_index % len(self._api_keys) if self._api_keys else 0
            self._mark_key_attempt(key_index)
            try:
                client = self._client()
                resp = await client.chat.completions.create(**kwargs)
                message = self._extract_message_or_raise(resp)
                self._mark_key_success(key_index)
                if attempt > 1:
                    logger.warning("%s recovered on retry %d", call_type, attempt)
                return message
            except Exception as e:
                if isinstance(e, ValueError):
                    last_error = str(e)
                else:
                    last_error = format_exception_details(e)
                logger.warning("%s attempt %d failed: %s", call_type, attempt, last_error)
                is_rate_limit = self._is_rate_limit_error(e)
                is_quota = self._is_quota_exhausted_error(e)
                is_permission_denied = self._is_permission_denied_key_error(e)
                is_auth_invalid = self._is_auth_invalid_key_error(e)
                self._mark_key_error(
                    key_index,
                    last_error,
                    rate_limit=is_rate_limit,
                    quota=is_quota,
                    permission_denied=is_permission_denied,
                    auth_invalid=is_auth_invalid,
                )
                should_rotate = is_rate_limit or is_permission_denied or is_auth_invalid
                if should_rotate and self._rotate_key():
                    if is_rate_limit:
                        reason = "rate limit/quota error"
                    elif is_permission_denied:
                        reason = "permission-denied key"
                    else:
                        reason = "auth-invalid key"
                    logger.warning("%s switching to next API key after %s; see %s", call_type, reason, self.key_usage_text_file)
                if attempt < max_attempts:
                    await asyncio.sleep(min(attempt, 3))
        raise RuntimeError(last_error)

    async def chat_with_image(
        self,
        image_base64: str,
        system_prompt: str,
        user_text: str,
        tools: Optional[List[dict]] = None,
        history_context: str = "",
        trace_metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[Any]:
        content: List[dict] = []
        if history_context:
            content.append({"type": "text", "text": history_context + "\n\n"})
        content.append({"type": "text", "text": user_text})
        mime = "image/png" if image_base64.startswith("iVBORw0KGgo") else "image/jpeg"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_base64}"},
        })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        kwargs: dict = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.default_extra_body:
            kwargs["extra_body"] = self._json_clone(self.default_extra_body)
        if tools:
            kwargs["tools"] = tools
            if self.tool_choice is not None:
                kwargs["tool_choice"] = self.tool_choice

        try:
            message = await self._request_message("chat_with_image", kwargs)
            self.last_error = None
            self._record_trace("chat_with_image", messages, tools, response_message=message, trace_metadata=trace_metadata)
            return message
        except Exception as e:
            self.last_error = str(e)
            logger.error("chat_with_image failed: %s", self.last_error)
            self._record_trace("chat_with_image", messages, tools, error=self.last_error, trace_metadata=trace_metadata)
            return None

    async def chat_with_images(
        self,
        image_base64_list: List[str],
        system_prompt: str,
        user_text: str,
        tools: Optional[List[dict]] = None,
        history_context: str = "",
        trace_metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[Any]:
        content: List[dict] = []
        if history_context:
            content.append({"type": "text", "text": history_context + "\n\n"})
        content.append({"type": "text", "text": user_text})
        for b64 in image_base64_list:
            mime = "image/png" if b64.startswith("iVBORw0KGgo") else "image/jpeg"
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        kwargs: dict = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.default_extra_body:
            kwargs["extra_body"] = self._json_clone(self.default_extra_body)
        if tools:
            kwargs["tools"] = tools
            if self.tool_choice is not None:
                kwargs["tool_choice"] = self.tool_choice
        try:
            message = await self._request_message("chat_with_images", kwargs)
            self.last_error = None
            self._record_trace("chat_with_images", messages, tools, response_message=message, trace_metadata=trace_metadata)
            return message
        except Exception as e:
            self.last_error = str(e)
            logger.error("chat_with_images failed: %s", self.last_error)
            self._record_trace("chat_with_images", messages, tools, error=self.last_error, trace_metadata=trace_metadata)
            return None

    async def chat_text_only(
        self,
        system_prompt: str,
        user_text: str,
        tools: Optional[List[dict]] = None,
        trace_metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        kwargs: dict = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.default_extra_body:
            kwargs["extra_body"] = self._json_clone(self.default_extra_body)
        if tools:
            kwargs["tools"] = tools
            if self.tool_choice is not None:
                kwargs["tool_choice"] = self.tool_choice

        try:
            message = await self._request_message("chat_text_only", kwargs)
            self.last_error = None
            self._record_trace("chat_text_only", messages, tools, response_message=message, trace_metadata=trace_metadata)
            return message
        except Exception as e:
            self.last_error = str(e)
            logger.error("chat_text_only failed: %s", self.last_error)
            self._record_trace("chat_text_only", messages, tools, error=self.last_error, trace_metadata=trace_metadata)
            return None


# 启动示例
if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

    async def demo():
        client = ModelClient(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            max_tokens=256,
        )
        msg = await client.chat_text_only(
            system_prompt="You are a helpful assistant.",
            user_text="Say hello in one sentence.",
        )
        if msg:
            print(msg.content)
        else:
            print("Request failed.")

    asyncio.run(demo())
