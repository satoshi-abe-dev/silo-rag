"""OpenAI-compatible client for a local LLM/embedding/VLM server (e.g. LM Studio).

Calls `/chat/completions` (text and image) and `/embeddings` directly via httpx, without
the openai package. Targets config.ai.base_url (LM Studio's localhost by default; any
OpenAI-compatible server such as Ollama works). No external network access.
Like meeting-minutes' llm_client.py, errors carry helpful hints for connection failures,
timeouts, and context-length overflows.
"""

from __future__ import annotations

from .config import AIConfig


class LLMConnectionError(RuntimeError):
    """Raised when the local LLM server is unreachable or returns an error."""


class LLMClient:
    def __init__(self, config: AIConfig):
        self.config = config
        import httpx

        self._httpx = httpx
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout,
            headers={"Authorization": f"Bearer {config.api_key}"},
        )

    # --- Low level ---------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = self._client.post(path, json=payload)
        except self._httpx.TimeoutException as exc:
            raise LLMConnectionError(
                f"LM Studio への応答がタイムアウトしました（{self.config.timeout}秒）。"
                "モデルのロードや生成に時間がかかっている可能性があります。"
                f"詳細: {exc}"
            ) from exc
        except self._httpx.RequestError as exc:
            raise LLMConnectionError(
                f"LM Studio ({self.config.base_url}) に接続できません。"
                "LM Studioが起動し、モデルがロードされているか確認してください。"
                f"詳細: {exc}"
            ) from exc

        if resp.status_code >= 400:
            body = resp.text[:500]
            msg = f"LM Studioがエラーを返しました（HTTP {resp.status_code}）: {body}"
            low = body.lower()
            if "no models loaded" in low or "model_not_found" in low:
                msg += f"\nLM Studioで対象モデルをロードしてください（base_url: {self.config.base_url}）。"
            elif "context length" in low or "context window" in low or "exceeds the context" in low:
                msg += "\n入力がモデルのコンテキスト長を超えています。チャンクを分割してください。"
            raise LLMConnectionError(msg)

        try:
            return resp.json()
        except ValueError as exc:
            raise LLMConnectionError(f"LM Studioの応答をJSONとして解釈できません: {resp.text[:500]}") from exc

    def _post_chat(self, payload: dict) -> str:
        data = self._post("/chat/completions", payload)
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = (message.get("content") or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMConnectionError(f"LM Studioの応答を解釈できません: {data}") from exc

        if not content:
            # Reasoning models (e.g. Qwen3) can spend all of a small max_tokens on thinking,
            # leaving content empty.
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            if reasoning:
                raise LLMConnectionError(
                    f"モデルが思考内容（{len(reasoning)}文字）のみを返し、回答本文が空でした。"
                    "max_tokensを増やしてください。"
                )
            # Empty without reasoning (e.g. a refusal): always raise, or callers may treat it
            # as success and overwrite existing data.
            raise LLMConnectionError(f"LM Studioが空の応答を返しました: {data}")

        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            # Truncated at max_tokens: raise, or synthetic reports get saved missing their
            # final sections (e.g. lessons learned).
            raise LLMConnectionError(
                f"応答がmax_tokens（{len(content)}文字生成した時点）で打ち切られました。"
                "max_tokensを増やすか、入力を短くしてください。"
            )
        return content

    # --- High level ------------------------------------------------------
    def chat(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Text-only chat completion (synthetic data and answer generation)."""
        payload = {
            "model": model or self.config.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        return self._post_chat(payload).strip()

    def describe_image(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 300,
    ) -> str:
        """Describe one image with the VLM (`config.vlm_model`); used for ingest captions.

        Raises if the VLM isn't loaded; errors are left for the caller to handle.
        """
        import base64

        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        payload = {
            "model": model or self.config.vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "stream": False,
        }
        return self._post_chat(payload).strip()

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """Embed a list of texts."""
        if not texts:
            return []
        payload = {"model": model or self.config.embed_model, "input": texts}
        data = self._post("/embeddings", payload)
        try:
            items = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in items]
        except (KeyError, TypeError) as exc:
            raise LLMConnectionError(f"LM Studioの埋め込み応答を解釈できません: {data}") from exc

    def ping(self) -> bool:
        """Lightweight reachability check (GET /models)."""
        try:
            resp = self._client.get("/models")
            return resp.status_code < 500
        except self._httpx.RequestError:
            return False

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
