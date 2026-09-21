"""ローカル LLM / 埋め込み / VLM サーバー（LM Studio 等）への OpenAI 互換クライアント。

`/chat/completions`（テキスト・画像入力とも） と `/embeddings` を httpx で直接叩くだけ。
openai パッケージには依存しない。接続先は config.server.base_url。既定は LM Studio の
http://localhost:1234/v1。Ollama など OpenAI 互換 API を出す他基盤に差し替えても動く。

外部ネットワークへは接続しない（base_url が localhost 前提）。
meeting-minutes プロジェクト（src/meeting_minutes/model/llm_client.py）の設計思想
（接続エラー・タイムアウト・コンテキスト長オーバーに親切なヒントを出す）を踏襲している。
"""

from __future__ import annotations

from .config import ServerConfig


class LLMConnectionError(RuntimeError):
    """ローカル LLM サーバーに接続できない／エラー応答のときに送出する。"""


class LLMClient:
    def __init__(self, config: ServerConfig):
        self.config = config
        import httpx

        self._httpx = httpx
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout,
            headers={"Authorization": f"Bearer {config.api_key}"},
        )

    # --- 低レベル ---------------------------------------------------------
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
            # Qwen3系などの推論モデルは、可視の回答を書く前に思考トークンを消費する。
            # max_tokensが小さいと思考だけで使い切り、contentが空のまま返ってくる。
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            if reasoning:
                raise LLMConnectionError(
                    f"モデルが思考内容（{len(reasoning)}文字）のみを返し、回答本文が空でした。"
                    "max_tokensを増やしてください。"
                )
            # reasoning も無いのに空文字（拒否応答など）が返ってきたケース。空のまま
            # 呼び出し元に返すと、呼び出し側が「成功」として扱い既存データを壊しかねない
            # ため、ここで必ずエラーにする。
            raise LLMConnectionError(f"LM Studioが空の応答を返しました: {data}")

        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            # max_tokensを使い切って途中で打ち切られた応答。合成データ生成では
            # 「トラブルシューティング・教訓」など末尾セクションが欠落したまま
            # 保存されてしまうため、黙って受理せず呼び出し側にエラーとして伝える。
            raise LLMConnectionError(
                f"応答がmax_tokens（{len(content)}文字生成した時点）で打ち切られました。"
                "max_tokensを増やすか、入力を短くしてください。"
            )
        return content

    # --- 高レベル -------------------------------------------------------
    def chat(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> str:
        """テキストのみのチャット補完。合成データ生成や回答生成に使う。"""
        payload = {
            "model": model or self.config.model,
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
        """画像1枚をvisionモデル（VLM）に説明させる。ingestの画像キャプション取得に使う。

        LM StudioにロードしたVLMを`config.vlm_model`で指定する（未ロードならエラーになる。
        呼び出し側で捕捉するかどうかは呼び出し側の判断に委ねる＝ここでは握りつぶさない）。
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
        """テキスト群を埋め込みベクトルに変換する。"""
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
        """サーバーに到達できるか軽く確認する（GET /models）。"""
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
