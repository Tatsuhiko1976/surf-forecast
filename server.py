#!/usr/bin/env python3
"""
Surf forecast local dev server.
Serves static files from the same directory and exposes /api/chat
as a Server-Sent Events streaming endpoint backed by the Anthropic API.
"""

import json
import os
import sys
import traceback
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not found. Run: pip3 install anthropic", file=sys.stderr)
    sys.exit(1)

PORT = int(os.environ.get("PORT", 3456))
BASE_DIR = Path(__file__).parent

SYSTEM_BASE = """あなたはサーフィン専門のAIアドバイザーです。
ユーザーは7.0フィートのミッドレングスボード（通称ミッドレン）を使用しています。

ミッドレングスに適した波の目安：
- 波高：0.5〜1.5m（腰〜肩程度）が最適
- 周期：8秒以上のうねりがある波が望ましい
- 風：無風〜弱いオフショア（陸から海への風）が最良
- 混雑：ローカル色が強いポイントや混雑している日は注意

回答のポイント：
- 具体的なスポット名を挙げてアドバイスする
- スコアや波高・周期・風向きの数値も参考に伝える
- 初心者〜中級者でも楽しめるような視点でアドバイスする
- 日本語で回答する
- 簡潔かつ有益な情報を伝える（300〜500字程度を目安に）
"""


class Handler(SimpleHTTPRequestHandler):
    """Serves static files from BASE_DIR and handles /api/chat POST."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #
    def log_message(self, fmt, *args):  # quieter logs
        print(f"[server] {self.address_string()} - {fmt % args}")

    # ------------------------------------------------------------------ #
    #  CORS helpers                                                        #
    # ------------------------------------------------------------------ #
    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    # ------------------------------------------------------------------ #
    #  POST /api/chat                                                      #
    # ------------------------------------------------------------------ #
    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
        except Exception:
            self.send_error(400, "Bad JSON")
            return

        message = body.get("message", "").strip()
        context = body.get("context", "").strip()
        history = body.get("history", [])

        if not message:
            self.send_error(400, "Empty message")
            return

        # Build system prompt (inject forecast data only on first turn)
        system = SYSTEM_BASE
        if context:
            system += f"\n\n【現在の波予報データ（参考）】\n{context}"

        # Build messages list
        msgs = list(history) + [{"role": "user", "content": message}]

        # --- SSE response ---
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self._cors_headers()
        self.end_headers()

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            err = json.dumps({"error": "ANTHROPIC_API_KEY が設定されていません。"})
            self.wfile.write(f"data: {err}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()
            return

        try:
            client = anthropic.Anthropic(api_key=api_key)
            with client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=1024,
                system=system,
                messages=msgs,
            ) as stream:
                for text in stream.text_stream:
                    payload = json.dumps({"text": text}, ensure_ascii=False)
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
        except anthropic.AuthenticationError:
            err = json.dumps({"error": "APIキーが無効です。ANTHROPIC_API_KEY を確認してください。"})
            self.wfile.write(f"data: {err}\n\n".encode())
        except Exception as e:
            traceback.print_exc()
            err = json.dumps({"error": f"エラーが発生しました: {e}"})
            self.wfile.write(f"data: {err}\n\n".encode())
        finally:
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except BrokenPipeError:
                pass


if __name__ == "__main__":
    server = HTTPServer(("", PORT), Handler)
    print(f"Surf forecast server running at http://localhost:{PORT}")
    print(f"Serving files from: {BASE_DIR}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY is not set — chat will return an error.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
