import json
import logging
import os
import re
import threading
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
from google import genai
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ローカルでは .env から読み込む（Render では環境変数が直接セットされるため無視される）
load_dotenv()

app = Flask(__name__, template_folder=".")
CORS(app)

# --- Basic 認証（非公開化）------------------------------------------------
# 公開リポジトリのためパスワードは環境変数で注入する。Render のダッシュボードで
# BASIC_AUTH_PASS（および任意で BASIC_AUTH_USER）を設定すること。
# 未設定時は「誤って公開しない」ため全アクセスを 503 でブロックする。
BASIC_AUTH_USER = os.environ.get("BASIC_AUTH_USER", "katsu")
BASIC_AUTH_PASS = os.environ.get("BASIC_AUTH_PASS")


# 認証不要のパス。
#   /warmup — UptimeRobot 等のキープアライブ用（機密情報を返さない）
#   /ask    — 公開中のブログ銀河（galaxy）から呼ぶため。裸で開けると Gemini の無料枠を
#             枯らされるので、ask() の冒頭で Origin・レート・日次上限の3段で絞っている
PUBLIC_PATHS = {"/warmup", "/ask"}


@app.before_request
def _require_basic_auth():
    if request.path in PUBLIC_PATHS:
        return None
    if not BASIC_AUTH_PASS:
        return Response(
            "非公開設定中です（管理者は環境変数 BASIC_AUTH_PASS を設定してください）",
            503,
        )
    auth = request.authorization
    if auth and auth.username == BASIC_AUTH_USER and auth.password == BASIC_AUTH_PASS:
        return None
    return Response(
        "認証が必要です",
        401,
        {"WWW-Authenticate": 'Basic realm="RIAT (private)"'},
    )
# --------------------------------------------------------------------------

GENAI_API_KEY    = os.environ.get("GENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not GENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("GENAI_API_KEY と PINECONE_API_KEY を .env に設定してください")

# gemini-3.1-flash-lite は無料枠RPDが500（2.5-flashの20の25倍）。RAG回答には十分な品質
GEN_MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
TOP_K = 3

client = genai.Client(api_key=GENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("seimeiron-blog")

logging.info("loading...")
with open("blog_data.json", "r", encoding="utf-8") as fl:
    content = fl.read()
clean_content = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', content)
all_articles = json.loads(clean_content)

# URL をキーにした高速検索用辞書（Pinecone のメタデータには本文が無いため全文をここから引く）
url_to_article = {a["url"]: a for a in all_articles}
logging.info("%d articles loaded", len(all_articles))


def sse(obj) -> str:
    """Server-Sent Events の1メッセージにエンコードする。"""
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


# --- /ask の流量制限 ------------------------------------------------------
# Origin チェックは他サイトからの埋め込みを防ぐだけで、認証ではない（非ブラウザからは
# 詐称できる）。実効的な保護は下のレート制限と日次上限のほう。
ALLOWED_ORIGINS = {
    "https://galaxy-wheat-zeta.vercel.app",
    "https://riat-blog-jiten-2.onrender.com",
    "http://localhost:5900",
    "http://127.0.0.1:5900",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
}
RATE_WINDOW_SEC = 600   # 10分あたり
RATE_MAX = 20           # 同一IPからの上限
DAILY_MAX = 300         # 全体の1日上限（ブログ辞典本体と Gemini 無料枠を共有するため）

_rate_lock = threading.Lock()
_ip_hits: dict[str, list[float]] = {}
_daily = {"date": None, "count": 0}


def _client_ip() -> str:
    fwd = request.headers.get("X-Forwarded-For", "")
    return fwd.split(",")[0].strip() if fwd else (request.remote_addr or "?")


def _check_quota():
    """許可されていれば None、拒否なら (本文, ステータス) を返す。"""
    origin = request.headers.get("Origin")
    if origin and origin not in ALLOWED_ORIGINS:
        return "この場所からは利用できません", 403

    now = time.time()
    today = time.strftime("%Y-%m-%d")
    with _rate_lock:
        if _daily["date"] != today:
            _daily["date"], _daily["count"] = today, 0
            _ip_hits.clear()
        if _daily["count"] >= DAILY_MAX:
            return "本日の利用枠を使い切りました。明日またお試しください", 429

        ip = _client_ip()
        hits = [t for t in _ip_hits.get(ip, []) if now - t < RATE_WINDOW_SEC]
        if len(hits) >= RATE_MAX:
            return "アクセスが集中しています。少し時間をおいてからお試しください", 429

        hits.append(now)
        _ip_hits[ip] = hits
        _daily["count"] += 1
    return None
# --------------------------------------------------------------------------


@app.route("/warmup", methods=["GET"])
def warmup():
    return jsonify({"status": "ok"})


@app.route("/")
def home():
    return send_file("index.html")


@app.route("/blog_data.json")
def blog_data():
    # フロントのローカル検索フォールバック用に全記事を配信
    return send_file("blog_data.json", mimetype="application/json")


@app.route("/ask", methods=["POST"])
def ask():
    denied = _check_quota()
    if denied:
        message, status = denied
        return jsonify({"response": message, "sources": []}), status

    data = request.json or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"response": "メッセージを入力してください", "sources": []}), 400
    if len(user_message) > 500:
        return jsonify({"response": "メッセージは500文字以内で入力してください", "sources": []}), 400

    def generate():
        try:
            # 1. 質問を埋め込み（768次元）
            embed_result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=user_message,
                config={"output_dimensionality": EMBED_DIM},
            )
            query_vector = embed_result.embeddings[0].values

            # 2. Pinecone で類似記事を検索
            search_result = index.query(
                vector=query_vector,
                top_k=TOP_K,
                include_metadata=True,
            )

            context_parts = []
            sources = []
            for match in search_result["matches"]:
                meta = match["metadata"]
                title = meta.get("title", "")
                url = meta.get("url", "")
                # URLで全文を引く（メタデータに本文は無いため）
                article_content = url_to_article.get(url, {}).get("content", meta.get("content", ""))
                context_parts.append("タイトル: " + title + "\n本文:\n" + article_content)
                sources.append({"title": title, "url": url})

            # 3. 出典を先に送る
            yield sse({"type": "sources", "sources": sources})

            context_text = "\n---\n".join(context_parts)
            prompt = (
                "以下のブログ記事を読んで質問に丁寧に答えてください。\n"
                "回答文にURLは含めないでください。\n\n"
                + context_text
                + "\n\n質問: "
                + user_message
            )

            # 4. Gemini の回答をストリーミング送出
            for chunk in client.models.generate_content_stream(
                model=GEN_MODEL,
                contents=prompt,
            ):
                if chunk.text:
                    yield sse({"type": "delta", "text": chunk.text})

            yield sse({"type": "done"})

        except Exception as e:
            logging.error("エラー発生 (message=%r): %s", user_message, e, exc_info=True)
            yield sse({"type": "error", "message": "しばらく待ってから再度お試しください"})

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
