import json
import logging
import os
import re
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


@app.before_request
def _require_basic_auth():
    # /warmup は UptimeRobot 等のキープアライブ用に認証不要（機密情報を返さない）
    if request.path == "/warmup":
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
