import json
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from google import genai
from pinecone import Pinecone

# ローカルでは .env から読み込む（Render では環境変数が直接セットされるため無視される）
load_dotenv()

app = Flask(__name__, template_folder=".")
CORS(app)

GENAI_API_KEY    = os.environ.get("GENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not GENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("GENAI_API_KEY と PINECONE_API_KEY を .env に設定してください")

client = genai.Client(api_key=GENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("seimeiron-blog")

print("loading...")
with open("blog_data.json", "r", encoding="utf-8") as fl:
    all_articles = json.load(fl)

# URL をキーにした高速検索用辞書
url_to_article = {a["url"]: a for a in all_articles}
print(str(len(all_articles)) + " articles loaded")


@app.route("/")
def home():
    return send_file("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_message = data.get("message")
    try:
        embed_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=user_message,
            config={"output_dimensionality": 768},
        )
        query_vector = embed_result.embeddings[0].values

        search_result = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True,
        )

        context_parts = []
        sources = []

        for match in search_result["matches"]:
            meta = match["metadata"]
            title = meta.get("title", "")
            url = meta.get("url", "")
            # URLで全文を引く（article_indexはメタデータに存在しないため）
            content = url_to_article.get(url, {}).get("content", meta.get("content", ""))
            context_parts.append("タイトル: " + title + "\n本文:\n" + content)
            sources.append({"title": title, "url": url})

        context_text = "\n---\n".join(context_parts)

        prompt = (
            "以下のブログ記事を読んで質問に丁寧に答えてください。\n"
            "回答文にURLは含めないでください。\n\n"
            + context_text
            + "\n\n質問: "
            + user_message
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return jsonify({"response": response.text, "sources": sources})

    except Exception as e:
        print(str(e))
        return jsonify({"response": "エラーが発生しました", "sources": []})


if __name__ == "__main__":
    app.run(port=5000)
