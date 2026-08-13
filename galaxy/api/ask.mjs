/* ブログ銀河の質問機能（RAG）。server.py の /ask をそのまま Vercel の関数へ移したもの。

   なぜ移したか: Render の無料インスタンスがスピンダウンからの復帰に失敗して無応答になる
   事故が続いた（接続は受けるが一切応答を返さず、再デプロイでしか復旧しない）。ギャラクシー
   自身が載っている Vercel で完結させれば、その経路ごと無くなる。

   SSE のイベント形式（sources / delta / done / error）は据え置き。フロントの streamAsk は
   エンドポイントの向き先だけ変えれば動く。

   依存パッケージは無し。Gemini も Pinecone も REST で足りる。 */

import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));

// server.py と同じ設定にそろえる（変えると Pinecone の既存ベクトルと噛み合わなくなる）
const GEN_MODEL = "gemini-3.1-flash-lite"; // 無料枠 RPD500
const EMBED_MODEL = "gemini-embedding-001";
const EMBED_DIM = 768;
const TOP_K = 3;
const PINECONE_HOST = "seimeiron-blog-hcmvlzp.svc.aped-4627-b74a.pinecone.io";
const GENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/models";

// 上流が返さないまま関数を占有しないよう、必ず上限を置く（Render の件と同じ轍を踏まない）
const EMBED_TIMEOUT_MS = 20000;
const QUERY_TIMEOUT_MS = 20000;
const GEN_TIMEOUT_MS = 45000;

/* 他サイトからの埋め込みを防ぐだけで、認証ではない（非ブラウザからは詐称できる）。
   server.py:90-102 と同じ顔ぶれ。 */
const ALLOWED_ORIGINS = new Set([
  "https://galaxy-wheat-zeta.vercel.app",
  "http://localhost:5900",
  "http://127.0.0.1:5900",
  "http://localhost:3000",
  "http://127.0.0.1:3000",
]);
const PREVIEW_ORIGIN_RE =
  /^https:\/\/galaxy-[a-z0-9]+-katsunoritoriumi-2409s-projects\.vercel\.app$/;

/* 記事全文。Pinecone のメタデータには本文が無いので URL から引く。
   コールドスタートの1回だけ展開し、以降は同じインスタンスで使い回される。 */
let articles = null;
function getArticles() {
  if (!articles) {
    const gz = fs.readFileSync(path.join(HERE, "_data", "articles.json.gz"));
    articles = JSON.parse(zlib.gunzipSync(gz).toString("utf-8"));
  }
  return articles;
}

const sse = (obj) => "data: " + JSON.stringify(obj) + "\n\n";

const jsonResponse = (body, status) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });

function originAllowed(origin) {
  if (!origin) return true; // 非ブラウザからの呼び出しは Origin が付かない
  return ALLOWED_ORIGINS.has(origin) || PREVIEW_ORIGIN_RE.test(origin);
}

async function embed(message, key) {
  const res = await fetch(
    `${GENAI_BASE}/${EMBED_MODEL}:embedContent?key=${encodeURIComponent(key)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: `models/${EMBED_MODEL}`,
        content: { parts: [{ text: message }] },
        outputDimensionality: EMBED_DIM,
      }),
      signal: AbortSignal.timeout(EMBED_TIMEOUT_MS),
    }
  );
  if (!res.ok) throw new Error("embed " + res.status + " " + (await res.text()).slice(0, 200));
  const data = await res.json();
  return data.embedding.values;
}

async function search(vector, key) {
  const res = await fetch(`https://${PINECONE_HOST}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Api-Key": key },
    body: JSON.stringify({ vector, topK: TOP_K, includeMetadata: true }),
    signal: AbortSignal.timeout(QUERY_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error("pinecone " + res.status + " " + (await res.text()).slice(0, 200));
  const data = await res.json();
  return data.matches || [];
}

/* Gemini のストリーミング応答を1片ずつ返す。SSE の中に SSE が来るので手で解く。 */
async function* generate(prompt, key) {
  const res = await fetch(
    `${GENAI_BASE}/${GEN_MODEL}:streamGenerateContent?alt=sse&key=${encodeURIComponent(key)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] }),
      signal: AbortSignal.timeout(GEN_TIMEOUT_MS),
    }
  );
  if (!res.ok) throw new Error("generate " + res.status + " " + (await res.text()).slice(0, 200));

  // Gemini の SSE は CRLF 区切りで返ってくる。LF に寄せてから区切らないと1つも取れない
  const texts = (chunk) => {
    const line = chunk.split("\n").find((l) => l.startsWith("data: "));
    if (!line) return [];
    const payload = line.slice(6).trim();
    if (!payload || payload === "[DONE]") return [];
    let obj;
    try {
      obj = JSON.parse(payload);
    } catch {
      return [];
    }
    // thoughtSignature だけの部品は text が空なので自然に落ちる
    return (obj?.candidates?.[0]?.content?.parts || []).map((p) => p.text).filter(Boolean);
  };

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    const chunks = buf.split("\n\n");
    buf = chunks.pop();
    for (const chunk of chunks) for (const t of texts(chunk)) yield t;
  }
  // 最後のイベントが区切りなしで終わることがあるので、残りも必ず見る
  for (const t of texts(buf)) yield t;
}

/* HTTP メソッド名でエクスポートすると Vercel は Web 標準の (Request) => Response として
   扱う。default エクスポートだと Node の (req, res) 形式になり、ストリーミングできない。 */
export async function POST(request) {
  if (!originAllowed(request.headers.get("origin"))) {
    return jsonResponse({ response: "この場所からは利用できません", sources: [] }, 403);
  }

  // 環境変数の登録経路によっては末尾に改行が紛れ込み、そのままだと鍵が無効になる
  const genaiKey = (process.env.GENAI_API_KEY || "").trim();
  const pineconeKey = (process.env.PINECONE_API_KEY || "").trim();
  if (!genaiKey || !pineconeKey) {
    return jsonResponse({ response: "サーバー設定が未完了です", sources: [] }, 503);
  }

  let data;
  try {
    data = await request.json();
  } catch {
    data = {};
  }
  const userMessage = String(data.message || "").trim();
  // ブログ銀河から「この星について聞く」場合に渡ってくる記事URL。
  // ベクトル検索は必ずしもその記事を引かないため、指定があれば文脈の先頭に固定する
  const focusUrl = String(data.url || "").trim();
  const table = getArticles();
  // 「この記事の要約」のように、その記事だけで完結する用途。
  // 埋め込みと Pinecone 検索を丸ごと省けるので API 消費も減る
  const focusOnly = Boolean(data.only) && Object.hasOwn(table, focusUrl);

  if (!userMessage) {
    return jsonResponse({ response: "メッセージを入力してください", sources: [] }, 400);
  }
  if (userMessage.length > 500) {
    return jsonResponse({ response: "メッセージは500文字以内で入力してください", sources: [] }, 400);
  }

  const enc = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (obj) => controller.enqueue(enc.encode(sse(obj)));
      try {
        let matches = [];
        if (!focusOnly) {
          const vector = await embed(userMessage, genaiKey);
          matches = await search(vector, pineconeKey);
        }

        const contextParts = [];
        const sources = [];
        const seen = new Set();
        const addArticle = (title, url, content) => {
          if (!url || seen.has(url)) return;
          seen.add(url);
          contextParts.push("タイトル: " + title + "\n本文:\n" + content);
          sources.push({ title, url });
        };

        // 読者が開いている記事は検索結果より先に、必ず文脈へ入れる
        let focusTitle = "";
        const focus = focusUrl ? table[focusUrl] : null;
        if (focus) {
          focusTitle = focus.t || "";
          addArticle(focusTitle, focusUrl, focus.c || "");
        }

        for (const match of matches) {
          const meta = match.metadata || {};
          const url = meta.url || "";
          const article = table[url];
          addArticle(meta.title || "", url, article ? article.c : meta.content || "");
        }

        send({ type: "sources", sources });

        const focusNote = focusTitle
          ? "読者は「" + focusTitle + "」という記事を読んでいます。" +
            "「この記事」とはこの記事を指します。まずこの記事の内容に基づいて答え、" +
            "足りない部分だけ他の記事で補ってください。\n"
          : "";
        const prompt =
          "以下のブログ記事を読んで質問に丁寧に答えてください。\n" +
          "回答文にURLは含めないでください。\n" +
          focusNote +
          "\n" +
          contextParts.join("\n---\n") +
          "\n\n質問: " +
          userMessage;

        for await (const text of generate(prompt, genaiKey)) {
          send({ type: "delta", text });
        }
        send({ type: "done" });
      } catch (e) {
        console.error("エラー発生 (message=%s): %s", userMessage, e && e.stack ? e.stack : e);
        send({ type: "error", message: "しばらく待ってから再度お試しください" });
      }
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
    },
  });
}
