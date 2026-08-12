# -*- coding: utf-8 -*-
"""ブログ銀河「この記事の要約」用の要約を事前生成する。

要約を毎回 Gemini に投げていると、閲覧者の数だけ API を消費し（1日300回の上限）、
同時利用も詰まる。クイズ（quiz_by_star.json）と同じく、生成は1回きりのバッチにして
静的 JSON として同梱する ＝ 実行時の API 呼び出しをゼロにする。

出力: galaxy/summaries.json  {"<記事URL>": {"s": "要約2〜3文", "p": ["要点", ...]}}
キャッシュ: galaxy/tools/article_summaries_cache.json（再実行時は生成済みを飛ばす）

使い方:
    venv/Scripts/python.exe galaxy/tools/build_summaries.py
    venv/Scripts/python.exe galaxy/tools/build_summaries.py --limit 8   # 試し打ち
"""
import argparse
import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from google import genai

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GALAXY_DIR = os.path.join(ROOT, "galaxy")
BLOG_PATH = os.path.join(ROOT, "blog_data.json")          # 実データ。読み取り専用
STARS_PATH = os.path.join(GALAXY_DIR, "galaxy_data.json")
CACHE_PATH = os.path.join(GALAXY_DIR, "tools", "article_summaries_cache.json")
OUT_PATH = os.path.join(GALAXY_DIR, "summaries.json")

GEN_MODEL = "gemini-3.1-flash-lite"  # 無料枠 RPD500（server.py と同じ）
BATCH = 4                   # 1リクエストあたりの記事数（大きくすると応答JSONが出力上限で切れる）
CONTENT_TRUNC = 6000        # 要約入力に使う本文の先頭文字数
SLEEP_BETWEEN_CALLS = 5     # 無料枠 RPM 対策
N_POINTS = 4                # 要点の数

load_dotenv(os.path.join(ROOT, ".env"))
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
if not GENAI_API_KEY:
    sys.exit("GENAI_API_KEY を .env に設定してください")


def load_articles():
    """blog_data.json を読む（制御文字が混ざっているので server.py と同じ処理で落とす）。"""
    with open(BLOG_PATH, "r", encoding="utf-8") as fl:
        raw = fl.read()
    return json.loads(re.sub(r"[\x00-\x1F\x7F-\x9F]", "", raw))


def load_targets():
    """銀河に出ている記事だけを対象にする（URLで blog_data.json と突き合わせる）。"""
    with open(STARS_PATH, "r", encoding="utf-8") as fl:
        stars = json.load(fl)["stars"]
    by_url = {a["url"]: a for a in load_articles()}
    targets, missing = [], []
    for s in stars:
        art = by_url.get(s["url"])
        if art:
            targets.append({"url": s["url"], "title": s["title"], "content": art.get("content", "")})
        else:
            missing.append(s["url"])
    if missing:
        print(f"警告: 本文が見つからない記事 {len(missing)} 件（例: {missing[:3]}）")
    return targets


def parse_json(text):
    """JSON応答を読む。応答の末尾に余分な文字が付くことがあるので、
    先頭の JSON 値だけを取り出す（実際に "Extra data" で中断した）。"""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        obj, _ = json.JSONDecoder().raw_decode(text)
        return obj


def call_gemini_json(client, prompt):
    """JSON応答を取得（400系は即中断、429/5xxのみ再試行）。build_galaxy.py と同じ方針。"""
    for attempt in (1, 2, 3):
        try:
            resp = client.models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
                config={"response_mime_type": "application/json", "max_output_tokens": 8192},
            )
            return parse_json(resp.text)
        except json.JSONDecodeError as e:
            if attempt < 3:
                print("  応答JSON不正、再生成:", str(e)[:80])
                time.sleep(SLEEP_BETWEEN_CALLS)
                continue
            raise
        except Exception as e:
            msg = str(e)
            if attempt < 3 and ("429" in msg or "500" in msg or "503" in msg):
                print("  リトライ待機30秒:", msg[:100])
                time.sleep(30)
                continue
            raise


def build_prompt(batch):
    items = [
        {"n": i, "title": b["title"], "body": b["content"][:CONTENT_TRUNC]}
        for i, b in enumerate(batch)
    ]
    return (
        "以下のブログ記事それぞれについて、はじめて読む人にも分かる日本語の要約を作ってください。\n"
        "記事の世界観（宇宙論・生命論）の用語はそのまま使って構いません。\n"
        '"summary" は記事全体の要旨を2〜3文で。挨拶・前置き・感想・締めの言葉は書かないでください。\n'
        f'"points" は記事の要点を{N_POINTS}つ、それぞれ1文で。行頭の記号（・や*）は付けないでください。\n'
        '出力はJSON配列: [{"n": 番号, "summary": "…", "points": ["…", "…"]}]\n\n'
        + json.dumps(items, ensure_ascii=False)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="先頭N件だけ生成（試し打ち用）")
    args = ap.parse_args()

    targets = load_targets()
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as fl:
            cache = json.load(fl)

    todo = [t for t in targets if t["url"] not in cache]
    if args.limit:
        todo = todo[: args.limit]
    print(f"対象 {len(targets)} 件 / キャッシュ済み {len(cache)} 件 / 生成 {len(todo)} 件")
    print(f"想定リクエスト数: {(len(todo) + BATCH - 1) // BATCH} 回")

    client = genai.Client(api_key=GENAI_API_KEY)
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        result = call_gemini_json(client, build_prompt(batch))
        for row in result:
            t = batch[int(row["n"])]
            points = [str(p).strip().lstrip("・*-").strip() for p in (row.get("points") or [])]
            cache[t["url"]] = {
                "s": str(row.get("summary", "")).strip(),
                "p": [p for p in points if p],
            }
        with open(CACHE_PATH, "w", encoding="utf-8") as fl:
            json.dump(cache, fl, ensure_ascii=False, indent=1)
        print(f"  {min(i + BATCH, len(todo))}/{len(todo)} 件完了")
        time.sleep(SLEEP_BETWEEN_CALLS)

    # 銀河に出ている記事ぶんだけを配信用に書き出す
    out = {t["url"]: cache[t["url"]] for t in targets if t["url"] in cache}
    with open(OUT_PATH, "w", encoding="utf-8") as fl:
        json.dump(out, fl, ensure_ascii=False, separators=(",", ":"))
    size = os.path.getsize(OUT_PATH) / 1024
    print(f"書き出し: {OUT_PATH}  {len(out)}/{len(targets)} 件  {size:.0f} KB")


if __name__ == "__main__":
    main()
