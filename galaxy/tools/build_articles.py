# -*- coding: utf-8 -*-
"""記事全文を Vercel 関数（galaxy/api/ask.js）へ同梱する形に書き出す。

回答の文脈には記事の全文が要るが、Pinecone のメタデータには本文が入っていないため、
server.py は blog_data.json を丸ごとメモリに載せて URL 引きしていた。関数側でも同じ
ことをするので、URL をキーにした辞書にして gzip で置く。

- 生の blog_data.json は 8.3MB。gzip なら日本語がよく縮むのでリポジトリが太らない
- 出力先を api/ 配下にするのは意図的。Vercel は api/ 配下を静的配信しないので、
  記事全文が公開URLで丸ごと落とせる状態にはならない

出力: galaxy/api/_data/articles.json.gz  {"<記事URL>": {"t": タイトル, "c": 本文}}

使い方:
    python galaxy/tools/build_articles.py
"""
import gzip
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLOG_PATH = os.path.join(ROOT, "blog_data.json")   # 実データ。読み取り専用
OUT_DIR = os.path.join(ROOT, "galaxy", "api", "_data")
OUT_PATH = os.path.join(OUT_DIR, "articles.json.gz")


def load_articles():
    """blog_data.json を読む（制御文字が混ざっているので server.py と同じ処理で落とす）。"""
    with open(BLOG_PATH, "r", encoding="utf-8") as fl:
        raw = fl.read()
    return json.loads(re.sub(r"[\x00-\x1F\x7F-\x9F]", "", raw))


def main():
    articles = load_articles()
    table = {}
    for a in articles:
        url = a.get("url", "")
        if not url:
            continue
        table[url] = {"t": a.get("title", ""), "c": a.get("content", "")}

    os.makedirs(OUT_DIR, exist_ok=True)
    body = json.dumps(table, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(OUT_PATH, "wb", compresslevel=9) as fl:
        fl.write(body)

    print("記事数      :", len(table))
    print("非圧縮      : %.2f MB" % (len(body) / 1024 / 1024))
    print("gzip 後     : %.2f MB" % (os.path.getsize(OUT_PATH) / 1024 / 1024))
    print("出力        :", OUT_PATH)
    if len(table) != len(articles):
        print("※ URL の無い記事を %d 件飛ばしました" % (len(articles) - len(table)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
