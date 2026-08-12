# -*- coding: utf-8 -*-
"""クイズプール（riat-quiz-frontend）を星＝記事に紐付けたデータを生成する。

各問題は source_urls（出典記事URL）を持つので、それを galaxy_data.json の star.url と
突き合わせれば「この記事から出た問題」を正確に引ける。直接ヒットしない星のために
星座（cluster）単位の在庫も同時に作る。

  入力（読み取りのみ）:
    riat-quiz-frontend/public/quiz_pool.json … 実データ。絶対に書き換えない
    galaxy/galaxy_data.json
  出力:
    galaxy/quiz_by_star.json

実行: python galaxy/tools/build_quiz_link.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GALAXY_DIR = os.path.join(ROOT, "galaxy")
POOL_PATH = os.path.join(os.path.dirname(ROOT), "riat-quiz-frontend", "public", "quiz_pool.json")
GALAXY_PATH = os.path.join(GALAXY_DIR, "galaxy_data.json")
OUT_PATH = os.path.join(GALAXY_DIR, "quiz_by_star.json")


def norm_url(u):
    return (u or "").strip().rstrip("/") + "/"


def main():
    if not os.path.exists(POOL_PATH):
        sys.exit(f"クイズプールが見つかりません: {POOL_PATH}")

    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    with open(GALAXY_PATH, encoding="utf-8") as f:
        galaxy = json.load(f)

    # 同じ問題が複数カテゴリ・複数記事に現れうるので、本文をキーに重複排除して index 参照にする
    questions = []
    seen = {}
    by_url = {}

    for arr in pool.values():
        for item in arr:
            key = item["question"]
            idx = seen.get(key)
            if idx is None:
                idx = len(questions)
                seen[key] = idx
                titles = item.get("source_titles") or []
                urls = item.get("source_urls") or []
                questions.append({
                    "q": item["question"],
                    "c": item["choices"],
                    "a": item["answer_index"],
                    "e": item.get("explanation", ""),
                    "st": titles[0] if titles else "",
                    "su": urls[0] if urls else "",
                })
            for u in item.get("source_urls") or []:
                by_url.setdefault(norm_url(u), []).append(idx)

    for lst in by_url.values():
        lst[:] = sorted(set(lst))

    # 星座単位の在庫（直接ヒットしない星のフォールバック用）
    by_cluster = {}
    direct_hits = 0
    for s in galaxy["stars"]:
        hits = by_url.get(norm_url(s["url"]))
        if not hits:
            continue
        direct_hits += 1
        by_cluster.setdefault(str(s["cluster"]), []).extend(hits)
    for lst in by_cluster.values():
        lst[:] = sorted(set(lst))

    # 星が存在しない記事URLは配信しても使えないので落とす
    star_urls = {norm_url(s["url"]) for s in galaxy["stars"]}
    by_url = {u: v for u, v in by_url.items() if u in star_urls}

    out = {"questions": questions, "byUrl": by_url, "byCluster": by_cluster}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    total_stars = len(galaxy["stars"])
    print(f"問題数: {len(questions)}（重複排除前 {sum(len(a) for a in pool.values())}）")
    print(f"直接ヒットした星: {direct_hits} / {total_stars}")
    missing = [c["id"] for c in galaxy["constellations"] if str(c["id"]) not in by_cluster]
    print("フォールバック在庫なしの星座:", missing if missing else "なし")
    print(f"出力: {OUT_PATH}  {os.path.getsize(OUT_PATH)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
