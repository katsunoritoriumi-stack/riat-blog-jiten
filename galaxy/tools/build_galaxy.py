# -*- coding: utf-8 -*-
"""RIATブログ・ギャラクシー データ生成パイプライン（1回きりのローカルバッチ）

blog_data.json（読み取りのみ）と Pinecone の登録済みベクトルから、
フロントが読む galaxy/galaxy_data.json を生成する。

  1. blog_data.json 読み込み（server.py と同じ制御文字クリーニング）
  2. Pinecone "seimeiron-blog" から 768次元ベクトルを全件 fetch
  3. KMeans でテーマクラスタに分割 → 対数螺旋（渦巻き銀河）の座標に配置
     （クラスタ＝腕上の半径帯、クラスタ内の並びは PCA 1次元投影で近い記事同士を隣接させる）
  4. Gemini で星座名（クラスタ名。実カテゴリー語彙をヒントに直感的な名前にする）と
     各記事の1行要約を生成（要約は summaries_cache.json にキャッシュし、再実行時は API を叩かない）
  5. galaxy_data.json を UTF-8 で出力

実行: venv/Scripts/python.exe galaxy/tools/build_galaxy.py
"""
import json
import os
import re
import sys
import time

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GALAXY_DIR = os.path.join(ROOT, "galaxy")
CACHE_PATH = os.path.join(GALAXY_DIR, "tools", "summaries_cache.json")
OUT_PATH = os.path.join(GALAXY_DIR, "galaxy_data.json")

GEN_MODEL = "gemini-3.1-flash-lite"  # 無料枠 RPD500（server.py と同じ）
N_CLUSTERS = 12
SUMMARY_BATCH = 10          # 1リクエストあたりの記事数（大きくすると応答JSONが出力上限で切れる）
SLEEP_BETWEEN_CALLS = 5     # 無料枠 RPM 対策
CONTENT_TRUNC = 1500        # 要約入力に使う本文の先頭文字数

# --- 渦巻き銀河の形状パラメータ ---
N_ARMS = 5                  # 腕の本数
ARM_TIGHTNESS = 0.26        # 対数螺旋の巻きつき具合（小さいほどきつく巻く）
CORE_RADIUS = 9.0           # 中心核（バルジ）の半径
GALAXY_RADIUS = 95.0        # 銀河全体の半径
ARM_SCATTER_ANGLE = 0.10    # 腕の太さ（角度方向のばらつき, ラジアン）
ARM_SCATTER_RADIUS = 2.6    # 腕の太さ（半径方向のばらつき）

# 実際のRIATブログのカテゴリー語彙（サイトのカテゴリー一覧から）。
# 星座名がこれらの実カテゴリーと直感的に結びつくようヒントとして渡す
KNOWN_CATEGORIES = [
    "創造主世界", "医学", "気学", "生命構造", "神界関連",
    "科学", "銀河史", "陰陽論", "龍神・聖地民族",
]

load_dotenv(os.path.join(ROOT, ".env"))
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not GENAI_API_KEY or not PINECONE_API_KEY:
    sys.exit("GENAI_API_KEY / PINECONE_API_KEY を .env に設定してください")


def load_articles():
    with open(os.path.join(ROOT, "blog_data.json"), "r", encoding="utf-8") as fl:
        content = fl.read()
    clean = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", content)
    articles = json.loads(clean)
    print(f"記事 {len(articles)} 件を読み込み")
    return articles


def fetch_vectors():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("seimeiron-blog")
    all_ids = []
    for page in index.list(limit=99):
        all_ids.extend(page)
    print(f"Pinecone ID {len(all_ids)} 件")
    vectors = {}
    for i in range(0, len(all_ids), 100):
        res = index.fetch(ids=all_ids[i : i + 100])
        for vid, v in res.vectors.items():
            vectors[vid] = v.values
    print(f"ベクトル {len(vectors)} 件を取得")
    return vectors


def assign_spiral_positions(labels, mat):
    """クラスタ（KMeans）と対数螺旋の腕を組み合わせて銀河状に配置する。

    各クラスタを1本の腕（N_ARMS本のいずれか）に割り当て、その腕の中での
    順番（内側〜外側）に応じて半径帯を決める。同じ腕に属するクラスタは
    中心核から銀河の縁まで連続してつながるため、腕全体が1本の流れる帯として
    見える（＝クラスタが飛び飛びの塊にならない）。
    クラスタ内部は PCA で1次元に投影した順序で腕の中に並べるため、
    似た内容の記事同士が腕の中で隣接する（意味的な近さが局所的に保たれる）。
    """
    rng = np.random.default_rng(42)
    n = len(labels)
    positions = np.zeros((n, 3), dtype=np.float32)

    cluster_ids = sorted(set(int(lb) for lb in labels))
    members = {cid: [] for cid in cluster_ids}
    for i, lb in enumerate(labels):
        members[int(lb)].append(i)

    # 各クラスタを腕に割り振り、腕ごとに「その腕の中で何番目か」を記録する。
    # これにより同じ腕のクラスタは半径方向に連続し、腕全体が中心核から
    # 銀河の縁まで途切れず伸びる1本の帯になる。
    arm_of = {}
    band_in_arm = {}
    arm_counts = [0] * N_ARMS
    for i, cid in enumerate(cluster_ids):
        arm = i % N_ARMS
        arm_of[cid] = arm
        band_in_arm[cid] = arm_counts[arm]
        arm_counts[arm] += 1

    for cid in cluster_ids:
        idxs = members[cid]
        m = len(idxs)
        arm = arm_of[cid]
        arm_offset = arm * (2 * np.pi / N_ARMS)
        band, n_bands = band_in_arm[cid], arm_counts[arm]
        b0, b1 = band / n_bands, (band + 1) / n_bands
        r0 = CORE_RADIUS + b0 * (GALAXY_RADIUS - CORE_RADIUS)
        r1 = CORE_RADIUS + b1 * (GALAXY_RADIUS - CORE_RADIUS)

        if m >= 3:
            proj = PCA(n_components=1, random_state=42).fit_transform(mat[idxs])[:, 0]
            order = np.argsort(proj)
        else:
            order = np.arange(m)

        for rank, oi in enumerate(order):
            i = idxs[oi]
            t = rank / max(1, m - 1)
            r = r0 + t * (r1 - r0) + rng.normal(0, ARM_SCATTER_RADIUS)
            r = max(CORE_RADIUS * 0.55, r)
            theta = arm_offset + np.log(r / CORE_RADIUS) / ARM_TIGHTNESS
            theta += rng.normal(0, ARM_SCATTER_ANGLE)
            thickness = 5.5 * np.exp(-r / 42.0) + 0.9
            y = rng.normal(0, thickness)
            positions[i] = [r * np.cos(theta), y, r * np.sin(theta)]

    return positions


def call_gemini_json(client, prompt):
    """JSON応答を1回リトライ付きで取得（400系は即中断、429/5xxのみ再試行）"""
    for attempt in (1, 2, 3):
        try:
            resp = client.models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "max_output_tokens": 8192,
                },
            )
            return json.loads(resp.text)
        except json.JSONDecodeError as e:
            # 応答が途中で切れた場合など。生成し直せば直ることが多い
            if attempt < 3:
                print("  応答JSON不正、再生成:", str(e)[:80])
                time.sleep(SLEEP_BETWEEN_CALLS)
                continue
            raise
        except Exception as e:
            msg = str(e)
            retryable = "429" in msg or "500" in msg or "503" in msg
            if attempt < 3 and retryable:
                print("  リトライ待機30秒:", msg[:100])
                time.sleep(30)
                continue
            raise


def gen_summaries(client, stars):
    """各記事の1行要約（40〜60字）を生成。キャッシュ済みはスキップ。"""
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as fl:
            cache = json.load(fl)
    todo = [s for s in stars if s["url"] not in cache]
    print(f"要約: キャッシュ {len(cache)} 件 / 生成対象 {len(todo)} 件")
    for i in range(0, len(todo), SUMMARY_BATCH):
        batch = todo[i : i + SUMMARY_BATCH]
        items = [
            {"n": j, "title": s["title"], "body": s["content"][:CONTENT_TRUNC]}
            for j, s in enumerate(batch)
        ]
        prompt = (
            "以下のブログ記事それぞれについて、内容の核心を伝える日本語の要約を"
            "60〜90字で書いてください。記事の世界観（宇宙論・生命論）の用語は"
            "そのまま使って構いません。\n"
            '出力はJSON配列: [{"n": 番号, "summary": "要約"}]\n\n'
            + json.dumps(items, ensure_ascii=False)
        )
        result = call_gemini_json(client, prompt)
        for row in result:
            s = batch[int(row["n"])]
            cache[s["url"]] = str(row["summary"]).strip()
        with open(CACHE_PATH, "w", encoding="utf-8") as fl:
            json.dump(cache, fl, ensure_ascii=False, indent=1)
        done = min(i + SUMMARY_BATCH, len(todo))
        print(f"  要約 {done}/{len(todo)} 件完了")
        time.sleep(SLEEP_BETWEEN_CALLS)
    return cache


def gen_constellations(client, stars, labels):
    """クラスタごとに星座名と一言説明を生成。

    初見の読者でも中身が直感的にわかる名前にするため、RIATブログの実際の
    カテゴリー語彙（医学・気学・生命構造…）をヒントとして渡し、
    抽象的すぎる造語ではなく「テーマが一目で伝わる」命名を優先させる。
    """
    clusters = {}
    for s, lb in zip(stars, labels):
        clusters.setdefault(int(lb), []).append(s["title"])
    payload = [
        {"cluster": cid, "titles": titles[:25]}
        for cid, titles in sorted(clusters.items())
    ]
    prompt = (
        "ブログ記事群をテーマごとの星団に分けました。各星団に、宇宙風の日本語名"
        "（〜銀河・〜星雲・〜星団・〜座、6〜10字）と、テーマの一言説明（30字以内）を"
        "付けてください。\n"
        "最優先事項: 初めてこのサイトを見る人が名前を見ただけでジャンルを直感的に"
        "理解できること。曖昧・抽象的すぎる名前（例:「原理星雲」「心磁星団」）は避け、"
        "具体的な話題（医学・健康、気学、生命の仕組み、神界・霊的存在、科学・物理、"
        "銀河や宇宙の歴史、陰陽論、龍神や聖地・民族の伝承、創造主・文明の起源、"
        "感染症・災害、放射線・エネルギーなど）を名前の中に必ず含めてください。\n"
        f"参考: このブログの実際のカテゴリーは {', '.join(KNOWN_CATEGORIES)} です。"
        "できるだけこれらの言葉、またはその言い換えを名前に取り込んでください。\n"
        '出力はJSON配列: [{"cluster": 番号, "name": "星座名", "desc": "説明"}]\n\n'
        + json.dumps(payload, ensure_ascii=False)
    )
    result = call_gemini_json(client, prompt)
    out = {int(r["cluster"]): {"name": r["name"], "desc": r["desc"]} for r in result}
    for cid in clusters:
        out.setdefault(cid, {"name": f"第{cid + 1}星団", "desc": ""})
    return out


def main():
    articles = load_articles()
    vectors = fetch_vectors()

    # ベクトルがある記事だけを星にする（順序は記事番号順 = blog_data.json の逆順対策で URL 昇順）
    stars = [a for a in articles if a["url"] in vectors]
    stars.sort(key=lambda a: a["url"])
    print(f"星になる記事: {len(stars)} 件")

    mat = np.array([vectors[s["url"]] for s in stars], dtype=np.float32)

    print(f"KMeans で {N_CLUSTERS} クラスタに分割中...")
    labels = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit_predict(mat)

    print(f"対数螺旋（腕 {N_ARMS} 本）に配置中...")
    pos3d = assign_spiral_positions(labels, mat)

    client = genai.Client(api_key=GENAI_API_KEY)
    summaries = gen_summaries(client, stars)
    constellations = gen_constellations(client, stars, labels)

    lengths = np.array([len(s["content"]) for s in stars], dtype=np.float32)
    sizes = 0.6 + 1.4 * (lengths - lengths.min()) / max(1.0, lengths.max() - lengths.min())

    def article_no(title):
        m = re.match(r"【(\d+)】", title)
        return int(m.group(1)) if m else None

    data = {
        "generated": time.strftime("%Y-%m-%d"),
        "constellations": [
            {"id": cid, "name": c["name"], "desc": c["desc"]}
            for cid, c in sorted(constellations.items())
        ],
        "stars": [
            {
                "no": article_no(s["title"]),
                "title": s["title"],
                "url": s["url"],
                "summary": summaries.get(s["url"], ""),
                "pos": [round(float(x), 2) for x in pos3d[i]],
                "cluster": int(labels[i]),
                "size": round(float(sizes[i]), 2),
            }
            for i, s in enumerate(stars)
        ],
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fl:
        json.dump(data, fl, ensure_ascii=False, separators=(",", ":"))
    print(f"出力完了: {OUT_PATH}（星 {len(data['stars'])} / 星座 {len(data['constellations'])}）")


if __name__ == "__main__":
    main()
