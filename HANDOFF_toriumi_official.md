# 引き継ぎ指示書 — 鳥海勝稚 オフィシャルサイト（toriumi-official）

最終更新: 2026-08-07

---

## 1. 概要

- **名称**: Katsunori Toriumi（**鳥海 勝稚** ※「勝典」ではない）オフィシャルサイト
- **コンセプト**: 「Tricky Multi-Creator ／ 銀河の海の渡り鳥」。
  **スクロールすると宇宙の奥へ進み、旅の途中に各セクションが現れる 1 ページ体験。**
- **本番URL**: https://toriumi-official.vercel.app （公開中・自動デプロイ）
- **場所**: `C:\Users\katsu\RIATブログ辞典\toriumi-official`（親リポ `riat-blog-jiten` 内）

## 2. 技術スタック

Next.js 16（App Router / Turbopack）+ React 19 + Tailwind v4 + framer-motion + lenis
＋ **React Three Fiber 9 / drei 10 / three r185**（ホームのみ）

`next.config.ts` は `output: "export"`（完全静的）／`images.unoptimized: true`／`trailingSlash: true`。
サーバー機能（API routes / server actions / next/image 最適化）は使えない。画像は `<img>`。

## 3. ローカル起動・デプロイ

```bash
cd toriumi-official && npm run dev   # http://localhost:4444
```

- preview ツール名は `toriumi-official`（`.claude/launch.json`、ポート 4444）
- **main に push → Vercel が自動ビルド＆本番反映（約30秒）**。Root Directory = `toriumi-official`
- 確認: `npx vercel ls toriumi-official --scope katsunoritoriumi-2409s-projects`

### ⚠️ push が通らないとき

素の `git push` は Git Credential Manager が対話プロンプトを開けず固まる。**URL にユーザー名を入れる**:

```bash
git push https://katsunoritoriumi-stack@github.com/katsunoritoriumi-stack/riat-blog-jiten.git main
```

`fetch` は認証不要なので「fetch は通るのに push だけ落ちる」という症状になる。**force push は禁止。**

### ⚠️ package.json の罠

リポジトリ root の `.gitignore` が `package.json` / `package-lock.json` を**グローバルに無視**している
（元の Flask プロジェクト用）。toriumi-official は `toriumi-official/.gitignore` に
`!package.json` `!package-lock.json` を入れて解決済み。**新しい Node プロジェクトを足すときは `git add -f` が必要。**

---

## 4. ホームの構造（ここが一番大事）

### 旅程は `lib/stations.ts` が唯一の出典

各ステーションの `id` / `scroll`（画面高さの倍数）/ 予告編コピーをここに置いている。
**見せ方（`components/ui/ZoomStage.tsx`）と宇宙（`components/Universe.tsx`）が同じ配分を読む。**
片方だけ変えると「セクションが出る位置」と「宇宙で何かが起きる位置」がずれる。

順番: home → manifesto → **universe（創造の座標軸）** → sound → digital → connect → report → end

### 宇宙は 1 つの永続 R3F シーン

`components/Universe.tsx` → `components/universe/Scene.tsx`。スクロール量がそのままカメラの前進距離。

| ファイル | 役割 |
|---|---|
| `universe/starfield.ts` | 無限に流れる星の回廊（頂点シェーダで再配置。CPU 負荷ゼロ） |
| `universe/Scene.tsx` | 天の川・星雲・ライト・スクロール駆動（Driver） |
| `universe/Flybys.tsx` | 航行区間ですれ違う天体、到着の予兆 |
| `universe/SolarSystem.tsx` | 創造の座標軸の太陽系 |
| `lib/planetTextures.ts` / `lib/spaceTextures.ts` | テクスチャの手続き生成（外部画像ファイルは持たない） |

`components/ConstellationMap.tsx` は**見出しとフォールバックだけの薄い DOM 層**。
太陽系の実体は宇宙側にある。クリックは背後のキャンバスへ通す（`pointer-events: none`）。

### 太陽系の見せ方（2026-08-07 に調整）

- **拡大率と見下ろす角度を画面の縦横比から毎回計算する**（`viewFor()`）。
  縦長（スマホ）ほど縮めて深く見下ろす。横に広い画面では上限 9 で頭打ち＝PC の見え方は従来どおり。
- **滞在中は 6 惑星を 60 度ずつのスロットへ整列させる**（`slot`）。速度がばらばらのままだと
  たまたま同じ方向に並んだ惑星が画面上で重なるため。割り当ては全 720 通り × 全回転角の
  総当たりで選んであり、最悪でも 0.64 単位の余白が残る。系全体はゆっくり回り続ける。
- **太陽系の近くでは通過天体を出さない**（`Flybys.tsx` の `systemBusy()`）。飛来天体は
  26〜88 単位のまま近くを通るので、縮んだ太陽系より大きく写ってしまう。
- 太陽の光輪は最内周の軌道より小さくし、深度判定を有効にしてある（惑星が飲み込まれないように）。

---

## 5. 表示速度（2026-08-07 に大幅改善）

**初期読み込み 17.9MB → 約1.0MB。**

| 項目 | before | after |
|---|---|---|
| `public/cosmos-loop.mp4` | 6,462,951 B（7.6Mbps） | 817,240 B（SSIM 0.974） |
| `public/hero-space.mp4` | 11,152,881 B（1080p/11Mbps） | 2,330,254 B（720p / SSIM 0.972） |
| hero-space の読み込み開始 | 222ms | 3,113ms（ブート明け） |
| `/` の初期チャンク合計 | 1,787,150 B | 874,076 B |
| MV ポスター(261KB) | トップ表示時に取得 | セクション表示まで取得しない |

- `components/UniverseGate.tsx` が three.js をブート明けにマウントする。
  待っている間は `universe/StaticSky.tsx`（CSS の静止星空）。**見え方は変わらない。**
- **踏んだ罠 ①**: `<video poster>` は `preload="none"` でも画面外でも**必ず**読み込まれる。
- **踏んだ罠 ②**: `display:none` の中では `loading="lazy"` が効かない（Chrome は即読み込む）。
  → 遅延させたい画像は「表示されるまで DOM に出さない」しかない。

---

## 6. SEO（2026-08-07 に一式導入）

- **公開URLの唯一の出典は `lib/site.ts`**。独自ドメインへ移すときは Vercel に
  環境変数 `NEXT_PUBLIC_SITE_URL` を入れるだけで、canonical・sitemap・robots・OG が全部切り替わる。
- `app/robots.ts` → `out/robots.txt` ／ `app/sitemap.ts` → `out/sitemap.xml`
- `app/icon.png`（512）／`app/apple-icon.png`（180）／`app/opengraph-image.jpg`（1200×630）
- **構造化データは `lib/jsonLd.ts` に集約**（`components/JsonLd.tsx` が埋め込む）
  - 全ページ: `Person`（**sameAs で SNS・ショップ 7 件を同一人物として束ねる**）／`WebSite`／
    `ProfessionalService`（サービス 6 件 ＋ `areaServed`）
  - ホーム: `WebPage` ＋ `VideoObject`（MV「星の彼方へ」）
  - /apps・/websites: `CollectionPage` ／ `BreadcrumbList` ／ `ItemList`
- 検証スクリプトは `out/` の HTML を node で読んで JSON.parse・必須項目・canonical・
  og:image・BOM 混入まで 3 ページぶん確認する方式（scratchpad の `checkld.mjs`）。

### ユーザー側の残作業

1. **Google Search Console に登録し sitemap.xml を送信**（所有権確認の meta タグが要るなら埋め込む）
2. **Google ビジネスプロフィールの作成**（地域で受注を狙うならサイトより先に効く）
3. 各 SNS のプロフィール「ウェブサイト」欄を本番URLに統一（sameAs は双方向で強くなる）
4. JSON-LD に**事業者名・対応エリア・住所を出すか**の確認（住所は実在のものが無いと入れられない）

### 効果が頭打ちになったら

「アプリ開発 諏訪」のようなサービス＋地域の語で上位を取るには、本来は**専用ページ**が要る。
今回はページ追加なしの範囲。次の一手は `/service/app` `/service/web` などの新設。

---

## 7. コンテンツの一元管理 → `lib/content.ts`

`SITE` / `LINKS` / `YOUTUBE` / `DOMAINS`（座標軸の 6 領域）/ `APPS` / `WEBSITES` / `WORKS` /
`MANIFESTO` / `ROLES` / `NIRAV_ITEMS`。文言・リンクはほぼ全部ここ。

**⚠️ `WORKS` と `APPS` の Voyage のリンクが `voyage-ai-travel-planner-8gw80ve11.vercel.app`
（ハッシュ付きの不変URL）になっている。これは旧版が永久表示される既知の罠で、正しくはスコープ付きURL。要修正。**

---

## 8. 検証のしかた（この環境固有）

Browser ペインが表示されていないとページがフレームを合成せず、
rAF / ResizeObserver / IntersectionObserver が全停止する。スクリーンショットも撮れない。

- **`window.dispatchEvent(new Event('resize'))` を JS で叩くとキャンバスは起きる**（実測）。
  `resize_window` ツールだけでは足りない。ここが分かっていないと延々ハマる。
- 起こしたあと `window.__universeSnapshot(prog)`（**開発ビルドのみ**）で任意のスクロール位置を
  1 フレーム手動描画して `toDataURL()` が取れる。ローカルの sink に POST して PNG にすると目視できる。
- IntersectionObserver は起きないままなので、IO 依存の遅延マウントは別の手段で確かめる。
- 数式・テクスチャ生成は node で直接動かして検証できる（DOM 非依存に書いてある理由）。

## 9. その他の既知の罠

- **lucide-react v1.21** はブランドアイコン（Youtube/Facebook/Instagram）を含まない → 代替アイコンを使う
- SVG で `Math.cos/sin` の座標は**丸める**（丸めないと SSR ハイドレーション不一致）
- `three.js` のライトは**カメラのレイヤーでしか絞れない**（オブジェクト単位で当てるライトを分けられない）
- `RingGeometry` の UV は平面展開なので、環のテクスチャを半径方向に貼るには
  UV を書き換える必要がある（`universe/ringGeometry.ts`）
