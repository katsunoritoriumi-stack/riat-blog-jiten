# 引き継ぎ指示書 — 鳥海勝稚 オフィシャルサイト（toriumi-official）

このドキュメントは、新しいチャットでこのプロジェクトの作業を続けるための指示書です。
まずこの内容を共有すれば、担当が変わっても状況を把握して続行できます。

---

## 1. プロジェクト概要
- **名称**: Katsunori Toriumi（**鳥海 勝稚** ※「勝典」ではない）オフィシャルサイト
- **コンセプト**: 「Tricky Multi-Creator ／ 銀河の海の渡り鳥」。宇宙的・スタイリッシュ。回転する銀河を全画面の永続背景に。
- **本番URL**: https://toriumi-official.vercel.app （公開中・自動デプロイ稼働中）
- **場所**: `C:\Users\katsu\RIATブログ辞典\toriumi-official`（親リポジトリ `riat-blog-jiten` 内のサブフォルダ）

## 2. 技術スタック
- Next.js 16（App Router / Turbopack）+ React 19 + Tailwind CSS v4
- framer-motion（アニメ）/ lenis（スムーススクロール）/ lucide-react（アイコン）
- **静的エクスポート構成**：`next.config.ts` に `output: "export"`、`images.unoptimized: true`
- フォント：ディスプレイ=**Sora**、ラベル=**Space Mono**、和文=Noto Sans JP / Noto Serif JP

## 3. ローカル起動
```
cd toriumi-official
npm install   # 初回のみ
npm run dev   # → http://localhost:4444
npm run build # 静的ビルド（out/ を生成）
```

## 4. デプロイ（自動）
- Vercelプロジェクト `toriumi-official`（team: `katsunoritoriumi-2409s-projects`）が GitHub `riat-blog-jiten` と連携済み。
- **main に push すると自動ビルド＆本番反映（約18〜20秒）**。
- 設定：Vercel の **Root Directory = `toriumi-official`**、リポジトリ root に `vercel.json`（`framework:null` / `buildCommand:"npm run build"` / `outputDirectory:"out"`）。
- **更新フロー**：ファイル編集 → `git add <files>` → `git commit` → `git push origin main`。
  - ※ push はときどき遅く2分でタイムアウトすることがある。その場合は `git log origin/main..main` で未pushを確認し、`git push origin main` を再実行。
- デプロイ確認：`cd toriumi-official && vercel ls toriumi-official`（先頭行の Production が `● Ready` なら反映済み）。

### ⚠️ 最重要の落とし穴（必読）
リポジトリ root の `.gitignore` が **`package.json` と `package-lock.json` をグローバルに無視**している（元の Flask プロジェクト用）。
- そのため新規Nodeプロジェクトの package.json は**普通の `git add` では追跡されない**。`git add -f` が必要。
- toriumi-official は `toriumi-official/.gitignore` に `!package.json` `!package-lock.json` を入れて解決済み。**このリポジトリで新しいNodeプロジェクトを足すときは要注意**。

## 5. コンテンツの一元管理 → `lib/content.ts`
文言・リンク・作品リストはほぼ全部ここ。ここを編集すれば全セクションに反映。
- `SITE`：氏名（鳥海 勝稚）、tagline（銀河の海の渡り鳥）、roleEn
- `LINKS`：SNS/ショップURL（facebook, instagram, youtube, line, baseQuantum=katsunoritor.thebase.in, baseToriumi=toriumi.thebase.in, produce=10-xi-teal, nirav）
- `YOUTUBE.thumbnail`：Sound & Vision に表示する Exodus チャンネルアート画像URL（yt3.googleusercontent.com）
- `DOMAINS`：星座マップ（創造の座標軸）の各カテゴリー。`href`（単一リンク）または `links[]`（複数リンク＝ポップオーバー）
- `WORKS`：Collection のプロジェクトカード（title/type/note/href/hue）
- `MANIFESTO`：宣言文（末尾に「— Shirankedo…」を別要素で表示）
- `SECTIONS`：ナビ項目

## 6. ページ構成（`app/page.tsx`）
`GalaxyBackground`（固定の回転銀河背景）→ `Navbar` → `Hero` → `Manifesto` → `ConstellationMap` → `SoundVisionSection` → `DigitalAISection` → `WorkSection` → `ConnectSection` → `Footer`
- **ナビ**：Home / Universe / Sound & Vision / Work / Connect
- **Hero**：Sora の「Katsunori（太）/ Toriumi（極細）」+ 銀河背景
- **Manifesto**：宣言文 + 「— Shirankedo…」
- **ConstellationMap（創造の座標軸）**：中心 **Connect→LINE**。外周6カテゴリー＝各星がリンク：
  - Art → toriumi.thebase.in
  - Youtube → Exodus チャンネル
  - Fashion →（クリックでポップオーバー）Original T-shirts=katsunoritor.thebase.in ／ Produced Brand=nirav.base.shop
  - Produce → 10-xi-teal.vercel.app
  - SNS →（ポップオーバー）Instagram ／ Facebook
  - Work → #work
  - ※複数リンクの星はクリックで選択メニュー。下半分(y>55)の星は**上向き**に開く（モバイル見切れ対策）。
- **SoundVisionSection（EXODUS）**：Exodus チャンネルアートのフレーム + 再生ボタン → YouTube
- **DigitalAISection**：eyebrow「Job request」/ 見出し「Make」/ カード3枚（アプリ開発・Web制作・動画制作）
- **WorkSection（Works / eyebrow: Collection）**：8件のプロジェクトカード。各カードに**mShotsのサイトスクショ**を表示
- **ConnectSection**：LINE / Instagram / YouTube / Facebook ＋ ショップ3枚（Original T-shirts=katsunoritor / Original Art=toriumi / Produced Brand=nirav）

## 7. Works（Collection）の8件
1. RIAT Quiz — https://riat-quiz-frontend.vercel.app/
2. Voyage（AI Travel）— https://voyage-ai-travel-planner-8gw80ve11.vercel.app/
3. 献立マスター — https://kondate-master.vercel.app/
4. 数秘計算 — https://suuhi-keisan-v2.vercel.app/
5. タネと菜園 — https://tanetosaien.vercel.app/
6. Dragon Shrine Checker — https://dragon-shrine-checker.vercel.app/
7. 静かな電子銀河 — https://shizuka-na-denshi-ginga.vercel.app/
8. RIAT ブログ事典（AI Search）— https://riat-blog-jiten-2.onrender.com/
- スクショは **WordPress mShots**：`https://s.wordpress.com/mshots/v1/{URLエンコード}?w=800&h=600`。初回は「Generating Preview…」→数十秒で実画像に差し替わる（サーバ側キャッシュ）。

## 8. コーディング上の注意（既知の罠）
- **静的エクスポート**なのでサーバー機能（API routes / server actions / next/image最適化）は不可。外部画像は `<img>` を使う（next/image は使わない）。
- **lucide-react v1.21** はブランドアイコン（Youtube/Facebook/Instagram）を含まない → Camera / PlayCircle / ThumbsUp / MessageCircle 等で代替。
- SVGで `Math.cos/sin` の座標は**丸める**（丸めないとSSRハイドレーション不一致）。
- Framer の `whileInView` リビールは、プレビュー環境がタブ非アクティブ時に `requestAnimationFrame` を間引くため**スクショで空に写る**ことがある（実ユーザーのブラウザでは正常）。検証はDOM（getComputedStyle/属性）でも行うと確実。
- `TextReveal` は長い英語見出しの折り返し対応済み（単語単位クリップ＋実体スペース）。
- **未使用（削除済みセクションの残骸）ファイル**：`QuantumArtGallery.tsx` / `NiravSection.tsx` / `Sigil.tsx` / `Magnetic.tsx` / `StarField.tsx` は page から未参照（Art ギャラリー・Produce セクションは削除済み）。消してよいが残してある。

## 9. 現状の未確定・今後の候補
- Works の各タイトル・説明は**URLスラッグからの推定**（WebFetchが使えなかったため）。正式名称に差し替え余地あり。
- 実写のプロフィール写真・経歴・作品画像は未提供。
- mShots の初回プレースホルダーが気になる場合は、各サイトのスクショを `public/` に置いて直接表示する方式に切り替え可能。

## 10. 参照リンク（氏の各種）
- BASE: katsunoritor.thebase.in（Tシャツ）/ toriumi.thebase.in（原画）/ nirav.base.shop（着るお守り）
- 10-xi-teal.vercel.app（Produce）/ YouTube @Exodus999-j7b / Instagram katsunoritoriumi / Facebook toriumikatsunori / LINE: line.me/ti/p/pcon_XLogZ

---
最終更新: 2026-06-30（本番稼働中）

---

## 11. 創造の座標軸の3D化（2026-07-31）

「創造の座標軸」を2D SVGから **React Three Fiber の3D天球図**に刷新。中心星 Connect(LINE) を唯一の光源とし、6ドメインが実際に公転する。

### ファイル
- `components/ConstellationMap.tsx` — ガワ（見出し・HUD枠・遅延ロード・フォールバック）。3D本体は持たない。
- `components/CelestialMap3D.tsx` — 3Dシーン本体。動的importでのみ読まれる。
- `lib/planetTextures.ts` — 惑星テクスチャの手続き生成（外部画像ファイルなし）。

### 追加依存
`three` / `@react-three/fiber` / `@react-three/drei` / `@react-three/postprocessing` / `postprocessing` / `camera-controls` / `@types/three`

### 設計と数値（実測）
- 3Dチャンクは **brotli 287KB・初期読み込みに含まれない**。初期JSは3D化前と同じ826KB(raw)。
- 読み込みは ①IntersectionObserver（400px手前）② 3秒後の先読みタイマー の二段構え。画面外では `frameloop="demand"` で描画停止。
- テクスチャは **512×256**。純生成 **619ms**・最長ブロック **91ms**。
  **解像度は上げないこと** — 1024×512にすると1枚450ms超のブロックになり実測合計3646msでカクつく。
- 質感の割当: art=砂漠 / youtube=ガス / fashion=大理石 / produce=クレーター / sns=森と海(大気層あり) / radio=氷 / 中心=プラズマ。
- 影は点光源のキューブシャドウ（6面レンダリング）のため**デスクトップのみ**。GodRaysのサンプル数もモバイルでは削減。

### 既知の罠（踏んだもの）
- **`camera-controls` を直接 import してはいけない**。drei が持つコピーと別物になり `install()` 済みでないため壊れる。型のみ `import type` で借り、`ACTION` 定数はインスタンスの `constructor` から取る。
- **ホイール/1本指タッチを3Dに食わせない**。全画面幅の3Dが縦スクロールを奪う。ホイール=ページスクロール、スマホは2本指で回転（`mouseButtons.wheel=NONE` / `touches.one=NONE`）。
- **Browserペインが閉じていると3Dは一切描画されない**。`document.hidden=true` でフレーム合成が止まり rAF / ResizeObserver / IntersectionObserver が全停止する（実測 raf=0）。canvasが300×150のまま中身が空になるが**コードのバグではない**。目視確認にはペインを開く必要がある。
- WebGL不可・3D例外時は `DomainListFallback`（全ドメインのリンク一覧）に退避。3Dの中にしかリンクが無い状態を作らないこと。

### 未確認
本番反映済み（commit `9b83251`）だが、**3Dの見た目そのものは未検証**（上記ペインの制約のため）。光量・軌道半径・惑星サイズ・カメラ距離は実際に見てから調整の余地あり。

---
最終更新: 2026-07-31（3D化を本番反映。見た目の微調整は未実施）

## 12. /voyage — Z軸フライスルー（2026-07-31）

スクロールでカメラが宇宙の奥へ進み、深さごとのステーションを通過する体験ページ。**トップとは別ページ**（トップの SIGNAL LOST 直後「Enter the Voyage」から入る）。

- `app/voyage/page.tsx` / `components/VoyageClient.tsx`（外枠）/ `components/VoyageScene.tsx`（3D）
- ステーション6つ: Voyage Start / Manifesto / Universe / Sound & Vision / Make / Final Report。文言とリンクは `lib/content.ts` の実データ。
- **ScrollControls は使っていない**。canvas 上にスクロール用オーバーレイ div を敷く仕様のため、canvas 内 `Html` のリンクがクリックできなくなる。代わりに `window.scrollY` からカメラZを駆動している。
- スペーサーは `pointer-events:none` でクリックを canvas へ素通し。中に `sr-only` の実テキストを持たせ、SEO・読み上げ・ページ内検索から本文が消えないようにしてある（静的HTMLに本文が出ていることを本番で確認済み）。
- カードの表示制御は毎フレームの setState ではなく DOM 直接更新。
- WebGL不可なら3Dを捨てて素の読み物ページとして表示。
- **未確認**: 3Dの見た目（Browserペインの制約）。ステーション間隔・カード出現距離・手ぶれ量は要調整の可能性あり。

---
最終更新: 2026-07-31（/voyage 追加。3Dの見た目調整は未実施）
