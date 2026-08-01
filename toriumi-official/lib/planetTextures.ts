/**
 * 惑星テクスチャの手続き生成。外部アセットを一切持たず、ノイズから
 * 「砂漠・ガス・氷・大理石・クレーター・大陸と海・プラズマ」の質感と、
 * 共通の岩石法線マップ・粗さマップ・雲のシェル・環をブラウザ内で焼き上げる。
 *
 * 構成：
 *   paint*()  … RGBA バッファを返す純粋関数。DOM にも THREE にも依存しないので
 *               node で直接叩いて PNG に落とし、目視で確かめられる
 *   make*()   … それを CanvasTexture に包む薄いラッパー（ブラウザ専用）
 *
 * ・横方向（経度）にシームレス：ノイズ格子を x 方向で周期化しているので継ぎ目が出ない
 * ・生成は 1 枚ずつ。呼び出し側で await を挟んでメインスレッドを譲ること
 */

import * as THREE from "three";
import {
  clamp01,
  makeFbm,
  makeWarpedFbm,
  mix,
  mulberry32,
  ramp,
  ridged,
  smoothstep,
  type RGB,
} from "./noise";

/**
 * 既定のテクスチャ解像度（横:縦 = 2:1 の正距円筒図法）。
 * 全部を 1024×512 にすると 7 枚で実測 3.6 秒かかったため、
 * 画面上で大きく映る主役（ガス惑星・大陸の惑星）だけを HI にする。
 */
export const TEX_W = 512;
export const TEX_H = 256;
export const TEX_W_HI = 1024;
export const TEX_H_HI = 512;

export type PlanetSkin =
  | "desert" // 乾いた赤い砂漠
  | "gas" // 縞と渦のガス惑星
  | "ice" // 冷たい青の氷惑星
  | "marble" // クラフト感のあるマーブル岩石
  | "cratered" // クレーターの多い灰褐色の岩石
  | "living" // 大陸と海の惑星
  | "plasma"; // 燃える中心星

/** 主役（高解像度で焼く）かどうか */
export const isHeroSkin = (skin: PlanetSkin) => skin === "gas" || skin === "living";

export function texSize(skin: PlanetSkin): [number, number] {
  return isHeroSkin(skin) ? [TEX_W_HI, TEX_H_HI] : [TEX_W, TEX_H];
}

/* ─────────────────────────────────────────────
   質感ごとの塗り分け
   ───────────────────────────────────────────── */

/** 極付近をなだらかに白く（氷冠）する係数 */
const polar = (v: number) => smoothstep(0.34, 0.06, Math.min(v, 1 - v));

/**
 * 大陸の高さ場。ディフューズと粗さマップの両方が同じ海岸線を見る必要があるので、
 * 式をここに一本化する（別々に書くとズレて、海の上が「陸の粗さ」になる）。
 * 生のノイズは 0.5 付近に固まって大陸と海の境が曖昧になるため、
 * コントラストを広げて分布を二山に寄せている。
 */
export function landHeight(seed: number) {
  const a = makeFbm(seed, 6, 5);
  const b = makeFbm(seed + 909, 5, 20);
  return (u: number, v: number) => clamp01((a(u, v) * 0.76 + b(u, v) * 0.24 - 0.5) * 2.2 + 0.5);
}
/**
 * これを超えたら陸。地球の海の割合に合わせた値。
 * landHeight の分布を二分探索して「海 70%」になる点を求めた（node での実測）。
 */
export const SEA_LEVEL = 0.67;

function shade(skin: PlanetSkin, seed: number) {
  const base = makeFbm(seed, 4);
  const detail = makeFbm(seed + 101, 4, 12);
  const warp = makeFbm(seed + 202, 2, 4);

  switch (skin) {
    case "desert": {
      // 風の筋。横に引き伸ばしたノイズで作る（正弦だと等間隔の縞に見えてしまう）
      const wind = makeFbm(seed + 606, 4, 10, 3.2);
      return (u: number, v: number): RGB => {
        const n = base(u, v) * 0.72 + detail(u, v) * 0.28;
        const c = ramp(
          [
            { at: 0, c: [52, 24, 16] },
            { at: 0.42, c: [124, 56, 34] },
            { at: 0.7, c: [180, 98, 58] },
            { at: 0.9, c: [214, 148, 100] },
            { at: 1, c: [232, 188, 146] },
          ],
          n
        );
        const streak = 0.88 + 0.24 * wind(u, v);
        const cap = polar(v) * 0.5; // 火星のような薄い極冠
        return [
          mix(c[0] * streak, 236, cap),
          mix(c[1] * streak, 238, cap),
          mix(c[2] * streak, 242, cap),
        ];
      };
    }

    case "gas": {
      /**
       * 木星の顔は「幅の不揃いな帯」＋「境界の乱流」で決まる。
       *   ・帯の位置は正弦波ではなく、緯度方向だけ細かいノイズ（zone）から取る。
       *     正弦だと等間隔になって作り物に見える
       *   ・境界の乱れは aspect を大きくして横に引き伸ばしたノイズで緯度をずらす。
       *     ずらす量は 0.05 程度まで。これを超えると帯が崩れて雲になる（実際になった）
       */
      const zone = makeFbm(seed, 4, 5, 4.5); // 列5・行22 → ほぼ緯度方向だけの profile
      const turb = makeFbm(seed + 313, 5, 6, 2.6); // 横長の乱流
      const fine = makeFbm(seed + 77, 5, 14, 1.6); // 細かい筋
      return (u: number, v: number): RGB => {
        const t1 = turb(u, v) - 0.5;
        const t2 = fine(u, v) - 0.5;
        // 帯の境界をうねらせる（緯度をわずかにずらす）
        const vv = clamp01(v + t1 * 0.05 + t2 * 0.012);
        // 帯そのもの。コントラストを上げて境界を締める
        let n = zone(u, vv) * 0.82 + turb(u, vv) * 0.11 + fine(u, vv) * 0.07;
        n = smoothstep(0.3, 0.7, n);

        let c = ramp(
          [
            { at: 0, c: [52, 34, 28] },
            { at: 0.2, c: [116, 68, 42] },
            { at: 0.4, c: [180, 120, 74] },
            { at: 0.58, c: [226, 186, 138] },
            { at: 0.76, c: [244, 226, 196] },
            { at: 1, c: [253, 248, 236] },
          ],
          n
        );
        // 極域は青灰色に寄せる（木星・土星ともに極は寒色）
        const pol = smoothstep(0.62, 1, Math.abs(v - 0.5) * 2);
        c = [mix(c[0], 118, pol * 0.5), mix(c[1], 132, pol * 0.5), mix(c[2], 156, pol * 0.5)];

        /**
         * 大赤斑。楕円の内側を渦として回し（中心ほど大きく回す）、
         * まわりに明るい襟を付けて「窪んで回っている」ように見せる。
         */
        const du = ((u - 0.5 + 0.5) % 1) - 0.5;
        const dv = v - 0.63;
        const rx = du / 0.15;
        const ry = dv / 0.062;
        const rr = Math.hypot(rx, ry);
        if (rr > 1.5) return c;
        const collar = smoothstep(1.5, 1.05, rr) * (1 - smoothstep(1.05, 0.92, rr));
        const spot = smoothstep(1.02, 0.15, rr);
        const swirl = 0.5 + 0.5 * Math.sin(Math.atan2(ry, rx) * 2 + (1 - rr) * 9 + t1 * 8);
        const spotC = ramp(
          [
            { at: 0, c: [122, 38, 24] },
            { at: 0.45, c: [186, 78, 44] },
            { at: 0.8, c: [224, 132, 88] },
            { at: 1, c: [242, 186, 148] },
          ],
          clamp01(swirl * 0.62 + (1 - rr) * 0.3)
        );
        const k = spot * 0.9;
        c = [mix(c[0], spotC[0], k), mix(c[1], spotC[1], k), mix(c[2], spotC[2], k)];
        return [
          mix(c[0], 250, collar * 0.5),
          mix(c[1], 238, collar * 0.5),
          mix(c[2], 214, collar * 0.5),
        ];
      };
    }

    case "ice":
      return (u: number, v: number): RGB => {
        const n = base(u, v);
        // 尖らせたノイズ＝氷の亀裂
        const crack = 1 - Math.abs(detail(u, v) * 2 - 1);
        const c = ramp(
          [
            { at: 0, c: [16, 38, 74] },
            { at: 0.45, c: [58, 106, 168] },
            { at: 0.75, c: [148, 194, 236] },
            { at: 1, c: [226, 244, 255] },
          ],
          n * 0.82 + crack * 0.18
        );
        const cap = polar(v);
        return [mix(c[0], 246, cap), mix(c[1], 252, cap), mix(c[2], 255, cap)];
      };

    case "marble":
      return (u: number, v: number): RGB => {
        // 大理石＝座標をノイズで歪ませた正弦縞
        const veins = 0.5 + 0.5 * Math.sin((u * 9 + v * 5) * Math.PI + base(u, v) * 14);
        const n = veins * 0.68 + detail(u, v) * 0.32;
        return ramp(
          [
            { at: 0, c: [58, 44, 62] },
            { at: 0.4, c: [128, 104, 118] },
            { at: 0.68, c: [186, 162, 168] },
            { at: 0.88, c: [226, 206, 200] },
            { at: 1, c: [246, 236, 226] },
          ],
          n
        );
      };

    case "cratered":
      return (u: number, v: number): RGB => {
        const n = base(u, v) * 0.7 + detail(u, v) * 0.3;
        return ramp(
          [
            { at: 0, c: [38, 34, 30] },
            { at: 0.42, c: [86, 78, 68] },
            { at: 0.72, c: [132, 118, 100] },
            { at: 1, c: [186, 172, 150] },
          ],
          n
        );
      };

    case "living": {
      /**
       * 大陸は「低い周波数で大まかな塊を作り、高い周波数で海岸線をぎざぎざにする」。
       * しきい値を跨いだところに浅瀬を挟むと、一気に地球らしくなる。
       */
      const land = landHeight(seed);
      const relief = ridged(makeFbm(seed + 1717, 4, 16));
      return (u: number, v: number): RGB => {
        const lat = Math.abs(v - 0.5) * 2;
        const h = land(u, v);
        const SEA = SEA_LEVEL;

        if (h < SEA) {
          // 深さで色を変える。岸に近いほど明るいターコイズ
          const d = clamp01((SEA - h) / 0.32);
          return ramp(
            [
              { at: 0, c: [72, 168, 178] }, // 浅瀬
              { at: 0.16, c: [34, 116, 168] },
              { at: 0.45, c: [18, 68, 130] },
              { at: 1, c: [6, 24, 62] }, // 深海
            ],
            d
          );
        }

        const a = clamp01((h - SEA) / 0.24); // 標高
        /**
         * 乾燥度。砂漠は赤道ではなく亜熱帯（緯度 20〜35°＝lat 0.3 付近）に帯として出る。
         * 最初は赤道を乾かしてしまい、熱帯雨林が砂漠になった。
         */
        const arid = Math.exp(-((lat - 0.3) ** 2) / 0.016);
        const dry = clamp01(relief(u, v) * 0.45 + arid * 0.72);
        const green = ramp(
          [
            { at: 0, c: [34, 84, 44] },
            { at: 0.4, c: [52, 110, 50] },
            { at: 0.75, c: [86, 118, 58] },
            { at: 1, c: [124, 128, 82] },
          ],
          a
        );
        const sand = ramp(
          [
            { at: 0, c: [148, 122, 82] },
            { at: 0.5, c: [176, 146, 100] },
            { at: 1, c: [198, 174, 136] },
          ],
          a
        );
        let c: RGB = [
          mix(green[0], sand[0], dry),
          mix(green[1], sand[1], dry),
          mix(green[2], sand[2], dry),
        ];
        // 高いところに雪
        const snow = smoothstep(0.82, 1, a);
        c = [mix(c[0], 236, snow), mix(c[1], 242, snow), mix(c[2], 250, snow)];
        const cap = polar(v);
        return [mix(c[0], 244, cap), mix(c[1], 248, cap), mix(c[2], 255, cap)];
      };
    }

    case "plasma":
      return (u: number, v: number): RGB => {
        // 対流セルのような粒状感＋暗い筋（黒点）
        const n = base(u, v) * 0.55 + detail(u, v) * 0.45;
        const fil = 1 - Math.abs(warp(u, v) * 2 - 1);
        const t = clamp01(n * 0.85 + fil * 0.3);
        return ramp(
          [
            { at: 0, c: [96, 22, 4] },
            { at: 0.35, c: [214, 78, 12] },
            { at: 0.62, c: [252, 158, 44] },
            { at: 0.85, c: [255, 226, 150] },
            { at: 1, c: [255, 252, 238] },
          ],
          t
        );
      };
  }
}

/**
 * クレーターを RGBA バッファへ直接描き込む（canvas のグラデーションを使わない）。
 * 内側は暗く、縁だけ明るいリムを持つ、という同心円の重ね塗り。
 */
function stampCraters(px: Uint8ClampedArray, W: number, H: number, seed: number, count: number) {
  const rnd = mulberry32(seed);
  /**
   * 内側が暗く、縁だけわずかに明るい。縁を強くすると石鹸の泡のように見えるので抑える
   * （最初 0.30 にしたら泡になった）。
   */
  const stops: { at: number; c: RGB; a: number }[] = [
    { at: 0, c: [22, 20, 18], a: 0.42 },
    { at: 0.66, c: [52, 48, 42], a: 0.2 },
    { at: 0.88, c: [212, 200, 178], a: 0.14 },
    { at: 1, c: [0, 0, 0], a: 0 },
  ];
  for (let i = 0; i < count; i++) {
    const cx = rnd() * W;
    const cy = rnd() * H;
    // 極付近は歪みが激しいので避ける
    if (cy < H * 0.12 || cy > H * 0.88) continue;
    // 小さいものが圧倒的に多く、大きいものは稀（3乗で偏らせる）
    const t = rnd() * rnd() * rnd();
    const r = (2 + t * 46) * (W / 1024);
    const y0 = Math.max(0, Math.floor(cy - r));
    const y1 = Math.min(H - 1, Math.ceil(cy + r));
    for (let y = y0; y <= y1; y++) {
      const dy = y - cy;
      const half = Math.sqrt(Math.max(0, r * r - dy * dy));
      const xs = Math.floor(cx - half);
      const xe = Math.ceil(cx + half);
      for (let x = xs; x <= xe; x++) {
        const d = Math.hypot(x - cx, dy) / r;
        if (d > 1) continue;
        // 経度方向はループするので巻き戻す
        const xi = ((x % W) + W) % W;
        let c: RGB = [0, 0, 0];
        let a = 0;
        for (let s = 0; s < stops.length - 1; s++) {
          if (d <= stops[s + 1].at) {
            const k = (d - stops[s].at) / (stops[s + 1].at - stops[s].at || 1);
            c = [
              mix(stops[s].c[0], stops[s + 1].c[0], k),
              mix(stops[s].c[1], stops[s + 1].c[1], k),
              mix(stops[s].c[2], stops[s + 1].c[2], k),
            ];
            a = mix(stops[s].a, stops[s + 1].a, k);
            break;
          }
        }
        if (a <= 0) continue;
        const o = (y * W + xi) * 4;
        px[o] = px[o] * (1 - a) + c[0] * a;
        px[o + 1] = px[o + 1] * (1 - a) + c[1] * a;
        px[o + 2] = px[o + 2] * (1 - a) + c[2] * a;
      }
    }
  }
}

/* ─────────────────────────────────────────────
   純粋なペインタ（node から直接叩ける）
   ───────────────────────────────────────────── */

/** カラー（ディフューズ）マップ */
export function paintDiffuse(skin: PlanetSkin, seed: number, W: number, H: number) {
  const px = new Uint8ClampedArray(W * H * 4);
  const f = shade(skin, seed);
  for (let y = 0; y < H; y++) {
    const v = (y + 0.5) / H;
    for (let x = 0; x < W; x++) {
      const c = f((x + 0.5) / W, v);
      const i = (y * W + x) * 4;
      px[i] = c[0];
      px[i + 1] = c[1];
      px[i + 2] = c[2];
      px[i + 3] = 255;
    }
  }
  if (skin === "cratered") stampCraters(px, W, H, seed + 555, Math.round(190 * (W / 512)));
  if (skin === "desert") stampCraters(px, W, H, seed + 556, Math.round(45 * (W / 512)));
  return px;
}

/**
 * 岩石質の共通法線マップ。
 * 高さの fBm を作り、隣接差分（Sobel 相当）から法線を求めて RGB に詰める。
 */
export function paintNormal(seed: number, W: number, H: number) {
  const f = makeFbm(seed, 5, 8);
  const rid = ridged(makeFbm(seed + 313, 4, 16));

  const height = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    const v = (y + 0.5) / H;
    for (let x = 0; x < W; x++) {
      const u = (x + 0.5) / W;
      height[y * W + x] = f(u, v) * 0.72 + rid(u, v) * 0.28;
    }
  }

  const px = new Uint8ClampedArray(W * H * 4);
  const strength = 2.6;
  for (let y = 0; y < H; y++) {
    const ym = y > 0 ? y - 1 : 0;
    const yp = y < H - 1 ? y + 1 : H - 1;
    for (let x = 0; x < W; x++) {
      const xm = (x - 1 + W) % W; // 経度方向はループ
      const xp = (x + 1) % W;
      const dx = (height[y * W + xp] - height[y * W + xm]) * strength;
      const dy = (height[yp * W + x] - height[ym * W + x]) * strength;
      const len = Math.hypot(dx, dy, 1);
      const i = (y * W + x) * 4;
      px[i] = ((-dx / len) * 0.5 + 0.5) * 255;
      px[i + 1] = ((-dy / len) * 0.5 + 0.5) * 255;
      px[i + 2] = (1 / len) * 0.5 * 255 + 127.5;
      px[i + 3] = 255;
    }
  }
  return px;
}

/**
 * 粗さマップ。海だけ滑らかにして光を鋭く返させる＝水面らしく見せるのが目的。
 * MeshStandardMaterial の roughnessMap は緑チャンネルを読むが、
 * 誤読を避けるため RGB すべてに同じ値を入れる。
 */
export function paintRoughness(skin: PlanetSkin, seed: number, W: number, H: number) {
  const px = new Uint8ClampedArray(W * H * 4);
  const land = landHeight(seed);
  const grain = makeFbm(seed + 4242, 4, 18);
  for (let y = 0; y < H; y++) {
    const v = (y + 0.5) / H;
    for (let x = 0; x < W; x++) {
      const u = (x + 0.5) / W;
      let r: number;
      if (skin === "living") {
        // 海は滑らかに（光を鋭く返す）、陸はざらつかせる。境目はディフューズと同じ式を見る
        r = mix(0.12, 0.92, smoothstep(SEA_LEVEL - 0.03, SEA_LEVEL + 0.05, land(u, v)));
      } else if (skin === "ice") {
        r = mix(0.2, 0.55, grain(u, v));
      } else {
        r = mix(0.72, 0.98, grain(u, v));
      }
      const b = clamp01(r) * 255;
      const i = (y * W + x) * 4;
      px[i] = b;
      px[i + 1] = b;
      px[i + 2] = b;
      px[i + 3] = 255;
    }
  }
  return px;
}

/**
 * 雲のシェル用テクスチャ（白＋アルファ）。
 * 本体より少し大きい半透明の球に貼り、別の速度で回すと一気に地球らしくなる。
 * 緯度ごとに雲量を変え（赤道と中緯度に多く、亜熱帯で少なく）、渦を巻かせている。
 */
export function paintClouds(seed: number, W: number, H: number) {
  const px = new Uint8ClampedArray(W * H * 4);
  // 雲も帯状に流れるので、横に引き伸ばしたノイズを使う
  const f = makeWarpedFbm(seed, 6, 9, [0.1, 0.02], 1.4);
  const wisp = makeFbm(seed + 555, 6, 26, 1.9);
  for (let y = 0; y < H; y++) {
    const v = (y + 0.5) / H;
    const lat = Math.abs(v - 0.5) * 2;
    // 赤道(0)と中緯度(0.6)で多く、亜熱帯(0.3)と極(1)で少なく
    const belt =
      0.62 * Math.exp(-((lat - 0.02) ** 2) / 0.02) +
      0.8 * Math.exp(-((lat - 0.58) ** 2) / 0.05) +
      0.3;
    for (let x = 0; x < W; x++) {
      const u = (x + 0.5) / W;
      const n = f(u, v) * 0.74 + wisp(u, v) * 0.26;
      // 「雲の無い抜け」をはっきり残す（一面の靄にしない）。被覆率は node で実測して詰めた
      const a = clamp01(smoothstep(0.36, 0.62, n * belt) * 1.25);
      const i = (y * W + x) * 4;
      px[i] = 255;
      px[i + 1] = 255;
      px[i + 2] = 255;
      px[i + 3] = a * 255;
    }
  }
  return px;
}

/**
 * 環のストリップ（横方向＝環の内→外）。
 * 濃淡の縞と、カッシーニの空隙にあたる大きな隙間を入れる。
 * RingGeometry の UV は既定では平面写像なので、貼る側で u を半径にし直すこと。
 */
export function paintRing(seed: number, W: number, H = 8) {
  const px = new Uint8ClampedArray(W * H * 4);
  const rnd = mulberry32(seed);
  const f = makeFbm(seed + 31, 6, 10);
  // 大きな空隙（中心位置と幅）
  const gaps: [number, number][] = [
    [0.42, 0.035],
    [0.63, 0.018],
    [0.86, 0.012],
  ];
  for (let x = 0; x < W; x++) {
    const t = (x + 0.5) / W; // 0=内側 1=外側
    let a = smoothstep(0, 0.06, t) * smoothstep(1, 0.9, t); // 内外の縁で消える
    // 細かい縞
    a *= 0.45 + 0.55 * f(t * 3, 0.5);
    a *= 0.7 + 0.3 * Math.sin(t * 220 + rnd() * 0.001);
    for (const [g, wdt] of gaps) {
      a *= 1 - 0.94 * Math.exp(-((t - g) ** 2) / (wdt * wdt));
    }
    const c = ramp(
      [
        { at: 0, c: [148, 130, 108] },
        { at: 0.45, c: [206, 190, 164] },
        { at: 0.75, c: [178, 164, 142] },
        { at: 1, c: [220, 208, 186] },
      ],
      f(t * 2 + 0.3, 0.5)
    );
    for (let y = 0; y < H; y++) {
      const i = (y * W + x) * 4;
      px[i] = c[0];
      px[i + 1] = c[1];
      px[i + 2] = c[2];
      px[i + 3] = clamp01(a) * 235;
    }
  }
  return px;
}

/* ─────────────────────────────────────────────
   THREE ラッパー（ブラウザ専用）
   ───────────────────────────────────────────── */

/** RGBA バッファを CanvasTexture に包む。宇宙側（lib/spaceTextures.ts）とも共有する */
export function bufferToTexture(
  px: Uint8ClampedArray,
  W: number,
  H: number,
  srgb: boolean,
  wrapU = true
) {
  const cv = document.createElement("canvas");
  cv.width = W;
  cv.height = H;
  const ctx = cv.getContext("2d")!;
  // createImageData + set で渡す（ImageData のコンストラクタは
  // Uint8ClampedArray のバッファ型に厳しく、型が通らない）
  const img = ctx.createImageData(W, H);
  img.data.set(px);
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  tex.wrapS = wrapU ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.anisotropy = 8;
  tex.needsUpdate = true;
  return tex;
}

export function makeDiffuse(skin: PlanetSkin, seed: number): THREE.CanvasTexture {
  const [w, h] = texSize(skin);
  return bufferToTexture(paintDiffuse(skin, seed, w, h), w, h, true);
}

export function makeCommonNormal(seed: number): THREE.CanvasTexture {
  return bufferToTexture(paintNormal(seed, TEX_W, TEX_H), TEX_W, TEX_H, false);
}

export function makeRoughness(skin: PlanetSkin, seed: number): THREE.CanvasTexture {
  const [w, h] = texSize(skin);
  return bufferToTexture(paintRoughness(skin, seed, w, h), w, h, false);
}

export function makeClouds(seed: number): THREE.CanvasTexture {
  return bufferToTexture(paintClouds(seed, TEX_W_HI, TEX_H_HI), TEX_W_HI, TEX_H_HI, true);
}

export function makeRing(seed: number): THREE.CanvasTexture {
  const W = 512;
  const H = 8;
  return bufferToTexture(paintRing(seed, W, H), W, H, false, false);
}
