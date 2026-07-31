/**
 * 惑星テクスチャの手続き生成。
 * 外部アセットを一切持たず、ノイズから「砂漠・ガス・氷・大理石・クレーター・森と海・
 * プラズマ」の質感と、共通の岩石法線マップをブラウザ内で焼き上げる。
 *
 * ・横方向（経度）にシームレス：ノイズ格子を x 方向で周期化しているので継ぎ目が出ない
 * ・生成は 1 枚ずつ。呼び出し側で await を挟んでメインスレッドを譲ること
 * ・public/textures/ に本物の画像を置いた場合は、CelestialMap3D 側でそちらが優先される
 */

import * as THREE from "three";

/**
 * テクスチャ解像度（横:縦 = 2:1 の正距円筒図法）。
 * 惑星は画面上でそれほど大きくならないため 512×256 で十分。
 * これ以上上げると1枚あたりの生成が100msを超え、体感でカクつく。
 */
const W = 512;
const H = 256;

/* ─────────────────────────────────────────────
   ノイズ基盤
   ───────────────────────────────────────────── */

/** 種付き擬似乱数（毎回同じ惑星が焼き上がるように） */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type Lattice = { w: number; h: number; g: Float32Array };

/** 値ノイズの格子。x 方向は周期的（＝経度方向でループする） */
function makeLattice(seed: number, w: number, h: number): Lattice {
  const rnd = mulberry32(seed);
  const g = new Float32Array(w * h);
  for (let i = 0; i < g.length; i++) g[i] = rnd();
  return { w, h, g };
}

/** u,v ∈ [0,1) を格子から双三次風（smoothstep）に補間して取り出す */
function sample(L: Lattice, u: number, v: number): number {
  const x = u * L.w;
  const y = v * L.h;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  const sx = fx * fx * (3 - 2 * fx);
  const sy = fy * fy * (3 - 2 * fy);
  const ix0 = ((x0 % L.w) + L.w) % L.w;
  const ix1 = (ix0 + 1) % L.w;
  const iy0 = Math.min(Math.max(y0, 0), L.h - 1);
  const iy1 = Math.min(iy0 + 1, L.h - 1);
  const r0 = iy0 * L.w;
  const r1 = iy1 * L.w;
  const a = L.g[r0 + ix0];
  const b = L.g[r0 + ix1];
  const c = L.g[r1 + ix0];
  const d = L.g[r1 + ix1];
  return (a + (b - a) * sx) * (1 - sy) + (c + (d - c) * sx) * sy;
}

/** オクターブを重ねた fBm。base は最粗オクターブの格子幅 */
function makeFbm(seed: number, octaves: number, base = 6) {
  const layers: Lattice[] = [];
  for (let i = 0; i < octaves; i++) {
    const w = base * 2 ** i;
    layers.push(makeLattice(seed + i * 7919, w, Math.max(2, w >> 1)));
  }
  return (u: number, v: number) => {
    let sum = 0;
    let amp = 1;
    let norm = 0;
    for (let i = 0; i < layers.length; i++) {
      sum += sample(layers[i], u, v) * amp;
      norm += amp;
      amp *= 0.5;
    }
    return sum / norm;
  };
}

/* ─────────────────────────────────────────────
   共通ユーティリティ
   ───────────────────────────────────────────── */

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smoothstep = (e0: number, e1: number, x: number) => {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
};
const mix = (a: number, b: number, t: number) => a + (b - a) * t;

type RGB = [number, number, number];

/** 色の階調表（stop は 0-1 の昇順） */
function ramp(stops: { at: number; c: RGB }[], t: number): RGB {
  const x = clamp01(t);
  for (let i = 0; i < stops.length - 1; i++) {
    const s0 = stops[i];
    const s1 = stops[i + 1];
    if (x <= s1.at) {
      const k = (x - s0.at) / (s1.at - s0.at || 1);
      return [mix(s0.c[0], s1.c[0], k), mix(s0.c[1], s1.c[1], k), mix(s0.c[2], s1.c[2], k)];
    }
  }
  return stops[stops.length - 1].c;
}

function newCanvas(w: number, h: number) {
  const cv = document.createElement("canvas");
  cv.width = w;
  cv.height = h;
  return cv;
}

function toTexture(cv: HTMLCanvasElement, srgb: boolean) {
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.anisotropy = 8;
  tex.needsUpdate = true;
  return tex;
}

/* ─────────────────────────────────────────────
   質感ごとの塗り分け
   ───────────────────────────────────────────── */

export type PlanetSkin =
  | "desert" // 乾いた赤い砂漠
  | "gas" // 赤橙の縞模様のガス惑星
  | "ice" // 冷たい青の氷惑星
  | "marble" // クラフト感のあるマーブル岩石
  | "cratered" // クレーターの多い灰褐色の岩石
  | "living" // 森と海の惑星
  | "plasma"; // 燃える中心星

/** 極付近をなだらかに白く（氷冠）する係数 */
const polar = (v: number) => smoothstep(0.34, 0.06, Math.min(v, 1 - v));

function shade(skin: PlanetSkin, seed: number) {
  const base = makeFbm(seed, 4);
  const detail = makeFbm(seed + 101, 4, 12);
  const warp = makeFbm(seed + 202, 2, 4);

  switch (skin) {
    case "desert":
      return (u: number, v: number): RGB => {
        const n = base(u, v) * 0.75 + detail(u, v) * 0.25;
        const c = ramp(
          [
            { at: 0, c: [46, 20, 14] },
            { at: 0.45, c: [122, 52, 32] },
            { at: 0.72, c: [178, 95, 56] },
            { at: 1, c: [224, 168, 118] },
          ],
          n
        );
        // 砂の筋を薄く重ねる
        const streak = 0.92 + 0.08 * Math.sin(v * 140 + warp(u, v) * 12);
        return [c[0] * streak, c[1] * streak, c[2] * streak];
      };

    case "gas":
      return (u: number, v: number): RGB => {
        // 緯度方向の縞を、経度方向のノイズで乱流のように歪ませる
        const w = warp(u, v);
        const band = v * 26 + w * 3.2 + Math.sin(u * Math.PI * 2 + w * 6) * 0.35;
        const t = 0.5 + 0.5 * Math.sin(band);
        const n = t * 0.72 + base(u, v) * 0.28;
        const c = ramp(
          [
            { at: 0, c: [78, 20, 8] },
            { at: 0.4, c: [168, 62, 22] },
            { at: 0.72, c: [232, 122, 52] },
            { at: 1, c: [255, 198, 140] },
          ],
          n
        );
        // 大赤斑のような渦をひとつ
        const dx = Math.abs(((u - 0.32 + 0.5) % 1) - 0.5) * 2.6;
        const dy = (v - 0.58) * 5.2;
        const spot = smoothstep(1, 0.15, Math.hypot(dx, dy));
        return [mix(c[0], 238, spot * 0.75), mix(c[1], 96, spot * 0.75), mix(c[2], 58, spot * 0.75)];
      };

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

    case "living":
      return (u: number, v: number): RGB => {
        const n = base(u, v) * 0.78 + detail(u, v) * 0.22;
        const sea = smoothstep(0.52, 0.46, n); // n が低いほど海
        const ocean = ramp(
          [
            { at: 0, c: [6, 20, 52] },
            { at: 0.6, c: [16, 62, 110] },
            { at: 1, c: [34, 118, 152] },
          ],
          n / 0.52
        );
        const land = ramp(
          [
            { at: 0, c: [42, 82, 44] },
            { at: 0.4, c: [72, 116, 52] },
            { at: 0.7, c: [126, 130, 74] },
            { at: 1, c: [156, 142, 112] },
          ],
          (n - 0.46) / 0.54
        );
        const c: RGB = [
          mix(land[0], ocean[0], sea),
          mix(land[1], ocean[1], sea),
          mix(land[2], ocean[2], sea),
        ];
        const cap = polar(v);
        return [mix(c[0], 248, cap), mix(c[1], 250, cap), mix(c[2], 255, cap)];
      };

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

/** クレーターを後から描き込む（横方向の継ぎ目にも回り込ませる） */
function stampCraters(ctx: CanvasRenderingContext2D, seed: number, count: number) {
  const rnd = mulberry32(seed);
  for (let i = 0; i < count; i++) {
    const cx = rnd() * W;
    const cy = rnd() * H;
    // 極付近は歪みが激しいので避ける
    if (cy < H * 0.12 || cy > H * 0.88) continue;
    // 半径は 1024px 基準で設計しているので解像度に合わせて縮める
    const r = (4 + rnd() * 26) * (1 + rnd() * rnd() * 2.2) * (W / 1024);
    for (const ox of [-W, 0, W]) {
      const x = cx + ox;
      if (x < -r || x > W + r) continue;
      const g = ctx.createRadialGradient(x, cy, r * 0.1, x, cy, r);
      g.addColorStop(0, "rgba(20,18,16,0.55)");
      g.addColorStop(0.62, "rgba(46,42,38,0.28)");
      g.addColorStop(0.86, "rgba(226,214,192,0.30)");
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, cy, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

/** 質感からカラー（ディフューズ）マップを焼く */
export function makeDiffuse(skin: PlanetSkin, seed: number): THREE.CanvasTexture {
  const cv = newCanvas(W, H);
  const ctx = cv.getContext("2d")!;
  const img = ctx.createImageData(W, H);
  const px = img.data;
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
  ctx.putImageData(img, 0, 0);
  if (skin === "cratered") stampCraters(ctx, seed + 555, 190);
  if (skin === "desert") stampCraters(ctx, seed + 556, 45);

  return toTexture(cv, true);
}

/**
 * 岩石質の共通法線マップ。
 * 高さの fBm を作り、隣接差分（Sobel 相当）から法線を求めて RGB に詰める。
 */
export function makeCommonNormal(seed: number): THREE.CanvasTexture {
  const f = makeFbm(seed, 5, 8);
  const ridged = makeFbm(seed + 313, 4, 16);

  // まず高さ場を作る（差分で 4 近傍を参照するので一旦配列に持つ）
  const height = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    const v = (y + 0.5) / H;
    for (let x = 0; x < W; x++) {
      const u = (x + 0.5) / W;
      const r = 1 - Math.abs(ridged(u, v) * 2 - 1); // 尖った稜線
      height[y * W + x] = f(u, v) * 0.72 + r * r * 0.28;
    }
  }

  const cv = newCanvas(W, H);
  const ctx = cv.getContext("2d")!;
  const img = ctx.createImageData(W, H);
  const px = img.data;
  const strength = 2.6;

  for (let y = 0; y < H; y++) {
    const ym = y > 0 ? y - 1 : 0;
    const yp = y < H - 1 ? y + 1 : H - 1;
    for (let x = 0; x < W; x++) {
      const xm = (x - 1 + W) % W; // 経度方向はループ
      const xp = (x + 1) % W;
      const dx = (height[y * W + xp] - height[y * W + xm]) * strength;
      const dy = (height[yp * W + x] - height[ym * W + x]) * strength;
      // 法線 = normalize(-dx, -dy, 1)
      const len = Math.hypot(dx, dy, 1);
      const i = (y * W + x) * 4;
      px[i] = ((-dx / len) * 0.5 + 0.5) * 255;
      px[i + 1] = ((-dy / len) * 0.5 + 0.5) * 255;
      px[i + 2] = (1 / len) * 0.5 * 255 + 127.5;
      px[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return toTexture(cv, false);
}
