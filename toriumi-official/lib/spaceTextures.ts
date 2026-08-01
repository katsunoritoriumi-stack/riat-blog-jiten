/**
 * 宇宙そのもののテクスチャを手続き生成する。
 * 惑星側（lib/planetTextures.ts）と同じ作法で、
 *   paint*() … RGBA バッファを返す純粋関数（node で PNG に落として目視できる）
 *   make*()  … CanvasTexture に包む薄いラッパー（ブラウザ専用）
 * に分けている。ノイズは lib/noise.ts を共有する。
 */

import * as THREE from "three";
import { clamp01, makeFbm, mix, mulberry32, ramp, ridged, smoothstep, type RGB } from "./noise";
import { bufferToTexture } from "./planetTextures";

/* ─────────────────────────────────────────────
   恒星のスプライト
   ───────────────────────────────────────────── */

/**
 * 1つの恒星。芯・にじみ・ハローを重ね、明るい星には回折光条（十字）を足す。
 * 色は頂点カラーで掛けるので、ここでは白（強度のみ）で焼く。
 *
 * 参考画像の「点が散っているだけでなく、明るい星が十字に伸びている」感じは
 * この光条があるかどうかで決まる。毎フレーム描くのではなく1回焼くだけなので費用はゼロ。
 */
export function paintStarSprite(S: number, spikes: boolean) {
  const px = new Uint8ClampedArray(S * S * 4);
  /**
   * 光条なしの星は「数 px の点」として描かれる。
   * 芯を細く作ると、その芯が 1px 未満になって画面上でほとんど消えてしまう
   * （実際に星空がスカスカになった）。点用は芯を太めにする。
   * 光条つきは大きく描かれるので、芯を締めて鋭さを出す。
   */
  const coreR = spikes ? 0.045 : 0.16;
  const glowR = spikes ? 0.13 : 0.32;
  const haloR = spikes ? 0.3 : 0.46;

  for (let y = 0; y < S; y++) {
    const dy = (y + 0.5) / S - 0.5;
    for (let x = 0; x < S; x++) {
      const dx = (x + 0.5) / S - 0.5;
      const r = Math.hypot(dx, dy);

      let v = Math.exp(-((r / coreR) ** 2)); // 芯
      v += Math.exp(-((r / glowR) ** 2)) * (spikes ? 0.4 : 0.3); // にじみ
      // ハローは弱く。強いと大きく描いたとき灰色の円盤が浮いて見える
      v += Math.exp(-((r / haloR) ** 2)) * 0.05;

      if (spikes) {
        /**
         * 縦横の光条。画面上では 10〜40px 程度で描かれるので、
         * 太さは 0.011 では 1px 未満になって消える。0.022 まで太らせ、
         * 長さも伸ばして「明るい星が十字に伸びている」と分かるようにする。
         */
        const h = Math.exp(-((dy / 0.022) ** 2)) * Math.exp(-Math.abs(dx) / 0.22) * 0.62;
        const w = Math.exp(-((dx / 0.022) ** 2)) * Math.exp(-Math.abs(dy) / 0.22) * 0.62;
        // 斜めは弱く
        const a = (dx + dy) * Math.SQRT1_2;
        const b = (dx - dy) * Math.SQRT1_2;
        const d1 = Math.exp(-((b / 0.016) ** 2)) * Math.exp(-Math.abs(a) / 0.12) * 0.2;
        const d2 = Math.exp(-((a / 0.016) ** 2)) * Math.exp(-Math.abs(b) / 0.12) * 0.2;
        v += h + w + d1 + d2;
      }

      // 枠の外を確実に 0 にして、四角い切れ目が出ないようにする
      v *= smoothstep(0.5, 0.42, r);
      const i = (y * S + x) * 4;
      const c = clamp01(v) * 255;
      px[i] = c;
      px[i + 1] = c;
      px[i + 2] = c;
      px[i + 3] = c;
    }
  }
  return px;
}

/* ─────────────────────────────────────────────
   星雲
   ───────────────────────────────────────────── */

/**
 * 加算合成で重ねる星雲の一片。
 * 1枚で完成させず、これを奥行き方向に何枚もずらして置くことで
 * 「中を通り抜ける」立体感を出す（1枚だと必ず貼り絵に見える）。
 */
export function paintNebula(seed: number, S: number, rgb: RGB) {
  const px = new Uint8ClampedArray(S * S * 4);
  const soft = makeFbm(seed, 5, 4);
  const fil = ridged(makeFbm(seed + 77, 5, 9));
  const rnd = mulberry32(seed + 4242);
  for (let y = 0; y < S; y++) {
    const v = (y + 0.5) / S;
    const dy = v - 0.5;
    for (let x = 0; x < S; x++) {
      const u = (x + 0.5) / S;
      const dx = u - 0.5;
      const r = Math.hypot(dx, dy) * 2; // 0=中心 1=縁
      const falloff = Math.pow(smoothstep(1, 0.05, r), 1.7);
      const n = soft(u, v) * 0.62 + fil(u, v) * 0.38;
      const a = clamp01((n - 0.3) * 1.8) * falloff;
      // 濃いところほど明るく、白へ寄る
      const c = ramp(
        [
          { at: 0, c: [rgb[0] * 0.35, rgb[1] * 0.35, rgb[2] * 0.4] },
          { at: 0.55, c: rgb },
          { at: 1, c: [mix(rgb[0], 255, 0.7), mix(rgb[1], 255, 0.7), mix(rgb[2], 255, 0.75)] },
        ],
        n
      );
      const i = (y * S + x) * 4;
      px[i] = c[0];
      px[i + 1] = c[1];
      px[i + 2] = c[2];
      px[i + 3] = a * 255;
    }
  }
  // 星雲の中に埋まった若い星
  for (let k = 0; k < Math.round(S / 6); k++) {
    const cx = S / 2 + (rnd() - 0.5) * S * 0.66;
    const cy = S / 2 + (rnd() - 0.5) * S * 0.66;
    const rad = 0.6 + rnd() * 1.8;
    const br = 0.45 + rnd() * 0.55;
    for (let y = Math.floor(cy - rad * 3); y <= cy + rad * 3; y++) {
      if (y < 0 || y >= S) continue;
      for (let x = Math.floor(cx - rad * 3); x <= cx + rad * 3; x++) {
        if (x < 0 || x >= S) continue;
        const g = Math.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (rad * rad)));
        if (g < 0.01) continue;
        const i = (y * S + x) * 4;
        px[i] = Math.max(px[i], 255 * g * br);
        px[i + 1] = Math.max(px[i + 1], 250 * g * br);
        px[i + 2] = Math.max(px[i + 2], 255 * g * br);
        px[i + 3] = Math.max(px[i + 3], 255 * g * br);
      }
    }
  }
  return px;
}

/* ─────────────────────────────────────────────
   天の川の帯（背面球に貼る正距円筒）
   ───────────────────────────────────────────── */

/**
 * 遠景の天の川。サイトの色（金の芯・紫青の裾）はここで保つ。
 * 中心に沿って暗いダストレーンを走らせると一気に写真らしくなる。
 */
export function paintMilkyBand(seed: number, W: number, H: number) {
  const px = new Uint8ClampedArray(W * H * 4);
  const wob = makeFbm(seed, 3, 4, 0.5); // 帯の中心のうねり
  const dens = makeFbm(seed + 31, 5, 8, 1.6); // 濃淡（横に伸びる）
  const dust = ridged(makeFbm(seed + 91, 5, 12, 2.2)); // 暗黒星雲
  const grain = makeFbm(seed + 707, 6, 40, 1.2);
  const rnd = mulberry32(seed + 5150);

  for (let y = 0; y < H; y++) {
    const v = (y + 0.5) / H;
    const lat = (v - 0.5) * 2; // -1..1
    for (let x = 0; x < W; x++) {
      const u = (x + 0.5) / W;
      const center = (wob(u, 0.5) - 0.5) * 0.26;
      const d = Math.abs(lat - center);
      const core = Math.exp(-((d / 0.09) ** 2));
      const wide = Math.exp(-((d / 0.24) ** 2));
      let a = (core * 0.8 + wide * 0.12) * (0.45 + 0.55 * dens(u, v));
      a *= 0.35 + 0.65 * grain(u, v);
      // ダストレーンで削る
      a *= 1 - 0.72 * dust(u, v) * core;

      const c = ramp(
        [
          { at: 0, c: [26, 30, 70] }, // 裾は青
          { at: 0.4, c: [86, 72, 150] }, // 紫
          { at: 0.72, c: [190, 150, 120] },
          { at: 1, c: [252, 232, 190] }, // 芯は金
        ],
        clamp01(core * 0.75 + dens(u, v) * 0.35)
      );
      const i = (y * W + x) * 4;
      px[i] = c[0];
      px[i + 1] = c[1];
      px[i + 2] = c[2];
      // 加算合成で貼るので控えめに。強いと画面全体が青く曇る
      px[i + 3] = clamp01(a) * 78;
    }
  }

  /*
    ここに微光星を焼き込むことはしない。
    半径 2600 の球に 1024px 幅で貼るので、1 ピクセルが 16 単位まで拡大され、
    星ではなく「横に伸びた滲み」として写る（実際にそう見えた）。
    星の密度は本物の点（components/universe/starfield.ts）で稼ぐ。
  */
  void rnd;
  return px;
}

/* ─────────────────────────────────────────────
   THREE ラッパー
   ───────────────────────────────────────────── */

export function makeStarSprite(spikes: boolean): THREE.CanvasTexture {
  const S = spikes ? 128 : 48;
  const t = bufferToTexture(paintStarSprite(S, spikes), S, S, false, false);
  t.anisotropy = 1;
  return t;
}

export function makeNebulaSprite(seed: number, rgb: RGB, size = 256): THREE.CanvasTexture {
  return bufferToTexture(paintNebula(seed, size, rgb), size, size, true, false);
}

export function makeMilkyBand(seed: number, W = 1024, H = 512): THREE.CanvasTexture {
  return bufferToTexture(paintMilkyBand(seed, W, H), W, H, true, true);
}

/** 星雲の色（旅の深度で移り変わる：青 → 紫 → 金） */
export const NEBULA_PALETTE: RGB[] = [
  [64, 128, 226], // 青
  [96, 96, 224], // 藍
  [148, 92, 220], // 紫
  [206, 92, 168], // 薄紅
  [232, 158, 88], // 橙金
];
