/**
 * 手続き生成の土台になる数式だけを集めたもの。
 *
 * DOM にも THREE にも依存しない。狙いは lib/flightMath.ts と同じで、
 * 「requestAnimationFrame が動かない環境でも素の node で検証できる」ようにするため。
 * 惑星テクスチャ（lib/planetTextures.ts）と宇宙テクスチャ（lib/spaceTextures.ts）が
 * ここを共有し、同じノイズから同じ絵が焼き上がる。
 */

/** 種付き擬似乱数（毎回同じものが焼き上がるように） */
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type Lattice = { w: number; h: number; g: Float32Array };

/** 値ノイズの格子。x 方向は周期的（＝経度方向でループする） */
export function makeLattice(seed: number, w: number, h: number): Lattice {
  const rnd = mulberry32(seed);
  const g = new Float32Array(w * h);
  for (let i = 0; i < g.length; i++) g[i] = rnd();
  return { w, h, g };
}

/** u,v ∈ [0,1) を格子から滑らかに補間して取り出す（u はループ、v は端で止める） */
export function sample(L: Lattice, u: number, v: number): number {
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

export type Field = (u: number, v: number) => number;

/**
 * オクターブを重ねた fBm。base は最粗オクターブの格子幅。
 *
 * aspect は格子の「行数 ÷ 列数」。正距円筒図（横:縦 = 2:1）を球に貼ると
 * aspect = 0.5 で球面上の模様が等方になる。0.5 より大きくすると
 * 行が細かく・列が粗くなり、模様が**経度方向に引き伸ばされる**。
 * ガス惑星の縞や風の筋は、これで作る（u を圧縮して呼ぶと経度の継ぎ目が壊れる）。
 */
export function makeFbm(seed: number, octaves: number, base = 6, aspect = 0.5): Field {
  const layers: Lattice[] = [];
  for (let i = 0; i < octaves; i++) {
    const w = base * 2 ** i;
    layers.push(makeLattice(seed + i * 7919, w, Math.max(2, Math.round(w * aspect))));
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

/**
 * ドメインワープした fBm。座標そのものを別のノイズでずらしてから引くと、
 * 直線的だった模様が渦を巻き、ちぎれ、流れる。ガス惑星の縞を本物らしくする要。
 * strength は経度方向・緯度方向のずらし量。
 */
export function makeWarpedFbm(
  seed: number,
  octaves: number,
  base: number,
  strength: [number, number],
  aspect = 0.5
): Field {
  const f = makeFbm(seed, octaves, base, aspect);
  const wu = makeFbm(seed + 4211, 3, Math.max(3, base >> 1), aspect);
  const wv = makeFbm(seed + 8677, 3, Math.max(3, base >> 1), aspect);
  return (u: number, v: number) => {
    const du = (wu(u, v) - 0.5) * strength[0];
    const dv = (wv(u, v) - 0.5) * strength[1];
    // u はループするので剰余で巻き戻す。v は範囲内に押し込む
    const uu = ((u + du) % 1 + 1) % 1;
    const vv = Math.min(0.999, Math.max(0, v + dv));
    return f(uu, vv);
  };
}

/** 尾根状ノイズ（1 - |2n-1|）。亀裂・稜線・星雲の筋に使う */
export function ridged(f: Field): Field {
  return (u, v) => {
    const r = 1 - Math.abs(f(u, v) * 2 - 1);
    return r * r;
  };
}

/* ── 小物 ────────────────────────────────── */

export const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

export const smoothstep = (e0: number, e1: number, x: number) => {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
};

export const mix = (a: number, b: number, t: number) => a + (b - a) * t;

export type RGB = [number, number, number];

/** 色の階調表（stop は 0-1 の昇順） */
export function ramp(stops: { at: number; c: RGB }[], t: number): RGB {
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

/**
 * 黒体輻射に寄せた恒星の色（温度 t: 0=赤い低温星 … 1=青白い高温星）。
 * 添付参考画像のように「大半は白〜青白、たまに橙」に見えるよう、
 * 呼び出し側で t の分布を高温寄りに偏らせて使う。
 */
export function starColor(t: number): RGB {
  return ramp(
    [
      { at: 0, c: [255, 176, 112] }, // 橙（低温）
      { at: 0.35, c: [255, 214, 170] }, // 黄
      { at: 0.62, c: [255, 248, 238] }, // 白
      { at: 0.84, c: [214, 232, 255] }, // 青白
      { at: 1, c: [170, 202, 255] }, // 青
    ],
    t
  );
}
