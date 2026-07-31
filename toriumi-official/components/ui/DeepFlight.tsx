"use client";

import { useEffect, useRef } from "react";
import {
  advance,
  cruiseSpeed,
  depthAlpha,
  flightVelocity,
  smoothstep,
  thrustStep,
  Z_FAR,
  Z_NEAR,
} from "@/lib/flightMath";

/**
 * 宇宙を前進する星屑（Canvas 2D）。
 * スクロールするとカメラが奥へ進み、点が手前へ流れてストリークを引く。
 * 止めても完全には停まらず、ゆっくりとした巡航に戻る。
 *
 * 実装メモ：
 * - 速度は rAF ループ内で window.scrollY を直接読む。scroll イベントにも lenis にも
 *   依存しない（lenis は prefers-reduced-motion 下では存在しないため）。
 * - Galaxy.tsx は星ごと・フレームごとに createRadialGradient していて重い。
 *   ここではスプライトを3枚だけ焼いて drawImage する。
 * - ストリークは色ごとに1回の stroke() へバッチする（描画3回で済む）。
 * - 数式は lib/flightMath.ts 側。rAF が動かない環境でも数値検証できるようにしてある。
 */

/**
 * 生成時に画面のどこまで広げて撒くか（画面半幅・半高に対する倍率）。
 * 世界座標で一様に撒くと、投影後はほとんどが画面外に落ちてスカスカになる
 * （実測で画面内 14%）。生成時の z に応じて視錐台に沿って撒くことで、
 * 画面上でほぼ一様な星空になり、近づくにつれて外へ流れていく。
 */
const SPAWN_SPREAD = 0.42;
/** 点の基準半径（世界単位）。画面半径は SIZE * focal / z */
const SIZE = 0.0026;

/** gold / violet / cyan — サイトのパレットに合わせる */
const HUES: [string, string][] = [
  ["#fceabb", "rgba(252,234,187,"],
  ["#a78bfa", "rgba(167,139,250,"],
  ["#5eead4", "rgba(94,234,212,"],
];

/* ── 航行中に通り過ぎる天体（星雲・惑星） ──────────────
   星屑よりずっと遠くから、ゆっくり近づいて脇を通り抜ける。
   セクションとセクションの間（何も表示されない航行区間）に
   「移動している」実感を与えるための要素。
   ───────────────────────────────────────────── */

const FLYBY_FAR = 5; // 飛来物の最遠。星屑(1.6)よりずっと遠く＝通過に時間がかかる
const FLYBY_NEAR = 0.16;

/** 星雲の色（サイトのパレット寄り） */
const NEBULA_RGB: [number, number, number][] = [
  [124, 58, 237], // 紫
  [56, 132, 214], // 青
  [214, 92, 148], // 薄紅
];
/** 惑星の色 */
const PLANET_RGB: [number, number, number][] = [
  [196, 138, 92], // 砂色
  [96, 132, 196], // 青
  [172, 150, 210], // 藤色（環あり）
];

function newSprite(size: number) {
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  return c;
}

/** もやのかたまりを重ねて星雲を焼く */
function makeNebulaSprite([r, g, b]: [number, number, number]): HTMLCanvasElement {
  const S = 256;
  const cv = newSprite(S);
  const x = cv.getContext("2d")!;
  x.globalCompositeOperation = "lighter";
  for (let i = 0; i < 7; i++) {
    const px = S / 2 + (Math.random() - 0.5) * S * 0.55;
    const py = S / 2 + (Math.random() - 0.5) * S * 0.55;
    const rad = S * (0.16 + Math.random() * 0.26);
    const grd = x.createRadialGradient(px, py, 0, px, py, rad);
    grd.addColorStop(0, `rgba(${r},${g},${b},0.5)`);
    grd.addColorStop(0.45, `rgba(${r},${g},${b},0.14)`);
    grd.addColorStop(1, "rgba(0,0,0,0)");
    x.fillStyle = grd;
    x.beginPath();
    x.arc(px, py, rad, 0, Math.PI * 2);
    x.fill();
  }
  // 中に散る小さな輝き
  for (let i = 0; i < 26; i++) {
    const px = S / 2 + (Math.random() - 0.5) * S * 0.7;
    const py = S / 2 + (Math.random() - 0.5) * S * 0.7;
    x.fillStyle = `rgba(255,255,255,${0.25 + Math.random() * 0.5})`;
    x.beginPath();
    x.arc(px, py, Math.random() * 1.4 + 0.4, 0, Math.PI * 2);
    x.fill();
  }
  return cv;
}

/** 光の当たった球として惑星を焼く（環つきも作れる） */
function makePlanetSprite([r, g, b]: [number, number, number], ring: boolean): HTMLCanvasElement {
  const S = 256;
  const cv = newSprite(S);
  const x = cv.getContext("2d")!;
  const cx = S / 2;
  const cy = S / 2;
  const R = S * 0.3;
  const lit = `rgb(${Math.min(255, r + 78)},${Math.min(255, g + 74)},${Math.min(255, b + 66)})`;
  const mid = `rgb(${r},${g},${b})`;
  const dark = `rgb(${Math.round(r * 0.14)},${Math.round(g * 0.14)},${Math.round(b * 0.2)})`;

  const drawRing = () => {
    x.save();
    x.translate(cx, cy);
    x.rotate(-0.32);
    x.strokeStyle = `rgba(${Math.min(255, r + 40)},${Math.min(255, g + 36)},${Math.min(255, b + 30)},0.5)`;
    x.lineWidth = S * 0.028;
    x.beginPath();
    x.ellipse(0, 0, R * 1.75, R * 0.46, 0, 0, Math.PI * 2);
    x.stroke();
    x.restore();
  };

  if (ring) drawRing(); // 奥側の環（このあと本体で半分隠れる）

  const grd = x.createRadialGradient(cx - R * 0.42, cy - R * 0.42, R * 0.05, cx, cy, R * 1.02);
  grd.addColorStop(0, lit);
  grd.addColorStop(0.45, mid);
  grd.addColorStop(1, dark);
  x.fillStyle = grd;
  x.beginPath();
  x.arc(cx, cy, R, 0, Math.PI * 2);
  x.fill();

  if (ring) {
    // 手前を通る側だけ描き足す（本体の下半分にかかる部分）
    x.save();
    x.beginPath();
    x.rect(0, cy + R * 0.12, S, S);
    x.clip();
    drawRing();
    x.restore();
  }
  return cv;
}

type Flyby = {
  kind: 0 | 1; // 0=星雲 1=惑星
  sprite: number;
  x: number;
  y: number;
  z: number;
  size: number; // 世界単位の半径
  rot: number;
  spin: number;
};

/** 遠すぎ／近すぎで消える（星屑より緩やかに出入りする） */
const flybyAlpha = (z: number) =>
  smoothstep(FLYBY_FAR, FLYBY_FAR * 0.5, z) * smoothstep(FLYBY_NEAR, FLYBY_NEAR * 3.4, z);

type P = {
  x: number;
  y: number;
  z: number;
  px: number; // 前フレームの投影位置（NaN = ストリークを引かない）
  py: number;
  hue: number;
  b: number;
};

/** 深度に応じた色の偏り：手前（浅い層）ほど金、奥へ行くほど青緑 */
function pickHue(prog: number): number {
  const r = Math.random();
  const cyanBias = 0.15 + prog * 0.4;
  if (r < 0.18) return 0;
  return r < 0.18 + (1 - cyanBias) * 0.82 ? 1 : 2;
}

export default function DeepFlight({ className = "" }: { className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const motionMq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const coarse = window.matchMedia("(pointer: coarse)").matches;
    let reduce = motionMq.matches;

    // 全画面のぼんやりした背景なので 2 ではなく 1.5 で十分（画素数 44% 減）
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);

    let w = 0;
    let h = 0;
    let focal = 0;
    let points: P[] = [];
    const mouse = { x: 0.5, y: 0.5, tx: 0.5, ty: 0.5 };

    // ── スプライトは最初に一度だけ焼く ──
    const sprites = HUES.map(([hex]) => {
      const s = document.createElement("canvas");
      s.width = 24;
      s.height = 24;
      const sc = s.getContext("2d")!;
      const g = sc.createRadialGradient(12, 12, 0, 12, 12, 12);
      g.addColorStop(0, "rgba(255,255,255,1)");
      g.addColorStop(0.35, hex);
      g.addColorStop(1, "rgba(0,0,0,0)");
      sc.fillStyle = g;
      sc.beginPath();
      sc.arc(12, 12, 12, 0, Math.PI * 2);
      sc.fill();
      return s;
    });

    // 飛来物のスプライトも最初に一度だけ焼く
    const nebulaSprites = NEBULA_RGB.map(makeNebulaSprite);
    const planetSprites = PLANET_RGB.map((c, i) => makePlanetSprite(c, i === 2));
    let flybys: Flyby[] = [];

    function spawnFlyby(f: Flyby, atFar: boolean) {
      f.z = atFar ? FLYBY_FAR : FLYBY_NEAR + Math.random() * (FLYBY_FAR - FLYBY_NEAR);
      // 画面中央（＝本文が乗る場所）を避け、脇を通るように配置する。
      // 位置は「z=1 のときの画面上の距離」で決めてから世界座標へ逆算する。
      const ang = Math.random() * Math.PI * 2;
      const half = Math.min(w, h) / 2;
      const spread =
        f.kind === 0 ? 0.3 + Math.random() * 0.75 : 0.5 + Math.random() * 0.85;
      f.x = (Math.cos(ang) * half * spread) / focal;
      f.y = (Math.sin(ang) * half * spread) / focal;
      f.rot = Math.random() * Math.PI * 2;
      f.spin = (Math.random() - 0.5) * 0.0022;
      f.size =
        f.kind === 0 ? 0.24 + Math.random() * 0.3 : 0.045 + Math.random() * 0.055;
      f.sprite = Math.floor(
        Math.random() * (f.kind === 0 ? nebulaSprites.length : planetSprites.length)
      );
    }

    function spawn(p: P, prog: number, atFar: boolean) {
      p.z = atFar ? Z_FAR : Z_NEAR + Math.random() * (Z_FAR - Z_NEAR);
      // その深さで画面に写る範囲へ撒く（視錐台に沿わせる）。
      // 画面上の位置 = 世界座標 * focal / z なので、逆算して z を掛ける。
      const rx = (Math.random() * 2 - 1) * w * SPAWN_SPREAD;
      const ry = (Math.random() * 2 - 1) * h * SPAWN_SPREAD;
      p.x = (rx * p.z) / focal;
      p.y = (ry * p.z) / focal;
      p.px = NaN; // ← 忘れると画面を横断する線が描かれる
      p.py = NaN;
      p.hue = pickHue(prog);
      p.b = 0.35 + Math.random() * 0.65;
    }

    function build() {
      const el = canvas as HTMLCanvasElement;
      const c = ctx as CanvasRenderingContext2D;
      w = el.clientWidth;
      h = el.clientHeight;
      if (w === 0 || h === 0) return;
      el.width = Math.round(w * dpr);
      el.height = Math.round(h * dpr);
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      focal = 0.9 * Math.min(w, h);

      // 生成した点のうち画面に写るのは実測 4割強。同時に 100 個前後が見える密度に合わせる
      let count = Math.round((w * h) / 3200);
      count = Math.max(180, Math.min(460, count));
      if (coarse) count = Math.round(count * 0.7);

      points = Array.from({ length: count }, () => {
        const p: P = { x: 0, y: 0, z: 0, px: NaN, py: NaN, hue: 0, b: 1 };
        spawn(p, 0, false);
        return p;
      });

      // 星雲と惑星。数を絞り、奥行きをばらけさせて「たまに通り過ぎる」密度にする
      const kinds: (0 | 1)[] = coarse ? [0, 0, 1, 1, 0] : [0, 0, 0, 0, 0, 1, 1, 1];
      flybys = kinds.map((kind, i) => {
        const f: Flyby = { kind, sprite: 0, x: 0, y: 0, z: 0, size: 0, rot: 0, spin: 0 };
        spawnFlyby(f, false);
        // 等間隔にずらして、同時に通り過ぎないようにする
        f.z = FLYBY_NEAR + ((i + 0.5) / kinds.length) * (FLYBY_FAR - FLYBY_NEAR);
        return f;
      });
    }

    let raf = 0;
    let last = 0;
    let lastY = 0;
    let thrust = 0;

    function frame(now: number) {
      const c = ctx as CanvasRenderingContext2D;
      if (w === 0 || h === 0) {
        raf = requestAnimationFrame(frame);
        return;
      }

      const dt = last ? Math.min(50, now - last) / 16.667 : 1;
      last = now;

      const y = window.scrollY;
      const dy = reduce ? 0 : (y - lastY) / Math.max(dt, 0.0001);
      lastY = y;

      const maxScroll =
        document.documentElement.scrollHeight - window.innerHeight;
      const prog = maxScroll > 0 ? Math.min(1, Math.max(0, y / maxScroll)) : 0;

      thrust = thrustStep(thrust, dy, dt);
      const vz = reduce ? 0 : flightVelocity(cruiseSpeed(prog), thrust, dt);

      mouse.x += (mouse.tx - mouse.x) * 0.04;
      mouse.y += (mouse.ty - mouse.y) * 0.04;
      const cx = w / 2 + (mouse.x - 0.5) * 40;
      const cy = h / 2 + (mouse.y - 0.5) * 28;

      c.clearRect(0, 0, w, h);

      // ── 飛来物を進める（描画は種類ごとに順序を分ける） ──
      for (const f of flybys) {
        if (vz > 0) {
          // 星屑より遅く進ませる＝遠くをゆっくり通り過ぎて見える
          const next = f.z - vz * 0.7;
          if (next <= FLYBY_NEAR) spawnFlyby(f, true);
          else f.z = next;
        }
        if (!reduce) f.rot += f.spin * dt;
      }

      const drawFlyby = (kind: 0 | 1) => {
        for (const f of flybys) {
          if (f.kind !== kind) continue;
          const a = flybyAlpha(f.z);
          if (a <= 0.004) continue;
          const k = focal / f.z;
          const sx = cx + f.x * k;
          const sy = cy + f.y * k;
          const rad = f.size * k;
          if (sx + rad < 0 || sx - rad > w || sy + rad < 0 || sy - rad > h) continue;
          const sprite = kind === 0 ? nebulaSprites[f.sprite] : planetSprites[f.sprite];
          c.globalAlpha = Math.min(1, a * (kind === 0 ? 0.72 : 1));
          c.save();
          c.translate(sx, sy);
          c.rotate(f.rot);
          c.drawImage(sprite, -rad, -rad, rad * 2, rad * 2);
          c.restore();
        }
        c.globalAlpha = 1;
      };

      // 星雲はいちばん奥。加算合成でにじませる
      c.globalCompositeOperation = "lighter";
      drawFlyby(0);

      // 色ごとにストリークを溜めて、最後にまとめて描く
      const streaks: number[][] = [[], [], []];
      const margin = Math.max(w, h) * 0.15;
      const maxStreak = Math.max(w, h) * 0.6;

      for (const p of points) {
        if (vz > 0) {
          const r = advance(p.z, vz);
          if (r.recycled) spawn(p, prog, true);
          else p.z = r.z;
        }

        const k = focal / p.z;
        const sx = cx + p.x * k;
        const sy = cy + p.y * k;

        if (sx < -margin || sx > w + margin || sy < -margin || sy > h + margin) {
          p.px = NaN;
          p.py = NaN;
          continue;
        }

        const alpha = p.b * depthAlpha(p.z);
        if (alpha <= 0.004) {
          p.px = sx;
          p.py = sy;
          continue;
        }

        // ストリーク（前フレーム位置との線分＝実質モーションブラー）
        if (!reduce && !Number.isNaN(p.px)) {
          const d = Math.hypot(sx - p.px, sy - p.py);
          if (d > 2 && d < maxStreak) {
            streaks[p.hue].push(p.px, p.py, sx, sy, alpha, k);
          }
        }

        // 頭（スプライト）
        // 下限 0.9px。これ以下だとスマホでサブピクセルになって見えなくなる
        const rad = Math.max(0.9, SIZE * k);
        c.globalAlpha = Math.min(1, alpha);
        c.drawImage(sprites[p.hue], sx - rad, sy - rad, rad * 2, rad * 2);

        p.px = sx;
        p.py = sy;
      }

      c.globalAlpha = 1;
      c.lineCap = "round";
      for (let hue = 0; hue < 3; hue++) {
        const arr = streaks[hue];
        if (arr.length === 0) continue;
        // 同じ色はまとめて1パス。線幅と不透明度は平均で代表させる
        let aSum = 0;
        let kSum = 0;
        c.beginPath();
        for (let i = 0; i < arr.length; i += 6) {
          c.moveTo(arr[i], arr[i + 1]);
          c.lineTo(arr[i + 2], arr[i + 3]);
          aSum += arr[i + 4];
          kSum += arr[i + 5];
        }
        const n = arr.length / 6;
        c.strokeStyle = HUES[hue][1] + Math.min(0.85, (aSum / n) * 0.9) + ")";
        c.lineWidth = Math.max(0.4, Math.min(2.2, SIZE * (kSum / n) * 1.1));
        c.stroke();
      }

      // 惑星は実体なので最後に不透明で描く（星が透けない）
      c.globalCompositeOperation = "source-over";
      drawFlyby(1);

      if (!reduce) raf = requestAnimationFrame(frame);
    }

    function onMove(e: MouseEvent) {
      mouse.tx = e.clientX / window.innerWidth;
      mouse.ty = e.clientY / window.innerHeight;
    }

    function start() {
      if (raf || reduce) return;
      // 復帰1フレーム目に巨大な dy を拾ってワープしないよう基準を取り直す
      last = 0;
      lastY = window.scrollY;
      thrust = 0;
      raf = requestAnimationFrame(frame);
    }
    function stop() {
      cancelAnimationFrame(raf);
      raf = 0;
    }

    function onVisibility() {
      if (document.hidden) stop();
      else start();
    }

    function onMotionChange() {
      reduce = motionMq.matches;
      stop();
      if (reduce) frame(performance.now()); // 静止1枚だけ描く
      else start();
    }

    build();
    lastY = window.scrollY;
    if (reduce) {
      frame(performance.now());
    } else {
      window.addEventListener("mousemove", onMove);
      start();
    }

    document.addEventListener("visibilitychange", onVisibility);
    motionMq.addEventListener("change", onMotionChange);
    const ro = new ResizeObserver(() => {
      build();
      if (reduce) frame(performance.now());
    });
    ro.observe(canvas);

    return () => {
      stop();
      window.removeEventListener("mousemove", onMove);
      document.removeEventListener("visibilitychange", onVisibility);
      motionMq.removeEventListener("change", onMotionChange);
      ro.disconnect();
    };
  }, []);

  return (
    <canvas
      ref={ref}
      aria-hidden="true"
      className={`block h-full w-full ${className}`}
    />
  );
}
