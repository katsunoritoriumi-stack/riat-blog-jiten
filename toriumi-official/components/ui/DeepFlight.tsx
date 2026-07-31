"use client";

import { useEffect, useRef } from "react";
import {
  advance,
  cruiseSpeed,
  depthAlpha,
  flightVelocity,
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
      c.globalCompositeOperation = "lighter";

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

      c.globalCompositeOperation = "source-over";
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
