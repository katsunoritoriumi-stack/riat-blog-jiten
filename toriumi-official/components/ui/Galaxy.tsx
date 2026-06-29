"use client";

import { useEffect, useRef } from "react";

type Star = {
  r: number;       // radius from core
  a: number;       // base angle
  size: number;
  hue: number;
  bright: number;
  tw: number;      // twinkle phase
};

/**
 * 渦巻銀河（spiral galaxy）。差動回転する複数の腕に星を配置し、
 * 円盤を傾けて見たパースで描画。中心はゴールドのコア、腕は紫〜シアン。
 * Canvas 2D のみで軽量に「銀河系」をヒーローに組み込む。
 */
export default function Galaxy({
  className = "",
  density = 1,
}: {
  className?: string;
  density?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    let w = 0;
    let h = 0;
    let R = 0;
    let stars: Star[] = [];
    const arms = 3;
    const spin = 3.1;            // how much the arms wind
    const mouse = { x: 0.5, y: 0.5, tx: 0.5, ty: 0.5 };

    function build() {
      const el = canvas as HTMLCanvasElement;
      const c = ctx as CanvasRenderingContext2D;
      w = el.clientWidth;
      h = el.clientHeight;
      el.width = w * dpr;
      el.height = h * dpr;
      c.setTransform(dpr, 0, 0, dpr, 0, 0);

      R = Math.min(w, h) * 0.5;
      const count = Math.floor(((w * h) / 1600) * density);
      stars = Array.from({ length: count }, () => {
        const t = Math.pow(Math.random(), 0.55);     // concentrate toward core
        const r = t * R;
        const arm = Math.floor(Math.random() * arms);
        const armAngle = (arm / arms) * Math.PI * 2;
        const scatter = (1 - t) * 0.6 + 0.05;
        const a = armAngle + (r / R) * spin * Math.PI + (Math.random() - 0.5) * scatter * Math.PI;
        const hue = t < 0.22 ? 45 : t < 0.55 ? 275 : 190; // gold → violet → cyan
        const bright = (1 - t) * 0.6 + 0.22 + Math.random() * 0.2;
        const size = (1 - t) * 1.3 + 0.35 + Math.random() * 0.7;
        return { r, a, size, hue, bright, tw: Math.random() * Math.PI * 2 };
      });
    }

    let raf = 0;
    let t = 0;

    function frame() {
      const c = ctx as CanvasRenderingContext2D;
      t += reduce ? 0 : 0.0011;
      mouse.x += (mouse.tx - mouse.x) * 0.04;
      mouse.y += (mouse.ty - mouse.y) * 0.04;

      const cx = w / 2 + (mouse.x - 0.5) * 34;
      const cy = h / 2 + (mouse.y - 0.5) * 22;
      const tilt = 0.5; // squash y for disc perspective

      c.clearRect(0, 0, w, h);

      // core glow
      const coreR = R * 0.5;
      const core = c.createRadialGradient(cx, cy, 0, cx, cy, coreR);
      core.addColorStop(0, "rgba(255, 240, 200, 0.9)");
      core.addColorStop(0.25, "rgba(246, 211, 101, 0.4)");
      core.addColorStop(0.6, "rgba(124, 58, 237, 0.12)");
      core.addColorStop(1, "rgba(124, 58, 237, 0)");
      c.fillStyle = core;
      c.beginPath();
      c.arc(cx, cy, coreR, 0, Math.PI * 2);
      c.fill();

      c.globalCompositeOperation = "lighter";
      for (const s of stars) {
        // differential rotation: inner spins faster
        const a = s.a + t * (1.4 - (s.r / R) * 0.7);
        const x = cx + Math.cos(a) * s.r;
        const y = cy + Math.sin(a) * s.r * tilt;
        const tw = reduce ? 1 : 0.6 + 0.4 * Math.sin(t * 26 + s.tw);
        const alpha = s.bright * tw;
        const rad = s.size * 3;
        const g = c.createRadialGradient(x, y, 0, x, y, rad);
        g.addColorStop(0, `hsla(${s.hue}, 92%, 74%, ${alpha})`);
        g.addColorStop(1, `hsla(${s.hue}, 92%, 60%, 0)`);
        c.fillStyle = g;
        c.beginPath();
        c.arc(x, y, rad, 0, Math.PI * 2);
        c.fill();
      }
      c.globalCompositeOperation = "source-over";

      if (!reduce) raf = requestAnimationFrame(frame);
    }

    function onMove(e: MouseEvent) {
      mouse.tx = e.clientX / window.innerWidth;
      mouse.ty = e.clientY / window.innerHeight;
    }

    build();
    frame();
    window.addEventListener("mousemove", onMove);
    const ro = new ResizeObserver(() => {
      build();
      if (reduce) frame();
    });
    ro.observe(canvas);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("mousemove", onMove);
      ro.disconnect();
    };
  }, [density]);

  return <canvas ref={ref} aria-hidden="true" className={`block h-full w-full ${className}`} />;
}
