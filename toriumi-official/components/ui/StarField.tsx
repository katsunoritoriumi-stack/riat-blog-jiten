"use client";

import { useEffect, useRef } from "react";

type Star = {
  x: number;
  y: number;
  z: number; // depth 0..1 for parallax
  r: number;
  base: number; // base brightness
  tw: number; // twinkle phase
  hue: number;
};

/**
 * 量子パーティクルフィールド：星々のきらめき + 漂うネビュラ + マウス追従。
 * Canvas 2D のみ（three.js 不使用）で軽量に宇宙感を出す。
 */
export default function StarField({
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
    let stars: Star[] = [];
    const mouse = { x: 0.5, y: 0.5, tx: 0.5, ty: 0.5 };

    const palette = [40, 250, 280, 190, 330]; // gold, violet, indigo, cyan, magenta hues

    function build() {
      const el = canvas as HTMLCanvasElement;
      const context = ctx as CanvasRenderingContext2D;
      w = el.clientWidth;
      h = el.clientHeight;
      el.width = w * dpr;
      el.height = h * dpr;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);

      const count = Math.floor(((w * h) / 9000) * density);
      stars = Array.from({ length: count }, () => {
        const z = Math.random();
        return {
          x: Math.random() * w,
          y: Math.random() * h,
          z,
          r: (Math.random() * 1.1 + 0.3) * (0.5 + z),
          base: Math.random() * 0.5 + 0.3,
          tw: Math.random() * Math.PI * 2,
          hue: palette[Math.floor(Math.random() * palette.length)],
        };
      });
    }

    let raf = 0;
    let t = 0;

    function frame() {
      const context = ctx as CanvasRenderingContext2D;
      t += 0.012;
      mouse.x += (mouse.tx - mouse.x) * 0.05;
      mouse.y += (mouse.ty - mouse.y) * 0.05;
      const ox = (mouse.x - 0.5) * 40;
      const oy = (mouse.y - 0.5) * 40;

      context.clearRect(0, 0, w, h);

      for (const s of stars) {
        const twinkle = reduce ? 1 : 0.55 + Math.sin(t * 1.6 + s.tw) * 0.45;
        const px = s.x + ox * s.z;
        const py = s.y + oy * s.z;
        const alpha = Math.max(0, Math.min(1, s.base * twinkle));

        // glow
        context.beginPath();
        const grd = context.createRadialGradient(px, py, 0, px, py, s.r * 6);
        grd.addColorStop(0, `hsla(${s.hue}, 90%, 75%, ${alpha})`);
        grd.addColorStop(1, `hsla(${s.hue}, 90%, 60%, 0)`);
        context.fillStyle = grd;
        context.arc(px, py, s.r * 6, 0, Math.PI * 2);
        context.fill();

        // core
        context.beginPath();
        context.fillStyle = `hsla(${s.hue}, 95%, 92%, ${alpha})`;
        context.arc(px, py, s.r, 0, Math.PI * 2);
        context.fill();
      }

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

  return (
    <canvas
      ref={ref}
      aria-hidden="true"
      className={`block h-full w-full ${className}`}
    />
  );
}
