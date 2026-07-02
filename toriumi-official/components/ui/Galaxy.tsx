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

type Meteor = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;    // 残りフレーム
  maxLife: number;
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

    // 流星（渡り鳥）：まれに一筋、銀河を横切る
    let meteors: Meteor[] = [];
    let nextMeteor = 300 + Math.random() * 300; // frames (~60fps)

    function spawnMeteor() {
      const fromLeft = Math.random() < 0.5;
      const angle = (Math.random() * 0.25 + 0.12) * Math.PI; // 浅い落下角
      const speed = 7 + Math.random() * 5;
      meteors.push({
        x: fromLeft ? -40 : w * (0.3 + Math.random() * 0.7),
        y: Math.random() * h * 0.35,
        vx: Math.cos(angle) * speed * (fromLeft ? 1 : -1),
        vy: Math.sin(angle) * speed,
        life: 90,
        maxLife: 90,
      });
    }

    function drawMeteors(c: CanvasRenderingContext2D) {
      if (--nextMeteor <= 0) {
        spawnMeteor();
        nextMeteor = 500 + Math.random() * 700; // 8〜20秒に一度
      }
      meteors = meteors.filter((m) => m.life > 0);
      for (const m of meteors) {
        m.x += m.vx;
        m.y += m.vy;
        m.life--;
        const fade = Math.sin((m.life / m.maxLife) * Math.PI); // in→out
        const tail = 14;
        const g = c.createLinearGradient(
          m.x, m.y,
          m.x - m.vx * tail, m.y - m.vy * tail
        );
        g.addColorStop(0, `rgba(252, 234, 187, ${0.85 * fade})`);
        g.addColorStop(0.3, `rgba(167, 139, 250, ${0.3 * fade})`);
        g.addColorStop(1, "rgba(167, 139, 250, 0)");
        c.strokeStyle = g;
        c.lineWidth = 1.4;
        c.lineCap = "round";
        c.beginPath();
        c.moveTo(m.x, m.y);
        c.lineTo(m.x - m.vx * tail, m.y - m.vy * tail);
        c.stroke();
        // head
        c.fillStyle = `rgba(255, 248, 225, ${fade})`;
        c.beginPath();
        c.arc(m.x, m.y, 1.6, 0, Math.PI * 2);
        c.fill();
      }
    }

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
      if (!reduce) drawMeteors(c);
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
