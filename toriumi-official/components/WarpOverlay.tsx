"use client";

import { useEffect, useRef } from "react";
import { getLenis } from "@/lib/lenisBridge";
import { isBootDone } from "@/lib/bootGate";
import { setDepthBypass } from "@/lib/depthBypass";
import { playSfx } from "@/lib/sfx";

/**
 * ハイパースペース・ワープによるページ内ナビゲーション。
 * capture-phase の document click を 1 本フックし、`a[href^="#"]` のクリックを
 * 星が放射状に流れる全画面 Canvas 演出（約 750ms）に差し替える。
 * 暗転のピーク（350ms）で lenis の instant スクロール → 明転して着地。
 * - 外部リンク / _blank / button には一切影響しない
 * - reduced-motion・演出中・ブート前は素通し（通常のアンカージャンプ）
 */

const DURATION = 750; // ms
const PEAK = 350; // 暗転ピーク＝瞬間移動のタイミング
const STAR_COUNT = 170;

type WarpStar = {
  angle: number;
  dist: number; // 中心からの距離（0-1、対角半径比）
  speed: number;
  width: number;
  hue: number; // 0=白金 / 1=紫
};

export default function WarpOverlay() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const activeRef = useRef(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let raf = 0;

    function warpTo(el: HTMLElement) {
      const c = canvas as HTMLCanvasElement;
      const ctx = c.getContext("2d");
      if (!ctx) {
        el.scrollIntoView();
        return;
      }
      activeRef.current = true;

      // キャンバス準備（演出のたびにサイズ確保）
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = window.innerWidth;
      const h = window.innerHeight;
      c.width = w * dpr;
      c.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      c.style.opacity = "1";
      c.style.pointerEvents = "auto"; // 演出中の誤クリックを遮断

      const cx = w / 2;
      const cy = h / 2;
      const maxR = Math.hypot(cx, cy);

      const stars: WarpStar[] = Array.from({ length: STAR_COUNT }, () => ({
        angle: Math.random() * Math.PI * 2,
        dist: Math.random() * 0.85 + 0.05,
        speed: Math.random() * 0.9 + 0.35,
        width: Math.random() * 1.6 + 0.5,
        hue: Math.random(),
      }));

      const lenis = getLenis();
      lenis?.stop();
      playSfx("warp");

      // 到着アニメで縮んでいるセクションを lenis が掴むと着地位置がずれるため、
      // ワープ中だけ全セクションを無変換へ戻す（暗転ピークまで 350ms の猶予がある）
      setDepthBypass(true);

      let jumped = false;
      const start = performance.now();

      const easeInCubic = (t: number) => t * t * t;
      const frame = (now: number) => {
        const t = Math.min(1, (now - start) / DURATION);

        // 暗転：0 → 0.95（ピーク）→ 0 の山なり
        const dark =
          t < PEAK / DURATION
            ? (t / (PEAK / DURATION)) * 0.95
            : (1 - (t - PEAK / DURATION) / (1 - PEAK / DURATION)) * 0.95;

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = `rgba(3, 2, 10, ${dark})`;
        ctx.fillRect(0, 0, w, h);

        // 星ストリーク：進行に応じて加速・伸長
        const stretch = easeInCubic(t < 0.5 ? t * 2 : (1 - t) * 2); // 山なりに伸びる
        ctx.lineCap = "round";
        for (const s of stars) {
          s.dist += s.speed * 0.035; // 外へ加速
          if (s.dist > 1.1) s.dist -= 1.05;
          const r0 = s.dist * maxR;
          const len = (s.speed * 90 + 30) * stretch;
          const x0 = cx + Math.cos(s.angle) * r0;
          const y0 = cy + Math.sin(s.angle) * r0;
          const x1 = cx + Math.cos(s.angle) * (r0 + len);
          const y1 = cy + Math.sin(s.angle) * (r0 + len);
          const alpha = Math.min(1, stretch * 1.4) * (0.35 + s.dist * 0.6);
          ctx.strokeStyle =
            s.hue < 0.55
              ? `rgba(252, 234, 187, ${alpha})`
              : `rgba(167, 139, 250, ${alpha})`;
          ctx.lineWidth = s.width;
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(x1, y1);
          ctx.stroke();
        }

        // 暗転ピークで瞬間移動（1回だけ）
        if (!jumped && now - start >= PEAK) {
          jumped = true;
          const l = getLenis();
          if (l) l.scrollTo(el, { immediate: true, force: true });
          else el.scrollIntoView();
        }

        if (t < 1) {
          raf = requestAnimationFrame(frame);
        } else {
          // 終了処理
          ctx.clearRect(0, 0, w, h);
          c.style.opacity = "0";
          c.style.pointerEvents = "none";
          getLenis()?.start();
          setDepthBypass(false);
          activeRef.current = false;
        }
      };
      raf = requestAnimationFrame(frame);
    }

    function onDocClick(e: MouseEvent) {
      const target = e.target as Element | null;
      const a = target?.closest?.('a[href^="#"]') as HTMLAnchorElement | null;
      if (!a) return;
      const id = a.getAttribute("href")!.slice(1);
      if (!id) return;
      const el = document.getElementById(id);
      if (!el) return;
      // ワープ演出中の追加クリックは握りつぶす（途中のネイティブジャンプ防止）
      if (activeRef.current) {
        e.preventDefault();
        return;
      }
      // reduced-motion / ブート前は通常ジャンプに任せる
      if (reduce || !isBootDone()) return;
      e.preventDefault();
      history.pushState(null, "", `#${id}`);
      warpTo(el);
    }

    document.addEventListener("click", onDocClick, true);
    return () => {
      document.removeEventListener("click", onDocClick, true);
      cancelAnimationFrame(raf);
      getLenis()?.start();
      setDepthBypass(false);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="fixed inset-0 z-[70] h-full w-full opacity-0 transition-opacity duration-150"
      style={{ pointerEvents: "none" }}
    />
  );
}
