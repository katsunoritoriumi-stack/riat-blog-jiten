"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { playSfx } from "@/lib/sfx";

/**
 * 徘徊UFOイースターエッグ。
 * まれに小さな UFO が画面を横切る。クリック/タップで「捕獲」すると
 * ビームと共に隠しメッセージを残して飛び去る。
 * - 初回はロード後 ~30s（`?ufo` クエリで 1.5s に短縮＝デバッグ用）
 * - 以降 60–120s 間隔、捕獲後は 90s+ のクールダウン
 * - document.hidden 中は延期／reduced-motion では出現しない
 * - z-40（Navbar=50 の下）。乱数はすべてマウント後（ハイドレーション安全）
 */

type Phase = "hidden" | "flying" | "caught";

const MESSAGES = [
  "交信成功 — “Shirankedo...”",
  "👽 観測ログ: この星の創造主を追跡中",
  "SIGNAL: 銀河の海の渡り鳥、確認",
];

const UFO_W = 76; // 表示幅(px)

function UfoSvg() {
  return (
    <svg width={UFO_W} height={UFO_W * 0.55} viewBox="0 0 76 42" fill="none" aria-hidden="true">
      {/* dome */}
      <ellipse cx="38" cy="15" rx="13" ry="10" fill="url(#dome)" opacity="0.9" />
      {/* body */}
      <ellipse cx="38" cy="24" rx="34" ry="10" fill="url(#body)" />
      <ellipse cx="38" cy="21.5" rx="34" ry="8" fill="url(#bodyTop)" />
      {/* lights */}
      {[14, 26, 38, 50, 62].map((x, i) => (
        <circle key={x} cx={x} cy="26.5" r="2" fill="#5eead4" opacity="0.9">
          <animate
            attributeName="opacity"
            values="0.25;1;0.25"
            dur="1.2s"
            begin={`${i * 0.18}s`}
            repeatCount="indefinite"
          />
        </circle>
      ))}
      <defs>
        <linearGradient id="dome" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#c4b5fd" stopOpacity="0.85" />
          <stop offset="1" stopColor="#7c3aed" stopOpacity="0.35" />
        </linearGradient>
        <linearGradient id="body" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#9ca3af" />
          <stop offset="1" stopColor="#374151" />
        </linearGradient>
        <linearGradient id="bodyTop" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#e5e7eb" />
          <stop offset="1" stopColor="#6b7280" />
        </linearGradient>
      </defs>
    </svg>
  );
}

export default function WanderingUfo() {
  const [phase, setPhase] = useState<Phase>("hidden");
  const [flight, setFlight] = useState<{
    fromLeft: boolean;
    topVh: number;
    dur: number;
  } | null>(null);
  const [caught, setCaught] = useState<{ x: number; topVh: number; msg: string } | null>(null);

  const posXRef = useRef(0); // 飛行中の現在 x（捕獲位置の記録用）
  const timerRef = useRef(0);
  const exitTimerRef = useRef(0);
  const disabledRef = useRef(false);

  const spawn = useCallback(() => {
    const fromLeft = Math.random() < 0.5;
    setFlight({
      fromLeft,
      topVh: 12 + Math.random() * 45,
      dur: 12 + Math.random() * 5,
    });
    setPhase("flying");
  }, []);

  const schedule = useCallback(
    (delay: number) => {
      window.clearTimeout(timerRef.current);
      timerRef.current = window.setTimeout(() => {
        if (disabledRef.current) return;
        if (document.hidden) {
          schedule(30000); // タブ非表示中は延期
          return;
        }
        spawn();
      }, delay);
    },
    [spawn]
  );

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      disabledRef.current = true;
      return;
    }
    const debug = window.location.search.includes("ufo");
    schedule(debug ? 1500 : 30000);
    return () => {
      window.clearTimeout(timerRef.current);
      window.clearTimeout(exitTimerRef.current);
    };
  }, [schedule]);

  /** 捕まらずに横断し終えた */
  function onFlightComplete() {
    setPhase("hidden");
    setFlight(null);
    schedule(60000 + Math.random() * 60000);
  }

  /** 捕獲！ */
  function capture() {
    if (phase !== "flying" || !flight) return;
    playSfx("capture");
    const vw = window.innerWidth;
    // メッセージが画面外に出ないよう捕獲位置をクランプ
    const x = Math.min(Math.max(posXRef.current, 12), vw - UFO_W - 12);
    setCaught({ x, topVh: flight.topVh, msg: MESSAGES[Math.floor(Math.random() * MESSAGES.length)] });
    setPhase("caught");
    setFlight(null);
    exitTimerRef.current = window.setTimeout(() => {
      setPhase("hidden");
      setCaught(null);
      schedule(90000 + Math.random() * 30000); // クールダウン
    }, 3400);
  }

  if (phase === "hidden") return null;

  const vw = typeof window !== "undefined" ? window.innerWidth : 1200;

  return (
    <>
      {/* ── 飛行中 ── */}
      {phase === "flying" && flight && (
        <motion.div
          className="fixed left-0 z-40"
          style={{ top: `${flight.topVh}vh` }}
          initial={{ x: flight.fromLeft ? -UFO_W - 20 : vw + 20 }}
          animate={{ x: flight.fromLeft ? vw + 20 : -UFO_W - 20 }}
          transition={{ duration: flight.dur, ease: "linear" }}
          onUpdate={(latest) => {
            if (typeof latest.x === "number") posXRef.current = latest.x;
          }}
          onAnimationComplete={onFlightComplete}
        >
          {/* ふわふわ上下 */}
          <motion.div
            animate={{ y: [0, -12, 2, -8, 0] }}
            transition={{ duration: 4.2, repeat: Infinity, ease: "easeInOut" }}
          >
            <motion.button
              onClick={capture}
              whileHover={{ scale: 1.15, rotate: -4 }}
              whileTap={{ scale: 0.9 }}
              className="relative block cursor-pointer"
              aria-label="UFOを捕獲する"
            >
              {/* 44px タップ領域 */}
              <span className="absolute left-1/2 top-1/2 h-12 w-24 -translate-x-1/2 -translate-y-1/2" />
              <UfoSvg />
            </motion.button>
          </motion.div>
        </motion.div>
      )}

      {/* ── 捕獲リアクション ── */}
      <AnimatePresence>
        {phase === "caught" && caught && (
          <motion.div
            key="caught"
            className="pointer-events-none fixed z-40"
            style={{ left: caught.x, top: `${caught.topVh}vh` }}
            initial={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -140, transition: { duration: 0.6, ease: "easeIn" } }}
          >
            {/* 驚いて一瞬跳ねる UFO */}
            <motion.div
              initial={{ y: 0 }}
              animate={{ y: [0, -14, 0], rotate: [0, -6, 4, 0] }}
              transition={{ duration: 0.7, ease: "easeOut" }}
            >
              <UfoSvg />
            </motion.div>

            {/* ビーム */}
            <motion.div
              initial={{ opacity: 0, scaleY: 0 }}
              animate={{ opacity: [0, 0.9, 0.7], scaleY: 1 }}
              transition={{ duration: 0.5, delay: 0.15 }}
              className="absolute left-1/2 top-[26px] h-28 w-16 origin-top -translate-x-1/2"
              style={{
                clipPath: "polygon(38% 0%, 62% 0%, 100% 100%, 0% 100%)",
                background:
                  "linear-gradient(to bottom, rgba(252,234,187,0.75), rgba(252,234,187,0.05))",
                filter: "blur(1px)",
              }}
            />

            {/* 隠しメッセージ */}
            <motion.div
              initial={{ opacity: 0, y: 8, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.5, ease: [0.22, 1, 0.36, 1] }}
              className="glass absolute left-1/2 top-[140px] -translate-x-1/2 whitespace-nowrap rounded-full px-4 py-2 font-mono text-[11px] tracking-wider text-aurum-200"
            >
              {caught.msg}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
