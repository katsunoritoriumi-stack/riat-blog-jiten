"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import SectionHeader from "./ui/SectionHeader";
import { DOMAINS } from "@/lib/content";

/* ─────────────────────────────────────────────
   装飾ジオメトリ（SSR ハイドレーション対策で全て丸めた決定値）
   ───────────────────────────────────────────── */
const r2 = (v: number) => Math.round(v * 100) / 100;

/** 背景の微細な星々：シード付き擬似乱数で常に同じ空 */
const BG_STARS = (() => {
  let s = 20260702;
  const rnd = () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
  return Array.from({ length: 64 }, (_, i) => ({
    x: r2(3 + rnd() * 94),
    y: r2(3 + rnd() * 94),
    r: r2(0.1 + rnd() * 0.28),
    o: r2(0.15 + rnd() * 0.45),
    tw: i % 4 === 0, // 4つに1つは瞬く
    d: r2(rnd() * 4),
  }));
})();

/** 外周リングの方位目盛り（15°刻み・90°ごとに長い目盛り） */
const TICKS = Array.from({ length: 24 }, (_, i) => {
  const a = (i * 15 * Math.PI) / 180;
  const inner = i % 6 === 0 ? 44.6 : 45.9;
  return {
    x1: r2(50 + Math.cos(a) * inner),
    y1: r2(50 + Math.sin(a) * inner),
    x2: r2(50 + Math.cos(a) * 47.2),
    y2: r2(50 + Math.sin(a) * 47.2),
    major: i % 6 === 0,
  };
});

/** 方位ラベル（天球図の経度表記） */
const DEGREES = [
  { label: "0°", x: 50, y: 5.2, anchor: "middle" },
  { label: "90°", x: 95.5, y: 51, anchor: "end" },
  { label: "180°", x: 50, y: 96.4, anchor: "middle" },
  { label: "270°", x: 4.5, y: 51, anchor: "start" },
] as const;

/** 同心軌道リング半径 */
const ORBITS = [15, 27, 39, 47.2];

export default function ConstellationMap() {
  const center = DOMAINS.find((d) => d.key === "connect")!;
  const outer = DOMAINS.filter((d) => d.key !== "connect");
  const [openKey, setOpenKey] = useState<string | null>(null);

  // 外周星座線：隣接する星同士を結ぶ（content.ts の並び＝時計回り）
  const perimeter = outer.map((d, i) => {
    const next = outer[(i + 1) % outer.length];
    return { key: `peri-${d.key}`, x1: d.x, y1: d.y, x2: next.x, y2: next.y };
  });

  const dotClass =
    "block h-3 w-3 rounded-full transition-all duration-300 group-hover:scale-150";
  const dotStyle = {
    background: "radial-gradient(circle, #c4b5fd, #7c3aed)",
    boxShadow: "0 0 12px rgba(167,139,250,0.7)",
  } as const;
  const labelClass =
    "pointer-events-none absolute left-1/2 top-full mt-2.5 -translate-x-1/2 whitespace-nowrap font-mono text-[10px] uppercase tracking-widest text-nebula-300/75 transition-colors group-hover:text-aurum-200 sm:text-xs";

  /** 星の回折スパイク（十字の光条） */
  const spikes = (gold = false) => (
    <>
      <span
        className={`pointer-events-none absolute left-1/2 top-1/2 h-px -translate-x-1/2 -translate-y-1/2 bg-gradient-to-r from-transparent to-transparent ${
          gold ? "w-10 via-aurum-200/80" : "w-7 via-nebula-200/60"
        }`}
      />
      <span
        className={`pointer-events-none absolute left-1/2 top-1/2 w-px -translate-x-1/2 -translate-y-1/2 bg-gradient-to-b from-transparent to-transparent ${
          gold ? "h-10 via-aurum-200/80" : "h-7 via-nebula-200/60"
        }`}
      />
    </>
  );

  return (
    <section id="universe" className="relative mx-auto max-w-7xl px-6 py-28 sm:py-36">
      <div className="mb-12 max-w-2xl">
        <SectionHeader
          eyebrow="The Universe of Creation"
          titleEn="One Creator, Many Worlds"
          titleJp="創造の座標軸"
        />
      </div>

      {/* star chart — 深宇宙の天球図。各星がリンク */}
      <div className="relative mx-auto aspect-square w-full max-w-2xl overflow-hidden rounded-3xl border border-nebula-500/20 bg-void-950/60 backdrop-blur-sm">
        {/* 奥行きの星雲ウォッシュ＋ヴィネット */}
        <div className="pointer-events-none absolute inset-0 nebula-bg opacity-60" />
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_45%,rgba(3,2,10,0.7)_100%)]" />

        <svg viewBox="0 0 100 100" className="absolute inset-0 h-full w-full">
          {/* ── 背景の星々 ── */}
          {BG_STARS.map((s, i) => (
            <circle
              key={`bg-${i}`}
              cx={s.x}
              cy={s.y}
              r={s.r}
              fill="#e7e3ff"
              opacity={s.tw ? undefined : s.o}
              className={s.tw ? "star-twinkle" : undefined}
              style={s.tw ? { animationDelay: `${s.d}s` } : undefined}
            />
          ))}

          {/* ── 天球グリッド：同心軌道（外側2本は逆方向に微回転）── */}
          <g className="map-rotate">
            <circle cx="50" cy="50" r={ORBITS[3]} fill="none" stroke="rgba(167,139,250,0.16)" strokeWidth="0.22" strokeDasharray="0.6 1.8" />
            <circle cx="50" cy="50" r={ORBITS[1]} fill="none" stroke="rgba(167,139,250,0.1)" strokeWidth="0.18" strokeDasharray="0.4 2.2" />
          </g>
          <g className="map-rotate-rev">
            <circle cx="50" cy="50" r={ORBITS[2]} fill="none" stroke="rgba(167,139,250,0.12)" strokeWidth="0.18" strokeDasharray="1.4 2.6" />
          </g>
          <circle cx="50" cy="50" r={ORBITS[0]} fill="none" stroke="rgba(240,180,41,0.14)" strokeWidth="0.18" strokeDasharray="0.3 1.6" />

          {/* ── 座標軸（十字）── */}
          <line x1="50" y1="4" x2="50" y2="96" stroke="rgba(167,139,250,0.07)" strokeWidth="0.18" />
          <line x1="4" y1="50" x2="96" y2="50" stroke="rgba(167,139,250,0.07)" strokeWidth="0.18" />
          {/* 中心のクロスヘア */}
          <line x1="46.5" y1="50" x2="53.5" y2="50" stroke="rgba(240,180,41,0.35)" strokeWidth="0.2" />
          <line x1="50" y1="46.5" x2="50" y2="53.5" stroke="rgba(240,180,41,0.35)" strokeWidth="0.2" />

          {/* ── 方位目盛りとラベル ── */}
          {TICKS.map((t, i) => (
            <line
              key={`tick-${i}`}
              x1={t.x1}
              y1={t.y1}
              x2={t.x2}
              y2={t.y2}
              stroke={t.major ? "rgba(240,180,41,0.4)" : "rgba(167,139,250,0.25)"}
              strokeWidth={t.major ? 0.3 : 0.18}
            />
          ))}
          {DEGREES.map((d) => (
            <text
              key={d.label}
              x={d.x}
              y={d.y}
              textAnchor={d.anchor}
              fill="rgba(167,139,250,0.45)"
              style={{ fontSize: "2.6px", fontFamily: "var(--font-mono)", letterSpacing: "0.1em" }}
            >
              {d.label}
            </text>
          ))}

          {/* ── 外周星座線：星から星へ ── */}
          {perimeter.map((l, i) => (
            <motion.line
              key={l.key}
              x1={l.x1}
              y1={l.y1}
              x2={l.x2}
              y2={l.y2}
              stroke="rgba(167,139,250,0.14)"
              strokeWidth={0.22}
              strokeDasharray="1.2 1.6"
              initial={{ pathLength: 0, opacity: 0 }}
              whileInView={{ pathLength: 1, opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 1.2, delay: 1.2 + i * 0.1, ease: "easeOut" }}
            />
          ))}

          {/* ── 結線：ビューに入ると中心から描き上がる ── */}
          {outer.map((d, i) => (
            <motion.line
              key={`l-${d.key}`}
              x1={center.x}
              y1={center.y}
              x2={d.x}
              y2={d.y}
              stroke="rgba(167,139,250,0.22)"
              strokeWidth={0.3}
              initial={{ pathLength: 0, opacity: 0 }}
              whileInView={{ pathLength: 1, opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 1.1, delay: 0.25 + i * 0.12, ease: "easeOut" }}
            />
          ))}
          {/* ── 光のパルス：中心から各星へ、時差をつけて静かに走る ── */}
          {outer.map((d, i) => (
            <motion.circle
              key={`p-${d.key}`}
              r={0.7}
              fill="#fceabb"
              initial={{ opacity: 0 }}
              animate={{
                cx: [center.x, (center.x + d.x) / 2, d.x],
                cy: [center.y, (center.y + d.y) / 2, d.y],
                opacity: [0, 0.85, 0],
              }}
              transition={{
                duration: 2.4,
                delay: 2 + i * 1.7,
                repeat: Infinity,
                repeatDelay: 8.5,
                ease: "easeInOut",
              }}
            />
          ))}
        </svg>

        {/* ── HUD コーナーブラケット＋観測メタ情報 ── */}
        <span className="pointer-events-none absolute left-4 top-4 h-4 w-4 rounded-tl border-l border-t border-aurum-300/40" />
        <span className="pointer-events-none absolute right-4 top-4 h-4 w-4 rounded-tr border-r border-t border-aurum-300/40" />
        <span className="pointer-events-none absolute bottom-4 left-4 h-4 w-4 rounded-bl border-b border-l border-aurum-300/40" />
        <span className="pointer-events-none absolute bottom-4 right-4 h-4 w-4 rounded-br border-b border-r border-aurum-300/40" />
        <span className="pointer-events-none absolute left-7 top-6 font-mono text-[8px] uppercase tracking-[0.25em] text-nebula-300/50 sm:text-[9px]">
          Celestial Map — K.T.
        </span>
        <span className="pointer-events-none absolute bottom-6 right-7 font-mono text-[8px] uppercase tracking-[0.25em] text-nebula-300/50 sm:text-[9px]">
          7 Domains · 1 Core
        </span>

        {/* outer category stars */}
        {outer.map((d, i) => {
          const external = d.href?.startsWith("http");
          const wrapStyle = { left: `${d.x}%`, top: `${d.y}%` };
          const reveal = {
            initial: { opacity: 0, scale: 0 },
            whileInView: { opacity: 1, scale: 1 },
            viewport: { once: true },
            transition: { duration: 0.5, delay: i * 0.08 },
          } as const;

          // multi-link star → toggles a small popover
          if (d.links) {
            const open = openKey === d.key;
            // lower-half stars open the popover upward so it never clips
            const up = d.y > 55;
            return (
              <motion.div
                key={d.key}
                {...reveal}
                className="group absolute z-20 -translate-x-1/2 -translate-y-1/2"
                style={wrapStyle}
              >
                <button
                  onClick={() => setOpenKey(open ? null : d.key)}
                  className="block"
                  aria-label={d.titleEn}
                >
                  <span className="pointer-events-none absolute left-1/2 top-1/2 h-6 w-6 -translate-x-1/2 -translate-y-1/2 animate-pulse-glow rounded-full bg-nebula-400/30 blur-[6px]" />
                  {spikes()}
                  <span className={`${dotClass} ${open ? "scale-150" : ""}`} style={dotStyle} />
                  <span className={labelClass}>{d.titleEn}</span>
                </button>
                {open && (
                  <div
                    className={`absolute left-1/2 flex -translate-x-1/2 flex-col gap-1 rounded-xl glass p-2 ${
                      up ? "bottom-full mb-7" : "top-full mt-7"
                    }`}
                  >
                    {d.links.map((l) => (
                      <a
                        key={l.href}
                        href={l.href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="whitespace-nowrap rounded-lg px-3 py-1.5 text-xs text-nebula-200 transition-colors hover:bg-aurum-400/10 hover:text-aurum-200"
                      >
                        {l.label}
                      </a>
                    ))}
                  </div>
                )}
              </motion.div>
            );
          }

          // single-link star
          return (
            <motion.a
              key={d.key}
              href={d.href}
              target={external ? "_blank" : undefined}
              rel={external ? "noopener noreferrer" : undefined}
              {...reveal}
              whileHover={{ scale: 1.2 }}
              className="group absolute -translate-x-1/2 -translate-y-1/2"
              style={wrapStyle}
              aria-label={d.titleEn}
            >
              <span className="pointer-events-none absolute left-1/2 top-1/2 h-6 w-6 -translate-x-1/2 -translate-y-1/2 animate-pulse-glow rounded-full bg-nebula-400/30 blur-[6px]" />
              {spikes()}
              <span className={dotClass} style={dotStyle} />
              <span className={labelClass}>{d.titleEn}</span>
            </motion.a>
          );
        })}

        {/* center node → LINE */}
        <motion.a
          href={center.href}
          target="_blank"
          rel="noopener noreferrer"
          initial={{ opacity: 0, scale: 0 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4 }}
          whileHover={{ scale: 1.15 }}
          className="group absolute -translate-x-1/2 -translate-y-1/2"
          style={{ left: `${center.x}%`, top: `${center.y}%` }}
          aria-label="LINE でつながる"
        >
          {/* 回転する破線リング：中心＝Connect を静かに強調 */}
          <span className="pointer-events-none absolute -inset-2.5 animate-spin-slow rounded-full border border-dashed border-aurum-300/40" />
          {spikes(true)}
          <span
            className="block h-6 w-6 animate-pulse-glow rounded-full"
            style={{
              background: "radial-gradient(circle, #fceabb, #f0b429)",
              boxShadow: "0 0 26px 4px rgba(240,180,41,0.8)",
            }}
          />
          <span className="pointer-events-none absolute left-1/2 top-full mt-2 -translate-x-1/2 whitespace-nowrap font-mono text-[10px] uppercase tracking-[0.2em] text-aurum-200 sm:text-xs">
            Connect
          </span>
        </motion.a>
      </div>
    </section>
  );
}
