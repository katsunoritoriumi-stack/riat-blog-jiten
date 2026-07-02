"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import SectionHeader from "./ui/SectionHeader";
import { DOMAINS } from "@/lib/content";

export default function ConstellationMap() {
  const center = DOMAINS.find((d) => d.key === "connect")!;
  const outer = DOMAINS.filter((d) => d.key !== "connect");
  const [openKey, setOpenKey] = useState<string | null>(null);

  const dotClass =
    "block h-3 w-3 rounded-full transition-all duration-300 group-hover:scale-150";
  const dotStyle = {
    background: "radial-gradient(circle, #c4b5fd, #7c3aed)",
    boxShadow: "0 0 12px rgba(167,139,250,0.7)",
  } as const;
  const labelClass =
    "pointer-events-none absolute left-1/2 top-full mt-2 -translate-x-1/2 whitespace-nowrap font-mono text-[10px] uppercase tracking-widest text-nebula-300/75 transition-colors group-hover:text-aurum-200 sm:text-xs";

  return (
    <section id="universe" className="relative mx-auto max-w-7xl px-6 py-28 sm:py-36">
      <div className="mb-12 max-w-2xl">
        <SectionHeader
          eyebrow="The Universe of Creation"
          titleEn="One Creator, Many Worlds"
          titleJp="創造の座標軸"
        />
      </div>

      {/* star map — each star is a link */}
      <div className="relative mx-auto aspect-square w-full max-w-2xl overflow-hidden rounded-3xl glass nebula-bg">
        <svg viewBox="0 0 100 100" className="absolute inset-0 h-full w-full">
          {/* 結線：ビューに入ると中心から描き上がる */}
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
          {/* 光のパルス：中心から各星へ、時差をつけて静かに走る */}
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
