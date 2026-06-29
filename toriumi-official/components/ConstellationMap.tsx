"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import SectionHeader from "./ui/SectionHeader";
import { DOMAINS } from "@/lib/content";

export default function ConstellationMap() {
  const center = DOMAINS.find((d) => d.key === "connect")!;
  const outer = DOMAINS.filter((d) => d.key !== "connect");
  const [active, setActive] = useState<string>(outer[0].key);
  const activeDomain = outer.find((d) => d.key === active)!;

  return (
    <section id="universe" className="relative mx-auto max-w-7xl px-6 py-28 sm:py-36">
      <div className="mb-16 max-w-2xl">
        <SectionHeader
          eyebrow="The Universe of Creation"
          titleEn="One Creator, Many Worlds"
          titleJp="創造の宇宙"
        />
        <p className="mt-6 text-nebula-200/70">
          ひとつの創造のエネルギーが、いくつもの領域に立ち現れる。星をなぞるように、彼の宇宙を巡ってみてください。
        </p>
      </div>

      <div className="grid items-center gap-10 lg:grid-cols-[1.4fr_1fr]">
        {/* star map */}
        <div className="relative aspect-square w-full overflow-hidden rounded-3xl glass nebula-bg">
          <svg viewBox="0 0 100 100" className="absolute inset-0 h-full w-full">
            {outer.map((d) => (
              <line
                key={`l-${d.key}`}
                x1={center.x}
                y1={center.y}
                x2={d.x}
                y2={d.y}
                stroke="rgba(167,139,250,0.25)"
                strokeWidth={active === d.key ? 0.5 : 0.25}
              />
            ))}
          </svg>

          {/* outer nodes (selectable) */}
          {outer.map((d, i) => {
            const isActive = active === d.key;
            return (
              <motion.button
                key={d.key}
                onMouseEnter={() => setActive(d.key)}
                onClick={() => setActive(d.key)}
                initial={{ opacity: 0, scale: 0 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.08 }}
                className="group absolute -translate-x-1/2 -translate-y-1/2"
                style={{ left: `${d.x}%`, top: `${d.y}%` }}
                aria-label={d.titleJp}
              >
                <span
                  className={`block h-3 w-3 rounded-full transition-all duration-300 ${isActive ? "scale-150" : ""}`}
                  style={{
                    background: "radial-gradient(circle, #c4b5fd, #7c3aed)",
                    boxShadow: isActive
                      ? "0 0 22px rgba(167,139,250,0.95)"
                      : "0 0 12px rgba(167,139,250,0.7)",
                  }}
                />
                <span
                  className={`pointer-events-none absolute left-1/2 top-full mt-2 -translate-x-1/2 whitespace-nowrap font-mono text-[10px] uppercase tracking-widest transition-opacity sm:text-xs ${
                    isActive ? "text-aurum-200 opacity-100" : "text-nebula-300/70 opacity-70"
                  }`}
                >
                  {d.titleEn}
                </span>
              </motion.button>
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

        {/* active detail */}
        <motion.div
          key={activeDomain.key}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="rounded-3xl glass p-8 sm:p-10"
        >
          <span className="font-mono text-sm uppercase tracking-cosmic text-aurum-300/80">
            {activeDomain.titleEn}
          </span>
          <h3 className="mt-3 font-serif text-3xl text-nebula-100 sm:text-4xl">
            {activeDomain.titleJp}
          </h3>
          <p className="mt-5 text-lg leading-relaxed text-nebula-200/80">
            {activeDomain.blurb}
          </p>
          <div className="mt-8 flex flex-wrap gap-2">
            {outer.map((d) => (
              <button
                key={d.key}
                onClick={() => setActive(d.key)}
                className={`rounded-full border px-3 py-1 text-xs transition-colors ${
                  d.key === active
                    ? "border-aurum-400/60 text-aurum-200"
                    : "border-nebula-500/20 text-nebula-300/60 hover:text-nebula-200"
                }`}
              >
                {d.titleJp}
              </button>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
