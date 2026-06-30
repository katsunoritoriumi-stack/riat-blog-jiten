"use client";

import { motion } from "framer-motion";
import SectionHeader from "./ui/SectionHeader";
import { DOMAINS } from "@/lib/content";

export default function ConstellationMap() {
  const center = DOMAINS.find((d) => d.key === "connect")!;
  const outer = DOMAINS.filter((d) => d.key !== "connect");

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
          {outer.map((d) => (
            <line
              key={`l-${d.key}`}
              x1={center.x}
              y1={center.y}
              x2={d.x}
              y2={d.y}
              stroke="rgba(167,139,250,0.22)"
              strokeWidth={0.3}
            />
          ))}
        </svg>

        {/* outer category stars */}
        {outer.map((d, i) => {
          const external = d.href?.startsWith("http");
          return (
            <motion.a
              key={d.key}
              href={d.href}
              target={external ? "_blank" : undefined}
              rel={external ? "noopener noreferrer" : undefined}
              initial={{ opacity: 0, scale: 0 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.08 }}
              whileHover={{ scale: 1.2 }}
              className="group absolute -translate-x-1/2 -translate-y-1/2"
              style={{ left: `${d.x}%`, top: `${d.y}%` }}
              aria-label={d.titleEn}
            >
              <span
                className="block h-3 w-3 rounded-full transition-all duration-300 group-hover:scale-150"
                style={{
                  background: "radial-gradient(circle, #c4b5fd, #7c3aed)",
                  boxShadow: "0 0 12px rgba(167,139,250,0.7)",
                }}
              />
              <span className="pointer-events-none absolute left-1/2 top-full mt-2 -translate-x-1/2 whitespace-nowrap font-mono text-[10px] uppercase tracking-widest text-nebula-300/75 transition-colors group-hover:text-aurum-200 sm:text-xs">
                {d.titleEn}
              </span>
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
