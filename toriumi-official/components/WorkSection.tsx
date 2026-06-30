"use client";

import { motion } from "framer-motion";
import { ArrowUpRight } from "lucide-react";
import SectionHeader from "./ui/SectionHeader";
import { WORKS } from "@/lib/content";

/** 各サイトのトップ画面サムネイル（WordPress mShots・無料・キー不要） */
function shot(url: string) {
  return `https://s.wordpress.com/mshots/v1/${encodeURIComponent(url)}?w=800&h=600`;
}

export default function WorkSection() {
  return (
    <section id="work" className="relative mx-auto max-w-7xl px-6 py-28 sm:py-36">
      <div className="mb-12">
        <SectionHeader eyebrow="Collection" titleEn="Works" />
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
        {WORKS.map((work, i) => (
          <motion.a
            key={work.title}
            href={work.href}
            target={work.href ? "_blank" : undefined}
            rel={work.href ? "noopener noreferrer" : undefined}
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-40px" }}
            transition={{ duration: 0.5, delay: i * 0.06 }}
            whileHover={{ y: -6 }}
            className="group relative flex aspect-[4/3] flex-col justify-between overflow-hidden rounded-xl border border-nebula-500/15 bg-void-800/60"
          >
            {/* site screenshot */}
            {work.href && (
              <img
                src={shot(work.href)}
                alt={`${work.title} のスクリーンショット`}
                loading="lazy"
                className="absolute inset-0 h-full w-full object-cover object-top opacity-70 transition-all duration-500 group-hover:scale-105 group-hover:opacity-90"
              />
            )}
            {/* legibility gradient */}
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-void-950 via-void-950/40 to-void-950/10" />
            <div
              className="pointer-events-none absolute -right-8 -top-10 h-24 w-24 rounded-full opacity-40 blur-2xl transition-opacity duration-500 group-hover:opacity-70"
              style={{ background: work.hue }}
            />

            <div className="relative flex items-center justify-between p-4">
              <span className="font-mono text-[10px] text-nebula-100/80">
                {String(i + 1).padStart(2, "0")}
              </span>
              <ArrowUpRight
                size={16}
                className="text-nebula-100/70 transition-all group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:text-aurum-200"
              />
            </div>

            <div className="relative p-4">
              <span
                className="font-mono text-[9px] uppercase tracking-wider"
                style={{ color: work.hue }}
              >
                {work.type}
              </span>
              <h3 className="mt-0.5 font-display text-base font-medium leading-tight text-nebula-50 drop-shadow">
                {work.title}
              </h3>
            </div>
          </motion.a>
        ))}
      </div>
    </section>
  );
}
