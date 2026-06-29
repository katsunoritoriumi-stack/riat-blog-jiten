"use client";

import { motion } from "framer-motion";
import { ArrowUpRight } from "lucide-react";
import SectionHeader from "./ui/SectionHeader";
import { NIRAV_ITEMS, LINKS } from "@/lib/content";

export default function NiravSection() {
  return (
    <section id="nirav" className="relative overflow-hidden py-28 sm:py-36">
      {/* backdrop */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(60%_60%_at_80%_20%,rgba(244,114,182,0.12),transparent_60%)]" />

      <div className="mx-auto grid max-w-7xl gap-14 px-6 lg:grid-cols-[1fr_1.1fr] lg:items-center">
        <div>
          <SectionHeader
            eyebrow="Wearable Holy Art"
            titleEn="Produce"
            titleJp="着るお守り"
          />
          <a
            href={LINKS.produce}
            target="_blank"
            rel="noopener noreferrer"
            className="group mt-8 inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-iris-rose/80 to-nebula-500/80 px-6 py-3 text-sm font-medium text-white transition-opacity hover:opacity-90"
          >
            PRODUCE STORE へ
            <ArrowUpRight size={16} className="transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
          </a>
        </div>

        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
          {NIRAV_ITEMS.map((item, i) => (
            <motion.a
              key={item.title}
              href={LINKS.produce}
              target="_blank"
              rel="noopener noreferrer"
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.07 }}
              whileHover={{ y: -5 }}
              className="group flex flex-col rounded-2xl glass p-5 transition-colors hover:border-iris-rose/40"
            >
              <span className="font-serif text-lg text-nebula-100">{item.title}</span>
              <span className="text-[11px] uppercase tracking-wider text-iris-rose/70">{item.en}</span>
              <span className="mt-3 text-xs text-nebula-300/70">{item.note}</span>
              <span className="mt-4 text-[11px] text-aurum-300/70">¥4,900</span>
            </motion.a>
          ))}
        </div>
      </div>
    </section>
  );
}
