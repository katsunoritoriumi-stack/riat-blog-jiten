"use client";

import { motion } from "framer-motion";
import { Camera, ThumbsUp, PlayCircle, Store, ArrowUpRight, Gift, MessageCircle, Radio } from "lucide-react";
import SectionHeader from "./ui/SectionHeader";
import { LINKS } from "@/lib/content";

// SNS とショップを 8 項目に統合（携帯では隙間なく 2×4、PC では 4×2）
const links = [
  { icon: MessageCircle, label: "LINE", handle: "友だち追加", href: LINKS.line },
  { icon: Camera, label: "Instagram", handle: "@katsunoritoriumi", href: LINKS.instagram },
  { icon: PlayCircle, label: "YouTube", handle: "Exodus チャンネル", href: LINKS.youtube },
  { icon: Radio, label: "Radio", handle: "stand.fm 音声配信", href: LINKS.radio },
  { icon: ThumbsUp, label: "Facebook", handle: "Katsunori Toriumi", href: LINKS.facebook },
  { icon: Store, label: "Original T-shirts", handle: "オリジナル Tシャツ", href: LINKS.baseQuantum, accent: true },
  { icon: Store, label: "Original Art", handle: "原画・デジタル", href: LINKS.baseToriumi, accent: true },
  { icon: Gift, label: "Produced Brand", handle: "着るお守り / NIRAV", href: LINKS.nirav, accent: true },
];

export default function ConnectSection() {
  return (
    <section data-section="connect" className="relative mx-auto max-w-7xl px-6 py-28 sm:py-36">
      <div className="mb-16 text-center">
        <SectionHeader
          eyebrow="Reach the Universe"
          titleEn="Connect"
          align="center"
        />
      </div>

      {/* 8 項目を隙間なく：携帯 2×4／PC 4×2 */}
      <div className="grid grid-cols-2 gap-3 sm:gap-4 md:grid-cols-4">
        {links.map((s, i) => (
          <LinkCard key={s.label} {...s} i={i} />
        ))}
      </div>
    </section>
  );
}

function LinkCard({
  icon: Icon,
  label,
  handle,
  href,
  i,
  accent = false,
}: {
  icon: typeof Store;
  label: string;
  handle: string;
  href: string;
  i: number;
  accent?: boolean;
}) {
  return (
    <motion.a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay: (i % 3) * 0.08 }}
      whileHover={{ y: -4 }}
      className={`sheen group relative flex flex-col gap-3 overflow-hidden rounded-2xl glass p-4 transition-colors sm:p-5 ${
        accent ? "hover:border-aurum-400/40" : "hover:border-nebula-400/40"
      }`}
    >
      <div className="flex items-center justify-between">
        <span
          className={`flex h-10 w-10 items-center justify-center rounded-xl transition-transform duration-500 group-hover:-rotate-6 group-hover:scale-110 ${
            accent ? "bg-aurum-400/15 text-aurum-200" : "bg-nebula-600/30 text-nebula-200"
          }`}
        >
          <Icon size={19} />
        </span>
        <ArrowUpRight
          size={15}
          className="text-nebula-300/45 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:text-aurum-200"
        />
      </div>
      <div className="min-w-0">
        <p className="truncate font-serif text-[15px] text-nebula-100 sm:text-base">{label}</p>
        <p className="truncate text-[11px] text-nebula-300/60">{handle}</p>
      </div>
    </motion.a>
  );
}
