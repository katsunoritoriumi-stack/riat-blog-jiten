"use client";

import { motion } from "framer-motion";
import { Camera, ThumbsUp, PlayCircle, Store, ArrowUpRight, Gift, MessageCircle } from "lucide-react";
import SectionHeader from "./ui/SectionHeader";
import { LINKS } from "@/lib/content";

const socials = [
  { icon: MessageCircle, label: "LINE", handle: "友だち追加", href: LINKS.line },
  { icon: Camera, label: "Instagram", handle: "@katsunoritoriumi", href: LINKS.instagram },
  { icon: PlayCircle, label: "YouTube", handle: "Exodus チャンネル", href: LINKS.youtube },
  { icon: ThumbsUp, label: "Facebook", handle: "Katsunori Toriumi", href: LINKS.facebook },
];

const shops = [
  { icon: Store, label: "Art Store", note: "アート 原画・デジタル", href: LINKS.baseQuantum },
  { icon: Store, label: "k.toriumi Store", note: "アート・グッズ", href: LINKS.baseToriumi },
  { icon: Gift, label: "Produce — 着るお守り", note: "Wearable Holy Art", href: LINKS.produce },
];

export default function ConnectSection() {
  return (
    <section id="connect" className="relative mx-auto max-w-7xl px-6 py-28 sm:py-36">
      <div className="mb-16 text-center">
        <SectionHeader
          eyebrow="Reach the Universe"
          titleEn="Connect"
          align="center"
        />
      </div>

      {/* social */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {socials.map((s, i) => (
          <LinkCard key={s.label} {...s} i={i} />
        ))}
      </div>

      {/* shops */}
      <div className="mt-5 grid gap-4 sm:grid-cols-3">
        {shops.map((s, i) => (
          <LinkCard key={s.label} {...s} handle={s.note} i={i + 3} accent />
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
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay: (i % 3) * 0.08 }}
      whileHover={{ y: -5 }}
      className={`group flex items-center justify-between gap-4 rounded-2xl glass p-6 transition-colors ${
        accent ? "hover:border-aurum-400/40" : "hover:border-nebula-400/40"
      }`}
    >
      <div className="flex items-center gap-4">
        <span
          className={`flex h-12 w-12 items-center justify-center rounded-xl ${
            accent ? "bg-aurum-400/15 text-aurum-200" : "bg-nebula-600/30 text-nebula-200"
          }`}
        >
          <Icon size={22} />
        </span>
        <div>
          <p className="font-serif text-base text-nebula-100">{label}</p>
          <p className="text-xs text-nebula-300/60">{handle}</p>
        </div>
      </div>
      <ArrowUpRight
        size={18}
        className="text-nebula-300/50 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:text-aurum-200"
      />
    </motion.a>
  );
}
