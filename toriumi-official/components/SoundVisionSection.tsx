"use client";

import { motion } from "framer-motion";
import { Play, ArrowUpRight } from "lucide-react";
import SectionHeader from "./ui/SectionHeader";
import { YOUTUBE } from "@/lib/content";

export default function SoundVisionSection() {
  return (
    <section id="sound" className="relative mx-auto max-w-5xl px-6 py-28 sm:py-36">
      <div className="mb-14 max-w-2xl">
        <SectionHeader eyebrow="Sound & Vision" titleEn="EXODUS" titleJp="音楽と映像" />
      </div>

      {/* video frame */}
      <motion.a
        href={YOUTUBE.channelUrl}
        target="_blank"
        rel="noopener noreferrer"
        initial={{ opacity: 0, scale: 0.96 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.7 }}
        className="group relative flex aspect-video items-center justify-center overflow-hidden rounded-3xl border border-nebula-500/20"
        style={{
          background:
            "radial-gradient(120% 120% at 50% 0%, rgba(124,58,237,0.3), transparent 55%), linear-gradient(160deg, #0c0a26, #03020a)",
        }}
      >
        <div className="pointer-events-none absolute inset-0 opacity-60 [background:repeating-linear-gradient(115deg,transparent,transparent_22px,rgba(167,139,250,0.05)_23px)]" />
        <div className="relative flex flex-col items-center gap-4 text-center">
          <span className="flex h-20 w-20 items-center justify-center rounded-full bg-aurum-400/15 ring-1 ring-aurum-400/40 transition-transform group-hover:scale-110">
            <Play size={30} className="translate-x-0.5 text-aurum-200" fill="currentColor" />
          </span>
          <span className="font-display text-xl tracking-widest text-nebula-100">{YOUTUBE.channelName}</span>
          <span className="text-xs uppercase tracking-cosmic text-nebula-300/60">YouTube · @Exodus999</span>
        </div>
      </motion.a>

      <div className="mt-8 flex justify-center">
        <a
          href={YOUTUBE.channelUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="group inline-flex items-center gap-2 text-sm text-aurum-200"
        >
          チャンネルを見る
          <ArrowUpRight size={16} className="transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
        </a>
      </div>
    </section>
  );
}
