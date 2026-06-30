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
        className="group relative flex aspect-video items-center justify-center overflow-hidden rounded-3xl border border-nebula-500/20 bg-void-950"
      >
        {/* Exodus channel art */}
        <img
          src={YOUTUBE.thumbnail}
          alt="Exodus チャンネル"
          loading="lazy"
          className="absolute inset-0 h-full w-full object-cover transition-transform duration-700 group-hover:scale-105"
        />
        <div className="pointer-events-none absolute inset-0 bg-void-950/35 transition-colors duration-500 group-hover:bg-void-950/20" />
        <div className="relative flex flex-col items-center gap-4 text-center">
          <span className="flex h-20 w-20 items-center justify-center rounded-full bg-void-950/40 ring-1 ring-aurum-200/70 backdrop-blur-sm transition-transform group-hover:scale-110">
            <Play size={30} className="translate-x-0.5 text-aurum-100" fill="currentColor" />
          </span>
          <span className="text-xs uppercase tracking-cosmic text-nebula-100/80 drop-shadow">YouTube · @Exodus999</span>
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
