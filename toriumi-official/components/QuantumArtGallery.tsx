"use client";

import { useRef } from "react";
import { motion, useMotionValue, useSpring } from "framer-motion";
import { ArrowUpRight } from "lucide-react";
import SectionHeader from "./ui/SectionHeader";
import Magnetic from "./ui/Magnetic";
import Sigil from "./ui/Sigil";
import { ARTWORKS, LINKS } from "@/lib/content";

export default function QuantumArtGallery() {
  const premium = ARTWORKS.filter((a) => a.tier === "premium");
  const digital = ARTWORKS.filter((a) => a.tier === "digital");

  return (
    <section id="quantum-art" className="relative mx-auto max-w-7xl px-6 py-28 sm:py-36">
      <div className="mb-16 flex flex-col gap-8 lg:flex-row lg:items-end lg:justify-between">
        <SectionHeader
          eyebrow="Signature Series"
          titleEn="Art"
        />
        <Magnetic className="self-start" strength={0.5}>
          <a
            href={LINKS.baseQuantum}
            target="_blank"
            rel="noopener noreferrer"
            className="group inline-flex items-center gap-2 rounded-full border border-aurum-400/40 px-5 py-2.5 text-sm text-aurum-200 transition-colors hover:bg-aurum-400/10"
          >
            作品を見る・購入する
            <ArrowUpRight size={16} className="transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
          </a>
        </Magnetic>
      </div>

      {/* premium tier */}
      <div className="mb-6 flex items-center gap-4">
        <span className="font-display text-sm uppercase tracking-cosmic text-aurum-300/70">Premium Works</span>
        <span className="h-px flex-1 bg-gradient-to-r from-aurum-400/40 to-transparent" />
      </div>
      <div className="grid grid-cols-2 gap-4 sm:gap-5 md:grid-cols-3 lg:grid-cols-5">
        {premium.map((art, i) => (
          <ArtCard key={art.title} art={art} i={i} large />
        ))}
      </div>

      {/* digital tier */}
      <div className="mb-6 mt-16 flex items-center gap-4">
        <span className="font-display text-sm uppercase tracking-cosmic text-nebula-300/70">Design</span>
        <span className="h-px flex-1 bg-gradient-to-r from-nebula-400/40 to-transparent" />
      </div>
      <div className="grid grid-cols-2 gap-4 sm:gap-5 md:grid-cols-3 lg:grid-cols-5">
        {digital.map((art, i) => (
          <ArtCard key={art.title} art={art} i={i} />
        ))}
      </div>
    </section>
  );
}

function ArtCard({
  art,
  i,
  large = false,
}: {
  art: (typeof ARTWORKS)[number];
  i: number;
  large?: boolean;
}) {
  const cardRef = useRef<HTMLAnchorElement>(null);
  const rx = useMotionValue(0);
  const ry = useMotionValue(0);
  const srx = useSpring(rx, { damping: 18, stiffness: 220 });
  const sry = useSpring(ry, { damping: 18, stiffness: 220 });

  function onMove(e: React.MouseEvent<HTMLAnchorElement>) {
    const el = cardRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width - 0.5;
    const py = (e.clientY - r.top) / r.height - 0.5;
    ry.set(px * 16);
    rx.set(-py * 16);
  }
  function reset() {
    rx.set(0);
    ry.set(0);
  }

  return (
    <motion.a
      ref={cardRef}
      href={LINKS.baseQuantum}
      target="_blank"
      rel="noopener noreferrer"
      onMouseMove={onMove}
      onMouseLeave={reset}
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ duration: 0.6, delay: i * 0.06 }}
      whileHover={{ y: -6 }}
      style={{ rotateX: srx, rotateY: sry, transformPerspective: 900 }}
      className="group relative block overflow-hidden rounded-2xl border border-nebula-500/15 bg-void-800/40 p-1 [transform-style:preserve-3d]"
    >
      <div
        className="relative flex aspect-square items-center justify-center overflow-hidden rounded-xl"
        style={{
          background: `radial-gradient(120% 120% at 50% 0%, ${art.hue}22, transparent 60%), #07061a`,
        }}
      >
        {/* shine sweep */}
        <div className="pointer-events-none absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/10 to-transparent transition-transform duration-700 group-hover:translate-x-full" />
        <div
          className="absolute inset-0 opacity-0 transition-opacity duration-500 group-hover:opacity-100"
          style={{ background: `radial-gradient(circle at 50% 50%, ${art.hue}26, transparent 70%)` }}
        />
        <Sigil
          type={art.sigil}
          hue={art.hue}
          className={`relative transition-transform duration-700 group-hover:rotate-[8deg] group-hover:scale-110 ${
            large ? "h-24 w-24 sm:h-28 sm:w-28" : "h-16 w-16 sm:h-20 sm:w-20"
          }`}
        />
        {art.tier === "premium" && (
          <span className="absolute right-2 top-2 rounded-full bg-aurum-400/15 px-2 py-0.5 text-[9px] tracking-wider text-aurum-200">
            ¥100,000
          </span>
        )}
      </div>
      <div className="px-2 py-3">
        <p className="font-serif text-sm text-nebula-100 sm:text-base">{art.title}</p>
        {art.sub && <p className="mt-0.5 text-[11px] text-nebula-300/60">{art.sub}</p>}
      </div>
    </motion.a>
  );
}
