"use client";

import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import Link from "next/link";
import { ArrowLeft, ArrowUpRight } from "lucide-react";
import GalaxyBackground from "./GalaxyBackground";
import CustomCursor from "./ui/CustomCursor";
import Footer from "./Footer";
import type { WorkItem } from "@/lib/content";

/** 各サイトのトップ画面サムネイル（WordPress mShots・無料・キー不要） */
function shot(url: string) {
  return `https://s.wordpress.com/mshots/v1/${encodeURIComponent(url)}?w=800&h=600`;
}

type Props = {
  eyebrow: string;
  titleEn: string;
  titleJp: string;
  items: WorkItem[];
};

/**
 * 「創造の座標軸」の Work から辿るサブページ（App / Website 一覧）の共通シェル。
 * ホーム同様の銀河背景の上に、作品カードを並べる。
 */
export default function ShowcaseGrid({ eyebrow, titleEn, titleJp, items }: Props) {
  return (
    <>
      <GalaxyBackground />
      <CustomCursor />
      <div className="relative z-10 min-h-screen">
        <header className="mx-auto flex max-w-7xl items-center px-5 py-6 sm:px-8">
          <Link
            href="/#universe"
            className="link-sweep group inline-flex items-center gap-2 font-mono text-xs uppercase tracking-wider text-nebula-300/80 transition-colors hover:text-aurum-200"
          >
            <ArrowLeft
              size={15}
              className="transition-transform group-hover:-translate-x-0.5"
            />
            創造の座標軸へ戻る
          </Link>
        </header>

        <main className="mx-auto max-w-7xl px-6 pb-28 pt-10 sm:px-8 sm:pt-16">
          <div className="mb-12 max-w-2xl">
            <motion.span
              initial={{ opacity: 0, letterSpacing: "0.1em" }}
              animate={{ opacity: 1, letterSpacing: "0.3em" }}
              transition={{ duration: 0.8 }}
              className="mb-3 block font-mono text-[11px] uppercase text-aurum-300/80 sm:text-xs"
            >
              ✦ {eyebrow}
            </motion.span>
            <motion.h1
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.1 }}
              className="font-display text-4xl font-extrabold leading-[1.02] tracking-tight gradient-aurum sm:text-5xl md:text-6xl"
            >
              {titleEn}
            </motion.h1>
            <motion.p
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.2 }}
              className="mt-3 font-serif text-lg text-nebula-300 sm:text-xl"
            >
              {titleJp}
            </motion.p>
          </div>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
            {items.map((item, i) => (
              <ShowcaseCard key={item.title} item={item} i={i} />
            ))}
          </div>
        </main>

        <Footer />
      </div>
    </>
  );
}

/** マウス追従の 3D チルト＋光沢スイープ付きカード（WorkSection と同意匠） */
function ShowcaseCard({ item, i }: { item: WorkItem; i: number }) {
  const px = useMotionValue(0.5);
  const py = useMotionValue(0.5);
  const rotateX = useSpring(useTransform(py, [0, 1], [7, -7]), {
    stiffness: 180,
    damping: 20,
  });
  const rotateY = useSpring(useTransform(px, [0, 1], [-7, 7]), {
    stiffness: 180,
    damping: 20,
  });

  function onMouseMove(e: React.MouseEvent<HTMLAnchorElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    px.set((e.clientX - r.left) / r.width);
    py.set((e.clientY - r.top) / r.height);
  }
  function onMouseLeave() {
    px.set(0.5);
    py.set(0.5);
  }

  return (
    <motion.a
      href={item.href}
      target={item.href ? "_blank" : undefined}
      rel={item.href ? "noopener noreferrer" : undefined}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
      style={{ rotateX, rotateY, transformPerspective: 700 }}
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: i * 0.08 }}
      whileHover={{ y: -6 }}
      className="sheen group relative flex aspect-[4/3] flex-col justify-between overflow-hidden rounded-xl border border-nebula-500/15 bg-void-800/60 transition-colors duration-500 hover:border-nebula-400/40"
    >
      {item.href && (
        <img
          src={shot(item.href)}
          alt={`${item.title} のスクリーンショット`}
          loading="lazy"
          className="absolute inset-0 h-full w-full object-cover object-top opacity-70 transition-all duration-500 group-hover:scale-105 group-hover:opacity-90"
        />
      )}
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-void-950 via-void-950/40 to-void-950/10" />
      <div
        className="pointer-events-none absolute -right-8 -top-10 h-24 w-24 rounded-full opacity-40 blur-2xl transition-opacity duration-500 group-hover:opacity-70"
        style={{ background: item.hue }}
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
          style={{ color: item.hue }}
        >
          {item.type}
        </span>
        <h3 className="mt-0.5 font-display text-base font-medium leading-tight text-nebula-50 drop-shadow">
          {item.title}
        </h3>
        <p className="mt-1 text-xs leading-snug text-nebula-300/70">{item.note}</p>
      </div>
    </motion.a>
  );
}
