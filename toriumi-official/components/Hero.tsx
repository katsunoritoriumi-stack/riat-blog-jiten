"use client";

import {
  motion,
  useScroll,
  useTransform,
  useMotionValue,
  useSpring,
} from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import { SITE } from "@/lib/content";

/** UFO が画面左へ抜けきる頃合い（動画尺 8s のうち）。ここでテキストが宿る。 */
const REVEAL_AT = 4.8;

export default function Hero() {
  const ref = useRef<HTMLElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [revealed, setRevealed] = useState(false);

  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end start"],
  });
  const yText = useTransform(scrollYProgress, [0, 1], ["0%", "60%"]);
  const opacity = useTransform(scrollYProgress, [0, 0.7], [1, 0]);
  const scale = useTransform(scrollYProgress, [0, 1], [1, 0.94]);

  // マウス追従の微細な 3D パララックス（テキストが宿ったあとに効く）
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const rotateX = useSpring(useTransform(my, [-0.5, 0.5], [2.5, -2.5]), {
    stiffness: 60,
    damping: 20,
  });
  const rotateY = useSpring(useTransform(mx, [-0.5, 0.5], [-2.5, 2.5]), {
    stiffness: 60,
    damping: 20,
  });

  function onMouseMove(e: React.MouseEvent) {
    mx.set(e.clientX / window.innerWidth - 0.5);
    my.set(e.clientY / window.innerHeight - 0.5);
  }

  // 動画の進行に同期してテキストを宿す。再生できない環境ではフォールバックで必ず表示。
  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      setRevealed(true);
      return;
    }
    const fallback = window.setTimeout(() => setRevealed(true), 7000);
    return () => window.clearTimeout(fallback);
  }, []);

  const onTimeUpdate = () => {
    const v = videoRef.current;
    if (v && v.currentTime >= REVEAL_AT) setRevealed(true);
  };

  return (
    <section
      id="home"
      ref={ref}
      onMouseMove={onMouseMove}
      className="relative flex h-[100svh] min-h-[640px] items-center justify-center overflow-hidden"
    >
      {/* ── full-bleed cinematic backdrop (UFO appearing in space) — 一度だけ再生 ── */}
      <motion.video
        ref={videoRef}
        style={{ scale }}
        className="pointer-events-none absolute inset-0 h-full w-full object-cover"
        src="/hero-space.mp4"
        autoPlay
        muted
        playsInline
        preload="auto"
        onTimeUpdate={onTimeUpdate}
        onEnded={() => setRevealed(true)}
        aria-hidden="true"
      />

      {/* base darken */}
      <div className="pointer-events-none absolute inset-0 bg-void-950/35" />
      {/* テキストが宿るとき、焼き込み映像を静かに沈めて文字を持ち上げる追い暗転 */}
      <motion.div
        className="pointer-events-none absolute inset-0 bg-void-950"
        initial={{ opacity: 0 }}
        animate={{ opacity: revealed ? 0.45 : 0 }}
        transition={{ duration: 1.6, ease: "easeInOut" }}
      />
      {/* vignette + 下端フェード（次セクションへ繋ぐ） */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_30%,rgba(3,2,10,0.85)_92%)]" />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-40 bg-gradient-to-b from-transparent to-void-950" />

      {/* soft glow behind the name */}
      <motion.div
        className="pointer-events-none absolute left-1/2 top-1/2 h-[70vmin] w-[70vmin] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[radial-gradient(circle,rgba(124,58,237,0.22),transparent_65%)]"
        initial={{ opacity: 0 }}
        animate={{ opacity: revealed ? 1 : 0 }}
        transition={{ duration: 1.8 }}
      />

      {/* ── テキスト：UFO の退場に同期して、もとからそこに在ったように宿る ── */}
      <motion.div
        style={{ y: yText, opacity, rotateX, rotateY, transformPerspective: 1000 }}
        className="relative z-10 flex flex-col items-center px-6 text-center"
      >
        <motion.span
          initial={{ opacity: 0, y: 8, filter: "blur(6px)", letterSpacing: "0.2em" }}
          animate={
            revealed
              ? { opacity: 1, y: 0, filter: "blur(0px)", letterSpacing: "0.42em" }
              : {}
          }
          transition={{ duration: 1.6, ease: [0.22, 1, 0.36, 1] }}
          className="mb-7 font-mono text-[10px] uppercase text-aurum-300/85 sm:text-xs"
        >
          ✦ Tricky Multi-Creator ✦
        </motion.span>

        <motion.h1
          initial={{ opacity: 0, filter: "blur(14px)", scale: 1.04 }}
          animate={revealed ? { opacity: 1, filter: "blur(0px)", scale: 1 } : {}}
          transition={{ duration: 1.9, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
          className="font-display leading-[0.9]"
        >
          <span className="name-wave block text-[14vw] font-semibold tracking-[-0.02em] gradient-aurum sm:text-7xl md:text-8xl lg:text-[8.5rem]">
            Katsunori
          </span>
          <span
            className="name-wave -mt-1 block text-[15vw] font-extralight tracking-[0.03em] gradient-nebula sm:text-[5rem] md:text-9xl lg:text-[9.5rem]"
            style={{ animationDelay: "1.2s" }}
          >
            Toriumi
          </span>
        </motion.h1>

        {/* a.k.a KIEJI */}
        <motion.p
          initial={{ opacity: 0, y: 10, filter: "blur(6px)" }}
          animate={
            revealed ? { opacity: 1, y: 0, filter: "blur(0px)" } : {}
          }
          transition={{ duration: 1.4, delay: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="mt-5 flex items-center gap-3 font-mono text-xs uppercase tracking-[0.4em] sm:text-sm"
        >
          <span className="h-px w-8 bg-gradient-to-r from-transparent to-aurum-300/60" />
          <span className="text-nebula-300/70">a.k.a</span>
          <span className="gradient-aurum font-bold tracking-[0.5em]">KIEJI</span>
          <span className="h-px w-8 bg-gradient-to-l from-transparent to-aurum-300/60" />
        </motion.p>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: revealed ? 1 : 0 }}
          transition={{ duration: 1.2, delay: 0.8 }}
          className="mt-7 max-w-xl font-serif text-base text-nebula-200/85 sm:text-lg"
        >
          {SITE.nameJp} — {SITE.taglineJp}
        </motion.p>
      </motion.div>

      {/* scroll cue */}
      <motion.a
        href="#universe"
        style={{ opacity }}
        className="absolute bottom-8 left-1/2 z-10 -translate-x-1/2 text-aurum-300/70"
        initial={{ opacity: 0 }}
        animate={
          revealed
            ? { opacity: [0, 0.7, 0.7], y: [0, 10, 0] }
            : { opacity: 0 }
        }
        transition={{
          opacity: { duration: 1, delay: 1 },
          y: { duration: 2, repeat: Infinity, delay: 1 },
        }}
        aria-label="次のセクションへ"
      >
        <ChevronDown size={28} />
      </motion.a>
    </section>
  );
}
