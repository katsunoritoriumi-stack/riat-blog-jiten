"use client";

import { motion, useScroll, useTransform } from "framer-motion";
import { useRef } from "react";
import { ChevronDown } from "lucide-react";
import { SITE } from "@/lib/content";

export default function Hero() {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end start"],
  });
  const yText = useTransform(scrollYProgress, [0, 1], ["0%", "60%"]);
  const opacity = useTransform(scrollYProgress, [0, 0.7], [1, 0]);

  return (
    <section
      id="home"
      ref={ref}
      className="relative flex h-[100svh] min-h-[640px] items-center justify-center overflow-hidden"
    >
      {/* soft glow to lift the name off the galaxy */}
      <div className="pointer-events-none absolute left-1/2 top-1/2 h-[70vmin] w-[70vmin] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[radial-gradient(circle,rgba(124,58,237,0.18),transparent_65%)]" />

      <motion.div
        style={{ y: yText, opacity }}
        className="relative z-10 flex flex-col items-center px-6 text-center"
      >
        <motion.span
          initial={{ opacity: 0, letterSpacing: "0.1em" }}
          animate={{ opacity: 1, letterSpacing: "0.42em" }}
          transition={{ duration: 1.4, delay: 0.2 }}
          className="mb-7 font-mono text-[10px] uppercase text-aurum-300/80 sm:text-xs"
        >
          ✦ Tricky Multi-Creator ✦
        </motion.span>

        <h1 className="font-display leading-[0.92]">
          <motion.span
            initial={{ opacity: 0, y: 40, filter: "blur(12px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 1.2, delay: 0.4, ease: [0.22, 1, 0.36, 1] }}
            className="block text-[15vw] font-medium tracking-[-0.01em] sm:text-7xl md:text-8xl lg:text-[8.5rem]"
          >
            <span className="gradient-aurum">Katsunori</span>
          </motion.span>
          <motion.span
            initial={{ opacity: 0, y: 40, filter: "blur(12px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 1.2, delay: 0.56, ease: [0.22, 1, 0.36, 1] }}
            className="-mt-1 block text-[16vw] font-light italic tracking-[-0.01em] sm:text-[5rem] md:text-9xl lg:text-[9.5rem]"
          >
            <span className="gradient-nebula">Toriumi</span>
          </motion.span>
        </h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 1 }}
          className="mt-7 max-w-xl font-serif text-base text-nebula-200/90 sm:text-lg"
        >
          {SITE.nameJp} — {SITE.taglineJp}
        </motion.p>
      </motion.div>

      {/* scroll cue */}
      <motion.a
        href="#universe"
        style={{ opacity }}
        className="absolute bottom-8 left-1/2 z-10 -translate-x-1/2 text-aurum-300/70"
        initial={{ y: 0 }}
        animate={{ y: [0, 10, 0] }}
        transition={{ duration: 2, repeat: Infinity }}
        aria-label="次のセクションへ"
      >
        <ChevronDown size={28} />
      </motion.a>
    </section>
  );
}
