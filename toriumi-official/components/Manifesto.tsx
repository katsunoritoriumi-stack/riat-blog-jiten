"use client";

import { motion } from "framer-motion";
import TextReveal from "./ui/TextReveal";
import { MANIFESTO } from "@/lib/content";

export default function Manifesto() {
  return (
    <section className="relative mx-auto max-w-4xl px-6 py-32 sm:py-40">
      <motion.span
        initial={{ opacity: 0, letterSpacing: "0.1em" }}
        whileInView={{ opacity: 1, letterSpacing: "0.4em" }}
        viewport={{ once: true }}
        transition={{ duration: 0.9 }}
        className="mb-10 block text-center font-display text-xs uppercase tracking-cosmic text-aurum-300/70"
      >
        {MANIFESTO.eyebrow}
      </motion.span>

      <div className="space-y-2 text-center">
        {MANIFESTO.lines.map((line, i) => (
          <TextReveal
            key={i}
            as="p"
            delay={i * 0.1}
            className="font-serif text-2xl font-medium leading-snug text-nebula-100 sm:text-4xl md:text-[2.7rem]"
          >
            {line}
          </TextReveal>
        ))}
      </div>

      <motion.p
        initial={{ opacity: 0, y: 24 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-60px" }}
        transition={{ duration: 1, delay: 0.3 }}
        className="mx-auto mt-12 max-w-2xl text-center text-base leading-loose text-nebula-200/70 sm:text-lg"
      >
        {MANIFESTO.body}
      </motion.p>

      <motion.p
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.9, delay: 0.5 }}
        className="mt-8 text-right font-display text-lg font-extralight italic tracking-wide text-aurum-300/80 sm:text-2xl"
      >
        — Shirankedo<span className="text-aurum-400/50">...</span>
      </motion.p>

      {/* divider glyph */}
      <div className="mt-16 flex justify-center">
        <div className="h-px w-40 bg-gradient-to-r from-transparent via-aurum-400/60 to-transparent" />
      </div>
    </section>
  );
}
