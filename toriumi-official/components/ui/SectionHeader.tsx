"use client";

import { motion } from "framer-motion";
import TextReveal from "./TextReveal";

type Props = {
  eyebrow: string;
  titleEn: string;
  titleJp?: string;
  align?: "left" | "center";
};

export default function SectionHeader({ eyebrow, titleEn, titleJp, align = "left" }: Props) {
  const alignCls = align === "center" ? "items-center text-center" : "items-start text-left";
  return (
    <div className={`flex flex-col gap-3 ${alignCls}`}>
      <motion.span
        initial={{ opacity: 0, letterSpacing: "0.1em" }}
        whileInView={{ opacity: 1, letterSpacing: "0.3em" }}
        viewport={{ once: true }}
        transition={{ duration: 0.8 }}
        className="font-mono text-[11px] sm:text-xs uppercase text-aurum-300/80"
      >
        ✦ {eyebrow}
      </motion.span>
      <TextReveal
        as="h2"
        className="font-display text-4xl sm:text-5xl md:text-6xl font-extrabold leading-[1.02] tracking-tight gradient-aurum"
      >
        {titleEn}
      </TextReveal>
      {titleJp && (
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7, delay: 0.15 }}
          className="font-serif text-lg sm:text-xl text-nebula-300"
        >
          {titleJp}
        </motion.p>
      )}
    </div>
  );
}
