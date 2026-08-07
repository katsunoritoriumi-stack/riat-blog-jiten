"use client";

import { motion } from "framer-motion";

/**
 * 作品の見出し。
 *
 * 「音楽と映像」には独立した作品が複数ある（MV とアルバム）。
 * それぞれ別のステーションに分かれて 1 画面ずつ与えられているので、
 * 通し番号を振って「これは何番目の、何という作品か」を各画面で示す。
 */
export default function WorkLabel({
  index,
  kind,
  title,
  sub,
}: {
  index: number;
  kind: string;
  title: string;
  sub: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6 }}
      className="mb-6 flex flex-col gap-2"
    >
      <span className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-cosmic text-aurum-300/80">
        <span className="tabular-nums text-nebula-300/50">{String(index).padStart(2, "0")}</span>
        <span className="h-px w-6 bg-aurum-300/40" />
        {kind}
      </span>
      <h3 className="font-display text-2xl font-bold leading-tight tracking-tight text-nebula-50 sm:text-3xl">
        {title}
      </h3>
      <p className="font-serif text-sm text-nebula-300">{sub}</p>
    </motion.div>
  );
}
