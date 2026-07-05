"use client";

import { motion, type Variants } from "framer-motion";

/**
 * サイト全体を締めくくる終章。
 * 「このポータルサイト＝終活レポート」という主旨を、
 * すべてを見終えた最後にドラマティックに明かす。
 */
const container: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.16, delayChildren: 0.1 } },
};
const line: Variants = {
  hidden: { opacity: 0, y: 26, filter: "blur(10px)" },
  show: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: { duration: 1.2, ease: [0.22, 1, 0.36, 1] },
  },
};

export default function LegacyStatement() {
  return (
    <section id="report" className="relative overflow-hidden py-36 sm:py-52">
      {/* 上下のヘアラインで"特別な通信"として隔てる */}
      <div className="mx-auto mb-20 h-px w-40 bg-gradient-to-r from-transparent via-aurum-400/50 to-transparent sm:mb-28" />

      {/* ドラマティックな中心グロー */}
      <div className="pointer-events-none absolute left-1/2 top-1/2 h-[55vmin] w-[92vmin] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[radial-gradient(ellipse,rgba(124,58,237,0.16),transparent_70%)]" />

      <motion.div
        variants={container}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: "-120px" }}
        className="relative mx-auto max-w-4xl px-6 text-center"
      >
        {/* eyebrow */}
        <motion.div
          variants={line}
          className="mb-12 flex items-center justify-center gap-4"
        >
          <span className="h-px w-10 bg-gradient-to-r from-transparent to-aurum-300/60" />
          <span className="font-mono text-[11px] uppercase tracking-[0.45em] text-aurum-300/85">
            Final Report
          </span>
          <span className="h-px w-10 bg-gradient-to-l from-transparent to-aurum-300/60" />
        </motion.div>

        {/* 終活レポートの一文（詩的な改行で緩急を作る） */}
        <p className="font-serif text-xl leading-loose text-nebula-100/95 sm:text-2xl sm:leading-[2.15] md:text-[1.7rem]">
          <motion.span variants={line} className="block">
            このポータルサイトは、
          </motion.span>
          <motion.span variants={line} className="mt-2 block">
            <span className="gradient-aurum font-medium text-glow-aurum">
              陽化（老化）を極めた惑星テラ（地球）
            </span>
            での
            <br className="hidden sm:block" />
            ミッションを締めくくるに当たり、
          </motion.span>
          <motion.span variants={line} className="mt-2 block">
            <span className="gradient-nebula">Katsunori&nbsp;Toriumi</span>
            が自己のミッションを
            <br className="hidden sm:block" />
            まとめ上げる為に作成した
          </motion.span>
          <motion.span variants={line} className="mt-3 block">
            <span className="gradient-aurum font-semibold text-glow-aurum">
              終活レポート
            </span>
            として
            <motion.span
              aria-hidden
              animate={{ opacity: [0.3, 0.9, 0.3] }}
              transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
              className="text-aurum-300"
            >
              …
            </motion.span>
          </motion.span>
        </p>

        <motion.p
          variants={line}
          className="mt-14 font-mono text-[10px] uppercase tracking-[0.4em] text-nebula-300/50 sm:text-xs"
        >
          Katsunori Toriumi &nbsp;a.k.a&nbsp; KIEJI
        </motion.p>
      </motion.div>
    </section>
  );
}
