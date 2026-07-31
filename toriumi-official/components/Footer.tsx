"use client";

import { motion } from "framer-motion";
import { SITE } from "@/lib/content";

/**
 * 通信が途絶えた（SIGNAL LOST）あとに、最後に残る署名。
 * 名前を文字で置くのではなく、ロゴがゆっくり浮かび上がって終わる。
 */
export default function Footer() {
  return (
    <footer className="relative px-6 pb-10 pt-8">
      <div className="mx-auto flex max-w-3xl flex-col items-center text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.92, filter: "blur(10px)" }}
          whileInView={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
          viewport={{ once: true, margin: "-10%" }}
          transition={{ duration: 2.2, ease: [0.22, 1, 0.36, 1] }}
          className="relative flex justify-center"
        >
          {/* 背後でゆっくり明滅する光 */}
          <motion.span
            aria-hidden="true"
            animate={{ opacity: [0.25, 0.5, 0.25] }}
            transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
            className="pointer-events-none absolute left-1/2 top-1/2 -z-10 -translate-x-1/2 -translate-y-1/2 rounded-full"
            style={{
              width: "190%",
              height: "190%",
              background: "radial-gradient(circle, rgba(124,58,237,0.55), rgba(3,2,10,0.35) 45%, transparent 70%)",
            }}
          />
          <img
            src="/logo-ktoriumi.webp"
            alt={`${SITE.nameEn} — K.TORIUMI`}
            width={846}
            height={1024}
            loading="lazy"
            className="logo-drift block h-[17svh] max-h-[180px] min-h-[110px] w-auto"
          />
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 1.6, delay: 0.9 }}
          style={{ textShadow: "0 1px 6px rgba(0,0,0,0.9)" }}
          className="mt-5 font-mono text-[10px] uppercase tracking-[0.4em] text-nebula-200/65"
        >
          {SITE.roleEn}
        </motion.p>

        <motion.span
          aria-hidden="true"
          initial={{ scaleX: 0 }}
          whileInView={{ scaleX: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 1.4, delay: 1.1, ease: "easeOut" }}
          className="mt-4 block h-px w-24 bg-gradient-to-r from-transparent via-aurum-300/45 to-transparent"
        />

        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 1.4, delay: 1.3 }}
          style={{ textShadow: "0 1px 6px rgba(0,0,0,0.9)" }}
          className="mt-4 text-[11px] text-nebula-200/55"
        >
          © {new Date().getFullYear()} {SITE.nameEn}. All works belong to the artist.
        </motion.p>
      </div>
    </footer>
  );
}
