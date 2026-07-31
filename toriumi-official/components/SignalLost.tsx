"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { motion, useInView } from "framer-motion";

/**
 * 信号途絶の終章。終活レポート（LegacyStatement）の直後、Footer の前。
 * ビューに入るとスタティックノイズが一瞬走り、
 * 「SIGNAL LOST — END OF TRANSMISSION」が明滅して静かに残る。
 * reduced-motion では演出なしの静的表示。
 */
export default function SignalLost() {
  const ref = useRef<HTMLElement>(null);
  const inView = useInView(ref, { once: true, margin: "-120px" });
  const [burst, setBurst] = useState(false); // ノイズバースト中
  const [settled, setSettled] = useState(false); // 明滅を終えて静止

  useEffect(() => {
    if (!inView) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setSettled(true);
      return;
    }
    setBurst(true);
    const t1 = window.setTimeout(() => setBurst(false), 900);
    const t2 = window.setTimeout(() => setSettled(true), 2400);
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
    };
  }, [inView]);

  return (
    <section
      ref={ref}
      className="scanlines relative overflow-hidden py-28 sm:py-36"
      aria-label="通信終了"
    >
      {/* スタティックノイズのバースト（一瞬だけ） */}
      {burst && (
        <div
          className="static-noise pointer-events-none absolute inset-0"
          style={{ animation: "static-burst 0.9s steps(2) 1 both" }}
        />
      )}

      <div className="relative mx-auto max-w-3xl px-6 text-center">
        {/* SIGNAL LOST 本文 */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : {}}
          transition={{ duration: 0.3, delay: 0.2 }}
          className={`font-mono text-base uppercase tracking-[0.5em] sm:text-xl ${
            settled ? "text-nebula-300/60" : "animate-flicker text-nebula-100"
          }`}
        >
          Signal Lost
        </motion.p>

        <motion.p
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : {}}
          transition={{ duration: 1, delay: 1.4 }}
          className="mt-6 font-mono text-[10px] uppercase tracking-[0.4em] text-nebula-300/40 sm:text-xs"
        >
          — End of Transmission —
        </motion.p>

        {/* 消えゆく信号レベルバー */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : {}}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mt-10 flex items-end justify-center gap-1.5"
          aria-hidden="true"
        >
          {[5, 9, 14, 10, 6].map((h, i) => (
            <motion.span
              key={i}
              initial={{ scaleY: 1 }}
              animate={inView ? { scaleY: [1, 0.7, 0.4, 0.15] } : {}}
              transition={{ duration: 2.6, delay: 0.8 + i * 0.25, ease: "easeOut" }}
              className="w-1 origin-bottom rounded-t bg-nebula-400/50"
              style={{ height: h * 2 }}
            />
          ))}
        </motion.div>

        {/* 交信は途絶えた。別の航路で宇宙の奥へ向かう導線 */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : {}}
          transition={{ duration: 1.2, delay: 2.8 }}
          className="mt-16"
        >
          <Link
            href="/voyage/"
            className="group inline-flex flex-col items-center gap-3"
            aria-label="Voyage — スクロールで宇宙の奥へ進む航行体験へ"
          >
            <span className="font-mono text-[9px] uppercase tracking-[0.4em] text-nebula-300/40">
              — Re-establishing another route —
            </span>
            <span className="relative font-mono text-xs uppercase tracking-[0.35em] text-aurum-200/90 transition-colors group-hover:text-aurum-100 sm:text-sm">
              Enter the Voyage
              <span className="absolute -bottom-2 left-0 h-px w-full origin-left scale-x-0 bg-aurum-300/70 transition-transform duration-500 group-hover:scale-x-100" />
            </span>
          </Link>
        </motion.div>
      </div>
    </section>
  );
}
