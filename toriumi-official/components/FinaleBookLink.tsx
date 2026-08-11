"use client";

import { motion } from "framer-motion";
import { ArrowUpRight } from "lucide-react";

/**
 * 終章の絵の中にある「宇宙生命論」の本を、そのままリンクにする。
 *
 * 絵（FinaleBackdrop）とは別レイヤーにしてある。絵は文字の下に敷きたいが、
 * このリンクは文字の上に出さないと押せないため。
 * 位置がずれないよう、寄りの動きは両者とも同じ CSS アニメーション
 * （.finale-zoom）で動かしている。framer で別々に動かすと同期しない。
 *
 * 座標は globals.css の .finale-frame / .finale-book が持つ。
 * 絵を差し替えたらそこの数字だけ直せばよい。
 *
 * 外側の overflow は hidden ではなく clip。hidden はスクロール可能な箱を作るので、
 * リンクを Tab でフォーカスしたりブラウザが scrollIntoView したりすると
 * この層だけが縦にずれ、リンクが本から外れる（実測で 48px ずれた）。
 * clip なら切り取るだけでスクロールしない。
 */

const HREF = "https://seimeiron.com/riat-blog/";

export default function FinaleBookLink() {
  return (
    <div
      className="pointer-events-none absolute inset-0 z-30 overflow-clip"
      style={{ containerType: "size" }}
    >
      <div className="finale-frame finale-zoom">
        <motion.a
          href={HREF}
          target="_blank"
          rel="noopener noreferrer"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 1.4, delay: 0.6 }}
          aria-label="宇宙生命論のブログを別のタブで開く"
          className="finale-book group pointer-events-auto absolute block"
          style={{ left: "var(--bx)", top: "var(--by)", width: "var(--bw)", height: "var(--bh)" }}
        >
          {/* ホバーで表紙に光が差す。角の出ない楕円なので、絵の中で紙が起きるように見える */}
          <span
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-500 group-hover:opacity-100"
            style={{
              background:
                "radial-gradient(ellipse at 50% 40%, rgba(252,234,187,0.3), rgba(252,234,187,0.08) 55%, transparent 78%)",
            }}
          />

          {/*
            押せる一点を示す目印。本の右上に置く（表題や紋章を隠さない位置）。
            静止した絵の中では、光の強弱より「動き」のほうが確実に気づかれる。
            波紋を 2 つ、半周ずらして出し続ける。
          */}
          <span
            aria-hidden="true"
            className="pointer-events-none absolute right-[7%] top-[6%] block h-3 w-3 sm:h-3.5 sm:w-3.5"
          >
            <span className="hotspot-ping absolute inset-0 rounded-full border border-aurum-200/90" />
            <span className="hotspot-ping hotspot-ping-2 absolute inset-0 rounded-full border border-aurum-200/90" />
            <span className="hotspot-core absolute inset-[22%] rounded-full bg-aurum-200 transition-transform duration-300 group-hover:scale-125" />
          </span>

          {/* 目印から文字へ、細い線を一本だけ引く */}
          <span
            aria-hidden="true"
            className="book-thread pointer-events-none absolute left-1/2 top-full h-4 w-px -translate-x-1/2 bg-gradient-to-b from-aurum-200/90 to-transparent sm:h-6"
          />

          {/*
            文字。枠も背景も付けない代わりに、影を濃くして絵の上でも読めるようにする。
            「読む」と動詞で書いて、押した先に何があるかを明示する。
          */}
          <span
            className="absolute left-1/2 top-full mt-4 flex -translate-x-1/2 items-center gap-1.5 whitespace-nowrap font-mono text-[10px] uppercase tracking-[0.24em] text-aurum-100 transition-all duration-300 group-hover:tracking-[0.32em] sm:mt-6 sm:text-[11px]"
            style={{ textShadow: "0 1px 8px rgba(3,2,10,1), 0 0 22px rgba(3,2,10,0.95)" }}
          >
            宇宙生命論を読む
            <ArrowUpRight
              size={12}
              className="transition-transform duration-300 group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
            />
          </span>
        </motion.a>
      </div>
    </div>
  );
}
