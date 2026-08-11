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
 */

const HREF = "https://seimeiron.com/riat-blog/";

export default function FinaleBookLink() {
  return (
    <div
      className="pointer-events-none absolute inset-0 z-30 overflow-hidden"
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
          {/* 本の縁をなぞる光。ゆっくり明滅して「押せる」ことを知らせる */}
          <span
            aria-hidden="true"
            className="book-glow absolute inset-0 rounded-[6px] border border-aurum-200/70 transition-[border-color,box-shadow] duration-300 group-hover:border-aurum-100"
          />

          {/* 角の印。額装のように四隅だけを光らせて、絵を邪魔しない */}
          <span aria-hidden="true" className="pointer-events-none absolute inset-0">
            {[
              "left-0 top-0 border-l-2 border-t-2 rounded-tl-[6px]",
              "right-0 top-0 border-r-2 border-t-2 rounded-tr-[6px]",
              "left-0 bottom-0 border-b-2 border-l-2 rounded-bl-[6px]",
              "right-0 bottom-0 border-b-2 border-r-2 rounded-br-[6px]",
            ].map((c) => (
              <span
                key={c}
                className={`absolute h-3 w-3 border-aurum-100/90 transition-all duration-300 group-hover:h-4 group-hover:w-4 ${c}`}
              />
            ))}
          </span>

          {/* ホバーでほんのり持ち上げる（絵の一部が起きるように） */}
          <span
            aria-hidden="true"
            className="absolute inset-0 rounded-[6px] bg-aurum-200/0 transition-colors duration-300 group-hover:bg-aurum-200/12"
          />

          {/* 何のリンクなのかを言葉で置く。本の真下、絵の外には出さない */}
          <span className="absolute left-1/2 top-full mt-2 flex -translate-x-1/2 items-center gap-1.5 whitespace-nowrap rounded-full border border-aurum-200/45 bg-void-950/75 px-3 py-1.5 font-mono text-[9px] uppercase tracking-cosmic text-aurum-100 shadow-[0_2px_14px_rgba(3,2,10,0.8)] backdrop-blur-sm transition-colors duration-300 group-hover:border-aurum-100 group-hover:bg-void-950/90 sm:text-[10px]">
            宇宙生命論を読む
            <ArrowUpRight size={11} className="transition-transform duration-300 group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
          </span>
        </motion.a>
      </div>
    </div>
  );
}
