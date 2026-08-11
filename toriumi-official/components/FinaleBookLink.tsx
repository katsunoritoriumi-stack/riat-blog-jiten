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
          {/*
            本そのものが内側から呼吸するように光る。輪郭線は描かない。
            枠で囲むと「絵の上に貼った UI」に見えてしまうため。
            はみ出した光は絵に溶けるので、少し大きめに置いてぼかす。
          */}
          <span
            aria-hidden="true"
            className="book-glow pointer-events-none absolute -inset-[12%] transition-opacity duration-500 group-hover:opacity-90"
          />

          {/* ホバーで表紙にだけ光が差す。角の出ない楕円で、紙が起きるように */}
          <span
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-500 group-hover:opacity-100"
            style={{
              background:
                "radial-gradient(ellipse at 50% 40%, rgba(252,234,187,0.22), rgba(252,234,187,0.06) 55%, transparent 75%)",
            }}
          />

          {/*
            栞。本の下端から細い光が一本だけ垂れて、そのまま文字につながる。
            吹き出しやボタンの形にしないことで、絵の中の小道具のように見せる。
          */}
          <span
            aria-hidden="true"
            className="book-thread pointer-events-none absolute left-1/2 top-full h-5 w-px -translate-x-1/2 bg-gradient-to-b from-aurum-200/80 to-transparent sm:h-7"
          />

          <span className="absolute left-1/2 top-full mt-5 flex -translate-x-1/2 items-center gap-1.5 whitespace-nowrap font-mono text-[9px] uppercase tracking-[0.28em] text-aurum-100/85 transition-colors duration-500 group-hover:text-aurum-100 sm:mt-7 sm:text-[10px]"
            style={{ textShadow: "0 1px 10px rgba(3,2,10,0.95), 0 0 24px rgba(3,2,10,0.9)" }}
          >
            宇宙生命論
            <ArrowUpRight
              size={11}
              className="opacity-70 transition-transform duration-500 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:opacity-100"
            />
          </span>
        </motion.a>
      </div>
    </div>
  );
}
