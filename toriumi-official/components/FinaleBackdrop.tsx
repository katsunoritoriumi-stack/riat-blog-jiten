"use client";

import { motion, useReducedMotion } from "framer-motion";

/**
 * 終章（SIGNAL LOST → ロゴの署名）の背景。
 * 画面いっぱいに絵を敷き、その上に文字とロゴが乗る。
 *
 * 絵は明るい領域（左上の惑星・右のロケット）と暗い領域（中央下の人物の膝元）が
 * はっきり分かれているので、
 *   ・上下から締める暗幕で SIGNAL LOST とロゴの座を作る
 *   ・中央だけ薄く残して、人物と惑星は見えるようにする
 * という二段構えで可読性と絵の見せ場を両立させている。
 */
export default function FinaleBackdrop() {
  const reduce = useReducedMotion();

  return (
    <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
      {/* 絵。ゆっくり寄っていく（宇宙の只中に留まっている感じ） */}
      <motion.img
        src="/finale-cosmos.webp"
        alt=""
        aria-hidden="true"
        width={1672}
        height={941}
        loading="lazy"
        initial={reduce ? undefined : { scale: 1.12 }}
        animate={reduce ? undefined : { scale: 1 }}
        transition={{ duration: 26, ease: "linear" }}
        className="absolute inset-0 h-full w-full object-cover"
        // 縦長の画面では中央だけを切り出すと人物が外れるので、少し左寄りを芯にする
        style={{ objectPosition: "44% center" }}
      />

      {/* 上下から締める暗幕：見出しと署名の座を作る */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(180deg, rgba(3,2,10,0.92) 0%, rgba(3,2,10,0.42) 26%, rgba(3,2,10,0.28) 52%, rgba(3,2,10,0.86) 84%, rgba(3,2,10,0.97) 100%)",
        }}
      />
      {/* 四隅を落として画面の中心へ視線を集める */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at 46% 52%, transparent 24%, rgba(3,2,10,0.5) 78%, rgba(3,2,10,0.86) 100%)",
        }}
      />
      {/* サイト全体の紫を一枚かぶせて、絵を世界観に馴染ませる */}
      <div
        className="absolute inset-0 mix-blend-soft-light"
        style={{ background: "linear-gradient(180deg, rgba(124,58,237,0.5), rgba(38,20,90,0.35))" }}
      />
    </div>
  );
}
