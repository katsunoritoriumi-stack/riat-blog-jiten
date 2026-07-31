"use client";

import { useRef, type ReactNode } from "react";
import { motion, useReducedMotion, useScroll, useTransform } from "framer-motion";
import { arrivalOpacity, arrivalScale } from "@/lib/flightMath";
import { useDepthBypass } from "@/lib/depthBypass";

/**
 * セクションを「宇宙の奥から近づいてくる」ように登場させるラッパー。
 *
 * 設計上の決めごと：
 * - perspective + translateZ ではなく scale を使う。画面と平行な内容では両者は
 *   数学的に同じ結果になり、perspective は position:fixed の包含ブロックと
 *   スタッキングコンテキストを増やすだけ損をする。
 * - transform-origin は 50% 0%。セクションは viewport より高い（1200〜2500px）ので、
 *   中心原点だと画面外へ縮んで「ポン」と出てしまう。上端を留めると、先頭の1画面ぶんが
 *   小さく現れて実寸まで膨らむ＝狙いどおりの見え方になる。
 * - 進捗 p <= 0（まだ画面下に入っていない）と p >= 1（到着済み）では必ず無変換に戻す
 *   （lib/flightMath.ts の arrivalScale / arrivalOpacity）。これにより
 *     ・静的HTML／JS無効時に今までと同じ見た目で出る
 *     ・未到達セクションの getBoundingClientRect() が実レイアウト値になる
 *       （lenis の scrollTo・scrollIntoView・IntersectionObserver が狂わない）
 *     ・画面外のセクションが GPU レイヤー化されない
 *   が同時に満たせる。
 * - will-change は手で付けない。1枚あたり数十MBのレイヤーになるため、
 *   framer が「全 transform が既定値なら transform: none を出す」挙動に任せる。
 * - filter: blur() は使わない。最大面積に毎フレーム最重量の処理になる割に、
 *   奥行き感は背景のストリークで足りている。
 *
 * 注意：この中に position: fixed の要素を置かないこと。transform された祖先は
 * fixed の包含ブロックになるため、画面固定が壊れる。
 */
export default function DepthArrival({
  children,
  from = 0.74,
  end = 0.55,
  disabled = false,
}: {
  children: ReactNode;
  /** 到着開始時のスケール。1 に近いほど控えめ */
  from?: number;
  /** 到着が完了する位置（viewport 高さに対する割合。小さいほど手前まで引っ張る） */
  end?: number;
  /** このセクションだけ演出を切る */
  disabled?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const reduce = useReducedMotion();
  const bypass = useDepthBypass();

  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start end", `start ${end}`],
  });
  const scale = useTransform(scrollYProgress, (p) => arrivalScale(p, from));
  const opacity = useTransform(scrollYProgress, arrivalOpacity);

  const off = disabled || !!reduce || bypass;

  return (
    // 外側は絶対に変換しない（useScroll の計測対象。変換すると自分の測定に効いてしまう）
    <div ref={ref}>
      <motion.div
        style={off ? undefined : { scale, opacity, transformOrigin: "50% 0%" }}
      >
        {children}
      </motion.div>
    </div>
  );
}
