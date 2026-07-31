"use client";

import { useEffect, useRef, useState } from "react";
import { motion, useScroll, useTransform, type MotionValue } from "framer-motion";

function Char({
  char,
  progress,
  range,
}: {
  char: string;
  progress: MotionValue<number>;
  range: [number, number];
}) {
  const opacity = useTransform(progress, range, [0.16, 1]);
  return <motion.span style={{ opacity }}>{char}</motion.span>;
}

/**
 * スクロールに連動して一文字ずつ「灯る」テキスト。
 * 和文（スペースなし）でも自然に効くよう文字単位で分割する。
 * 読み進める速度と光が同期する、静かな読書体験のための演出。
 */
export default function ScrubText({
  text,
  className = "",
}: {
  text: string;
  className?: string;
}) {
  const ref = useRef<HTMLParagraphElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start 0.9", "end 0.5"],
  });
  const chars = Array.from(text);

  /**
   * ZoomStage のステーション内では要素が画面に固定されるため、
   * スクロール連動の進捗が動かず文字が薄いまま止まってしまう。
   * その場合は素直に全文を表示する（登場演出はステーション側が担当する）。
   */
  const [pinned, setPinned] = useState(false);
  useEffect(() => {
    setPinned(!!ref.current?.closest("[data-station]"));
  }, []);

  if (pinned) {
    return (
      <p ref={ref} className={className}>
        {text}
      </p>
    );
  }

  return (
    <p ref={ref} className={className}>
      {chars.map((ch, i) => (
        <Char
          key={i}
          char={ch}
          progress={scrollYProgress}
          range={[i / chars.length, Math.min(1, (i + 1.5) / chars.length)]}
        />
      ))}
    </p>
  );
}
