"use client";

import { motion } from "framer-motion";

type Props = {
  text: string;
  /** グラデーションの3色（文字列全体で連続するよう各文字に分配） */
  colors: [string, string, string];
  /** 光の波が通るときのグロー色 */
  glow: string;
  delay?: number;
  className?: string;
};

/**
 * 宇宙的なネームロゴ。
 * - 一文字ずつ 3D 回転＋ブラーで「虚空から星が生まれる」ように立ち上がる
 * - 立ち上がり後は明度の波（name-wave）が文字列を端から端へ流れ続ける
 * - ホバーで文字が個別にふわりと浮く（Tricky な遊び）
 * 文字ごとに background-position をずらし、全体でひと続きのグラデーションを再現。
 */
export default function StarName({
  text,
  colors,
  glow,
  delay = 0,
  className = "",
}: Props) {
  const chars = Array.from(text);
  const n = chars.length;

  return (
    <span className={`inline-block ${className}`} style={{ perspective: "600px" }}>
      {chars.map((ch, i) => (
        <motion.span
          key={i}
          initial={{ opacity: 0, y: "0.55em", rotateX: -75, filter: "blur(10px)" }}
          animate={{ opacity: 1, y: 0, rotateX: 0, filter: "blur(0px)" }}
          transition={{
            duration: 0.9,
            delay: delay + i * 0.055,
            ease: [0.22, 1, 0.36, 1],
          }}
          whileHover={{ y: -6, transition: { duration: 0.25 } }}
          className="name-wave inline-block"
          style={{
            backgroundImage: `linear-gradient(120deg, ${colors[0]}, ${colors[1]}, ${colors[2]})`,
            backgroundSize: `${n * 100}% 100%`,
            backgroundPosition: n > 1 ? `${(i / (n - 1)) * 100}% 0` : "0 0",
            WebkitBackgroundClip: "text",
            backgroundClip: "text",
            color: "transparent",
            ["--wave-glow" as string]: glow,
            animationDelay: `${2.2 + i * 0.16}s`,
          }}
        >
          {ch}
        </motion.span>
      ))}
    </span>
  );
}
