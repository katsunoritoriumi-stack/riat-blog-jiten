"use client";

import { motion, useScroll, useTransform } from "framer-motion";
import Galaxy from "./ui/Galaxy";
import DeepFlight from "./ui/DeepFlight";

/**
 * サイト全体に固定表示する宇宙の背景。
 * スクロールすると「奥へ飛んでいく」ように見せるため、二層で構成する：
 *   ① 渦巻銀河 — 徐々に拡大しながら薄れる＝背後へ飛び去っていく遠景
 *   ② DeepFlight — カメラの前を流れていく星屑（ストリークが速度を表す）
 * 可読性のため、暗幕を①と②の間に、ヴィネットを最前面に置く。
 */
export default function GalaxyBackground() {
  const { scrollYProgress } = useScroll();
  const galaxyScale = useTransform(scrollYProgress, [0, 1], [1, 1.45]);
  const galaxyOpacity = useTransform(scrollYProgress, [0, 0.55], [0.9, 0.18]);

  return (
    <div className="fixed inset-0 z-0 overflow-hidden bg-void-950" aria-hidden="true">
      {/* ① 遠ざかっていく銀河。
          センタリングは Tailwind の -translate-* ではなく motion style 側で行う
          （同一要素で class の transform と framer の transform は衝突するため） */}
      <motion.div
        style={{ x: "-50%", y: "-50%", scale: galaxyScale, opacity: galaxyOpacity }}
        className="absolute left-1/2 top-1/2 aspect-square w-[160vmin]"
      >
        <Galaxy density={0.4} />
      </motion.div>

      {/* 暗幕は銀河の上・星屑の下。こうするとストリークが減光されずに立つ */}
      <div className="absolute inset-0 bg-void-950/55" />

      {/* ② 前を流れる星屑 */}
      <DeepFlight className="absolute inset-0" />

      {/* ヴィネットは最前面（画面端でストリークが落ちる） */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_30%,rgba(3,2,10,0.85)_85%)]" />
    </div>
  );
}
