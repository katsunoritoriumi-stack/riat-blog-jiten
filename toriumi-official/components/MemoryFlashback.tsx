"use client";

import { useEffect, useRef, useState } from "react";
import { motion, useMotionValueEvent, useScroll, useTransform } from "framer-motion";
import { CARD_HOLD, CARD_IN, CARD_OUT, CARD_SHARP } from "./ui/ZoomStage";
import { STATIONS, computeBands } from "@/lib/stations";

/**
 * 記憶の回想。
 *
 * 「地球での転生回数」を数える場面の裏で、いくつもの生の断片が漂う。
 * はっきり見せるものではないので、次の三つで“思い出しかけている”状態に留める：
 *   ・mix-blend-mode: screen … 素材は黒地なので黒が完全に抜け、
 *     背景の宇宙がそのまま透ける。四角い動画に見えない。
 *   ・弱いぼかしと彩度落とし … 焦点が合わない記憶のように
 *   ・楕円のマスク … 縁を溶かして、画面の中央付近にだけ像が残る
 *
 * ⚠ ZoomStage の中に置いてはいけない。
 * mix-blend-mode は「祖先が作る合成グループ」の内側としか混ざらない。
 * 予告編カードは transform（scale）を持つので独立したグループになり、
 * その中に置くと背景の宇宙と混ざらず、黒地がそのまま矩形として乗ってしまう。
 * だから宇宙（z-0）と ZoomStage（z-10）の間に、独立した層として敷いている。
 *
 * どの場面で流すかは lib/stations.ts の caption.memoryVideo が唯一の出典。
 * 出入りの刻みも予告編カードと同じ定数を使うので、文字と映像がずれない。
 */

const BANDS = computeBands(STATIONS.map((s) => s.scroll));
const INDEX = STATIONS.findIndex((s) => s.caption?.memoryVideo);
const SRC = INDEX >= 0 ? STATIONS[INDEX].caption!.memoryVideo! : null;

/**
 * いちばん濃いときでもこのくらい。回想なので像は結ばせきらない。
 * 背景が暗い場面なので、0.34 では写真がはっきり出すぎた（合成して見比べて決めた）。
 */
const PEAK = 0.24;

export default function MemoryFlashback() {
  const { scrollYProgress } = useScroll();
  const band = INDEX >= 0 ? BANDS[INDEX] : null;
  const videoRef = useRef<HTMLVideoElement>(null);

  const q = useTransform(scrollYProgress, (v) =>
    band ? (v - band.start) / band.span : -1
  );
  const opacity = useTransform(
    q,
    [CARD_IN, CARD_SHARP, CARD_HOLD, CARD_OUT],
    [0, PEAK, PEAK, 0]
  );

  // 画面に出ている間だけ DOM に置く。通り過ぎれば読み込みも再生も止まる
  const [live, setLive] = useState(false);
  useMotionValueEvent(q, "change", (v) => {
    const next = v > CARD_IN - 0.04 && v < CARD_OUT + 0.04;
    setLive((p) => (p === next ? p : next));
  });

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !live) return;
    // 実時間よりゆっくり流して、回想の速度にする
    v.playbackRate = 0.75;
    const p = v.play();
    if (p && typeof p.catch === "function") p.catch(() => {});
  }, [live]);

  if (!SRC || !band) return null;

  const fade =
    "radial-gradient(ellipse 68% 58% at 50% 50%, #000 30%, rgba(0,0,0,0.55) 62%, transparent 88%)";

  return (
    <motion.div
      aria-hidden="true"
      style={{ opacity, display: live ? undefined : "none" }}
      className="pointer-events-none fixed inset-0 z-[1] overflow-clip"
    >
      {live && (
        <video
          ref={videoRef}
          src={SRC}
          muted
          loop
          playsInline
          preload="none"
          className="absolute inset-0 h-full w-full object-cover"
          style={{
            mixBlendMode: "screen",
            filter: "blur(1.6px) saturate(0.72) contrast(0.95)",
            maskImage: fade,
            WebkitMaskImage: fade,
          }}
        />
      )}
      {/* うっすら紫をかぶせて、サイトの空気に馴染ませる */}
      <span
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse 70% 60% at 50% 50%, rgba(124,58,237,0.10), transparent 75%)",
        }}
      />
    </motion.div>
  );
}
