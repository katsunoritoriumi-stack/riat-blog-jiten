"use client";

import { useEffect, useRef, useState } from "react";
import { motion, useMotionValueEvent, useScroll, useTransform } from "framer-motion";
import { CARD_HOLD, CARD_IN, CARD_OUT, CARD_SHARP } from "./ui/ZoomStage";
import { STATIONS, computeBands } from "@/lib/stations";

/**
 * 記憶の回想。
 *
 * 「地球での転生回数」を数える場面の裏で、いくつもの生の断片が漂う。
 * 素材は黒地に写真が散らばった映像なので、mix-blend-mode: screen で黒を抜き、
 * 背景の宇宙をそのまま透かす。四角い動画には見えない。
 *
 * ⚠ screen 合成は「いちばん外側の箱」に掛ける。
 * position: fixed の箱は、それだけで合成グループを作ってしまう。
 * つまり中の映像に screen を掛けても、混ざる相手はこの箱の中だけで、
 * 背景の宇宙までは届かない。黒はそのまま黒として乗り、宇宙が暗く沈む
 * （ヘッドレス Chrome で計測：下地 [145,153,180] が [26,28,33] まで落ちた）。
 * 箱自身に screen を掛けると、箱ごと宇宙に混ざって黒が正しく抜ける。
 * 中の二層にも screen を残してあるのは、層同士も足し算で重ねるため。
 *
 * ── 二層で流している理由 ──
 * 一本だけだと画面に出ている写真の数が物足りない。かといって同じ映像を
 * そのまま重ねると、写真同士が同じ場所でぶつかって濁る。
 * そこで二層を「別の場所・別の時間」に振り分けてある：
 *   ・中央の層 … いまと同じ楕円のマスク。手前の記憶としてはっきり見せる
 *   ・外周の層 … 中央を抜いたリングのマスク。左右反転して 5.7 秒ずらし、
 *     少し外へ逃がして、奥の記憶としてやわらかく見せる
 * マスクが中央とリングで食い違うので、重なる余地はリングの内側の縁だけに限られる。
 * 実測（6fps・84 コマ・写真の面積で計測）では、見えている写真のうち
 * 重なるのは 7.2%。同じ映像をただ重ねた場合は 31.8% だった。
 * 写真の量はこれで 202%（ほぼ 2 倍）になる。
 *
 * ⚠ ZoomStage の中に置いてはいけない。
 * mix-blend-mode は「祖先が作る合成グループ」の内側としか混ざらない。
 * 予告編カードは transform（scale）を持つので独立したグループになり、
 * その中に置くと背景の宇宙と混ざらず、黒地がそのまま矩形として乗ってしまう。
 * だから宇宙（z-0）と ZoomStage（z-10）の間に、独立した層として敷いている。
 * 同じ理由で、出入りのフェードは外側の箱ではなく映像自身の opacity に持たせる。
 *
 * どの場面で流すかは lib/stations.ts の caption.memoryVideo が唯一の出典。
 * 出入りの刻みも予告編カードと同じ定数を使うので、文字と映像がずれない。
 */

const BANDS = computeBands(STATIONS.map((s) => s.scroll));
const INDEX = STATIONS.findIndex((s) => s.caption?.memoryVideo);
const SRC = INDEX >= 0 ? STATIONS[INDEX].caption!.memoryVideo! : null;

/**
 * いちばん濃いときの不透明度。
 * 0.24 → 0.45 → 0.62 と上げてきて、まだ薄いという判断でここまで来た。
 * 文字の真後ろは予告編カード側の暗幕（このレイヤーより手前）が守るので、
 * ここを濃くしてもカウンターの可読性は落ちない。
 */
const CENTER_PEAK = 0.72;
/**
 * 外周の層。ここは画面の端＝視界のふちなので、中央よりはっきり下げる。
 * 素材の端には大きな写真が多く、中央と同じ濃さにすると
 * 回想ではなく写真の壁になってカウンターが読みにくくなる（実際にそうなった）。
 */
const RING_PEAK = 0.5;

/** 外周の層をずらす秒数。素材は 14 秒なので、ほぼ反対側の場面になる */
const RING_OFFSET_S = 5.7;
/** 実時間よりゆっくり流して、回想の速度にする */
const RATE = 0.75;

/** 中央：いまと同じ楕円。縁を溶かして、画面の中央付近に像を残す */
const MASK_CENTER =
  "radial-gradient(ellipse 68% 58% at 50% 50%, #000 30%, rgba(0,0,0,0.55) 62%, transparent 88%)";
/**
 * 外周：中央をくり抜いたリング。中央の層とぶつからない場所だけに出る。
 *
 * ⚠ 半径は 50%／50% から広げないこと。
 * radial-gradient の「ellipse 66% 60%」は箱の縁までの割合ではなく、
 * 中心からの半径が幅の 66%・高さの 60% という意味。中心は 50% にあるので、
 * 半径が 50% を超えると、色止め 100% の透明が箱の外側に落ちてしまう。
 * つまり箱の縁ではまだ濃度が残り、そこが直線の切れ目として見える。
 * この層は translate でずらしてあるため、その切れ目が画面の中に入る。
 * 実際に画面の右 90% の位置へ縦線が出た（66% のとき縁の濃度 0.87）。
 * 50% なら左右も上下も、縁でちょうど 0 になる。
 */
const MASK_RING =
  "radial-gradient(ellipse 50% 50% at 50% 50%, transparent 16%, rgba(0,0,0,0.6) 42%, #000 74%, transparent 100%)";

export default function MemoryFlashback() {
  const { scrollYProgress } = useScroll();
  const band = INDEX >= 0 ? BANDS[INDEX] : null;
  const centerRef = useRef<HTMLVideoElement>(null);
  const ringRef = useRef<HTMLVideoElement>(null);

  const q = useTransform(scrollYProgress, (v) =>
    band ? (v - band.start) / band.span : -1
  );
  const steps: [number, number, number, number] = [
    CARD_IN,
    CARD_SHARP,
    CARD_HOLD,
    CARD_OUT,
  ];
  const opCenter = useTransform(q, steps, [0, CENTER_PEAK, CENTER_PEAK, 0]);
  const opRing = useTransform(q, steps, [0, RING_PEAK, RING_PEAK, 0]);
  const opWash = useTransform(q, steps, [0, 1, 1, 0]);

  // 画面に出ている間だけ DOM に置く。通り過ぎれば読み込みも再生も止まる
  const [live, setLive] = useState(false);
  useMotionValueEvent(q, "change", (v) => {
    const next = v > CARD_IN - 0.04 && v < CARD_OUT + 0.04;
    setLive((p) => (p === next ? p : next));
  });

  useEffect(() => {
    if (!live) return;
    for (const v of [centerRef.current, ringRef.current]) {
      if (!v) continue;
      v.playbackRate = RATE;
      const p = v.play();
      if (p && typeof p.catch === "function") p.catch(() => {});
    }
  }, [live]);

  if (!SRC || !band) return null;

  return (
    <div
      aria-hidden="true"
      style={{ display: live ? undefined : "none", mixBlendMode: "screen" }}
      className="pointer-events-none fixed inset-0 z-[1] overflow-clip"
    >
      {live && (
        <>
          {/* 外周＝奥の記憶。左右反転・時間差・少し外へ逃がして、輪郭もやわらかく */}
          <motion.video
            ref={ringRef}
            src={SRC}
            muted
            loop
            playsInline
            preload="none"
            onLoadedMetadata={(e) => {
              // 中央の層と別の場面を映すための時間差。頭出しは一度だけでよい
              try {
                e.currentTarget.currentTime = RING_OFFSET_S;
              } catch {}
            }}
            style={{
              opacity: opRing,
              mixBlendMode: "screen",
              filter: "blur(2.2px) saturate(0.8) contrast(0.95)",
              maskImage: MASK_RING,
              WebkitMaskImage: MASK_RING,
              transform: "translate(-10%, 8%) scaleX(-1)",
            }}
            className="absolute inset-0 h-full w-full object-cover"
          />
          {/* 中央＝手前の記憶。ぼかしを浅くして、何が映っているか分かる程度に */}
          <motion.video
            ref={centerRef}
            src={SRC}
            muted
            loop
            playsInline
            preload="none"
            style={{
              opacity: opCenter,
              mixBlendMode: "screen",
              filter: "blur(1px) saturate(0.9) contrast(1.05)",
              maskImage: MASK_CENTER,
              WebkitMaskImage: MASK_CENTER,
            }}
            className="absolute inset-0 h-full w-full object-cover"
          />
        </>
      )}
      {/* うっすら紫をかぶせて、サイトの空気に馴染ませる */}
      <motion.span
        className="absolute inset-0"
        style={{
          opacity: opWash,
          background:
            "radial-gradient(ellipse 70% 60% at 50% 50%, rgba(124,58,237,0.10), transparent 75%)",
        }}
      />
    </div>
  );
}
