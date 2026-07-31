"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import {
  motion,
  useMotionValue,
  useMotionValueEvent,
  useReducedMotion,
  useScroll,
  useTransform,
  type MotionValue,
} from "framer-motion";

/**
 * スクロールで宇宙の奥へ進む「ステーション」方式のページ。
 *
 * ページは流れない。画面いっぱいの固定ステージがあり、スクロール量が
 * 「奥へ押し込む力」になる：
 *   ・いま見ているセクションは拡大しながら通り過ぎて消える（カメラが通過する）
 *   ・同時に次のセクションが中央から小さく現れ、実寸まで育って静止する
 *   ・静止している間は普通に読めて、リンクも押せる
 *
 * スクロール量は本文と別に「スペーサー」で作る。スペーサーには各セクションの id を
 * 付けてあるので、Navbar のアンカー・ワープ演出・スクロールスパイがそのまま動く
 * （lenis の scrollTo は実レイアウト上のスペーサーを掴む）。
 */

export type StationDef = {
  /** アンカー用 id。Navbar の SECTIONS と一致させる */
  id: string;
  node: ReactNode;
  /** このステーションに割り当てるスクロール量（画面高さの倍数） */
  scroll?: number;
};

/** 進捗 q の意味：0 で到着完了、0.75 まで静止、1 を超えると通過しきる */
const ENTER_FROM = -0.4; // ここから奥に現れ始める
const ENTER_TO = -0.02; // ここで実寸に到着
const HOLD_TO = 0.76; // ここまで静止（読める・押せる）
const EXIT_TO = 1.1; // ここで完全に通過
const SCALE_FAR = 0.52; // 現れ始めの大きさ
const SCALE_PAST = 2.15; // 通り過ぎるときの大きさ
const BLUR_MAX = 9; // ピント外れの最大量(px)

function StationStage({
  children,
  progress,
  start,
  span,
  blur,
  last,
}: {
  children: ReactNode;
  progress: MotionValue<number>;
  start: number;
  span: number;
  blur: boolean;
  /** 最後のステーションは通過させない（次が無いので静止したまま終わる） */
  last: boolean;
}) {
  // q < 0: まだ奥　0〜0.76: 到着して静止　> 0.76: 通過中
  const q = useTransform(progress, (v) => (v - start) / span);

  const holdTo = last ? EXIT_TO : HOLD_TO;
  const travel = useTransform(
    q,
    [ENTER_FROM, ENTER_TO, holdTo, EXIT_TO],
    [SCALE_FAR, 1, 1, last ? 1 : SCALE_PAST]
  );

  /**
   * 画面1枚に収まらない中身は、収まる倍率まで縮めて表示する。
   * 低いビューポート（横向きスマホ・小さいノート）でも切れないようにするための保険。
   */
  const stageRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  const fit = useMotionValue(1);
  const measure = useCallback(() => {
    const stage = stageRef.current;
    const inner = innerRef.current;
    if (!stage || !inner) return;
    const avail = stage.clientHeight;
    const need = inner.scrollHeight;
    if (!avail || !need) return;
    // 0.62 を下限に。これ以上小さくすると読めないので、そこからは内部スクロールに任せる
    const f = need > avail ? Math.max(0.62, avail / need) : 1;
    if (Math.abs(fit.get() - f) > 0.004) fit.set(f);
  }, [fit]);

  const scale = useTransform([travel, fit], ([t, f]: number[]) => t * f);
  const opacity = useTransform(q, [ENTER_FROM, ENTER_TO, holdTo, EXIT_TO], [0, 1, 1, last ? 1 : 0]);
  const blurPx = useTransform(q, [ENTER_FROM, ENTER_TO, holdTo, EXIT_TO], [BLUR_MAX, 0, 0, last ? 0 : BLUR_MAX]);
  const filter = useTransform(blurPx, (v) => (v < 0.15 ? "none" : `blur(${v.toFixed(1)}px)`));

  // 画面に関係ない間は display:none にする。
  // これで裏の 3D キャンバスや whileInView が無駄に走らず、
  // 表示されたタイミングで各セクション本来のリビールが動く。
  const [live, setLive] = useState(() => {
    const v = q.get();
    return v > ENTER_FROM - 0.05 && (last || v < EXIT_TO + 0.05);
  });
  // 触れるのは静止している間だけ
  const [interactive, setInteractive] = useState(() => {
    const v = q.get();
    return v > ENTER_TO - 0.1 && (last || v < HOLD_TO + 0.05);
  });

  useMotionValueEvent(q, "change", (v) => {
    const nextLive = v > ENTER_FROM - 0.05 && (last || v < EXIT_TO + 0.05);
    setLive((p) => (p === nextLive ? p : nextLive));
    const nextInteractive = v > ENTER_TO - 0.1 && (last || v < HOLD_TO + 0.05);
    setInteractive((p) => (p === nextInteractive ? p : nextInteractive));
  });

  // 表示されたタイミングと、画面サイズが変わったときに測り直す
  useLayoutEffect(() => {
    if (live) measure();
  }, [live, measure]);
  useEffect(() => {
    const stage = stageRef.current;
    window.addEventListener("resize", measure);
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(measure) : null;
    if (ro && stage) {
      ro.observe(stage);
      if (innerRef.current) ro.observe(innerRef.current);
    }
    return () => {
      window.removeEventListener("resize", measure);
      ro?.disconnect();
    };
  }, [measure]);

  return (
    <motion.div
      ref={stageRef}
      data-station
      aria-hidden={!interactive}
      style={{
        scale,
        opacity,
        filter: blur ? filter : undefined,
        display: live ? undefined : "none",
        pointerEvents: interactive ? "auto" : "none",
      }}
      className="zoom-station absolute inset-0 flex items-center justify-center overflow-y-auto overscroll-contain"
    >
      <div ref={innerRef} className="w-full">
        {children}
      </div>
    </motion.div>
  );
}

export default function ZoomStage({ stations }: { stations: StationDef[] }) {
  const reduce = useReducedMotion();
  const { scrollYProgress } = useScroll();

  // 各ステーションのスクロール帯を、重みから求める
  const weights = stations.map((s) => s.scroll ?? 1);
  const total = weights.reduce((a, b) => a + b, 0);
  let acc = 0;
  const bands = weights.map((wgt) => {
    const start = acc / total;
    acc += wgt;
    return { start, span: wgt / total };
  });

  // モバイル（粗いポインタ）ではフルスクリーンのぼかしが重いので切る
  const blur =
    typeof window !== "undefined" ? !window.matchMedia("(pointer: coarse)").matches : true;

  if (reduce) {
    return (
      <div>
        {stations.map((s) => (
          <div key={s.id} id={s.id} data-station>
            {s.node}
          </div>
        ))}
      </div>
    );
  }

  return (
    <>
      {/* JS が動かない場合は、固定ステージをやめて普通に縦積みで読ませる */}
      <noscript>
        <style>{`
          .zoom-stage { position: static !important; height: auto !important; }
          .zoom-station { position: static !important; display: block !important;
            opacity: 1 !important; transform: none !important; filter: none !important;
            pointer-events: auto !important; overflow: visible !important; }
          .zoom-spacers { display: none !important; }
        `}</style>
      </noscript>

      {/* スクロール量を作るスペーサー。アンカーの着地点でもある */}
      <div className="zoom-spacers" aria-hidden="true">
        {stations.map((s, i) => (
          <div key={s.id} id={s.id} style={{ height: `${weights[i] * 100}svh` }} />
        ))}
      </div>

      {/* 固定ステージ（Navbar の下・銀河背景の上） */}
      <div className="zoom-stage pointer-events-none fixed inset-0 z-10">
        {stations.map((s, i) => (
          <StationStage
            key={s.id}
            progress={scrollYProgress}
            start={bands[i].start}
            span={bands[i].span}
            blur={blur}
            last={i === stations.length - 1}
          >
            {s.node}
          </StationStage>
        ))}
      </div>
    </>
  );
}
