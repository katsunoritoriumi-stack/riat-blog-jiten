"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import {
  cubicBezier,
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
  /**
   * このステーションを通り過ぎたあとの「航行区間」に出す予告編のコピー。
   * 何も表示されない時間を、次の場面への引きとして使う。
   */
  caption?: { en: string; jp: string };
};

/**
 * 1ステーションの帯（q: 0〜1）の使い方。
 *
 *   -0.18 ┄┄ 0 ────────── 0.34 ─────── 0.52 ┄┄┄┄┄┄┄┄┄ 0.82(=次の出現開始)
 *     現れる    静止して読める      通り過ぎる      何も無い＝航行区間
 *
 * 通り過ぎたあとに「何も無い区間」をわざと空けている。ここでは星屑だけが流れ、
 * 次のセクションはまだ現れない＝宇宙を移動している時間になる。
 * この空白が無いと、セクションが切れ目なく続いて「旅している感じ」が出ない。
 */
const ENTER_FROM = -0.18; // ここから奥に現れ始める
const ENTER_TO = 0; // ここで実寸に到着
const HOLD_TO = 0.34; // ここまで静止（読める・押せる）
const EXIT_TO = 0.52; // ここで完全に通過（以降 0.82 までは何も無い）
const SCALE_FAR = 0.34; // 現れ始めの大きさ（遠いほど旅の距離を感じる）
const SCALE_PAST = 2.6; // 通り過ぎるときの大きさ
const BLUR_MAX = 11; // ピント外れの最大量(px)

/** 近づくときは減速して着地、去るときは加速して抜けていく */
const EASE_APPROACH = cubicBezier(0.16, 0.62, 0.24, 1); // easeOut 寄り
const EASE_DEPART = cubicBezier(0.62, 0, 0.9, 0.4); // easeIn 寄り

/* 予告編のタイトルカードが出ている区間（帯の 0.52〜0.81＝航行区間の内側） */
const CARD_IN = 0.53;
const CARD_SHARP = 0.62;
const CARD_HOLD = 0.72;
const CARD_OUT = 0.81;

/**
 * 航行区間に差し込む予告編のタイトルカード。
 * 大きな字間の大文字が、ぼけた状態から一度に焦点を結び（＝打ち込まれ）、
 * しばらく留まってから静かに引いていく。映画の予告編の呼吸に合わせている。
 */
function TrailerCard({
  progress,
  start,
  span,
  en,
  jp,
}: {
  progress: MotionValue<number>;
  start: number;
  span: number;
  en: string;
  jp: string;
}) {
  const q = useTransform(progress, (v) => (v - start) / span);
  const stops = [CARD_IN, CARD_SHARP, CARD_HOLD, CARD_OUT];
  const opacity = useTransform(q, stops, [0, 1, 1, 0]);
  const scale = useTransform(q, stops, [1.18, 1, 1, 0.97]);
  const blurPx = useTransform(q, stops, [18, 0, 0, 9]);
  const filter = useTransform(blurPx, (v) => (v < 0.2 ? "none" : `blur(${v.toFixed(1)}px)`));
  const ruleScale = useTransform(q, stops, [0, 1, 1, 1]);

  const [live, setLive] = useState(() => {
    const v = q.get();
    return v > CARD_IN - 0.02 && v < CARD_OUT + 0.02;
  });
  useMotionValueEvent(q, "change", (v) => {
    const next = v > CARD_IN - 0.02 && v < CARD_OUT + 0.02;
    setLive((p) => (p === next ? p : next));
  });

  return (
    <motion.div
      style={{ opacity, scale, filter, display: live ? undefined : "none" }}
      className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center px-8 text-center"
    >
      {/* 文字が沈まないよう、角の出ないぼかし暗幕を敷く */}
      <span
        aria-hidden
        className="pointer-events-none absolute left-1/2 top-1/2 -z-10 -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{
          width: "120%",
          height: "60%",
          background:
            "radial-gradient(closest-side, rgba(3,2,10,0.82), rgba(3,2,10,0.4) 55%, rgba(3,2,10,0) 100%)",
          filter: "blur(18px)",
        }}
      />
      <p
        className="font-display text-[clamp(1.1rem,4.2vw,2.6rem)] font-light uppercase leading-[1.25] tracking-[0.2em] text-nebula-50 sm:tracking-[0.3em]"
        style={{ textShadow: "0 2px 18px rgba(0,0,0,0.9), 0 0 44px rgba(124,58,237,0.35)" }}
      >
        {en}
      </p>
      <motion.span
        aria-hidden
        style={{ scaleX: ruleScale }}
        className="mt-5 block h-px w-28 bg-gradient-to-r from-transparent via-aurum-300/70 to-transparent"
      />
      <p
        className="mt-5 font-serif text-[clamp(0.7rem,2.4vw,0.95rem)] tracking-[0.2em] text-nebula-200/70"
        style={{ textShadow: "0 1px 10px rgba(0,0,0,0.9)" }}
      >
        {jp}
      </p>
    </motion.div>
  );
}

function StationStage({
  children,
  progress,
  start,
  span,
  blurMax,
  last,
}: {
  children: ReactNode;
  progress: MotionValue<number>;
  start: number;
  span: number;
  /** ピント外れの最大量(px)。0 ならぼかさない */
  blurMax: number;
  /** 最後のステーションは通過させない（次が無いので静止したまま終わる） */
  last: boolean;
}) {
  // q < 0: まだ奥　0〜0.76: 到着して静止　> 0.76: 通過中
  const q = useTransform(progress, (v) => (v - start) / span);

  const holdTo = last ? EXIT_TO : HOLD_TO;
  const easing = { ease: [EASE_APPROACH, EASE_APPROACH, EASE_DEPART] };
  const travel = useTransform(
    q,
    [ENTER_FROM, ENTER_TO, holdTo, EXIT_TO],
    [SCALE_FAR, 1, 1, last ? 1 : SCALE_PAST],
    easing
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
  const opacity = useTransform(
    q,
    [ENTER_FROM, ENTER_TO, holdTo, EXIT_TO],
    [0, 1, 1, last ? 1 : 0],
    easing
  );
  const blurPx = useTransform(
    q,
    [ENTER_FROM, ENTER_TO, holdTo, EXIT_TO],
    [blurMax, 0, 0, last ? 0 : blurMax],
    easing
  );
  // filter は必ず style に含める。外してしまうと、サーバーHTMLに焼き込まれた
  // blur(...) を framer が上書きできず、ぼけたまま固まる（モバイルで発生した不具合）
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
        filter,
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

  /**
   * フルスクリーンのぼかしはモバイルで重いので、細かいポインタ（＝PC）でだけ有効にする。
   * 描画中に matchMedia を読むとサーバーとクライアントで結果が食い違い、
   * サーバー側のぼかしが焼き付いてしまうため、マウント後に切り替える。
   */
  const [blurMax, setBlurMax] = useState(0);
  useEffect(() => {
    const mq = window.matchMedia("(pointer: coarse)");
    const sync = () => setBlurMax(mq.matches ? 0 : BLUR_MAX);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);

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
            blurMax={blurMax}
            last={i === stations.length - 1}
          >
            {s.node}
          </StationStage>
        ))}

        {/* 航行区間に差し込む予告編のコピー */}
        {stations.map((s, i) =>
          s.caption && i < stations.length - 1 ? (
            <TrailerCard
              key={`cap-${s.id}`}
              progress={scrollYProgress}
              start={bands[i].start}
              span={bands[i].span}
              en={s.caption.en}
              jp={s.caption.jp}
            />
          ) : null
        )}
      </div>
    </>
  );
}
