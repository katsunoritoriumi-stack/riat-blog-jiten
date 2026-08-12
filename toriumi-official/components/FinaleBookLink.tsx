"use client";

import { motion } from "framer-motion";
import { ArrowUpRight } from "lucide-react";

/**
 * 終章の絵の中の本を指し示す HUD。
 *
 * 本そのものを押させるのではなく、SF の計器表示のような引き出し線を絵の上に引き、
 * その線と先端の照準・文字がリンクになっている。
 *
 * 座標は絵のピクセルそのまま。
 * .finale-frame（globals.css）が「絵の比率のまま画面を覆う最小の箱」なので、
 * その中に絵と同じ viewBox の SVG を敷けば、線の端は絵の狙った場所に必ず載る。
 * 絵とこの層は同じ CSS アニメーション（.finale-zoom）で動くのでズレない。
 *
 * 外側の overflow は hidden ではなく clip。hidden はスクロール可能な箱を作るので、
 * リンクを Tab でフォーカスすると層だけがずれて指し示す先が狂う（実測で 48px ずれた）。
 *
 * 絵を差し替えたら、下の SPEC の数字（本の紋章＝終端の位置など）を測り直す。
 */

const HREF = "https://seimeiron.com/riat-blog/";

type Spec = {
  /** 絵の実寸。SVG の viewBox にそのまま使う */
  w: number;
  h: number;
  /** 終端（本の紋章の中心） */
  tip: [number, number];
  /** 水平に走る部分の折れ点 */
  elbow: [number, number];
  /** 斜めの線の先 */
  tail: [number, number];
  /** 線の太さ・照準の大きさ（絵の実寸に対して決める） */
  stroke: number;
  r: number;
  /** 文字の大きさ（絵の高さに対する比） */
  font: number;
  /** 文字を線の先にどう付けるか。中央寄せは横幅が足りない画面用 */
  align: "start" | "middle";
  /** 文字を線の先からどれだけ上へ逃がすか（絵の高さに対する比） */
  lift: number;
};

/**
 * スマホ：本は画面のやや左。絵自体も左へずらしてある（globals.css の --ox）。
 *   文字は線の先の真上に中央寄せで置く。左寄せだと画面右にはみ出し、
 *   下げすぎるとページ末尾のロゴと重なるため、この位置に逃がしている。
 * PC：本は左下。右側が宇宙で空いているので、そちらへ引き出して左寄せ。
 * どちらも「斜め → 水平 → 照準」で、参考にした計器表示と同じ運びにしてある。
 */
const MOBILE: Spec = {
  w: 1100,
  h: 1310,
  tip: [484, 984],
  elbow: [566, 984],
  tail: [676, 878],
  stroke: 3.2,
  r: 13,
  font: 0.0135,
  align: "middle",
  lift: 0.045,
};

/**
 * PC は引き出しを短くして、絵の左寄りで完結させている。
 * 長く伸ばすとページ末尾のロゴ（画面中央）に文字の頭が触れるため。
 * 絵の 50% は画面の中央に一致するので、文字の右端が 45% を超えないようにしてある。
 */
const DESKTOP: Spec = {
  w: 1672,
  h: 942,
  tip: [244, 796],
  elbow: [356, 796],
  tail: [452, 706],
  stroke: 2.8,
  r: 11,
  font: 0.018,
  align: "start",
  lift: 0.03,
};

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
          transition={{ duration: 1, delay: 0.4 }}
          aria-label="宇宙生命論のブログを別のタブで開く"
          className="group pointer-events-none absolute inset-0 block"
        >
          <Pointer spec={MOBILE} className="md:hidden" />
          <Pointer spec={DESKTOP} className="hidden md:block" />
        </motion.a>
      </div>
    </div>
  );
}

function Pointer({ spec, className }: { spec: Spec; className: string }) {
  const { w, h, tip, elbow, tail, stroke, r, font, align, lift } = spec;
  /** 斜め → 水平 → 照準。参考にした計器表示と同じ運び */
  const line = `M ${tail[0]} ${tail[1]} L ${elbow[0]} ${elbow[1]} L ${tip[0] + r} ${tip[1]}`;
  /** 斜めに沿わせる細い相棒の線（計器表示によくある二重線） */
  const dx = elbow[0] - tail[0];
  const dy = elbow[1] - tail[1];
  const len = Math.hypot(dx, dy);
  const off = stroke * 2.6;
  const nx = (-dy / len) * off;
  const ny = (dx / len) * off;
  const line2 = `M ${tail[0] + nx + dx * 0.06} ${tail[1] + ny + dy * 0.06} L ${
    elbow[0] + nx - dx * 0.34
  } ${elbow[1] + ny - dy * 0.34}`;

  /** 目盛りの四角。文字と反対側（線の外）へ並べる */
  const tick = stroke * 2.1;
  const gap = tick * 1.9;
  const tx = tail[0] + (dx / len) * -tick * 3.2;
  const ty = tail[1] + (dy / len) * -tick * 3.2;

  /** 文字を入れる箱の幅。8 文字＋矢印＋字間ぶんを見込む */
  const widthBox = h * font * 9.2;

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className={`absolute inset-0 h-full w-full ${className}`}
      fill="none"
      aria-hidden="true"
    >
      <defs>
        <filter id={`hud-glow-${w}`} x="-40%" y="-40%" width="180%" height="180%">
          <feDropShadow dx="0" dy="0" stdDeviation={stroke * 1.1} floodColor="#fceabb" floodOpacity="0.4" />
        </filter>
      </defs>

      {/*
        当たり判定。細い線のままだと押しにくいので、透明な太い線を重ねる。
        これと文字だけが pointer-events を持つ（絵の他の場所は押せない）。
      */}
      <path
        d={line}
        stroke="transparent"
        strokeWidth={stroke * 9}
        strokeLinecap="round"
        className="pointer-events-auto cursor-pointer"
      />

      {/*
        全体を薄めに置く。絵の主役は本と人物なので、案内は気づける最小限にとどめる。
        カーソルを載せたときだけはっきりさせて、押せることを確かめられるようにする。
      */}
      <g
        filter={`url(#hud-glow-${w})`}
        className="opacity-[0.62] transition-opacity duration-500 group-hover:opacity-100"
      >
        {/* 引き出し線 */}
        <path
          d={line}
          pathLength={100}
          stroke="#fceabb"
          strokeWidth={stroke}
          strokeLinecap="square"
          className="hud-line transition-[stroke] duration-300 group-hover:stroke-white"
        />
        {/* 斜めに沿う二重線 */}
        <path
          d={line2}
          pathLength={100}
          stroke="#fceabb"
          strokeWidth={stroke * 0.6}
          strokeLinecap="square"
          opacity={0.7}
          className="hud-line hud-line-2"
        />

        {/* 終端の照準：芯・回るアーク・広がる波紋 */}
        <circle cx={tip[0]} cy={tip[1]} r={stroke * 1.5} fill="#fceabb" />
        <circle
          cx={tip[0]}
          cy={tip[1]}
          r={r}
          stroke="#fceabb"
          strokeWidth={stroke * 0.8}
          strokeDasharray={`${r * 2.6} ${r * 1.5}`}
          className="hud-arc"
        />
        <circle
          cx={tip[0]}
          cy={tip[1]}
          r={r}
          stroke="#fceabb"
          strokeWidth={stroke * 0.7}
          className="hud-ping"
        />

        {/* 目盛りの四角 3 つ */}
        {[0, 1, 2].map((i) => (
          <rect
            key={i}
            x={tx - (dx / len) * gap * i - tick / 2}
            y={ty - (dy / len) * gap * i - tick / 2}
            width={tick}
            height={tick}
            fill="#fceabb"
            className={`hud-tick ${i === 1 ? "hud-tick-2" : i === 2 ? "hud-tick-3" : ""}`}
          />
        ))}
      </g>

      {/*
        文字。線の先に付けて、矢印の一部として押せるようにする。
        foreignObject の箱は widthBox ぶん取り、中身を寄せて位置を決める
        （中央寄せのときに箱を中心に置くと、はみ出しの計算が単純になる）。
      */}
      <foreignObject
        x={align === "middle" ? tail[0] - widthBox / 2 : tail[0] - stroke * 2}
        y={tail[1] - h * lift - h * font * 1.9}
        width={widthBox}
        height={h * font * 2.4}
        className="overflow-visible"
      >
        <span
          className={`pointer-events-auto inline-flex cursor-pointer items-center gap-2 whitespace-nowrap uppercase tracking-[0.2em] text-aurum-200/75 transition-all duration-300 group-hover:tracking-[0.26em] group-hover:text-aurum-100 ${
            align === "middle" ? "w-full justify-center" : ""
          }`}
          style={{
            // 計器表示らしい幾何学的な書体。読み込み前は等幅で代用する
            fontFamily: "var(--font-hud), var(--font-mono), ui-monospace, monospace",
            fontSize: `${h * font}px`,
            textShadow: "0 1px 8px rgba(3,2,10,1), 0 0 22px rgba(3,2,10,0.95)",
          }}
        >
          Travel Guide
          <ArrowUpRight
            size={h * font * 1.15}
            className="transition-transform duration-300 group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
          />
        </span>
      </foreignObject>
    </svg>
  );
}
