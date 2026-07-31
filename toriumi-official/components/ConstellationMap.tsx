"use client";

import { Component, type ReactNode, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { motion } from "framer-motion";
import SectionHeader from "./ui/SectionHeader";
import { DOMAINS } from "@/lib/content";

/**
 * 創造の座標軸。中身は three.js 製の 3D 天球図（CelestialMap3D）。
 * three.js は重い（gzip 約230KB）ので、
 *   ① next/dynamic + ssr:false でメインバンドルから切り離し
 *   ② セクションが画面に近づいてから読み込み開始（IntersectionObserver）
 * とすることで、トップ表示（ブート演出・UFO動画）の速度を落とさない。
 * 画面外に出ている間は描画ループを止める（電池・GPU 節約）。
 */

const CelestialMap3D = dynamic(() => import("./CelestialMap3D"), {
  ssr: false,
  loading: () => <MapPlaceholder />,
});

/** 読み込み中に出る、静かな星の待機画面 */
function MapPlaceholder() {
  return (
    <div className="absolute inset-0 flex items-center justify-center">
      <div className="flex flex-col items-center gap-3">
        <span className="h-2 w-2 animate-pulse-glow rounded-full bg-aurum-300" />
        <span className="font-mono text-[9px] uppercase tracking-[0.35em] text-nebula-300/45">
          Establishing orbital link
        </span>
      </div>
    </div>
  );
}

/**
 * WebGL が使えない／3D の初期化に失敗した場合の代替表示。
 * 3D の中にしかリンクが無いと、この状況で各ドメインへの導線が丸ごと消えてしまうため、
 * 素のリンク一覧に必ず退避できるようにしておく。
 */
function DomainListFallback() {
  const items = DOMAINS.filter((d) => !d.hidden);
  return (
    <div className="absolute inset-0 flex flex-col justify-center gap-6 px-8 py-10">
      <p className="text-center font-mono text-[9px] uppercase tracking-[0.35em] text-nebula-300/45">
        Orbital view unavailable — direct links
      </p>
      <ul className="mx-auto grid w-full max-w-lg grid-cols-2 gap-x-6 gap-y-3">
        {items.map((d) => {
          const links = d.links ?? (d.href ? [{ label: d.titleJp, href: d.href }] : []);
          return (
            <li key={d.key} className="border-t border-nebula-500/20 pt-2">
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-aurum-300/70">
                {d.titleEn}
              </p>
              <div className="mt-1 flex flex-col gap-0.5">
                {links.map((l) =>
                  l.href.startsWith("/") ? (
                    <Link
                      key={l.href}
                      href={l.href}
                      className="text-xs text-nebula-200/80 underline decoration-nebula-400/30 underline-offset-4 hover:text-aurum-200"
                    >
                      {l.label}
                    </Link>
                  ) : (
                    <a
                      key={l.href}
                      href={l.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-nebula-200/80 underline decoration-nebula-400/30 underline-offset-4 hover:text-aurum-200"
                    >
                      {l.label}
                    </a>
                  )
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/** 3D 側で例外が出てもページ全体を巻き込まないための境界 */
class MapBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? <DomainListFallback /> : this.props.children;
  }
}

/** WebGL が実際に取れるかを一度だけ判定する */
function detectWebGL(): boolean {
  try {
    const cv = document.createElement("canvas");
    const gl = cv.getContext("webgl2") || cv.getContext("webgl");
    // 判定用に取った context は即返す（同時に持てる数に上限があるため）
    gl?.getExtension("WEBGL_lose_context")?.loseContext();
    return !!gl;
  } catch {
    return false;
  }
}

export default function ConstellationMap() {
  const boxRef = useRef<HTMLDivElement>(null);
  const [near, setNear] = useState(false); // 読み込み開始
  const [visible, setVisible] = useState(true); // 画面内＝描画ループを回す
  const [webgl, setWebgl] = useState<boolean | null>(null);

  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;

    setWebgl(detectWebGL());

    // ① 画面の 400px 手前まで来たら即読み込み
    const pre = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setNear(true);
          pre.disconnect();
        }
      },
      { rootMargin: "400px" }
    );
    pre.observe(el);

    // ② スクロールされなくても、表示が落ち着いた頃に裏で先読みしておく
    //    （到達したときには読み込み済み＝待ち時間ゼロ。初期表示はすでに終わっている）
    const idle = window.setTimeout(() => setNear(true), 3000);

    // ③ 画面外の間は描画ループを止める（電池・GPU 節約）
    const vis = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), {
      rootMargin: "80px",
    });
    vis.observe(el);

    return () => {
      pre.disconnect();
      vis.disconnect();
      window.clearTimeout(idle);
    };
  }, []);

  const domainCount = DOMAINS.filter((d) => d.key !== "connect" && !d.hidden).length;

  return (
    <section data-section="universe" className="relative mx-auto max-w-[1700px] px-3 py-28 sm:px-6 sm:py-36">
      <div className="mb-12 max-w-2xl">
        <SectionHeader
          eyebrow="The Universe of Creation"
          titleEn="One Creator, Many Worlds"
          titleJp="創造の座標軸"
        />
        <p className="mt-4 flex items-center gap-2 font-mono text-[11px] tracking-widest text-aurum-300/70">
          <span className="inline-block h-2 w-2 animate-pulse-glow rounded-full bg-aurum-300" />
          ドラッグで視点を回し、各星をタップしてそれぞれの世界へ
        </p>
      </div>

      {/* star chart — 公転する天球図。各星がリンク */}
      <motion.div
        ref={boxRef}
        initial={{ opacity: 0, y: 24 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-25%" }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="relative mx-auto h-[clamp(420px,66vh,780px)] w-full max-w-[min(1500px,96vw)] overflow-hidden rounded-3xl border border-nebula-500/20 bg-void-950/60 backdrop-blur-sm"
      >
        {/* 奥行きの星雲ウォッシュ＋ヴィネット */}
        <div className="pointer-events-none absolute inset-0 nebula-bg opacity-50" />
        <div className="pointer-events-none absolute inset-0 z-[1] bg-[radial-gradient(circle_at_center,transparent_55%,rgba(3,2,10,0.75)_100%)]" />

        {webgl === false ? (
          <DomainListFallback />
        ) : near && webgl ? (
          <MapBoundary>
            <CelestialMap3D paused={!visible} />
          </MapBoundary>
        ) : (
          <MapPlaceholder />
        )}

        {/* ── HUD コーナーブラケット＋観測メタ情報 ── */}
        <span className="pointer-events-none absolute left-4 top-4 z-[2] h-4 w-4 rounded-tl border-l border-t border-aurum-300/40" />
        <span className="pointer-events-none absolute right-4 top-4 z-[2] h-4 w-4 rounded-tr border-r border-t border-aurum-300/40" />
        <span className="pointer-events-none absolute bottom-4 left-4 z-[2] h-4 w-4 rounded-bl border-b border-l border-aurum-300/40" />
        <span className="pointer-events-none absolute bottom-4 right-4 z-[2] h-4 w-4 rounded-br border-b border-r border-aurum-300/40" />
        <span className="pointer-events-none absolute left-7 top-6 z-[2] font-mono text-[8px] uppercase tracking-[0.25em] text-nebula-300/50 sm:text-[9px]">
          Celestial Map — K.T.
        </span>
        <span className="pointer-events-none absolute bottom-6 right-7 z-[2] font-mono text-[8px] uppercase tracking-[0.25em] text-nebula-300/50 sm:text-[9px]">
          {domainCount} Domains · 1 Core
        </span>
      </motion.div>
    </section>
  );
}
