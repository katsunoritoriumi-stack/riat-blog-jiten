"use client";

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { DOMAINS, LINKS, MANIFESTO, SITE, YOUTUBE } from "@/lib/content";

/**
 * Voyage ページの外枠。
 * - 3D は固定配置（position: fixed）。その上に高さだけを持つスペーサーを重ね、
 *   ページ本来の縦スクロールをカメラの前進に変換する。
 * - スペーサーは pointer-events: none。クリックは canvas 内の HUD リンクに素通しする。
 * - スペーサーの中には実テキストを入れてある（視覚的には出さないが、検索エンジン・
 *   スクリーンリーダー・ページ内検索からは読める）。3D の中だけに本文がある状態を作らない。
 */

const PAGES = 7; // 100vh × PAGES ぶんスクロールできる

const VoyageScene = dynamic(() => import("./VoyageScene"), {
  ssr: false,
  loading: () => null,
});

/** WebGL が使えるか（使えなければ 3D は諦めて読み物として成立させる） */
function detectWebGL(): boolean {
  try {
    const cv = document.createElement("canvas");
    const gl = cv.getContext("webgl2") || cv.getContext("webgl");
    gl?.getExtension("WEBGL_lose_context")?.loseContext();
    return !!gl;
  } catch {
    return false;
  }
}

const STATION_LABELS = [
  "Voyage Start",
  "Manifesto",
  "Universe",
  "Sound & Vision",
  "Make",
  "Final Report",
];

export default function VoyageClient() {
  const [webgl, setWebgl] = useState<boolean | null>(null);
  const barRef = useRef<HTMLDivElement>(null);
  const idxRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    setWebgl(detectWebGL());
  }, []);

  // 進捗 HUD（React state を毎フレーム更新しないよう DOM を直接触る）
  useEffect(() => {
    const onScroll = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const p = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
      if (barRef.current) barRef.current.style.transform = `scaleX(${p})`;
      if (idxRef.current) {
        const i = Math.min(STATION_LABELS.length - 1, Math.round(p * (STATION_LABELS.length - 1)));
        const label = `${String(i + 1).padStart(2, "0")} ${STATION_LABELS[i]}`;
        if (idxRef.current.textContent !== label) idxRef.current.textContent = label;
      }
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, []);

  const sections = [
    {
      h: SITE.nameEn,
      p: `a.k.a KIEJI — ${SITE.taglineJp}`,
      links: [] as { label: string; href: string }[],
    },
    { h: "創造の意思 — Manifesto", p: MANIFESTO.lines.join("") + MANIFESTO.body, links: [] },
    {
      h: "創造の座標軸 — Universe",
      p: "ひとりの中に同時に存在する、六つの世界。",
      links: DOMAINS.filter((d) => d.key !== "connect" && !d.hidden).map((d) => ({
        label: d.titleEn,
        href: d.href ?? d.links?.[0]?.href ?? "/",
      })),
    },
    {
      h: "EXODUS — Sound & Vision",
      p: "音楽と映像。声と光で綴る、もうひとつの次元。",
      links: [
        { label: "YouTube · @Exodus999", href: YOUTUBE.channelUrl },
        { label: "Radio · stand.fm", href: LINKS.radio },
      ],
    },
    {
      h: "つくる — Make",
      p: "アプリ開発／Web制作／動画制作。イメージを、動く形へ。",
      links: [{ label: "ご依頼・ご相談（LINE）", href: LINKS.line }],
    },
    {
      h: "終活レポート — Final Report",
      p: "このポータルサイトは、陽化（老化）を極めた惑星テラ（地球）でのミッションを締めくくるに当たり、Katsunori Toriumi が自己の記憶をまとめ上げる為に作成した終活レポートとして…",
      links: [],
    },
  ];

  // WebGL が無い場合は、素の読み物ページとして普通に表示する
  if (webgl === false) {
    return (
      <main className="mx-auto max-w-3xl px-6 py-24">
        <p className="font-mono text-[10px] uppercase tracking-[0.35em] text-aurum-300/70">
          Voyage — 3D unavailable
        </p>
        {sections.map((s) => (
          <section key={s.h} className="mt-14 border-t border-nebula-500/20 pt-6">
            <h2 className="font-serif text-2xl text-nebula-100">{s.h}</h2>
            <p className="mt-3 text-sm leading-relaxed text-nebula-200/70">{s.p}</p>
            <div className="mt-3 flex flex-col gap-1">
              {s.links.map((l) => (
                <a
                  key={l.href}
                  href={l.href}
                  target={l.href.startsWith("http") ? "_blank" : undefined}
                  rel={l.href.startsWith("http") ? "noopener noreferrer" : undefined}
                  className="text-sm text-aurum-200 underline decoration-aurum-400/40 underline-offset-4"
                >
                  {l.label}
                </a>
              ))}
            </div>
          </section>
        ))}
        <Link href="/" className="mt-16 inline-block text-sm text-aurum-200 underline underline-offset-4">
          ← サイトへ戻る
        </Link>
      </main>
    );
  }

  return (
    <>
      {/* 3D は背面に固定 */}
      <div className="fixed inset-0 z-0 bg-[#030509]">{webgl && <VoyageScene />}</div>

      {/* スクロール量を生むためのスペーサー。本文は実テキストで持つ（クリックは素通し） */}
      <div className="pointer-events-none relative z-10">
        {sections.map((s, i) => (
          <section
            key={s.h}
            aria-hidden={false}
            style={{ height: `${(PAGES / sections.length) * 100}vh` }}
            className="flex items-center justify-center"
          >
            {/* 視覚的には 3D 側の HUD が担当するので、ここは読み上げ・検索向け */}
            <div className="sr-only">
              <h2>{s.h}</h2>
              <p>{s.p}</p>
              {s.links.map((l) => (
                <a key={l.href} href={l.href}>
                  {l.label}
                </a>
              ))}
            </div>
            {i === 0 && (
              <span className="font-mono text-[10px] uppercase tracking-[0.4em] text-[#6ff2d6]/40">
                scroll
              </span>
            )}
          </section>
        ))}
      </div>

      {/* HUD クローム（通常の DOM。クリック可） */}
      <div className="fixed inset-x-0 top-0 z-20 flex items-center justify-between px-5 py-4">
        <Link
          href="/"
          className="pointer-events-auto font-mono text-[10px] uppercase tracking-[0.3em] text-[#6ff2d6]/80 transition-colors hover:text-[#6ff2d6]"
        >
          ← Toriumi Official
        </Link>
        <span
          ref={idxRef}
          className="font-mono text-[10px] uppercase tracking-[0.3em] text-[#6ff2d6]/60"
        >
          01 Voyage Start
        </span>
      </div>

      {/* 進捗バー */}
      <div className="fixed inset-x-0 bottom-0 z-20 h-px bg-[#6ff2d6]/15">
        <div
          ref={barRef}
          className="h-full origin-left bg-[#6ff2d6]/80"
          style={{ transform: "scaleX(0)" }}
        />
      </div>
    </>
  );
}
