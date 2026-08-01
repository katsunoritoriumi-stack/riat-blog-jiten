"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import SectionHeader from "./ui/SectionHeader";
import { DOMAINS } from "@/lib/content";

/**
 * 創造の座標軸。
 *
 * 太陽系そのものは背景の宇宙（components/universe/SolarSystem.tsx）の中に浮かんでいて、
 * このステーションが近づくと奥から現れる。だからここはもうカードでも 3D の入れ物でもなく、
 * 「その領域に着いたことを告げる見出し」と、
 * 「WebGL が使えないときにドメインへ辿り着くための退避経路」だけを持つ薄い層。
 *
 * 惑星のクリックは背後のキャンバスが受け取るので、この層は
 * pointer-events を通す（見出しだけが浮いている状態にする）。
 */

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

/**
 * WebGL が使えない場合の代替表示。
 * 3D の中にしかリンクが無いと、この状況で各ドメインへの導線が丸ごと消えてしまうため、
 * 素のリンク一覧に必ず退避できるようにしておく。
 */
function DomainListFallback() {
  const items = DOMAINS.filter((d) => !d.hidden);
  return (
    <div className="pointer-events-auto mx-auto mt-10 w-full max-w-2xl">
      <p className="text-center font-mono text-[9px] uppercase tracking-[0.35em] text-nebula-300/45">
        Orbital view unavailable — direct links
      </p>
      <ul className="mt-6 grid grid-cols-2 gap-x-6 gap-y-3">
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

export default function ConstellationMap() {
  const [webgl, setWebgl] = useState<boolean | null>(null);
  useEffect(() => setWebgl(detectWebGL()), []);

  const domainCount = DOMAINS.filter((d) => d.key !== "connect" && !d.hidden).length;

  return (
    <section
      data-section="universe"
      // クリックは背後の宇宙（惑星）へ通す。見出しだけが浮いている
      className="pointer-events-none relative mx-auto flex min-h-[100svh] max-w-5xl flex-col justify-between px-6 py-24 sm:py-28"
    >
      <div className="max-w-2xl">
        <SectionHeader
          eyebrow="The Universe of Creation"
          titleEn="One Creator, Many Worlds"
          titleJp="創造の座標軸"
        />
        <p className="mt-4 flex items-center gap-2 font-mono text-[11px] tracking-widest text-aurum-300/70">
          <span className="inline-block h-2 w-2 animate-pulse-glow rounded-full bg-aurum-300" />
          目の前に浮かぶ星をタップして、それぞれの世界へ
        </p>
      </div>

      {webgl === false && <DomainListFallback />}

      <p className="self-end font-mono text-[8px] uppercase tracking-[0.25em] text-nebula-300/45 sm:text-[9px]">
        {domainCount} Domains · 1 Core
      </p>
    </section>
  );
}
