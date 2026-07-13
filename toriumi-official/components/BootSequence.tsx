"use client";

import { useEffect, useRef, useState } from "react";
import { markBootDone } from "@/lib/bootGate";

/**
 * 交信ブート演出（初回訪問のみ）。
 * 深宇宙からの通信を受信するタイピング演出 約2.5秒 → フェード → 本編（UFO動画）へ。
 * - サーバーHTMLでは常に「表示」でレンダリング（ハイドレーション不一致なし・Heroのチラ見え防止）
 * - sessionStorage("boot-seen") がある同一タブ再訪、または reduced-motion では即スキップ
 * - クリック / キー入力で即スキップ
 */

const LINES = [
  "> INCOMING TRANSMISSION ...",
  "> SOURCE: DEEP SPACE — VECTOR 0°7'KIEJI",
  "> DECRYPTING ▓▓▓▓▓░░░░░ 52%",
  "> SIGNAL LOCKED. OPENING CHANNEL —",
];

/** 1文字あたりの表示間隔(ms)。全文で約2.2秒になる速度。 */
const CHAR_MS = 16;

export default function BootSequence() {
  const [gone, setGone] = useState(false); // unmount 済み
  const [fading, setFading] = useState(false); // フェードアウト中
  const [typed, setTyped] = useState(""); // タイプ済みテキスト（改行含む）
  const finishedRef = useRef(false);
  const finishRef = useRef<() => void>(() => {}); // クリックスキップ用

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const seen = sessionStorage.getItem("boot-seen");

    const finish = (instant = false) => {
      if (finishedRef.current) return;
      finishedRef.current = true;
      try {
        sessionStorage.setItem("boot-seen", "1");
      } catch {}
      if (instant) {
        markBootDone();
        setGone(true);
      } else {
        setFading(true);
        markBootDone(); // フェードと同時に本編（UFO動画）を始動
        window.setTimeout(() => setGone(true), 450);
      }
    };

    finishRef.current = () => finish();

    if (seen || reduce) {
      finish(true);
      return;
    }

    // タイピング演出
    const full = LINES.join("\n");
    let i = 0;
    const typer = window.setInterval(() => {
      i += 1;
      setTyped(full.slice(0, i));
      if (i >= full.length) {
        window.clearInterval(typer);
        window.setTimeout(() => finish(), 500); // 読み終わりの間
      }
    }, CHAR_MS);

    // スキップ（クリック / キー）
    const skip = () => finish();
    window.addEventListener("keydown", skip);

    // 保険：何があっても6秒で必ず明ける
    const failsafe = window.setTimeout(() => finish(), 6000);

    return () => {
      window.clearInterval(typer);
      window.clearTimeout(failsafe);
      window.removeEventListener("keydown", skip);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (gone) return null;

  return (
    <div
      onClick={() => finishRef.current()}
      className={`scanlines fixed inset-0 z-[80] flex cursor-pointer items-center justify-center bg-void-950 transition-opacity duration-500 ${
        fading ? "opacity-0" : "opacity-100"
      }`}
      aria-hidden="true"
    >
      {/* 走査バー */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="scan-bar" />
      </div>

      {/* 通信テキスト */}
      <div className="w-full max-w-xl px-8">
        <pre className="whitespace-pre-wrap font-mono text-[11px] leading-loose tracking-[0.15em] text-nebula-200/90 sm:text-sm">
          {typed}
          <span className="caret-blink text-aurum-300">▌</span>
        </pre>
        <p className="mt-8 text-center font-mono text-[9px] uppercase tracking-[0.4em] text-nebula-300/40">
          tap to skip
        </p>
      </div>
    </div>
  );
}
