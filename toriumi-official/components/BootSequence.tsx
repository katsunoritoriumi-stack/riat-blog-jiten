"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { markBootDone } from "@/lib/bootGate";
import { renderBootAmbienceUrl, renderBootTransitionUrl } from "@/lib/bootAudio";

/**
 * 交信ブート演出（初回訪問のみ）。
 * 深宇宙からの通信を受信するタイピング演出 約2.5秒
 * → CRT が落ちるような一瞬の暗転トランジション（効果音付き）→ 本編（UFO動画）へ。
 * - サーバーHTMLでは常に「表示」でレンダリング（ハイドレーション不一致なし・Heroのチラ見え防止）
 * - sessionStorage("boot-seen") がある同一タブ再訪、または reduced-motion では即スキップ
 * - クリック / キー入力で即スキップ
 * - 音はブラウザの自動再生ポリシー上、ページ内で最初のジェスチャー（クリック・キー等）が
 *   起きるまで鳴らせないことがある。鳴らせない場合は無音のまま視覚演出のみ進行する。
 */

const LINES = [
  "> INCOMING TRANSMISSION ...",
  "> SOURCE: DEEP SPACE — VECTOR 0°7'KIEJI",
  "> DECRYPTING ▓▓▓▓▓░░░░░ 52%",
  "> SIGNAL LOCKED. OPENING CHANNEL —",
];
const FULL_TEXT = LINES.join("\n");

/** 1文字あたりの表示間隔(ms)。全文で約2.2秒になる速度。 */
const CHAR_MS = 16;
const TYPE_DURATION_SEC = (FULL_TEXT.length * CHAR_MS) / 1000;
const POST_TYPE_PAUSE_SEC = 0.5;
const AMBIENCE_DURATION_SEC = TYPE_DURATION_SEC + POST_TYPE_PAUSE_SEC + 0.3;

type Phase = "typing" | "burst" | "collapse";

export default function BootSequence() {
  const [gone, setGone] = useState(false); // unmount 済み
  const [phase, setPhase] = useState<Phase>("typing");
  const [typed, setTyped] = useState(""); // タイプ済みテキスト（改行含む）
  const finishedRef = useRef(false);
  const finishRef = useRef<() => void>(() => {}); // クリックスキップ用

  const ambienceRef = useRef<HTMLAudioElement | null>(null);
  const transitionRef = useRef<HTMLAudioElement | null>(null);
  const unlockedRef = useRef(false);

  // 音の事前レンダリング＋「最初のジェスチャーで解錠」対応
  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const seen = sessionStorage.getItem("boot-seen");
    if (reduce || seen) return; // 視覚演出自体を出さないので音も不要

    let cancelled = false;
    const urls: string[] = [];

    const tryUnlock = () => {
      if (unlockedRef.current) return;
      unlockedRef.current = true;
      // 実ジェスチャー内で再生を試みる＝以後のプログラム的な再生も許可されやすくなる（iOS含む）
      ambienceRef.current?.play().catch(() => {});
      const t = transitionRef.current;
      if (t) {
        t.play()
          .then(() => {
            t.pause();
            t.currentTime = 0;
          })
          .catch(() => {});
      }
    };
    window.addEventListener("pointerdown", tryUnlock, { once: true });
    window.addEventListener("keydown", tryUnlock, { once: true });

    renderBootAmbienceUrl(AMBIENCE_DURATION_SEC, TYPE_DURATION_SEC).then((url) => {
      if (cancelled || !url) return;
      urls.push(url);
      const el = new Audio(url);
      el.preload = "auto";
      el.volume = 0.85;
      ambienceRef.current = el;
      el.play().catch(() => {}); // 自動再生できれば即鳴る。できなければ上のジェスチャーで解錠
    });
    renderBootTransitionUrl().then((url) => {
      if (cancelled || !url) return;
      urls.push(url);
      const el = new Audio(url);
      el.preload = "auto";
      el.volume = 0.9;
      transitionRef.current = el;
    });

    return () => {
      cancelled = true;
      window.removeEventListener("pointerdown", tryUnlock);
      window.removeEventListener("keydown", tryUnlock);
      urls.forEach((u) => URL.revokeObjectURL(u));
    };
  }, []);

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
        return;
      }

      // アンビエンスをふわっと止め、トランジション・スティングを鳴らす
      const amb = ambienceRef.current;
      if (amb) {
        const startVol = amb.volume;
        const fadeStart = performance.now();
        const fadeMs = 220;
        const fade = () => {
          const t = Math.min(1, (performance.now() - fadeStart) / fadeMs);
          amb.volume = startVol * (1 - t);
          if (t < 1) requestAnimationFrame(fade);
          else amb.pause();
        };
        requestAnimationFrame(fade);
      }
      transitionRef.current?.play().catch(() => {});

      // CRT が落ちるような暗転トランジション：静電ノイズのバースト → 中央へ収束して消える
      setPhase("burst");
      markBootDone(); // ここから本編（UFO動画）を裏で始動
      window.setTimeout(() => setPhase("collapse"), 130);
      window.setTimeout(() => setGone(true), 130 + 520);
    };

    finishRef.current = () => finish();

    if (seen || reduce) {
      finish(true);
      return;
    }

    // タイピング演出
    let i = 0;
    const typer = window.setInterval(() => {
      i += 1;
      setTyped(FULL_TEXT.slice(0, i));
      if (i >= FULL_TEXT.length) {
        window.clearInterval(typer);
        window.setTimeout(() => finish(), POST_TYPE_PAUSE_SEC * 1000);
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

  const closing = phase !== "typing";

  return (
    <motion.div
      onClick={() => finishRef.current()}
      animate={
        phase === "collapse"
          ? { scaleY: 0.015, opacity: 0 }
          : { scaleY: 1, opacity: 1 }
      }
      transition={{ duration: 0.52, ease: [0.76, 0, 0.24, 1] }}
      style={{ transformOrigin: "50% 50%" }}
      className="scanlines fixed inset-0 z-[80] flex cursor-pointer items-center justify-center bg-void-950"
      aria-hidden="true"
    >
      {/* 走査バー */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="scan-bar" />
      </div>

      {/* 切り替え時の静電ノイズ・バースト */}
      {closing && (
        <div
          className="static-noise pointer-events-none absolute inset-0"
          style={{ animation: "static-burst 0.42s steps(2) 1 both" }}
        />
      )}

      {/* CRT が落ちる瞬間の明るい水平ライン */}
      {phase === "collapse" && (
        <motion.div
          initial={{ opacity: 0.9 }}
          animate={{ opacity: 0 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          className="pointer-events-none absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-gradient-to-r from-transparent via-aurum-100 to-transparent"
          style={{ boxShadow: "0 0 24px 2px rgba(252,234,187,0.8)" }}
        />
      )}

      {/* 通信テキスト */}
      <motion.div
        animate={{ opacity: closing ? 0 : 1 }}
        transition={{ duration: 0.18 }}
        className="w-full max-w-xl px-8"
      >
        <pre className="whitespace-pre-wrap font-mono text-[11px] leading-loose tracking-[0.15em] text-nebula-200/90 sm:text-sm">
          {typed}
          <span className="caret-blink text-aurum-300">▌</span>
        </pre>
        <p className="mt-8 text-center font-mono text-[9px] uppercase tracking-[0.4em] text-nebula-300/40">
          tap to skip
        </p>
      </motion.div>
    </motion.div>
  );
}
