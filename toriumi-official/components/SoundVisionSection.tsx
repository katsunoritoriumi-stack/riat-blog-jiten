"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Play, Pause, Volume2, VolumeX, ArrowUpRight } from "lucide-react";
import SectionHeader from "./ui/SectionHeader";
import { YOUTUBE } from "@/lib/content";
import { setSoundOn } from "@/lib/soundStore";

/**
 * このサイトのテーマソング（ミュージックビデオ）を置く場所。
 *
 * 実装上の要点：
 * - 21MB あるので preload="none"。ポスターには既存の EXODUS チャンネルアートを使い、
 *   再生ボタンを押して初めて読み込みが始まる（勝手に通信量を使わせない）。
 * - 最初のクリックが「音を鳴らす許可」になるので、そこで音ありで頭から再生する。
 * - ZoomStage は画面外のセクションを display:none にするが、それだけでは音は止まらない。
 *   IntersectionObserver で見えなくなったら必ず一時停止する。
 * - 再生を始めたらサイト全体の soundOn を立て、他の効果音と足並みを揃える。
 */
export default function SoundVisionSection() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const [started, setStarted] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);

  // 画面から外れたら止める（ステーションが切り替わっても曲が流れ続けないように）
  useEffect(() => {
    const el = boxRef.current;
    const v = videoRef.current;
    if (!el || !v) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (!e.isIntersecting && !v.paused) v.pause();
      },
      { threshold: 0.15 }
    );
    io.observe(el);
    return () => {
      io.disconnect();
      v.pause();
    };
  }, []);

  /**
   * 再生開始。スマホで音が出なかったので次の点を守っている：
   * - preload="none" の状態で currentTime を触ると読み込み前の要素を操作することになり、
   *   iOS では再生が始まらないことがある。頭出しはしない（そもそも初回は先頭から）。
   * - play() はクリックハンドラの中で「同期的に」呼ぶ。await などを挟むと
   *   ユーザー操作の許可が切れて弾かれる。
   * - それでも音ありが拒否された端末では、消音で再生だけ通し、
   *   音声ボタンで鳴らせる状態にしておく（無音のまま黙って失敗させない）。
   */
  function start() {
    const v = videoRef.current;
    if (!v) return;
    setStarted(true);
    setSoundOn(true);
    v.muted = false;
    setMuted(false);
    const p = v.play();
    if (p && typeof p.catch === "function") {
      p.catch(() => {
        v.muted = true;
        setMuted(true);
        void v.play();
      });
    }
  }

  function togglePlay() {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) void v.play();
    else v.pause();
  }

  function toggleMute() {
    const v = videoRef.current;
    if (!v) return;
    v.muted = !v.muted;
    setMuted(v.muted);
    if (!v.muted) setSoundOn(true);
  }

  return (
    <section data-section="sound" className="relative mx-auto max-w-5xl px-6 py-28 sm:py-36">
      <div className="mb-14 max-w-2xl">
        <SectionHeader eyebrow="Sound & Vision" titleEn="EXODUS" titleJp="音楽と映像" />
      </div>

      <motion.div
        ref={boxRef}
        initial={{ opacity: 0, scale: 0.96 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.7 }}
        className="group relative aspect-video overflow-hidden rounded-3xl border border-nebula-500/20 bg-void-950"
      >
        <video
          ref={videoRef}
          className="absolute inset-0 h-full w-full object-cover"
          src="/theme-mv.mp4"
          poster={YOUTUBE.thumbnail}
          preload="none"
          playsInline
          loop
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
        />

        {/* 未再生：ポスターの上に、光が周回する再生ボタン */}
        {!started && (
          <button
            onClick={start}
            aria-label="テーマソングを再生"
            className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-center"
          >
            <span className="pointer-events-none absolute inset-0 bg-void-950/45 transition-colors duration-500 group-hover:bg-void-950/25" />
            <span className="orbit-ring relative flex h-20 w-20 items-center justify-center rounded-full bg-void-950/40 ring-1 ring-aurum-200/70 backdrop-blur-sm transition-transform group-hover:scale-110">
              <Play size={30} className="translate-x-0.5 text-aurum-100" fill="currentColor" />
            </span>
            <span className="relative flex items-center gap-3 text-xs uppercase tracking-cosmic text-nebula-100/85 drop-shadow">
              <span className="eq" aria-hidden="true">
                <span /><span /><span /><span /><span />
              </span>
              Theme — play with sound
              <span className="eq" aria-hidden="true">
                <span /><span /><span /><span /><span />
              </span>
            </span>
          </button>
        )}

        {/* 再生後：邪魔をしない最小限の操作 */}
        {started && (
          <div className="absolute inset-x-0 bottom-0 flex items-center justify-end gap-2 bg-gradient-to-t from-void-950/85 to-transparent p-3 opacity-0 transition-opacity duration-300 focus-within:opacity-100 group-hover:opacity-100">
            <button
              onClick={togglePlay}
              aria-label={playing ? "一時停止" : "再生"}
              className="flex h-9 w-9 items-center justify-center rounded-full bg-void-950/60 text-nebula-100 ring-1 ring-nebula-400/30 backdrop-blur-sm transition-colors hover:text-aurum-200"
            >
              {playing ? <Pause size={16} /> : <Play size={16} className="translate-x-px" />}
            </button>
            <button
              onClick={toggleMute}
              aria-label={muted ? "音を出す" : "消音"}
              className="flex h-9 w-9 items-center justify-center rounded-full bg-void-950/60 text-nebula-100 ring-1 ring-nebula-400/30 backdrop-blur-sm transition-colors hover:text-aurum-200"
            >
              {muted ? <VolumeX size={16} /> : <Volume2 size={16} />}
            </button>
          </div>
        )}
      </motion.div>

      <div className="mt-8 flex justify-center">
        <a
          href={YOUTUBE.channelUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="group inline-flex items-center gap-2 text-sm text-aurum-200"
        >
          チャンネルを見る
          <ArrowUpRight size={16} className="transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
        </a>
      </div>
    </section>
  );
}
