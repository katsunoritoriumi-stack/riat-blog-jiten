"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Play, Pause, Volume2, VolumeX, ArrowUpRight, Loader2 } from "lucide-react";
import SectionHeader from "./ui/SectionHeader";
import WorkLabel from "./ui/WorkLabel";
import { YOUTUBE } from "@/lib/content";
import { claimPlayback, registerMedia, setDucked } from "@/lib/mediaBus";
import { setSoundOn } from "@/lib/soundStore";

/**
 * このサイトのテーマソング（ミュージックビデオ）を置く場所。
 *
 * アルバム（AlbumSection）は別のステーションにある。
 * 1 画面に両方を積むと画面 2.6 枚ぶんの高さになり、ZoomStage は中身を
 * 中央に置くので上下が同時に切れる（実際にそうなって直した）。
 *
 * 実装上の要点：
 * - 21MB あるので preload="none"。ポスターは MV 用に描き下ろした
 *   /theme-mv-poster.webp（自前で持つので外部ホストの都合で消えない）。
 *   再生ボタンを押して初めて動画の読み込みが始まる（勝手に通信量を使わせない）。
 * - 最初のクリックが「音を鳴らす許可」になるので、そこで音ありで頭から再生する。
 * - ZoomStage は画面外のセクションを display:none にするが、それだけでは音は止まらない。
 *   IntersectionObserver で見えなくなったら必ず一時停止する。
 * - 再生を始めたらサイト全体の soundOn を立て、他の効果音と足並みを揃える。
 * - この端末で再生できない／読み込みに失敗したときは、黙って止まらず必ず画面に出す。
 *   以前 AV1 でエンコードした動画を置いてしまい、iPhone では押しても無反応に見えていた。
 */
export default function SoundVisionSection() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const [started, setStarted] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [buffering, setBuffering] = useState(false);
  const [failed, setFailed] = useState(false);
  // 一度でも実際に映像が出たか（ポスター画像を引っ込める判断に使う）
  const [hasPlayed, setHasPlayed] = useState(false);
  // このセクションが一度でも画面に入ったか（ポスター画像を読み始める合図）
  const [seen, setSeen] = useState(false);

  /** アルバムが鳴り出したら MV は黙る。その入口を mediaBus に預けておく */
  const stopSelf = useCallback(() => {
    videoRef.current?.pause();
  }, []);

  useEffect(() => registerMedia(stopSelf), [stopSelf]);

  // 画面から外れたら止める（ステーションが切り替わっても曲が流れ続けないように）
  useEffect(() => {
    const el = boxRef.current;
    const v = videoRef.current;
    if (!el || !v) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) setSeen(true);
        else if (!v.paused) v.pause();
      },
      { threshold: 0.15 }
    );
    io.observe(el);

    /**
     * ポスター表示の保険。
     * 通常は上の IntersectionObserver で十分だが、万一それが働かない環境でも
     * 絵が出ないまま終わらないように、1画面ぶんスクロールしたら読み始める。
     * ここまで来ていれば最初の描画はとうに終わっているので、速度には響かない。
     */
    const onScroll = () => {
      if (window.scrollY > window.innerHeight) {
        setSeen(true);
        window.removeEventListener("scroll", onScroll);
      }
    };
    window.addEventListener("scroll", onScroll, { passive: true });

    return () => {
      io.disconnect();
      window.removeEventListener("scroll", onScroll);
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
   * - play() が拒否されたのが「音の許可」ではなく「再生そのものの失敗」だったときは
   *   消音での再試行も失敗するので、そこで諦めて失敗表示に切り替える。
   */
  function start() {
    const v = videoRef.current;
    if (!v) return;
    claimPlayback(stopSelf);
    setStarted(true);
    setFailed(false);
    setBuffering(true);
    setSoundOn(true);
    v.muted = false;
    setMuted(false);
    const p = v.play();
    if (p && typeof p.catch === "function") {
      p.catch(() => {
        v.muted = true;
        setMuted(true);
        const retry = v.play();
        if (retry && typeof retry.catch === "function") {
          retry.catch(() => {
            setBuffering(false);
            setFailed(true);
          });
        }
      });
    }
  }

  function togglePlay() {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      claimPlayback(stopSelf);
      void v.play();
    } else v.pause();
  }

  function toggleMute() {
    const v = videoRef.current;
    if (!v) return;
    v.muted = !v.muted;
    setMuted(v.muted);
    if (!v.muted) setSoundOn(true);
  }

  return (
    <section data-section="sound" className="relative mx-auto max-w-5xl px-6 py-16 sm:py-20">
      <div className="mb-8 max-w-2xl sm:mb-10">
        <SectionHeader eyebrow="Sound & Vision" titleEn="EXODUS" titleJp="音楽と映像" />
      </div>

      {/*
        「音楽と映像」には独立した作品が 2 つある。
        次のステーションにアルバムが控えているので、どちらにも通し番号を振って対にする。
      */}
      <WorkLabel index={1} kind="Original MV" title="星の彼方へ" sub="Beyond the Stars" />

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
          preload="none"
          playsInline
          loop
          onPlay={() => {
            setPlaying(true);
            setDucked(true); // MV が鳴っている間、BGM を絞る
          }}
          onPause={() => {
            setPlaying(false);
            setDucked(false);
          }}
          onWaiting={() => setBuffering(true)}
          onPlaying={() => {
            setBuffering(false);
            setFailed(false);
            setHasPlayed(true);
          }}
          onError={() => {
            setBuffering(false);
            setFailed(true);
          }}
        />

        {/*
          ポスターは video の poster 属性ではなく <img> で出す。
          poster 属性は preload="none" でも、画面外でも必ず読み込まれてしまい、
          トップを開いただけで 261KB を取りにいっていた。
          さらに ZoomStage は表示外のセクションを display:none にするが、
          その中の loading="lazy" は効かない（Chrome は即読み込む。実測で確認）。
          なので「このセクションが一度画面に入るまで DOM に出さない」で止める。
        */}
        {seen && !hasPlayed && (
          <img
            src="/theme-mv-poster.webp"
            alt=""
            aria-hidden="true"
            decoding="async"
            width={1600}
            height={894}
            className="absolute inset-0 h-full w-full object-cover"
          />
        )}

        {/**
         * 未再生：ポスターの上に、光が周回する再生ボタン。
         *
         * ポスターが描き下ろしのキーアートになったので、置き方を二点だけ変えている。
         * - 暗幕を薄くする（絵を潰さない）
         * - ボタンを真ん中より少し下げる。ど真ん中は人物の顔にちょうど重なるため。
         *   pt はパーセントだと「幅」に対する比になる。カードは 16:9 固定なので
         *   高さの 0.32 ＝ 幅の 0.32×9/16 ＝ 18%。これで中心が高さの 66% に来る。
         * 説明文はカードの外へ出した。絵の中の「オリジナルMV」の文字と重なるため。
         */}
        {!started && !failed && (
          <button
            onClick={start}
            aria-label="テーマソングを音ありで再生"
            className="absolute inset-0 flex items-center justify-center pt-[18%]"
          >
            <span className="pointer-events-none absolute inset-0 bg-void-950/30 transition-colors duration-500 group-hover:bg-void-950/12" />
            <span className="orbit-ring relative flex h-20 w-20 items-center justify-center rounded-full bg-void-950/40 ring-1 ring-aurum-200/70 backdrop-blur-sm transition-transform group-hover:scale-110">
              <Play size={30} className="translate-x-0.5 text-aurum-100" fill="currentColor" />
            </span>
          </button>
        )}

        {/* 読み込み中：21MB あるので回線によっては数秒かかる。無反応に見せない */}
        {started && buffering && !failed && (
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-4 bg-void-950/45">
            <Loader2 size={30} className="animate-spin text-aurum-100" />
            <span className="font-mono text-[0.65rem] uppercase tracking-cosmic text-nebula-100/80">
              Loading transmission…
            </span>
          </div>
        )}

        {/* 再生できなかったとき：黙って固まらせず、必ず逃げ道を出す */}
        {failed && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-void-950/80 px-6 text-center">
            <span className="font-mono text-[0.65rem] uppercase tracking-cosmic text-aurum-200/80">
              Signal unavailable
            </span>
            <span className="max-w-xs font-serif text-sm leading-relaxed text-nebula-200/75">
              この端末では動画を再生できませんでした。
            </span>
            <a
              href={YOUTUBE.channelUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-full border border-aurum-200/40 px-5 py-2 text-xs uppercase tracking-cosmic text-aurum-100 transition-colors hover:bg-aurum-200/10"
            >
              YouTube で見る
              <ArrowUpRight size={14} />
            </a>
          </div>
        )}

        {/* 一時停止中：中央に大きな再生ボタン。止めたあと戻れないと困るので必ず出す */}
        {started && !failed && !playing && !buffering && (
          <button
            onClick={togglePlay}
            aria-label="再生"
            className="absolute inset-0 flex items-center justify-center bg-void-950/35 transition-colors hover:bg-void-950/20"
          >
            <span className="flex h-16 w-16 items-center justify-center rounded-full bg-void-950/55 ring-1 ring-aurum-200/70 transition-transform hover:scale-110">
              <Play size={26} className="translate-x-0.5 text-aurum-100" fill="currentColor" />
            </span>
          </button>
        )}

        {/* 再生後の操作列。ホバーのないスマホでも押せるよう、常に表示しておく */}
        {started && !failed && (
          <div className="absolute inset-x-0 bottom-0 flex items-center justify-end gap-2 bg-gradient-to-t from-void-950/85 to-transparent p-3">
            <button
              onClick={togglePlay}
              aria-label={playing ? "一時停止" : "再生"}
              className="flex h-11 w-11 items-center justify-center rounded-full bg-void-950/70 text-nebula-100 ring-1 ring-nebula-400/40 backdrop-blur-sm transition-colors hover:text-aurum-200"
            >
              {playing ? <Pause size={18} /> : <Play size={18} className="translate-x-px" />}
            </button>
            <button
              onClick={toggleMute}
              aria-label={muted ? "音を出す" : "消音"}
              className="flex h-11 w-11 items-center justify-center rounded-full bg-void-950/70 text-nebula-100 ring-1 ring-nebula-400/40 backdrop-blur-sm transition-colors hover:text-aurum-200"
            >
              {muted ? <VolumeX size={18} /> : <Volume2 size={18} />}
            </button>
          </div>
        )}
      </motion.div>

      {/* 「押すと音が鳴る」ことの予告。カードの中に置くと絵の文字と喧嘩するので外に出した */}
      {!started && !failed && (
        <p className="mt-6 flex items-center justify-center gap-3 text-xs uppercase tracking-cosmic text-nebula-100/75">
          <span className="eq" aria-hidden="true">
            <span /><span /><span /><span /><span />
          </span>
          Theme — play with sound
          <span className="eq" aria-hidden="true">
            <span /><span /><span /><span /><span />
          </span>
        </p>
      )}

      {/* この外部リンクは MV（YouTube 側）に属する。アルバムは次のステーション */}
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
