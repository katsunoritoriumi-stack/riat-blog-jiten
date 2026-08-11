"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Pause, Play, SkipBack, SkipForward, Volume2, VolumeX, Loader2 } from "lucide-react";
import { ALBUM, albumDuration, trackArt, trackCover, trackSrc, type Track } from "@/lib/content";
import { claimPlayback, registerMedia, setDucked } from "@/lib/mediaBus";
import { setSoundOn } from "@/lib/soundStore";

/**
 * アルバム再生機。
 *
 * 実装上の要点：
 * - audio 要素は 1 個だけ持ち、src を差し替えて使い回す。
 *   曲ごとに要素を作ると、iOS で「最初のタップで解錠された要素」以外は
 *   音を出せず、2 曲目から無音になる。
 * - src の差し替えと play() は必ずクリックハンドラの中で同期的に行う。
 *   間に await や setState を挟むとユーザー操作の許可が切れて弾かれる。
 * - preload="none"。1 曲 4〜7MB あるので、押されるまで 1 バイトも取りにいかない。
 * - 画面外に出たら止める。ZoomStage は表示外のセクションを display:none にするが、
 *   それだけでは音は止まらない。
 * - MV と同時に鳴らない。mediaBus で「鳴らす側が名乗り出たら他は止まる」ようにしてある。
 */

const fmt = (sec: number) => {
  if (!Number.isFinite(sec) || sec < 0) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
};

export default function AlbumPlayer() {
  const audioRef = useRef<HTMLAudioElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);

  const [index, setIndex] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [buffering, setBuffering] = useState(false);
  const [failed, setFailed] = useState(false);
  const [muted, setMuted] = useState(false);
  const [time, setTime] = useState(0);
  /** 実際に読み込めた長さ。読み込む前は data の実測値で代用する */
  const [liveDuration, setLiveDuration] = useState(0);
  /** 一度でも画面に入ったか（アートワークを読み始める合図） */
  const [seen, setSeen] = useState(false);

  const current = index === null ? null : ALBUM.tracks[index];
  const total = liveDuration || current?.duration || 0;

  /** 他の音源（MV）から止められたときの入口。mediaBus に登録する実体 */
  const stopSelf = useCallback(() => {
    audioRef.current?.pause();
  }, []);

  useEffect(() => registerMedia(stopSelf), [stopSelf]);

  // 画面から外れたら止める（ステーションが切り替わっても鳴り続けないように）
  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) setSeen(true);
        else audioRef.current?.pause();
      },
      { threshold: 0.1 }
    );
    io.observe(el);

    /**
     * アートワーク表示の保険。
     * display:none の中では loading="lazy" が効かず、かといって
     * IntersectionObserver が働かない環境で絵が出ないままなのも困る。
     * 1 画面ぶんスクロールしたら読み始める（そこまで来れば初回描画は終わっている）。
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
      audioRef.current?.pause();
    };
  }, []);

  /**
   * 指定の曲を鳴らす。
   * 同じ曲をもう一度押したときは頭出しせず、一時停止／再開のトグルにする
   * （曲の途中で誤って押しても最初に戻らない）。
   */
  function playAt(i: number) {
    const a = audioRef.current;
    const t = ALBUM.tracks[i];
    if (!a || !t) return;

    if (i === index) {
      if (a.paused) void a.play();
      else a.pause();
      return;
    }

    claimPlayback(stopSelf);
    setSoundOn(true);
    setIndex(i);
    setFailed(false);
    setBuffering(true);
    setTime(0);
    setLiveDuration(0);
    a.src = trackSrc(t);
    a.muted = muted;
    const p = a.play();
    if (p && typeof p.catch === "function") {
      p.catch(() => {
        // 音の許可が下りない端末では、消音で再生だけ通してボタンで鳴らせる状態にする
        a.muted = true;
        setMuted(true);
        const retry = a.play();
        if (retry && typeof retry.catch === "function") {
          retry.catch(() => {
            setBuffering(false);
            setFailed(true);
          });
        }
      });
    }
  }

  /** 再生／一時停止。まだ何も選んでいなければ 1 曲目から */
  function toggle() {
    const a = audioRef.current;
    if (!a) return;
    if (index === null) return playAt(0);
    claimPlayback(stopSelf);
    if (a.paused) void a.play();
    else a.pause();
  }

  /** 前後の曲へ。端では止まる（ループはしない） */
  function step(delta: number) {
    const base = index ?? 0;
    const next = base + delta;
    if (next < 0 || next >= ALBUM.tracks.length) return;
    playAt(next);
  }

  function toggleMute() {
    const a = audioRef.current;
    if (!a) return;
    a.muted = !a.muted;
    setMuted(a.muted);
    if (!a.muted) setSoundOn(true);
  }

  function seek(v: number) {
    const a = audioRef.current;
    if (!a || !Number.isFinite(a.duration)) return;
    a.currentTime = v;
    setTime(v);
  }

  const progress = total > 0 ? (time / total) * 100 : 0;

  return (
    <div
      ref={boxRef}
      className="rounded-3xl border border-nebula-500/20 bg-void-900/40 p-5 backdrop-blur-sm sm:p-7"
    >
      {/* 音源はひとつ。曲を変えるときは src を差し替える */}
      <audio
        ref={audioRef}
        preload="none"
        onPlay={() => {
          setPlaying(true);
          setDucked(true); // 曲が鳴っている間、BGM を絞る
        }}
        onPause={() => {
          setPlaying(false);
          setDucked(false);
        }}
        onWaiting={() => setBuffering(true)}
        onPlaying={() => {
          setBuffering(false);
          setFailed(false);
        }}
        onLoadedMetadata={(e) => setLiveDuration(e.currentTarget.duration)}
        onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
        onEnded={() => {
          // 最後の曲まで来たら止まる。勝手に頭へ戻って延々鳴り続けないように
          if (index !== null && index + 1 < ALBUM.tracks.length) playAt(index + 1);
          else setPlaying(false);
        }}
        onError={() => {
          setBuffering(false);
          setFailed(true);
        }}
      />

      <div className="grid gap-6 sm:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)] sm:gap-8">
        {/* ── ジャケット ── */}
        {/* スマホでは 1 画面に収めるためジャケットを小さめに置く */}
        <div className="relative mx-auto w-full max-w-[210px] sm:max-w-none">
          <div className="relative aspect-square overflow-hidden rounded-2xl border border-nebula-500/25 bg-void-950 shadow-[0_18px_50px_-18px_rgba(3,2,10,0.95)]">
            {/*
              アルバムの絵と、選ばれた曲の絵を重ねて置き、上を出し入れして入れ替える。
              img を 1 枚にして src を差し替えると、読み込みの一瞬だけ絵が消える。
            */}
            {seen && (
              <img
                src={ALBUM.cover}
                alt={`${ALBUM.titleEn} — アルバムジャケット`}
                decoding="async"
                width={900}
                height={900}
                className="absolute inset-0 h-full w-full object-cover"
              />
            )}
            {current && (
              <img
                key={current.id}
                src={trackCover(current)}
                alt={`${current.title} — ジャケット`}
                decoding="async"
                width={760}
                height={760}
                onLoad={(e) => e.currentTarget.classList.remove("opacity-0")}
                className="absolute inset-0 h-full w-full object-cover opacity-0 transition-opacity duration-700"
              />
            )}

            {/* 文字の座を作る暗幕。下を強めに落として、絵の情報量は残す */}
            <span
              className="pointer-events-none absolute inset-0"
              style={{
                background:
                  "linear-gradient(180deg, rgba(3,2,10,0.45) 0%, rgba(3,2,10,0.08) 34%, rgba(3,2,10,0.62) 74%, rgba(3,2,10,0.94) 100%)",
              }}
            />

            <button
              onClick={toggle}
              aria-label={playing ? "一時停止" : "アルバムを再生"}
              className="group absolute inset-0 flex items-center justify-center"
            >
              <span className="orbit-ring relative flex h-14 w-14 items-center justify-center rounded-full bg-void-950/50 ring-1 ring-aurum-200/70 backdrop-blur-sm transition-transform group-hover:scale-110 sm:h-16 sm:w-16">
                {buffering ? (
                  <Loader2 size={22} className="animate-spin text-aurum-100 sm:h-6 sm:w-6" />
                ) : playing ? (
                  <Pause size={22} className="text-aurum-100 sm:h-6 sm:w-6" fill="currentColor" />
                ) : (
                  <Play size={22} className="translate-x-0.5 text-aurum-100 sm:h-6 sm:w-6" fill="currentColor" />
                )}
              </span>
            </button>

            {/* ── ジャケットの題字 ── CD の表面のように、絵の中に組み込む */}
            <div className="pointer-events-none absolute inset-x-0 bottom-0 p-3 sm:p-4">
              <p
                className="font-display text-[clamp(0.95rem,3.4vw,1.5rem)] font-extrabold leading-[1.08] tracking-tight text-nebula-50"
                style={{ textShadow: "0 2px 12px rgba(3,2,10,0.95), 0 0 30px rgba(3,2,10,0.8)" }}
              >
                {ALBUM.titleEn}
              </p>
              <p
                className="mt-0.5 font-serif text-[10px] text-nebula-200/85 sm:text-xs"
                style={{ textShadow: "0 1px 8px rgba(3,2,10,0.95)" }}
              >
                {ALBUM.titleJp}
              </p>
              {/* いま鳴っている曲。ジャケットの絵が変わる理由を言葉でも示す */}
              <p
                className={`mt-1.5 truncate font-mono text-[9px] uppercase tracking-cosmic text-aurum-200/90 transition-opacity duration-500 sm:text-[10px] ${
                  current ? "opacity-100" : "opacity-0"
                }`}
                style={{ textShadow: "0 1px 8px rgba(3,2,10,0.95)" }}
              >
                {current ? `${String((index ?? 0) + 1).padStart(2, "0")} ${current.title}` : " "}
              </p>
            </div>
          </div>

          <p className="mt-4 text-center font-mono text-[10px] uppercase tracking-cosmic text-nebula-300/60 sm:text-left">
            {ALBUM.tracks.length} tracks · {fmt(albumDuration())}
          </p>
        </div>

        {/* ── 曲目 ── */}
        <div className="min-w-0">
          {/*
            曲目は 11 曲あるので、全部展開すると 1 画面に収まらない。ここだけ中でスクロールさせる。

            overscroll-y-auto が要る。contain にすると端まで来ても外へ連鎖せず、
            カードの上でページが進まなくなる（スマホで画面が固定される）。
            ホイール側は lenis の allowNestedScroll が同じ役割を果たす。
          */}
          <ol className="-mx-2 max-h-[15rem] overflow-y-auto overscroll-y-auto pr-1 sm:max-h-[19rem] lg:max-h-[23rem]">
            {ALBUM.tracks.map((t, i) => (
              <TrackRow
                key={t.id}
                track={t}
                n={i + 1}
                seen={seen}
                current={i === index}
                playing={i === index && playing}
                onClick={() => playAt(i)}
              />
            ))}
          </ol>
        </div>
      </div>

      {/* ── 操作バー ── */}
      <div className="mt-6 border-t border-nebula-500/15 pt-5">
        {failed && (
          <p className="mb-3 text-center font-mono text-[11px] uppercase tracking-cosmic text-aurum-200/80">
            Signal unavailable — この端末では再生できませんでした
          </p>
        )}

        <div className="flex items-center gap-3">
          <span className="w-10 shrink-0 text-right font-mono text-[11px] tabular-nums text-nebula-300/70">
            {fmt(time)}
          </span>
          <input
            type="range"
            className="seekbar min-w-0 flex-1"
            style={{ ["--p" as string]: `${progress}%` }}
            min={0}
            max={total || 1}
            step={0.1}
            value={Math.min(time, total || 1)}
            disabled={index === null}
            aria-label="再生位置"
            onChange={(e) => seek(Number(e.target.value))}
          />
          <span className="w-10 shrink-0 font-mono text-[11px] tabular-nums text-nebula-300/70">
            {fmt(total)}
          </span>
        </div>

        <div className="mt-4 grid grid-cols-[1fr_auto_1fr] items-center">
          <span />
          <div className="flex items-center gap-2">
            <TransportButton
              label="前の曲"
              onClick={() => step(-1)}
              disabled={index !== null && index === 0}
            >
              <SkipBack size={18} fill="currentColor" />
            </TransportButton>
            <button
              onClick={toggle}
              aria-label={playing ? "一時停止" : "再生"}
              className="flex h-14 w-14 items-center justify-center rounded-full bg-aurum-300/12 text-aurum-100 ring-1 ring-aurum-200/50 transition-colors hover:bg-aurum-300/22"
            >
              {buffering ? (
                <Loader2 size={22} className="animate-spin" />
              ) : playing ? (
                <Pause size={22} fill="currentColor" />
              ) : (
                <Play size={22} className="translate-x-0.5" fill="currentColor" />
              )}
            </button>
            <TransportButton
              label="次の曲"
              onClick={() => step(1)}
              disabled={index !== null && index === ALBUM.tracks.length - 1}
            >
              <SkipForward size={18} fill="currentColor" />
            </TransportButton>
          </div>
          <div className="flex justify-end">
            <TransportButton label={muted ? "音を出す" : "消音"} onClick={toggleMute}>
              {muted ? <VolumeX size={18} /> : <Volume2 size={18} />}
            </TransportButton>
          </div>
        </div>
      </div>
    </div>
  );
}

function TransportButton({
  label,
  onClick,
  disabled,
  children,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      className="flex h-11 w-11 items-center justify-center rounded-full text-nebula-200 transition-colors hover:bg-nebula-500/15 hover:text-aurum-200 disabled:pointer-events-none disabled:opacity-30"
    >
      {children}
    </button>
  );
}

function TrackRow({
  track,
  n,
  seen,
  current,
  playing,
  onClick,
}: {
  track: Track;
  n: number;
  seen: boolean;
  current: boolean;
  playing: boolean;
  onClick: () => void;
}) {
  return (
    <li>
      <motion.button
        initial={{ opacity: 0, x: -8 }}
        whileInView={{ opacity: 1, x: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.45, delay: Math.min(n, 6) * 0.04 }}
        onClick={onClick}
        aria-current={current ? "true" : undefined}
        className={`flex w-full items-center gap-3 rounded-xl px-2 py-2 text-left transition-colors ${
          current ? "bg-nebula-500/14 ring-1 ring-aurum-300/25" : "hover:bg-nebula-500/8"
        }`}
      >
        <span className="w-5 shrink-0 text-center font-mono text-[11px] tabular-nums text-nebula-300/55">
          {playing ? (
            <span className="eq !h-3" aria-hidden="true">
              <span />
              <span />
              <span />
            </span>
          ) : (
            String(n).padStart(2, "0")
          )}
        </span>

        <span className="relative h-11 w-11 shrink-0 overflow-hidden rounded-lg bg-void-950 ring-1 ring-nebula-500/20">
          {seen && (
            <img
              src={trackArt(track)}
              alt=""
              aria-hidden="true"
              decoding="async"
              width={160}
              height={160}
              className="h-full w-full object-cover"
            />
          )}
        </span>

        <span className="min-w-0 flex-1">
          <span
            className={`block truncate text-[13px] font-medium tracking-wide ${
              current ? "text-aurum-100" : "text-nebula-100"
            }`}
          >
            {track.title}
          </span>
          {track.sub && (
            <span className="block truncate font-serif text-[11px] text-nebula-300/60">
              {track.sub}
            </span>
          )}
        </span>

        <span className="shrink-0 font-mono text-[11px] tabular-nums text-nebula-300/55">
          {fmt(track.duration)}
        </span>
      </motion.button>
    </li>
  );
}
