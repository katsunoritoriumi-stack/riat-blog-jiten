"use client";

import {
  motion,
  useScroll,
  useTransform,
  useMotionValue,
  useSpring,
} from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { ChevronDown, Volume2, VolumeX } from "lucide-react";
import { SITE } from "@/lib/content";
import {
  playMajesticIntro,
  renderHeroIntroUrl,
  type HeroAudioHandle,
} from "@/lib/heroAudio";
import { useSoundOn, setSoundOn } from "@/lib/soundStore";
import { useBootDone } from "@/lib/bootGate";
import { markHeroRevealed } from "@/lib/heroReveal";

/** UFO が画面左へ抜けきる頃合い（UFO動画尺 8s のうち）。ここでテキストが宿る。 */
const REVEAL_AT = 4.8;

export default function Hero() {
  const ref = useRef<HTMLElement>(null);
  const ufoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HeroAudioHandle | null>(null); // ライブ Web Audio（フォールバック）
  const audioElRef = useRef<HTMLAudioElement | null>(null); // 事前レンダリングした <audio>（iOS消音対応）
  const [revealed, setRevealed] = useState(false);
  const [ufoGone, setUfoGone] = useState(false); // UFO動画が退場し、蠢く宇宙が前面へ
  const soundOn = useSoundOn(); // グローバル共有（SFX等が参照）。書き込みは setSoundOn（store）
  const bootDone = useBootDone(); // ブート演出明けに UFO 動画を始動

  /**
   * 名前とロゴが立ち現れたことを外へ知らせる。
   * BGM はこの合図とスクロール開始の両方が揃ってから鳴り始める。
   * revealed を立てる場所が複数あるので、状態を見て一度だけ通知する。
   */
  useEffect(() => {
    if (revealed) markHeroRevealed();
  }, [revealed]);

  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end start"],
  });
  const yTextRaw = useTransform(scrollYProgress, [0, 1], ["0%", "60%"]);
  const opacityRaw = useTransform(scrollYProgress, [0, 0.7], [1, 0]);
  const scaleRaw = useTransform(scrollYProgress, [0, 1], [1, 0.94]);

  /**
   * ZoomStage のステーション内では Hero が画面に固定されるため、
   * このスクロール連動パララックスは意味を持たないうえ、
   * 表示が切り替わる瞬間に進捗が 1 のまま固まって
   * ロゴが 60% 下へずれたままになる不具合が出る（ロゴをクリックして
   * トップへ戻ったときに再現）。固定表示のときは素直に無効化する。
   */
  const [pinned, setPinned] = useState(false);
  useEffect(() => {
    setPinned(!!ref.current?.closest("[data-station]"));
  }, []);
  const yText = pinned ? 0 : yTextRaw;
  const opacity = pinned ? 1 : opacityRaw;
  const scale = pinned ? 1 : scaleRaw;

  // マウス追従の微細な 3D パララックス
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const rotateX = useSpring(useTransform(my, [-0.5, 0.5], [2.5, -2.5]), {
    stiffness: 60,
    damping: 20,
  });
  const rotateY = useSpring(useTransform(mx, [-0.5, 0.5], [-2.5, 2.5]), {
    stiffness: 60,
    damping: 20,
  });

  function onMouseMove(e: React.MouseEvent) {
    mx.set(e.clientX / window.innerWidth - 0.5);
    my.set(e.clientY / window.innerHeight - 0.5);
  }

  // reduced-motion では即・完成形を表示。
  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      setRevealed(true);
      setUfoGone(true);
    }
  }, []);

  // ブート演出が明けたら UFO 動画を頭から再生（muted なので gesture 不要）。
  // 再生できない環境でも 7 秒で必ずテキストを表示するフォールバック付き。
  useEffect(() => {
    if (!bootDone) return;
    const v = ufoRef.current;
    if (v && v.paused && !v.ended) {
      try {
        // preload="none" なのでここで初めて読み込みを始める（load() は頭出しも兼ねる）
        v.load();
        void v.play();
      } catch {}
    }
    const fallback = window.setTimeout(() => setRevealed(true), 7000);
    return () => window.clearTimeout(fallback);
  }, [bootDone]);

  // マウント時にイントロ音を WAV へ事前レンダリングし、<audio> を用意しておく。
  // （iOS のサイレントスイッチは Web Audio を無音化するが、メディア要素は鳴るため）
  useEffect(() => {
    let url: string | null = null;
    let cancelled = false;
    renderHeroIntroUrl().then((u) => {
      if (cancelled || !u) return;
      url = u;
      const el = new Audio(u);
      el.preload = "auto";
      el.setAttribute("playsinline", "");
      audioElRef.current = el;
    });
    return () => {
      cancelled = true;
      audioRef.current?.stop();
      audioElRef.current?.pause();
      if (url) URL.revokeObjectURL(url);
    };
  }, []);

  const onTimeUpdate = () => {
    const v = ufoRef.current;
    if (v && v.currentTime >= REVEAL_AT) setRevealed(true);
  };

  // UFO動画が終わったら、少し余韻を置いて前面から退場（背後の蠢く宇宙が現れる）
  const onEnded = () => {
    setRevealed(true);
    setUfoGone(true);
  };

  // イントロを頭から、荘厳なサウンドと共に再体験
  function replayWithSound() {
    audioRef.current?.stop();
    audioRef.current = null;
    audioElRef.current?.pause();
    setRevealed(false);
    setUfoGone(false);
    const v = ufoRef.current;
    if (v) {
      try {
        v.currentTime = 0;
        v.play();
      } catch {}
    }
    // まず <audio>（iOS消音でも鳴る）。無ければライブ Web Audio。
    const el = audioElRef.current;
    if (el) {
      try {
        el.currentTime = 0;
        void el.play();
      } catch {
        audioRef.current = playMajesticIntro();
      }
    } else {
      audioRef.current = playMajesticIntro();
    }
    setSoundOn(true);
  }

  function toggleSound() {
    if (soundOn) {
      audioRef.current?.stop();
      audioRef.current = null;
      audioElRef.current?.pause();
      setSoundOn(false);
    } else {
      replayWithSound();
    }
  }

  return (
    <section
      data-section="home"
      ref={ref}
      onMouseMove={onMouseMove}
      className="relative flex h-[100svh] min-h-[640px] items-center justify-center overflow-hidden"
    >
      {/* ── 背景レイヤー①：蠢く宇宙（常時ループ）。テキスト表示後も動き続ける ── */}
      <motion.video
        style={{ scale }}
        className="pointer-events-none absolute inset-0 h-full w-full object-cover"
        src="/cosmos-loop.mp4"
        poster="/cosmos-poster.webp"
        autoPlay
        muted
        loop
        playsInline
        preload="auto"
        aria-hidden="true"
      />

      {/*
        ── 背景レイヤー②：UFO登場（一度だけ再生）。終わると静かにフェードして①へ受け渡す ──

        この動画はブート演出が明けるまで再生しないので、最初から取りに行かない
        （preload="none"）。ブートが明けた時点で load() → play() する。
        その間ポスター画像（1コマ目）が出ているので、見え方はこれまでと変わらない。
      */}
      <motion.video
        ref={ufoRef}
        style={{ scale }}
        initial={{ opacity: 1 }}
        animate={{ opacity: ufoGone ? 0 : 1 }}
        transition={{ duration: 1.8, ease: "easeInOut" }}
        className="pointer-events-none absolute inset-0 h-full w-full object-cover"
        src="/hero-space.mp4"
        poster="/hero-poster.webp"
        muted
        playsInline
        preload="none"
        onTimeUpdate={onTimeUpdate}
        onEnded={onEnded}
        aria-hidden="true"
      />

      {/* base darken */}
      <div className="pointer-events-none absolute inset-0 bg-void-950/35" />
      {/* テキストが宿るとき、映像を少し沈めて文字を持ち上げる追い暗転 */}
      <motion.div
        className="pointer-events-none absolute inset-0 bg-void-950"
        initial={{ opacity: 0 }}
        animate={{ opacity: revealed ? 0.4 : 0 }}
        transition={{ duration: 1.6, ease: "easeInOut" }}
      />
      {/* vignette + 下端フェード（次セクションへ繋ぐ） */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_30%,rgba(3,2,10,0.85)_92%)]" />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-40 bg-gradient-to-b from-transparent to-void-950" />

      {/* soft glow behind the name */}
      <motion.div
        className="pointer-events-none absolute left-1/2 top-1/2 h-[70vmin] w-[70vmin] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[radial-gradient(circle,rgba(124,58,237,0.22),transparent_65%)]"
        initial={{ opacity: 0 }}
        animate={{ opacity: revealed ? 1 : 0 }}
        transition={{ duration: 1.8 }}
      />

      {/* ── テキスト：UFO の退場に同期して、もとからそこに在ったように宿る ── */}
      <motion.div
        style={{ y: yText, opacity, rotateX, rotateY, transformPerspective: 1000 }}
        className="relative z-10 flex flex-col items-center px-6 text-center"
      >
        <motion.span
          initial={{ opacity: 0, y: 8, filter: "blur(6px)", letterSpacing: "0.2em" }}
          animate={
            revealed
              ? { opacity: 1, y: 0, filter: "blur(0px)", letterSpacing: "0.42em" }
              : {}
          }
          transition={{ duration: 1.6, ease: [0.22, 1, 0.36, 1] }}
          className="mb-7 font-mono text-[10px] uppercase text-aurum-300/85 sm:text-xs"
        >
          ✦ Tricky Multi-Creator ✦
        </motion.span>

        <motion.h1
          initial={{ opacity: 0, filter: "blur(14px)", scale: 1.04 }}
          animate={revealed ? { opacity: 1, filter: "blur(0px)", scale: 1 } : {}}
          transition={{ duration: 1.9, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
          className="relative flex justify-center"
        >
          {/* 見出しの実体はロゴ画像。読み上げ・検索向けに文字も残す */}
          <span className="sr-only">Katsunori Toriumi — K.TORIUMI</span>
          {/* ロゴの背後にやわらかい星雲の光 */}
          <span
            aria-hidden="true"
            className="pointer-events-none absolute left-1/2 top-1/2 -z-10 -translate-x-1/2 -translate-y-1/2 rounded-full"
            style={{
              width: "135%",
              height: "135%",
              background: "radial-gradient(circle, rgba(124,58,237,0.3), transparent 68%)",
            }}
          />
          <img
            src="/logo-ktoriumi.webp"
            alt=""
            aria-hidden="true"
            width={846}
            height={1024}
            className="logo-drift block h-[25svh] max-h-[260px] w-auto sm:h-[34svh] sm:max-h-[400px]"
          />
        </motion.h1>

        {/* a.k.a KIEJI */}
        <motion.p
          initial={{ opacity: 0, y: 10, filter: "blur(6px)" }}
          animate={revealed ? { opacity: 1, y: 0, filter: "blur(0px)" } : {}}
          transition={{ duration: 1.4, delay: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="mt-5 flex items-center gap-3 font-mono text-xs uppercase tracking-[0.4em] sm:text-sm"
        >
          <span className="h-px w-8 bg-gradient-to-r from-transparent to-aurum-300/60" />
          <span className="text-nebula-300/70">a.k.a</span>
          <span className="gradient-aurum font-bold tracking-[0.5em]">KIEJI</span>
          <span className="h-px w-8 bg-gradient-to-l from-transparent to-aurum-300/60" />
        </motion.p>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: revealed ? 1 : 0 }}
          transition={{ duration: 1.2, delay: 0.8 }}
          className="mt-7 max-w-xl font-serif text-base text-nebula-200/85 sm:text-lg"
        >
          {SITE.nameJp} — {SITE.taglineJp}
        </motion.p>
      </motion.div>

      {/* scroll cue */}
      <motion.a
        href="#universe"
        style={{ opacity }}
        className="absolute bottom-8 left-1/2 z-10 -translate-x-1/2 text-aurum-300/70"
        initial={{ opacity: 0 }}
        animate={
          revealed ? { opacity: [0, 0.7, 0.7], y: [0, 10, 0] } : { opacity: 0 }
        }
        transition={{
          opacity: { duration: 1, delay: 1 },
          y: { duration: 2, repeat: Infinity, delay: 1 },
        }}
        aria-label="次のセクションへ"
      >
        <ChevronDown size={28} />
      </motion.a>

      {/* sound toggle — 荘厳なサウンドと共にイントロを再体験 */}
      <button
        onClick={toggleSound}
        className="group absolute bottom-7 right-6 z-20 flex items-center gap-2 rounded-full border border-nebula-400/25 bg-void-900/40 px-4 py-2 backdrop-blur-sm transition-colors hover:border-aurum-300/50 sm:right-8"
        aria-label={soundOn ? "サウンドを止める" : "荘厳なサウンドと共に再生"}
      >
        {!soundOn && (
          <span className="absolute inset-0 -z-10 animate-ping rounded-full bg-aurum-300/15" />
        )}
        {soundOn ? (
          <Volume2 size={15} className="text-aurum-200" />
        ) : (
          <VolumeX size={15} className="text-nebula-200/80 group-hover:text-aurum-200" />
        )}
        <span className="font-mono text-[9px] uppercase tracking-[0.25em] text-nebula-200/80 group-hover:text-aurum-200">
          {soundOn ? "Sound On" : "Sound"}
        </span>
      </button>
    </section>
  );
}
