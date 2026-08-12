"use client";

import { useEffect, useRef } from "react";
import { subscribeDuck } from "@/lib/mediaBus";
import { subscribeSound, getSoundOn } from "@/lib/soundStore";

/**
 * サイト全体の BGM。
 *
 * 鳴り始めの決まり：
 * 1. ページ上部のサウンドボタンが押される（= soundOn）と「点火待ち」になる。
 *    ここではまだ鳴らさない。イントロの効果音と重なって濁るため。
 * 2. 最初にスクロールが動いた瞬間から、ゆっくり音量を上げて入ってくる。
 *    旅が始まるのと同時に音楽が立ち上がる。
 * 3. 音量は小さめ（MAX_VOL）。主役は各セクションの作品なので、その下に敷く。
 *
 * 外部リンクへ出て戻ってきたとき：
 * - 別タブで開かれた場合、このページは生きたまま隠れるだけなので、
 *   隠れている間は止め、戻ってきたら鳴らし直す（裏で鳴り続けると鬱陶しい）。
 * - 同じタブで移動して戻った場合はページが読み直される。
 *   sessionStorage に点火済みの印を残しておき、次の読み込みで自動再開する。
 *   ただし操作なしの自動再生はブラウザに止められることがあるので、
 *   弾かれたら最初のスクロールやクリックで鳴らす保険を張る。
 *
 * MV やアルバムが鳴っている間は、ほとんど聞こえないところまで絞る（mediaBus の duck）。
 */

const SRC = "/bgm-neon-horizon.mp3";
/** 主張しすぎない。0.18 では大きいという判断で 3/5 に落とした */
const MAX_VOL = 0.108;
const DUCK_VOL = 0.012; // 作品が鳴っている間
const FADE_MS = 2200;
const ARM_KEY = "toriumi.bgm.armed";

export default function BackgroundMusic() {
  const ref = useRef<HTMLAudioElement>(null);
  /** 目標音量へ寄せていくためのタイマー */
  const fadeRef = useRef<number | null>(null);
  const duckedRef = useRef(false);
  /** サウンドボタンが押されたか（＝鳴らしてよいか） */
  const armedRef = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    el.volume = 0;

    const target = () => (duckedRef.current ? DUCK_VOL : MAX_VOL);

    /** 音量をなめらかに寄せる。ぶつ切りで鳴らさない */
    function fadeTo(to: number, ms: number) {
      if (fadeRef.current) window.clearInterval(fadeRef.current);
      const a = ref.current;
      if (!a) return;
      const from = a.volume;
      const t0 = performance.now();
      fadeRef.current = window.setInterval(() => {
        const cur = ref.current;
        if (!cur) return;
        const k = Math.min(1, (performance.now() - t0) / ms);
        cur.volume = Math.max(0, Math.min(1, from + (to - from) * k));
        if (k >= 1) {
          if (fadeRef.current) window.clearInterval(fadeRef.current);
          fadeRef.current = null;
          if (to === 0) cur.pause();
        }
      }, 50);
    }

    function start() {
      const a = ref.current;
      if (!a || !armedRef.current || document.hidden) return;
      // arm() の時点で音量 0 のまま鳴らし始めている場合がある。
      // そのときは play() を呼び直さず、音量を上げるだけでよい
      if (a.paused) {
        const p = a.play();
        if (p && typeof p.catch === "function") {
          p.catch(() => {
            // 自動再生を拒否された。次のユーザー操作で鳴らす
            armRetryOnGesture();
          });
        }
      }
      fadeTo(target(), FADE_MS);
    }

    /** 自動再生が弾かれたときの保険。一度きり */
    function armRetryOnGesture() {
      const retry = () => {
        window.removeEventListener("pointerdown", retry);
        window.removeEventListener("keydown", retry);
        start();
      };
      window.addEventListener("pointerdown", retry, { once: true, passive: true });
      window.addEventListener("keydown", retry, { once: true });
    }

    // ── 1. サウンドボタン → 点火待ち。実際に鳴るのはスクロールが動いてから ──
    let waitingForScroll = false;
    const onFirstScroll = () => {
      if (!waitingForScroll) return;
      waitingForScroll = false;
      window.removeEventListener("scroll", onFirstScroll);
      start();
    };

    function arm() {
      armedRef.current = true;
      try {
        sessionStorage.setItem(ARM_KEY, "1");
      } catch {
        /* プライベートモード等。鳴らせなくても致命的ではない */
      }
      /**
       * ここで「音量 0 のまま再生を始めて」おく。
       *
       * スクロールを待ってから play() すると、そこで初めて 5MB の取得が始まり、
       * 回線の細いスマホでは音が出るまで数秒空いてしまう（実機で発生）。
       * ボタンを押した瞬間はユーザー操作の最中なので、ここなら確実に再生を
       * 始められる。音量 0 なのでイントロの効果音とは重ならない。
       * スクロールが動いたら、あとは音量を上げるだけ＝待たずに鳴る。
       */
      const a = ref.current;
      if (a && a.paused && !document.hidden) {
        a.preload = "auto";
        a.volume = 0;
        const p = a.play();
        if (p && typeof p.catch === "function") p.catch(() => armRetryOnGesture());
      }

      if (window.scrollY > 4) {
        // すでにスクロール済みならすぐ入る
        start();
      } else {
        waitingForScroll = true;
        window.addEventListener("scroll", onFirstScroll, { passive: true });
      }
    }

    function disarm() {
      armedRef.current = false;
      waitingForScroll = false;
      window.removeEventListener("scroll", onFirstScroll);
      try {
        sessionStorage.removeItem(ARM_KEY);
      } catch {
        /* noop */
      }
      fadeTo(0, 700);
    }

    const syncToSoundStore = () => (getSoundOn() ? arm() : disarm());
    const unsubSound = subscribeSound(() => {
      if (getSoundOn() === armedRef.current) return;
      syncToSoundStore();
    });

    // ── 2. 前回この タブ で点火済みなら、戻ってきた扱いで鳴らし直す ──
    let restored = false;
    try {
      restored = sessionStorage.getItem(ARM_KEY) === "1";
    } catch {
      /* noop */
    }
    if (restored) {
      armedRef.current = true;
      start(); // 弾かれたら armRetryOnGesture が拾う
    } else if (getSoundOn()) {
      arm();
    }

    // ── 3. 別タブへ出ている間は止め、戻ってきたら鳴らし直す ──
    const onVisibility = () => {
      if (document.hidden) {
        ref.current?.pause();
      } else if (armedRef.current) {
        start();
      }
    };
    document.addEventListener("visibilitychange", onVisibility);

    // 戻る操作で復元されたとき（bfcache）も鳴らし直す
    const onPageShow = () => {
      if (armedRef.current) start();
    };
    window.addEventListener("pageshow", onPageShow);

    // ── 4. 作品が鳴っている間は絞る ──
    const unsubDuck = subscribeDuck((d) => {
      duckedRef.current = d;
      const a = ref.current;
      if (a && !a.paused) fadeTo(target(), d ? 500 : 1400);
    });

    return () => {
      unsubSound();
      unsubDuck();
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("pageshow", onPageShow);
      window.removeEventListener("scroll", onFirstScroll);
      if (fadeRef.current) window.clearInterval(fadeRef.current);
      ref.current?.pause();
    };
  }, []);

  return <audio ref={ref} src={SRC} loop preload="none" aria-hidden="true" />;
}
