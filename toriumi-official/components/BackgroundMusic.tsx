"use client";

import { useEffect, useRef } from "react";
import { subscribeDuck } from "@/lib/mediaBus";
import { subscribeSound, getSoundOn } from "@/lib/soundStore";
import { isHeroRevealed, subscribeHeroReveal } from "@/lib/heroReveal";

/**
 * サイト全体の BGM。
 *
 * 鳴り始めの決まり：
 * 1. ページ上部のサウンドボタンが押される（= soundOn）と「点火待ち」になる。
 *    ここではまだ鳴らさない。イントロの効果音と重なって濁るため。
 * 2. 最初の画面に名前とロゴが立ち現れる。ここから初めてスクロールを見張る。
 * 3. そこから SCROLL_TRIGGER_PX 以上スクロールしたら、ゆっくり音量を上げて入る。
 *    旅が始まるのと同時に音楽が立ち上がる。
 * 4. 音量は小さめ（MAX_VOL）。主役は各セクションの作品なので、その下に敷く。
 *
 * 外部リンクへ出て戻ってきたとき：
 * - 別タブで開かれた場合、このページは生きたまま隠れるだけなので、
 *   隠れている間は止め、戻ってきたら鳴らし直す（裏で鳴り続けると鬱陶しい）。
 * - 同じタブで移動して戻った場合はページが読み直される。
 *   sessionStorage に点火済みの印を残しておき、次の読み込みで自動再開する。
 *   ただし操作なしの自動再生はブラウザに止められることがあるので、
 *   弾かれたら最初のスクロールやクリックで鳴らす保険を張る。
 *
 * MV やアルバムが鳴っている間は譲る（mediaBus の duck）。
 * PC はほとんど聞こえないところまで絞り、スマホは完全に止める。
 */

const SRC = "/bgm-neon-horizon.mp3";
/** 主張しすぎない。0.18 → 0.108（3/5）→ さらに半分 */
const MAX_VOL = 0.054;
const DUCK_VOL = 0.006; // 作品が鳴っている間
const FADE_MS = 2200;
const ARM_KEY = "toriumi.bgm.armed";
/** ロゴが出たあと、これだけ動いたら「スクロールを始めた」とみなす */
const SCROLL_TRIGGER_PX = 24;
/**
 * 同じタブで戻ってきて位置も復元されている場合、ここより深ければ
 * 「旅の途中」とみなしてスクロールを待たずに鳴らし直す
 */
const RESUME_DEPTH_PX = 200;

/** 端末のスピーカーで聴く画面幅。ここでは絞るのではなく完全に止める */
const isPhone = () => window.matchMedia("(max-width: 767px)").matches;

export default function BackgroundMusic() {
  const ref = useRef<HTMLAudioElement>(null);
  /** 目標音量へ寄せていくためのタイマー */
  const fadeRef = useRef<number | null>(null);
  const duckedRef = useRef(false);
  /** サウンドボタンが押されたか（＝鳴らしてよいか） */
  const armedRef = useRef(false);
  /** 実際に音量を上げて鳴り出したか。復帰時に鳴らし直してよいかの判断に使う */
  const startedRef = useRef(false);

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

    /**
     * 音量 0 のまま再生だけ始めておく（先読み）。まだ聞こえない。
     *
     * スクロールを待ってから play() すると、そこで初めて 5MB の取得が始まり、
     * 回線の細いスマホでは音が出るまで数秒空いてしまう（実機で発生）。
     * ボタンを押した瞬間はユーザー操作の最中なので、ここなら確実に始められる。
     */
    function prime() {
      const a = ref.current;
      if (!a || !armedRef.current || document.hidden || !a.paused) return;
      a.preload = "auto";
      a.volume = 0;
      const p = a.play();
      if (p && typeof p.catch === "function") p.catch(() => retryOnGesture());
    }

    /**
     * 自動再生を拒まれたときの保険。一度きり。
     *
     * ⚠ ここで音量を上げてはいけない。
     * 以前は無条件に start() を呼んでいたため、ロゴが立ち現れる前の
     * 何気ないタップひとつで鳴り始めの条件を飛び越えていた（実測で再現）。
     * やり直すのは無音の先読みだけで、すでに鳴ってよい状態のときだけ音量を戻す。
     */
    function retryOnGesture() {
      const retry = () => {
        window.removeEventListener("pointerdown", retry);
        window.removeEventListener("keydown", retry);
        if (startedRef.current) start();
        else prime();
      };
      window.addEventListener("pointerdown", retry, { once: true, passive: true });
      window.addEventListener("keydown", retry, { once: true });
    }

    function start() {
      const a = ref.current;
      if (!a || !armedRef.current || document.hidden) return;
      startedRef.current = true;
      // prime() で音量 0 のまま鳴り始めている場合がある。
      // そのときは play() を呼び直さず、音量を上げるだけでよい
      if (a.paused) {
        const p = a.play();
        if (p && typeof p.catch === "function") p.catch(() => retryOnGesture());
      }
      fadeTo(target(), FADE_MS);
    }

    /**
     * ── 1. 鳴り始めの条件 ──
     * ①サウンドボタンが押されている ②最初の画面に名前とロゴが立ち現れた
     * ③そのあとで、はっきりとスクロールが動いた——の三つが揃ってから音量を上げる。
     *
     * ②を入れているのは、ブート明けすぐに鳴らすと UFO のワープ音と
     * 顕現の一撃に重なって濁るため。
     *
     * ③は「②のあとの動き」だけを数える。ここが以前の作りとの違い。
     * 以前は読み込み時点の scrollY と、それ以降の scroll をすべて数えていたが、
     * スマホではアドレスバーが隠れるだけでも scroll が飛び、指を置いた弾みの
     * わずかな揺れでも発火する。その結果、ロゴが立ち現れた瞬間に条件が揃って
     * 鳴り出していた（実機で発生）。
     * そこで、顕現の合図が来てから基準の位置を取り直し、そこから
     * SCROLL_TRIGGER_PX 以上動いたときだけ「スクロールを始めた」とみなす。
     */
    let baseline = 0;
    let watching = false;

    const onScroll = () => {
      if (Math.abs(window.scrollY - baseline) < SCROLL_TRIGGER_PX) return;
      stopWatching();
      start();
    };
    function stopWatching() {
      watching = false;
      window.removeEventListener("scroll", onScroll);
    }
    /** 「点火待ち」と「顕現済み」が揃った時点から、そこを基準にスクロールを見張る */
    function watchScroll() {
      if (watching || startedRef.current || !armedRef.current || !isHeroRevealed()) return;
      watching = true;
      baseline = window.scrollY;
      window.addEventListener("scroll", onScroll, { passive: true });
    }
    const unsubReveal = subscribeHeroReveal(watchScroll);

    function arm() {
      armedRef.current = true;
      try {
        sessionStorage.setItem(ARM_KEY, "1");
      } catch {
        /* プライベートモード等。鳴らせなくても致命的ではない */
      }
      // 音量 0 のまま先読みを始めておく。スクロールが動いたら音量を上げるだけ＝待たずに鳴る
      prime();
      watchScroll(); // 顕現が済んでいれば、この場から見張りに入る
    }

    function disarm() {
      armedRef.current = false;
      startedRef.current = false;
      stopWatching();
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
    /**
     * 再訪時も arm() を通す。
     * 以前は armedRef を直接立てて start() を呼んでいたが、それだと
     * スクロールの監視が登録されないうえ、armedRef が先に true になるせいで
     * あとからサウンドボタンが押されても「変化なし」と見なされて arm() が
     * 呼ばれず、鳴らないままになっていた（実機の再訪で発生）。
     */
    if (restored || getSoundOn()) arm();
    /**
     * 外部リンクから同じタブで戻ってきて、読み終えた位置まで復元されている場合は
     * 「旅の途中」なので、もう一度スクロールさせずにそのまま鳴らし直す。
     * 深さで見分けているので、頭から読み込んだときは（scrollY≈0）ここを通らない。
     */
    if (restored && window.scrollY > RESUME_DEPTH_PX) {
      stopWatching();
      start();
    }

    // ── 3. 別タブへ出ている間は止め、戻ってきたら鳴らし直す ──
    /** 一度鳴り出していて、いま作品に譲っていなければ鳴らし直してよい */
    const mayResume = () => startedRef.current && !duckedRef.current;

    const onVisibility = () => {
      if (document.hidden) {
        ref.current?.pause();
      } else if (armedRef.current && mayResume()) {
        start();
      }
    };
    document.addEventListener("visibilitychange", onVisibility);

    // 戻る操作で復元されたとき（bfcache）も鳴らし直す
    const onPageShow = () => {
      if (armedRef.current && mayResume()) start();
    };
    window.addEventListener("pageshow", onPageShow);

    /**
     * ── 4. 作品が鳴っている間は譲る ──
     *
     * PC は小さく敷いたままでよいが、スマホは完全に止める。
     * 端末のスピーカーは低い音量差を潰してしまい、絞っただけでは
     * 曲に混ざって聞こえてしまう（実機で発生）。加えて iOS Safari は
     * volume の書き換えそのものを無視するので、絞る指示が効かない。
     * どちらの事情も「止める」なら確実に解決する。
     */
    const unsubDuck = subscribeDuck((d) => {
      duckedRef.current = d;
      const a = ref.current;
      if (!a) return;

      if (isPhone()) {
        if (d) {
          if (fadeRef.current) window.clearInterval(fadeRef.current);
          fadeRef.current = null;
          a.pause();
          a.volume = 0;
        } else if (armedRef.current && startedRef.current && !document.hidden) {
          start(); // 曲が終わったら、また静かに敷き直す
        }
        return;
      }

      if (!a.paused) fadeTo(target(), d ? 500 : 1400);
    });

    return () => {
      unsubSound();
      unsubDuck();
      unsubReveal();
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("pageshow", onPageShow);
      stopWatching();
      if (fadeRef.current) window.clearInterval(fadeRef.current);
      ref.current?.pause();
    };
  }, []);

  return <audio ref={ref} src={SRC} loop preload="none" aria-hidden="true" />;
}
