"use client";

import { encodeWav } from "./heroAudio";
import { getSoundOn, subscribeSound } from "./soundStore";

/**
 * 短い UI 効果音（SFX）。Sound ON のときだけ鳴る。
 * iOS のサイレントスイッチは Web Audio を無音化するため、
 * OfflineAudioContext で WAV に事前レンダリングし <audio> で再生する（heroAudio と同方式）。
 * レンダリングは初回 soundOn=true のときに一度だけ（OFF の間はゼロコスト）。
 */

export type SfxName = "chime" | "warp" | "capture";

const urls: Partial<Record<SfxName, string>> = {};
let rendering = false;

type OC = typeof OfflineAudioContext;

function getOfflineCtx(): OC | null {
  if (typeof window === "undefined") return null;
  return (
    window.OfflineAudioContext ||
    (window as unknown as { webkitOfflineAudioContext: OC })
      .webkitOfflineAudioContext ||
    null
  );
}

/** 星クリックのチャイム：澄んだベル2部音（D6 系）0.5s */
function scheduleChime(ctx: BaseAudioContext) {
  const t0 = 0.01;
  const bus = ctx.createGain();
  bus.gain.value = 0.5;
  bus.connect(ctx.destination);
  [
    { f: 1174.66, g: 0.5, d: 0.45 }, // D6
    { f: 1760.0, g: 0.22, d: 0.32 }, // A6
  ].forEach((p) => {
    const o = ctx.createOscillator();
    o.type = "sine";
    o.frequency.value = p.f;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(p.g, t0 + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + p.d);
    o.connect(g).connect(bus);
    o.start(t0);
    o.stop(t0 + p.d + 0.05);
  });
}

/** ワープのウーッシュ：上昇バンドパスノイズ＋低域の押し出し 0.75s */
function scheduleWarp(ctx: BaseAudioContext) {
  const t0 = 0.01;
  const dur = 0.7;
  const buf = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * dur), ctx.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  const src = ctx.createBufferSource();
  src.buffer = buf;
  const bp = ctx.createBiquadFilter();
  bp.type = "bandpass";
  bp.Q.value = 1.2;
  bp.frequency.setValueAtTime(220, t0);
  bp.frequency.exponentialRampToValueAtTime(4200, t0 + dur * 0.55);
  bp.frequency.exponentialRampToValueAtTime(500, t0 + dur);
  const g = ctx.createGain();
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(0.35, t0 + 0.12);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  src.connect(bp).connect(g).connect(ctx.destination);
  src.start(t0);
  src.stop(t0 + dur);

  const sub = ctx.createOscillator();
  sub.type = "sine";
  sub.frequency.setValueAtTime(120, t0);
  sub.frequency.exponentialRampToValueAtTime(45, t0 + dur);
  const sg = ctx.createGain();
  sg.gain.setValueAtTime(0.0001, t0);
  sg.gain.exponentialRampToValueAtTime(0.3, t0 + 0.05);
  sg.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  sub.connect(sg).connect(ctx.destination);
  sub.start(t0);
  sub.stop(t0 + dur + 0.05);
}

/** UFO捕獲のピロリ：3音上昇ブリップ 0.5s */
function scheduleCapture(ctx: BaseAudioContext) {
  const notes = [660, 880, 1320]; // E5 → A5 → E6
  notes.forEach((f, i) => {
    const t = 0.01 + i * 0.11;
    const o = ctx.createOscillator();
    o.type = "triangle";
    o.frequency.value = f;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.4, t + 0.015);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.14);
    o.connect(g).connect(ctx.destination);
    o.start(t);
    o.stop(t + 0.2);
  });
}

const RECIPES: Record<SfxName, { dur: number; schedule: (c: BaseAudioContext) => void }> = {
  chime: { dur: 0.6, schedule: scheduleChime },
  warp: { dur: 0.85, schedule: scheduleWarp },
  capture: { dur: 0.55, schedule: scheduleCapture },
};

async function renderAll() {
  if (rendering || urls.chime) return;
  rendering = true;
  const OfflineCtx = getOfflineCtx();
  if (!OfflineCtx) {
    rendering = false;
    return;
  }
  const sr = 44100;
  for (const name of Object.keys(RECIPES) as SfxName[]) {
    try {
      const { dur, schedule } = RECIPES[name];
      const octx = new OfflineCtx(1, Math.ceil(sr * dur), sr);
      schedule(octx);
      const rendered = await octx.startRendering();
      const blob = new Blob([encodeWav(rendered)], { type: "audio/wav" });
      urls[name] = URL.createObjectURL(blob);
    } catch {
      /* 個別失敗は無視（該当SFXが無音になるだけ） */
    }
  }
  rendering = false;
}

// 初回 soundOn=true を検知して遅延レンダリング
if (typeof window !== "undefined") {
  subscribeSound(() => {
    if (getSoundOn()) void renderAll();
  });
}

/** SFX を再生。Sound OFF・未レンダリング時は何もしない。重ね再生OK。 */
export function playSfx(name: SfxName) {
  if (!getSoundOn()) return;
  const url = urls[name];
  if (!url) {
    void renderAll(); // 念のため（トグル前に呼ばれた場合）
    return;
  }
  try {
    const el = new Audio(url);
    el.volume = 0.9;
    void el.play();
  } catch {}
}
