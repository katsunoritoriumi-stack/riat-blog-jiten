/**
 * 交信ブート演出（BootSequence）のサウンド。
 *   ① アンビエンス：深宇宙の低いドローン＋テレメトリのブリップ音＋末尾の「信号ロック」チャイム
 *   ② トランジション・スティング：ブート→UFO本編への切り替え瞬間に鳴る、ノイズ・ザップ＋低音の一撃
 *
 * iOS 対策は heroAudio.ts と同方式：OfflineAudioContext で WAV に事前書き出しし、
 * <audio>（メディア再生）で鳴らす。サイレントスイッチでも鳴る。
 */

import { encodeWav } from "./heroAudio";

function getOfflineCtx(): typeof OfflineAudioContext | null {
  if (typeof window === "undefined") return null;
  return (
    window.OfflineAudioContext ||
    (window as unknown as { webkitOfflineAudioContext: typeof OfflineAudioContext })
      .webkitOfflineAudioContext ||
    null
  );
}

function makeNoise(ctx: BaseAudioContext, dur: number): AudioBuffer {
  const b = ctx.createBuffer(1, Math.max(1, Math.floor(ctx.sampleRate * dur)), ctx.sampleRate);
  const d = b.getChannelData(0);
  for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  return b;
}

/**
 * アンビエンスを合成する。durationSec 全体に対し、末尾 lockAtSec のあたりで
 * 「信号ロック」の2音チャイムを鳴らす（BootSequence の最終行が表示される頃合いに一致させる）。
 */
function scheduleAmbience(ctx: BaseAudioContext, durationSec: number, lockAtSec: number) {
  const master = ctx.createGain();
  master.gain.value = 0.55;
  master.connect(ctx.destination);

  // ── 低いドローン（深宇宙の"間"）：2声・ゆっくり呼吸するように音量が揺れる ──
  const droneBus = ctx.createGain();
  droneBus.gain.value = 0.09;
  const droneFilter = ctx.createBiquadFilter();
  droneFilter.type = "lowpass";
  droneFilter.frequency.value = 340;
  droneBus.connect(droneFilter).connect(master);

  [46, 69.3].forEach((f, i) => {
    const osc = ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.value = f;
    const g = ctx.createGain();
    g.gain.value = i === 0 ? 0.6 : 0.35;
    osc.connect(g).connect(droneBus);
    osc.start(0);
    osc.stop(durationSec + 0.3);

    // 呼吸するようなゆっくりのLFO
    const lfo = ctx.createOscillator();
    lfo.frequency.value = 0.12 + i * 0.03;
    const lfoGain = ctx.createGain();
    lfoGain.gain.value = i === 0 ? 0.25 : 0.15;
    lfo.connect(lfoGain).connect(g.gain);
    lfo.start(0);
    lfo.stop(durationSec + 0.3);
  });

  // ── テレメトリのブリップ（データ受信音）：ランダムな間隔で短いクリック ──
  let t = 0.15;
  while (t < durationSec - 0.3) {
    const click = ctx.createBufferSource();
    click.buffer = makeNoise(ctx, 0.02);
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass";
    bp.Q.value = 5;
    bp.frequency.value = 1800 + Math.random() * 1600;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.16, t + 0.006);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.05);
    click.connect(bp).connect(g).connect(master);
    click.start(t);
    click.stop(t + 0.06);
    t += 0.22 + Math.random() * 0.22;
  }

  // ── 信号ロック：2音の確認チャイム（最終行が表示される瞬間に合わせる） ──
  [
    { f: 659.25, at: lockAtSec },
    { f: 987.77, at: lockAtSec + 0.1 },
  ].forEach((n) => {
    const osc = ctx.createOscillator();
    osc.type = "triangle";
    osc.frequency.value = n.f;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, n.at);
    g.gain.exponentialRampToValueAtTime(0.32, n.at + 0.015);
    g.gain.exponentialRampToValueAtTime(0.0001, n.at + 0.4);
    osc.connect(g).connect(master);
    osc.start(n.at);
    osc.stop(n.at + 0.45);
  });

  // 末尾はふっと消える
  master.gain.setValueAtTime(0.55, Math.max(0, durationSec - 0.5));
  master.gain.linearRampToValueAtTime(0.0001, durationSec);
}

/** ブート→UFO本編への切り替え瞬間に鳴る、ノイズ・ザップ＋低音の一撃＋きらめき（約0.9秒） */
function scheduleTransition(ctx: BaseAudioContext) {
  const master = ctx.createGain();
  master.gain.value = 0.85;
  master.connect(ctx.destination);
  const t0 = 0.01;

  // (a) ノイズバースト：帯域が急上昇する"ザザッ"
  const noise = ctx.createBufferSource();
  noise.buffer = makeNoise(ctx, 0.3);
  const bp = ctx.createBiquadFilter();
  bp.type = "bandpass";
  bp.Q.value = 0.9;
  bp.frequency.setValueAtTime(300, t0);
  bp.frequency.exponentialRampToValueAtTime(5200, t0 + 0.22);
  const noiseG = ctx.createGain();
  noiseG.gain.setValueAtTime(0.0001, t0);
  noiseG.gain.exponentialRampToValueAtTime(0.4, t0 + 0.03);
  noiseG.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.28);
  noise.connect(bp).connect(noiseG).connect(master);
  noise.start(t0);
  noise.stop(t0 + 0.3);

  // (b) 上昇ザップ
  const zap = ctx.createOscillator();
  zap.type = "sawtooth";
  zap.frequency.setValueAtTime(220, t0);
  zap.frequency.exponentialRampToValueAtTime(3000, t0 + 0.22);
  const zapFilter = ctx.createBiquadFilter();
  zapFilter.type = "bandpass";
  zapFilter.Q.value = 3.5;
  zapFilter.frequency.setValueAtTime(500, t0);
  zapFilter.frequency.exponentialRampToValueAtTime(3600, t0 + 0.22);
  const zapG = ctx.createGain();
  zapG.gain.setValueAtTime(0.0001, t0);
  zapG.gain.exponentialRampToValueAtTime(0.28, t0 + 0.05);
  zapG.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.24);
  zap.connect(zapFilter).connect(zapG).connect(master);
  zap.start(t0);
  zap.stop(t0 + 0.3);

  // (c) 低音の一撃（画面が切り替わる瞬間の"ドン"）
  const thump = ctx.createOscillator();
  thump.type = "sine";
  const thumpAt = t0 + 0.16;
  thump.frequency.setValueAtTime(95, thumpAt);
  thump.frequency.exponentialRampToValueAtTime(34, thumpAt + 0.45);
  const thumpG = ctx.createGain();
  thumpG.gain.setValueAtTime(0.0001, thumpAt);
  thumpG.gain.exponentialRampToValueAtTime(0.75, thumpAt + 0.02);
  thumpG.gain.exponentialRampToValueAtTime(0.0001, thumpAt + 0.55);
  thump.connect(thumpG).connect(master);
  thump.start(thumpAt);
  thump.stop(thumpAt + 0.6);

  // (d) きらめき：ハイトーンの短いベル（"確立した"感触）
  const shimmerBus = ctx.createGain();
  shimmerBus.gain.setValueAtTime(0.0001, thumpAt);
  shimmerBus.gain.exponentialRampToValueAtTime(0.18, thumpAt + 0.05);
  shimmerBus.gain.exponentialRampToValueAtTime(0.0001, thumpAt + 0.7);
  shimmerBus.connect(master);
  [1567.98, 2093, 3135.96].forEach((f, i) => {
    const osc = ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.value = f;
    const g = ctx.createGain();
    g.gain.value = 0.5 - i * 0.13;
    osc.connect(g).connect(shimmerBus);
    osc.start(thumpAt);
    osc.stop(thumpAt + 0.75);
  });
}

async function renderUrl(
  channels: number,
  durationSec: number,
  build: (ctx: OfflineAudioContext) => void
): Promise<string | null> {
  const OfflineCtx = getOfflineCtx();
  if (!OfflineCtx) return null;
  try {
    const sr = 44100;
    const octx = new OfflineCtx(channels, Math.ceil(sr * durationSec), sr);
    build(octx);
    const rendered = await octx.startRendering();
    const blob = new Blob([encodeWav(rendered)], { type: "audio/wav" });
    return URL.createObjectURL(blob);
  } catch {
    return null;
  }
}

/** アンビエンスを WAV へ事前レンダリングし、Blob URL を返す。 */
export function renderBootAmbienceUrl(durationSec: number, lockAtSec: number): Promise<string | null> {
  return renderUrl(1, durationSec, (ctx) => scheduleAmbience(ctx, durationSec, lockAtSec));
}

/** トランジション・スティングを WAV へ事前レンダリングし、Blob URL を返す。 */
export function renderBootTransitionUrl(): Promise<string | null> {
  return renderUrl(2, 1.0, (ctx) => scheduleTransition(ctx));
}
