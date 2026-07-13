/**
 * Hero イントロの荘厳なサウンドエフェクト（手続き合成・音源ファイル不要）。
 * UFO が彼方から目の前へ到達（0.8s→2.4s）→ テキスト顕現（≈5.8s）に合わせた一筆書き：
 *   ・接近のウーッシュ＋上昇ドップラー → 到達の"光速ドロップアウト"の一撃（スターウォーズ的）
 *   ・低い開放和音のパッド（荘厳なドローン）がゆっくり満ちる
 *   ・ノイズ・ライザーで緊張を高め、顕現の瞬間にベルの開花＋サブの一撃＋天上のシマー
 *
 * iOS 対策：Web Audio API の音は iOS の消音（サイレント）スイッチで無音化される。
 * そこで OfflineAudioContext で WAV に事前書き出しし、<audio>（メディア再生）で鳴らす。
 * メディア要素はサイレントモードでも鳴るため、確実に音が出る。
 */

const REVEAL_TIME = 5.8; // 顕現の一撃（動画時間に対応）
const TOTAL = 14.5; // 全体尺（レンダリング長）

/**
 * 与えられた AudioContext（オンライン/オフライン両対応）にイントロ全体を組む。
 * master gain を返す（ライブ再生時のフェード停止に使う）。
 */
function scheduleIntro(ctx: BaseAudioContext, t0: number): GainNode {
  // ── master chain: gain → soft compressor → destination ──
  const master = ctx.createGain();
  master.gain.value = 0.0;
  const comp = ctx.createDynamicsCompressor();
  comp.threshold.value = -18;
  comp.knee.value = 24;
  comp.ratio.value = 3;
  comp.attack.value = 0.005;
  comp.release.value = 0.25;
  master.connect(comp).connect(ctx.destination);

  master.gain.setValueAtTime(0.0001, t0);
  master.gain.exponentialRampToValueAtTime(0.9, t0 + 1.2);
  master.gain.setValueAtTime(0.9, t0 + 9);
  master.gain.exponentialRampToValueAtTime(0.0001, t0 + 13);

  const makeNoise = (dur: number) => {
    const b = ctx.createBuffer(1, Math.floor(ctx.sampleRate * dur), ctx.sampleRate);
    const d = b.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    return b;
  };

  // ── 1) 荘厳な開放和音パッド（A1 / E2 / A2 / E3）──
  const padGain = ctx.createGain();
  padGain.gain.setValueAtTime(0.0001, t0);
  padGain.gain.exponentialRampToValueAtTime(0.16, t0 + 5.5);
  padGain.gain.setValueAtTime(0.16, t0 + 7);
  padGain.gain.exponentialRampToValueAtTime(0.05, t0 + 11);
  padGain.gain.exponentialRampToValueAtTime(0.0001, t0 + 13);
  const padFilter = ctx.createBiquadFilter();
  padFilter.type = "lowpass";
  padFilter.frequency.setValueAtTime(320, t0);
  padFilter.frequency.exponentialRampToValueAtTime(1400, t0 + REVEAL_TIME);
  padFilter.Q.value = 0.7;
  padGain.connect(padFilter).connect(master);

  const padVoices = [55, 82.41, 110, 164.81];
  padVoices.forEach((f, i) => {
    [0, 1].forEach((d) => {
      const osc = ctx.createOscillator();
      osc.type = i < 2 ? "sine" : "triangle";
      osc.frequency.value = f;
      osc.detune.value = d === 0 ? -5 : 6;
      const g = ctx.createGain();
      g.gain.value = i < 2 ? 0.5 : 0.28;
      osc.connect(g).connect(padGain);
      const lfo = ctx.createOscillator();
      lfo.frequency.value = 0.06 + i * 0.017;
      const lfoG = ctx.createGain();
      lfoG.gain.value = f * 0.004;
      lfo.connect(lfoG).connect(osc.frequency);
      osc.start(t0);
      lfo.start(t0);
      osc.stop(t0 + 13.2);
      lfo.stop(t0 + 13.2);
    });
  });

  // ── 1.5) スターウォーズ的ハイパースペース（UFO が彼方から目の前へ到達）──
  const APPROACH = t0 + 0.8;
  const ARRIVE = t0 + 2.4;

  // (a) 接近ウーッシュ
  const approach = ctx.createBufferSource();
  approach.buffer = makeNoise(ARRIVE - APPROACH + 0.3);
  const abp = ctx.createBiquadFilter();
  abp.type = "bandpass";
  abp.Q.value = 1.1;
  abp.frequency.setValueAtTime(240, APPROACH);
  abp.frequency.exponentialRampToValueAtTime(3600, ARRIVE);
  const aG = ctx.createGain();
  aG.gain.setValueAtTime(0.0001, APPROACH);
  aG.gain.exponentialRampToValueAtTime(0.22, ARRIVE);
  aG.gain.exponentialRampToValueAtTime(0.0001, ARRIVE + 0.5);
  approach.connect(abp).connect(aG).connect(master);
  approach.start(APPROACH);
  approach.stop(ARRIVE + 0.55);

  // (b) 接近ドップラー（迫るにつれ音程が上がる）
  const dop = ctx.createOscillator();
  dop.type = "sawtooth";
  dop.frequency.setValueAtTime(120, APPROACH);
  dop.frequency.exponentialRampToValueAtTime(900, ARRIVE - 0.05);
  const dlp = ctx.createBiquadFilter();
  dlp.type = "lowpass";
  dlp.frequency.setValueAtTime(700, APPROACH);
  dlp.frequency.exponentialRampToValueAtTime(2600, ARRIVE);
  dlp.Q.value = 3;
  const dG = ctx.createGain();
  dG.gain.setValueAtTime(0.0001, APPROACH);
  dG.gain.exponentialRampToValueAtTime(0.12, ARRIVE - 0.1);
  dG.gain.exponentialRampToValueAtTime(0.0001, ARRIVE + 0.3);
  dop.connect(dlp).connect(dG).connect(master);
  dop.start(APPROACH);
  dop.stop(ARRIVE + 0.4);

  // (c) 到達の一撃：光速からのドロップアウト（下降 vwoom）
  const drop = ctx.createOscillator();
  drop.type = "sawtooth";
  drop.frequency.setValueAtTime(2400, ARRIVE - 0.05);
  drop.frequency.exponentialRampToValueAtTime(90, ARRIVE + 0.6);
  const droplp = ctx.createBiquadFilter();
  droplp.type = "lowpass";
  droplp.frequency.setValueAtTime(3400, ARRIVE - 0.05);
  droplp.frequency.exponentialRampToValueAtTime(500, ARRIVE + 0.65);
  droplp.Q.value = 7;
  const dropG = ctx.createGain();
  dropG.gain.setValueAtTime(0.0001, ARRIVE - 0.05);
  dropG.gain.exponentialRampToValueAtTime(0.32, ARRIVE + 0.02);
  dropG.gain.exponentialRampToValueAtTime(0.0001, ARRIVE + 0.9);
  drop.connect(droplp).connect(dropG).connect(master);
  drop.start(ARRIVE - 0.05);
  drop.stop(ARRIVE + 1.0);

  // (d) 到達の轟き：ベース・インパクト
  const boom = ctx.createOscillator();
  boom.type = "sine";
  boom.frequency.setValueAtTime(80, ARRIVE);
  boom.frequency.exponentialRampToValueAtTime(32, ARRIVE + 0.9);
  const boomG = ctx.createGain();
  boomG.gain.setValueAtTime(0.0001, ARRIVE);
  boomG.gain.exponentialRampToValueAtTime(0.7, ARRIVE + 0.03);
  boomG.gain.exponentialRampToValueAtTime(0.0001, ARRIVE + 1.5);
  boom.connect(boomG).connect(master);
  boom.start(ARRIVE);
  boom.stop(ARRIVE + 1.6);

  // (e) 到達直後の高域シュワッ（星の尾）
  const streak = ctx.createBufferSource();
  streak.buffer = makeNoise(1.2);
  const shp = ctx.createBiquadFilter();
  shp.type = "highpass";
  shp.frequency.value = 3200;
  const streakG = ctx.createGain();
  streakG.gain.setValueAtTime(0.0001, ARRIVE);
  streakG.gain.exponentialRampToValueAtTime(0.08, ARRIVE + 0.12);
  streakG.gain.exponentialRampToValueAtTime(0.0001, ARRIVE + 1.1);
  streak.connect(shp).connect(streakG).connect(master);
  streak.start(ARRIVE);
  streak.stop(ARRIVE + 1.2);

  // ── 2) ノイズ・ライザー（顕現へ向けて上がる）──
  const noiseLen = 3.6;
  const noise = ctx.createBufferSource();
  noise.buffer = makeNoise(noiseLen);
  const riseStart = t0 + REVEAL_TIME - noiseLen + 0.2;
  const bp = ctx.createBiquadFilter();
  bp.type = "bandpass";
  bp.Q.value = 0.8;
  bp.frequency.setValueAtTime(200, riseStart);
  bp.frequency.exponentialRampToValueAtTime(6500, riseStart + noiseLen);
  const noiseGain = ctx.createGain();
  noiseGain.gain.setValueAtTime(0.0001, riseStart);
  noiseGain.gain.exponentialRampToValueAtTime(0.11, t0 + REVEAL_TIME - 0.05);
  noiseGain.gain.exponentialRampToValueAtTime(0.0001, t0 + REVEAL_TIME + 0.6);
  noise.connect(bp).connect(noiseGain).connect(master);
  noise.start(riseStart);
  noise.stop(riseStart + noiseLen);

  // ── 3) 顕現の一撃：ベルの開花（加算合成）──
  const bellT = t0 + REVEAL_TIME;
  const bellFund = 293.66;
  const partials = [
    { m: 1, g: 0.5, d: 3.4 },
    { m: 2.0, g: 0.32, d: 2.6 },
    { m: 3.01, g: 0.2, d: 2.1 },
    { m: 4.18, g: 0.13, d: 1.6 },
    { m: 5.43, g: 0.08, d: 1.2 },
  ];
  const bellBus = ctx.createGain();
  bellBus.gain.value = 0.5;
  bellBus.connect(master);
  partials.forEach((p) => {
    const osc = ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.value = bellFund * p.m;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, bellT);
    g.gain.exponentialRampToValueAtTime(p.g, bellT + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, bellT + p.d);
    osc.connect(g).connect(bellBus);
    osc.start(bellT);
    osc.stop(bellT + p.d + 0.1);
  });

  // ── 4) サブの一撃 ──
  const sub = ctx.createOscillator();
  sub.type = "sine";
  sub.frequency.setValueAtTime(90, bellT);
  sub.frequency.exponentialRampToValueAtTime(42, bellT + 0.5);
  const subG = ctx.createGain();
  subG.gain.setValueAtTime(0.0001, bellT);
  subG.gain.exponentialRampToValueAtTime(0.6, bellT + 0.03);
  subG.gain.exponentialRampToValueAtTime(0.0001, bellT + 1.4);
  sub.connect(subG).connect(master);
  sub.start(bellT);
  sub.stop(bellT + 1.5);

  // ── 5) 天上のシマー ──
  const shimmerBus = ctx.createGain();
  shimmerBus.gain.setValueAtTime(0.0001, bellT);
  shimmerBus.gain.exponentialRampToValueAtTime(0.06, bellT + 0.6);
  shimmerBus.gain.exponentialRampToValueAtTime(0.0001, bellT + 3);
  shimmerBus.connect(master);
  [880, 1318.5, 1760, 2637].forEach((f, i) => {
    const osc = ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.value = f;
    const g = ctx.createGain();
    g.gain.value = 0.5 - i * 0.1;
    osc.connect(g).connect(shimmerBus);
    osc.start(bellT);
    osc.stop(bellT + 3.1);
  });

  return master;
}

/** AudioBuffer → 16bit PCM WAV（ArrayBuffer） */
export function encodeWav(buffer: AudioBuffer): ArrayBuffer {
  const numCh = buffer.numberOfChannels;
  const sr = buffer.sampleRate;
  const len = buffer.length;
  const blockAlign = numCh * 2;
  const dataSize = len * blockAlign;
  const ab = new ArrayBuffer(44 + dataSize);
  const view = new DataView(ab);
  let p = 0;
  const wstr = (s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(p++, s.charCodeAt(i));
  };
  const u32 = (v: number) => {
    view.setUint32(p, v, true);
    p += 4;
  };
  const u16 = (v: number) => {
    view.setUint16(p, v, true);
    p += 2;
  };
  wstr("RIFF");
  u32(36 + dataSize);
  wstr("WAVE");
  wstr("fmt ");
  u32(16);
  u16(1);
  u16(numCh);
  u32(sr);
  u32(sr * blockAlign);
  u16(blockAlign);
  u16(16);
  wstr("data");
  u32(dataSize);
  const chans: Float32Array[] = [];
  for (let c = 0; c < numCh; c++) chans.push(buffer.getChannelData(c));
  for (let i = 0; i < len; i++) {
    for (let c = 0; c < numCh; c++) {
      const s = Math.max(-1, Math.min(1, chans[c][i]));
      view.setInt16(p, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      p += 2;
    }
  }
  return ab;
}

/**
 * イントロを WAV に事前レンダリングし、Blob URL を返す（マウント時に一度だけ）。
 * この URL を <audio> に流せば、iOS のサイレントモードでも鳴る。
 * 失敗時は null（呼び出し側はライブ Web Audio にフォールバック）。
 */
export async function renderHeroIntroUrl(): Promise<string | null> {
  try {
    const OfflineCtx =
      window.OfflineAudioContext ||
      (window as unknown as { webkitOfflineAudioContext: typeof OfflineAudioContext })
        .webkitOfflineAudioContext;
    if (!OfflineCtx) return null;
    const sampleRate = 44100;
    const length = Math.ceil(sampleRate * TOTAL);
    const octx = new OfflineCtx(2, length, sampleRate);
    scheduleIntro(octx, 0.05);
    const rendered = await octx.startRendering();
    const blob = new Blob([encodeWav(rendered)], { type: "audio/wav" });
    return URL.createObjectURL(blob);
  } catch {
    return null;
  }
}

export type HeroAudioHandle = {
  stop: () => void;
};

/**
 * ライブ再生（フォールバック）。OfflineAudioContext が使えない環境向け。
 * ※ iOS ではサイレントスイッチで無音になり得るため、可能なら renderHeroIntroUrl + <audio> を使う。
 */
export function playMajesticIntro(): HeroAudioHandle {
  const Ctx =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new Ctx();
  void ctx.resume?.();
  try {
    const unlock = ctx.createBufferSource();
    unlock.buffer = ctx.createBuffer(1, 1, 22050);
    unlock.connect(ctx.destination);
    unlock.start(0);
  } catch {}

  const master = scheduleIntro(ctx, ctx.currentTime + 0.15);

  let stopped = false;
  const cleanup = () => {
    if (ctx.state !== "closed") ctx.close();
  };
  const autoTimer = window.setTimeout(cleanup, TOTAL * 1000);

  return {
    stop() {
      if (stopped) return;
      stopped = true;
      window.clearTimeout(autoTimer);
      const now = ctx.currentTime;
      try {
        master.gain.cancelScheduledValues(now);
        master.gain.setValueAtTime(Math.max(master.gain.value, 0.0001), now);
        master.gain.exponentialRampToValueAtTime(0.0001, now + 0.4);
      } catch {}
      window.setTimeout(cleanup, 500);
    },
  };
}
