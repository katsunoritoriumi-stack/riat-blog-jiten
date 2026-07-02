/**
 * Hero イントロの荘厳なサウンドエフェクト（手続き合成・音源ファイル不要）。
 * UFO の登場〜テキストの顕現に合わせた 8 秒のシネマティックな一筆書き：
 *   ・低い開放和音のパッド（荘厳なドローン）がゆっくり満ちる
 *   ・フィルターの上がるノイズ・ライザーで緊張を高める
 *   ・顕現の瞬間（≈5.8s）にベルの開花＋サブの一撃（"名前が宿る"衝撃）
 *   ・天上のシマーが一瞬ひらいて、パッドは静かに消えていく
 *
 * ブラウザの自動再生ポリシーによりユーザー操作（クリック等）から呼ぶこと。
 */

const REVEAL_TIME = 5.8; // 顕現の一撃（動画時間に対応）

export type HeroAudioHandle = {
  stop: () => void;
};

export function playMajesticIntro(): HeroAudioHandle {
  const Ctx =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext })
      .webkitAudioContext;
  const ctx = new Ctx();
  const t0 = ctx.currentTime + 0.02;

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

  // マスターを軽く立ち上げ、末尾で消す
  master.gain.setValueAtTime(0.0001, t0);
  master.gain.exponentialRampToValueAtTime(0.9, t0 + 1.2);
  master.gain.setValueAtTime(0.9, t0 + 9);
  master.gain.exponentialRampToValueAtTime(0.0001, t0 + 13);

  const stops: Array<{ stop: (t?: number) => void }> = [];

  // ── 1) 荘厳な開放和音パッド（A1 / E2 / A2 / E3）──
  const padGain = ctx.createGain();
  padGain.gain.setValueAtTime(0.0001, t0);
  padGain.gain.exponentialRampToValueAtTime(0.16, t0 + 5.5); // ゆっくり満ちる
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
    // 各声部を微デチューンした2基のオシレーターで厚みを出す
    [0, 1].forEach((d) => {
      const osc = ctx.createOscillator();
      osc.type = i < 2 ? "sine" : "triangle";
      osc.frequency.value = f;
      osc.detune.value = d === 0 ? -5 : 6;
      const g = ctx.createGain();
      g.gain.value = i < 2 ? 0.5 : 0.28;
      osc.connect(g).connect(padGain);
      // ゆらぎ（LFO）で"蠢く"感覚を音にも
      const lfo = ctx.createOscillator();
      lfo.frequency.value = 0.06 + i * 0.017;
      const lfoG = ctx.createGain();
      lfoG.gain.value = f * 0.004;
      lfo.connect(lfoG).connect(osc.frequency);
      osc.start(t0);
      lfo.start(t0);
      osc.stop(t0 + 13.2);
      lfo.stop(t0 + 13.2);
      stops.push(osc, lfo);
    });
  });

  // ── 1.5) スターウォーズ的ハイパースペース（UFO が彼方から目の前へ到達）──
  //   動画：≈0.8s に光の中から出現 → 接近しながら巨大化 → ≈2.4s で画面いっぱい（目の前）
  //   接近のウーッシュが満ちていき、到達の瞬間に"光速からのドロップアウト"の一撃。
  const APPROACH = t0 + 0.8; // 彼方に出現
  const ARRIVE = t0 + 2.4; // 目の前に到達

  const makeNoise = (dur: number) => {
    const b = ctx.createBuffer(1, Math.floor(ctx.sampleRate * dur), ctx.sampleRate);
    const d = b.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    return b;
  };

  // (a) 接近ウーッシュ：帯域とゲインが到達へ向けて上がりきる（ヒュゥゥゥ…）
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
  stops.push(approach);

  // (b) 接近ドップラー：迫るにつれ音程が上がる（ドップラー効果）
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
  stops.push(dop);

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
  stops.push(drop);

  // (d) 到達の轟き：現実空間へのベース・インパクト
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
  stops.push(boom);

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
  stops.push(streak);

  // ── 2) ノイズ・ライザー（顕現へ向けてフィルターとゲインが上がる）──
  const noiseLen = 3.6;
  const buffer = ctx.createBuffer(1, ctx.sampleRate * noiseLen, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  const noise = ctx.createBufferSource();
  noise.buffer = buffer;
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
  stops.push(noise);

  // ── 3) 顕現の一撃：ベルの開花（加算合成）──
  const bellT = t0 + REVEAL_TIME;
  const bellFund = 293.66; // D4
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
    stops.push(osc);
  });

  // ── 4) サブの一撃（低い衝撃）──
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
  stops.push(sub);

  // ── 5) 天上のシマー（高い開放和音が一瞬ひらく）──
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
    stops.push(osc);
  });

  let stopped = false;
  function cleanup() {
    if (ctx.state !== "closed") ctx.close();
  }
  // 自然終了
  const autoTimer = window.setTimeout(cleanup, 14000);

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
