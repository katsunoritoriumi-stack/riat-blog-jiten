/**
 * 「宇宙を前進する」演出の数式だけを切り出したもの。
 * DOM にも React にも依存しないので、requestAnimationFrame が動かない環境でも
 * 素の node スクリプトで数値検証できる（canvas 側を薄く保つ目的も兼ねる）。
 */

/* ── 星屑の飛行 ───────────────────────────── */

/** カメラ直前（これを超えたら奥へ再配置） */
export const Z_NEAR = 0.08;
/** 生成される最も遠い距離 */
export const Z_FAR = 1.6;

/** ページ最上部での巡航速度（スクロールを止めても完全には停まらない） */
export const CRUISE_MIN = 0.0009;
/** ページ最下部での巡航速度 */
export const CRUISE_MAX = 0.0034;
/** スクロール量(px/フレーム) → 奥行き速度 の換算 */
export const PX_TO_Z = 0.000045;
/** 1フレームあたりの上限速度（速すぎると点が飛び飛びになる） */
export const V_MAX = 0.018;
/** thrust の1フレームあたり減衰率（0.15 ＝ 60fps で 1% まで約28フレーム≒0.47秒） */
export const THRUST_LERP = 0.15;

export const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** e0 → e1 に向かって 0→1 に滑らかに上がる（e0 > e1 の逆順も可） */
export function smoothstep(e0: number, e1: number, x: number): number {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
}

/** ページ深度 prog(0-1) に応じた巡航速度 */
export function cruiseSpeed(prog: number): number {
  return CRUISE_MIN + clamp01(prog) * (CRUISE_MAX - CRUISE_MIN);
}

/**
 * スクロール速度を平滑化した「推力」。
 * dyPerFrame はフレーム換算のスクロール量(px)。dt は 60fps を 1 とした経過フレーム数。
 * スクロールを止める（dyPerFrame=0）と 0 へ収束する＝止めれば漂いに戻る。
 */
export function thrustStep(thrust: number, dyPerFrame: number, dt: number): number {
  const target = dyPerFrame * PX_TO_Z;
  const k = 1 - Math.pow(1 - THRUST_LERP, Math.max(0, dt));
  return thrust + (target - thrust) * k;
}

/**
 * 実際に進む距離。前進方向にクランプする（上スクロールでは減速するだけで逆走しない）。
 * 戻り値は「このフレームで z から引く量」。
 */
export function flightVelocity(cruise: number, thrust: number, dt: number): number {
  const v = Math.max(cruise * 0.15, Math.min(V_MAX, cruise + thrust));
  return v * Math.max(0, dt);
}

/**
 * 点を1フレーム進める。カメラを通り過ぎたら最遠へ戻す。
 * recycled が true のとき、呼び出し側は x/y を撒き直し、前フレーム位置(px,py)を
 * 必ず NaN に戻すこと（戻さないと画面を横断するストリークが描かれる）。
 */
export function advance(
  z: number,
  vz: number,
  zNear = Z_NEAR,
  zFar = Z_FAR
): { z: number; recycled: boolean } {
  const next = z - vz;
  if (next <= zNear) return { z: zFar, recycled: true };
  return { z: next, recycled: false };
}

/** 遠すぎ／近すぎで消える明るさ係数 */
export function depthAlpha(z: number, zNear = Z_NEAR, zFar = Z_FAR): number {
  return smoothstep(zFar, zFar * 0.78, z) * smoothstep(zNear, zNear * 2.4, z);
}

/* ── セクションの「奥から到着」 ───────────────── */

const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);

/**
 * 到着中のスケール。
 * p <= 0（まだ画面下に入っていない）と p >= 1（到着済み）では必ず 1 を返す。
 * これにより、静的HTML／JS無効時／未到達セクションが常に無変換になり、
 * getBoundingClientRect() が実レイアウト値を返す（lenis の scrollTo が狂わない）。
 */
export function arrivalScale(p: number, from: number): number {
  if (p <= 0 || p >= 1) return 1;
  return from + (1 - from) * easeOutCubic(p);
}

/** 到着中の不透明度。スケールと同じく p <= 0 では 1（＝無加工）を返す。 */
export function arrivalOpacity(p: number): number {
  if (p <= 0) return 1;
  return clamp01(0.25 + 0.75 * Math.min(1, p / 0.5));
}
