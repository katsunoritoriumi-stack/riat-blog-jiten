/**
 * 旅の状態。React の再描画を挟まずに毎フレーム読み書きしたいので、
 * モジュール1つに可変オブジェクトとして置いている（context だと再描画が走る）。
 * 書き込むのは Scene の Driver ただ1つ。ほかは読むだけ。
 */

import { clamp01 } from "@/lib/flightMath";
import { STATION_BANDS, TOTAL_SCREENS } from "@/lib/stations";

/**
 * 旅の全長（ワールド単位）。
 * 全 23.9 画面ぶんのスクロールをこの距離に対応させる。
 * 1 画面あたり約 150 単位＝星の筒（780）の中を 5 画面かけて抜ける計算。
 */
export const CORRIDOR = TOTAL_SCREENS * 150;

export const flight = {
  /** ページ全体の進捗 0-1 */
  prog: 0,
  /** いまの深度（ワールド単位。カメラは z = -depth にいる） */
  depth: 0,
  /** 前進速度（ワールド単位／秒）。ストリークと画角に効く */
  speed: 0,
  /** 0-1 に正規化した速度。演出の強さに使う */
  rush: 0,
  /**
   * true の間はスクロールを読まない。
   * 検証用スナップショットで「任意の深度に固定して1フレーム描く」ためだけに使う。
   */
  locked: false,
};

export const depthOf = (prog: number) => clamp01(prog) * CORRIDOR;

/** ステーション帯の中の位置 q における深度 */
export function depthAt(index: number, q: number) {
  const b = STATION_BANDS[index];
  if (!b) return 0;
  return depthOf(b.start + b.span * q);
}

/** この速度で「速い」とみなす上限（ワールド単位／秒） */
export const RUSH_FULL = 900;
