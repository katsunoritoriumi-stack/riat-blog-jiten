"use client";

import { useSyncExternalStore } from "react";

/**
 * セクションの「奥からの到着」変換を一時的に無効化するフラグ。
 *
 * ワープ遷移（WarpOverlay）は lenis の scrollTo に要素を渡すが、lenis は
 * getBoundingClientRect() から移動先を計算する。到着アニメの途中で縮んでいる
 * セクションを掴むと着地位置がずれるため、ワープ中だけ全セクションを
 * 無変換に戻す。SSR スナップショットは false。
 */

let bypassed = false;
const listeners = new Set<() => void>();

export const isDepthBypassed = () => bypassed;

export function setDepthBypass(v: boolean) {
  if (bypassed === v) return;
  bypassed = v;
  listeners.forEach((l) => l());
}

export function subscribeDepthBypass(l: () => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function useDepthBypass(): boolean {
  return useSyncExternalStore(subscribeDepthBypass, isDepthBypassed, () => false);
}
