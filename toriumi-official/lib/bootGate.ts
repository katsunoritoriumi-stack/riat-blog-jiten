"use client";

import { useSyncExternalStore } from "react";

/**
 * ブート演出（交信シーケンス）の完了フラグ。
 * Hero は bootDone を待って UFO 動画の再生を開始する。
 * SSR スナップショットは false（サーバーHTMLではブートオーバーレイが表示状態）。
 */

let bootDone = false;
const listeners = new Set<() => void>();

export const isBootDone = () => bootDone;

export function markBootDone() {
  if (bootDone) return;
  bootDone = true;
  listeners.forEach((l) => l());
}

export function subscribeBoot(l: () => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function useBootDone(): boolean {
  return useSyncExternalStore(subscribeBoot, isBootDone, () => false);
}
