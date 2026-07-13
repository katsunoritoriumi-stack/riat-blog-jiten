"use client";

import { useSyncExternalStore } from "react";

/**
 * soundOn（サウンド有効フラグ）のグローバルストア。
 * Hero のトグルが唯一の書き込み元。ワープ演出や星クリックの SFX など、
 * React の外（素の関数）からも getSoundOn() で読めるように module pub/sub で持つ。
 */

let soundOn = false;
const listeners = new Set<() => void>();

export const getSoundOn = () => soundOn;

export function setSoundOn(v: boolean) {
  if (soundOn === v) return;
  soundOn = v;
  listeners.forEach((l) => l());
}

export function subscribeSound(l: () => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

/** React から購読するフック。SSR スナップショットは常に false（ハイドレーション安全）。 */
export function useSoundOn(): boolean {
  return useSyncExternalStore(subscribeSound, getSoundOn, () => false);
}
