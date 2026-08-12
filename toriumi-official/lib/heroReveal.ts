"use client";

import { useSyncExternalStore } from "react";

/**
 * Hero の顕現フラグ（最初の画面に名前とロゴが立ち現れたか）。
 *
 * BGM の鳴り始めをここに合わせている。
 * ブート明けすぐに鳴らすと、UFO のワープ音と顕現の一撃に重なって濁るので、
 * 「ロゴが出そろってから、スクロールが動いたら」という二段構えにしたい。
 * その片方の合図がこれ。bootGate と同じ作りにしてある。
 */

let revealed = false;
const listeners = new Set<() => void>();

export const isHeroRevealed = () => revealed;

export function markHeroRevealed() {
  if (revealed) return;
  revealed = true;
  listeners.forEach((l) => l());
}

export function subscribeHeroReveal(l: () => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function useHeroRevealed(): boolean {
  return useSyncExternalStore(subscribeHeroReveal, isHeroRevealed, () => false);
}
