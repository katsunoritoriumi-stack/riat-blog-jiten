"use client";

import { useSyncExternalStore } from "react";

/**
 * 「創造の座標軸」でいま選ばれている星（ドメイン）の key。
 *
 * 太陽系は背景の宇宙（R3F のツリー）の中にあり、リンクの一覧を出す DOM は
 * ConstellationMap 側にある。React のツリーが分かれているので、
 * lib/bootGate.ts と同じ形のモジュール級ストアで橋渡しする。
 *
 * 単一のリンクしか持たない星はそのまま開くので、ここに入るのは
 * 「行き先が複数ある星」だけ。
 */

let picked: string | null = null;
const listeners = new Set<() => void>();

export const getPickedDomain = () => picked;

export function setPickedDomain(key: string | null) {
  if (picked === key) return;
  picked = key;
  listeners.forEach((l) => l());
}

export function subscribePickedDomain(l: () => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function usePickedDomain(): string | null {
  return useSyncExternalStore(subscribePickedDomain, getPickedDomain, () => null);
}
