/**
 * 手続き生成テクスチャの共有キャッシュ。
 *
 * 太陽系と飛来天体が同じ質感を使うので、同じものを二度焼かないようにする
 * （ガス惑星は 1 枚 0.6 秒かかる。二度焼くとそのぶんスクロールが引っかかる）。
 *
 * 焼くのは必ず 1 枚ずつ、間に setTimeout を挟んで順番に。
 * まとめて焼くとメインスレッドが 1.6 秒固まる。
 */

import type * as THREE from "three";

const cache = new Map<string, THREE.Texture>();
/** 直列に並べるための鎖 */
let chain: Promise<unknown> = Promise.resolve();

export function bakeOnce(key: string, make: () => THREE.Texture): Promise<THREE.Texture> {
  const hit = cache.get(key);
  if (hit) return Promise.resolve(hit);

  const next = chain.then(
    () =>
      new Promise<THREE.Texture>((resolve) => {
        // 1 フレーム譲ってから焼く（連続で焼かない）
        window.setTimeout(() => {
          let t = cache.get(key);
          if (!t) {
            t = make();
            cache.set(key, t);
          }
          resolve(t);
        }, 16);
      })
  );
  chain = next.catch(() => undefined);
  return next;
}

/** すでに焼き上がっていれば即返す（描画中の同期取得用） */
export function peek(key: string): THREE.Texture | undefined {
  return cache.get(key);
}

/** ディフューズマップの鍵。太陽系と飛来天体が同じ質感を共有するために揃える */
export const diffuseKey = (skin: string, seed: number) => `diffuse:${skin}:${seed}`;
