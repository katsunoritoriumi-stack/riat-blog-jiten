"use client";

import type Lenis from "lenis";

/**
 * SmoothScroll が生成した lenis インスタンスを module-level で共有する橋。
 * ワープ演出が lenis.scrollTo / stop / start を使うために参照する。
 * reduced-motion 等で lenis が無い場合は null（呼び出し側でフォールバック）。
 */

let lenis: Lenis | null = null;

export const setLenis = (l: Lenis | null) => {
  lenis = l;
};

export const getLenis = (): Lenis | null => lenis;
