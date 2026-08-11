"use client";

import { useEffect, type ReactNode } from "react";
import Lenis from "lenis";
import { setLenis } from "@/lib/lenisBridge";

export default function SmoothScroll({ children }: { children: ReactNode }) {
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const lenis = new Lenis({
      duration: 1.3,
      easing: (t: number) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      touchMultiplier: 2,
      /**
       * ページの中に縦スクロールする箱がある（アルバムの曲目リスト）。
       * これが無いと lenis がホイールを全部ページ送りに使ってしまい、
       * 箱の中身が動かせない。逆に data-lenis-prevent で丸ごと譲ると、
       * 今度は箱の上でページが進まなくなる（スマホで画面が固定される）。
       * allowNestedScroll なら「箱が動ける間は箱、端まで来たらページ」になる。
       */
      allowNestedScroll: true,
    });
    setLenis(lenis); // ワープ演出などから scrollTo / stop / start を使えるよう共有

    let raf = 0;
    function loop(time: number) {
      lenis.raf(time);
      raf = requestAnimationFrame(loop);
    }
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      setLenis(null);
      lenis.destroy();
    };
  }, []);

  return <>{children}</>;
}
