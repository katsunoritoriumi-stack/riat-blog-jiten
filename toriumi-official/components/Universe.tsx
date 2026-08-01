"use client";

import { useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import Scene, { type Quality } from "./universe/Scene";

/**
 * ホームの背景そのものである、ひとつながりの宇宙。
 *
 * スクロール量がそのままカメラの前進距離になり、星の回廊を抜け、
 * 星雲の中を通り、決まった深度に浮かぶ太陽系（創造の座標軸）へ到達する。
 * ZoomStage（セクションの見せ方）とは lib/stations.ts の配分を共有していて、
 * 「セクションが出る位置」と「宇宙で何かが起きる位置」がずれないようにしてある。
 *
 * 置き場所はホームだけ。/apps と /websites は従来どおり Canvas 2D の
 * GalaxyBackground を使う（three.js を他のルートへ広げないため）。
 */
export default function Universe() {
  const [quality, setQuality] = useState<Quality | null>(null);
  const [dpr, setDpr] = useState(1);

  useEffect(() => {
    const coarse = window.matchMedia("(pointer: coarse)").matches;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    setDpr(coarse ? 1 : Math.min(window.devicePixelRatio || 1, 1.75));
    setQuality({
      coarse,
      // 参考画像のような密度を出したいが、モバイルは描画予算を優先して減らす
      stars: reduce ? 900 : coarse ? 1500 : 6000,
      bright: reduce ? 40 : coarse ? 70 : 260,
      nebulaLayers: coarse ? 5 : 9,
      fov: 60,
    });
  }, []);

  return (
    <div className="fixed inset-0 z-0 overflow-hidden bg-void-950" aria-hidden="true">
      {/*
        WebGL が立ち上がるまで（と、使えない環境で）出しておく静止した宇宙。
        これがあるので、SSR の HTML は軽いまま最初の一枚が真っ黒にならない。
      */}
      <div className="absolute inset-0 universe-fallback" />

      {quality && (
        <Canvas
          className="absolute inset-0"
          camera={{ fov: quality.fov, near: 0.5, far: 4200, position: [0, 0, 0] }}
          dpr={dpr}
          /**
           * 不透明にする。透過のままだと加算合成の結果が下の CSS 星空と混ざり、
           * 色が飽和したところに階調の段差が出る（実際に紫の塊になった）。
           */
          onCreated={({ gl }) => gl.setClearColor(0x04030c, 1)}
          gl={{
            antialias: false,
            alpha: false,
            powerPreference: "high-performance",
            // 検証用スナップショットのため開発時のみ保持する（本番では性能を優先）
            preserveDrawingBuffer: process.env.NODE_ENV !== "production",
          }}
        >
          <Scene quality={quality} />
        </Canvas>
      )}

      {/* 画面端を締めるヴィネット。DOM 側でやるのが一番安い */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_28%,rgba(2,2,8,0.82)_88%)]" />
    </div>
  );
}
