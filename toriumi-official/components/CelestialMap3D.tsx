"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { CameraControls, Stars, Html } from "@react-three/drei";
import { EffectComposer, Bloom, GodRays } from "@react-three/postprocessing";
// 型だけ借りる。実体は drei が持つもの（install 済み）を使う
import type CameraControlsImpl from "camera-controls";
import Link from "next/link";
import * as THREE from "three";
import { DOMAINS, type Domain } from "@/lib/content";
import { playSfx } from "@/lib/sfx";
import { makeCommonNormal, makeDiffuse, type PlanetSkin } from "@/lib/planetTextures";

/**
 * 創造の座標軸（3D版）— 「深淵なる宇宙と光の物語」。
 *
 * ・中心星 Connect(LINE) を唯一の光源とし、6つのドメインが実際に公転する
 * ・惑星は PBR（カラー＋共通法線マップ）。質感は lib/planetTextures.ts で手続き生成
 * ・Bloom と GodRays（中心星からの光芒）で映画的な光を作る
 * ・星をクリック → カメラがその星まで飛び、注記が開いて行き先が出る
 * ・three.js は重いので、このファイルは ConstellationMap 側から動的 import される
 *
 * 本物のテクスチャに差し替えたい場合は、SKINS の生成を
 * useLoader(THREE.TextureLoader, "/textures/xxx.webp") に置き換えるだけでよい。
 */

type PlanetConfig = {
  skin: PlanetSkin;
  seed: number;
  radius: number; // 軌道半径
  speed: number; // 公転角速度（rad/s）
  size: number;
  phase: number;
  roughness: number;
  metalness: number;
  normalScale: number; // 0 なら法線マップを使わない（ガス惑星）
  emissive?: string;
  emissiveIntensity?: number;
  atmosphere?: string; // 指定すると大気層を纏う
  accent: string; // ラベルとにじむ光の色
};

/** lib/content.ts の 6 ドメインに、仕様書の質感を割り当てる */
const PLANETS: Record<string, PlanetConfig> = {
  // 乾いた赤い砂漠惑星
  art: {
    skin: "desert",
    seed: 11,
    radius: 4.0,
    speed: 0.3,
    size: 0.55,
    phase: 0.2,
    roughness: 0.85,
    metalness: 0.05,
    normalScale: 2.5,
    accent: "#e8a06a",
  },
  // 滑らかな赤橙のガス惑星（わずかに自己発光）
  youtube: {
    skin: "gas",
    seed: 22,
    radius: 5.6,
    speed: 0.24,
    size: 0.8,
    phase: 2.1,
    roughness: 0.35,
    metalness: 0.1,
    normalScale: 0,
    emissive: "#ff4400",
    emissiveIntensity: 0.6,
    accent: "#ff8a4c",
  },
  // クラフト感のあるマーブル岩石
  fashion: {
    skin: "marble",
    seed: 33,
    radius: 7.2,
    speed: 0.19,
    size: 0.6,
    phase: 4.0,
    roughness: 0.9,
    metalness: 0.05,
    normalScale: 1.8,
    accent: "#d8c2cc",
  },
  // クレーターの多い荒々しい岩石
  produce: {
    skin: "cratered",
    seed: 44,
    radius: 8.8,
    speed: 0.155,
    size: 0.72,
    phase: 0.9,
    roughness: 0.95,
    metalness: 0.05,
    normalScale: 3.0,
    accent: "#bfae94",
  },
  // 森と海の惑星（大気層あり）
  sns: {
    skin: "living",
    seed: 55,
    radius: 10.4,
    speed: 0.13,
    size: 0.68,
    phase: 3.1,
    roughness: 0.8,
    metalness: 0.1,
    normalScale: 1.2,
    emissive: "#bbffcc",
    emissiveIntensity: 0.05,
    atmosphere: "#bbffcc",
    accent: "#9ff0c0",
  },
  // 冷たい青の氷惑星
  radio: {
    skin: "ice",
    seed: 66,
    radius: 12.0,
    speed: 0.11,
    size: 0.6,
    phase: 5.4,
    roughness: 0.7,
    metalness: 0.15,
    normalScale: 1.0,
    emissive: "#77aaff",
    emissiveIntensity: 0.1,
    accent: "#9cc6ff",
  },
};

const CORE_SIZE = 1.5;
const CAMERA_FOV = 45;

/**
 * drei の Html は distanceFactor / カメラ距離 で拡大する。
 * 惑星をクリックしてカメラが寄ると距離が 4 程度まで縮み、ラベルが 4 倍近くに
 * 膨れ上がって画面からはみ出す。見かけの倍率をこの範囲に収めるため、
 * drei が掛けた分を内側で打ち消す。
 */
const HTML_DISTANCE_FACTOR = 16;
const LABEL_SCALE_MIN = 0.6;
const LABEL_SCALE_MAX = 1.05;

/** カメラ距離から「内側で打ち消すべき倍率」を求める */
function labelCounterScale(distance: number): number {
  const applied = HTML_DISTANCE_FACTOR / Math.max(distance, 0.001);
  const wanted = Math.min(LABEL_SCALE_MAX, Math.max(LABEL_SCALE_MIN, applied));
  return wanted / applied;
}

type TextureSet = { normal: THREE.Texture; skins: Partial<Record<PlanetSkin, THREE.Texture>> };
type PositionMap = Map<string, THREE.Vector3>;

/* ─────────────────────────────────────────────
   惑星ラベル（計器の注記風）
   ───────────────────────────────────────────── */

/** ポップオーバー内のリンク1行。ホバーで短い罫線が伸びて文字が少し送られる */
function LinkRow({
  label,
  href,
  accent,
}: {
  label: string;
  href: string;
  accent: string;
}) {
  const cls =
    "group/row flex items-center gap-2 px-3 py-[7px] text-left font-mono text-[11px] tracking-wide text-nebula-100/90 transition-colors duration-300 hover:text-white";
  const inner = (
    <>
      <span
        aria-hidden
        className="h-px w-0 shrink-0 transition-all duration-300 group-hover/row:w-3"
        style={{ background: accent }}
      />
      <span className="transition-transform duration-300 group-hover/row:translate-x-0.5">
        {label}
      </span>
    </>
  );
  // 内部ページ（App / Website 一覧）は Next のクライアントルーティングで遷移
  return href.startsWith("/") ? (
    <Link href={href} className={cls}>
      {inner}
    </Link>
  ) : (
    <a href={href} target="_blank" rel="noopener noreferrer" className={cls}>
      {inner}
    </a>
  );
}

/**
 * 惑星に添えるラベル。
 * 箱で囲わず、惑星へ引き出し線を伸ばす「注記」の形にしている。
 * 色はその惑星自身の色を使うので、天体ごとに佇まいが変わる。
 */
function PlanetLabel({
  rootRef,
  data,
  accent,
  active,
  focused,
  onSelect,
  onHover,
}: {
  rootRef: React.RefObject<HTMLDivElement | null>;
  data: Domain;
  accent: string;
  active: boolean;
  focused: boolean;
  onSelect: () => void;
  onHover: (v: boolean) => void;
}) {
  const links = data.links ?? (data.href ? [{ label: "開く", href: data.href }] : []);

  return (
    <div
      ref={rootRef}
      className="flex select-none flex-col items-center"
      style={{ transformOrigin: "50% 100%" }}
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
    >
      {focused && (
        <div
          className="mb-3 w-[196px] overflow-hidden rounded-2xl backdrop-blur-md"
          style={{
            background:
              "linear-gradient(180deg, rgba(10,7,26,0.95) 0%, rgba(3,2,10,0.96) 100%)",
            border: `1px solid ${accent}26`,
            boxShadow: `0 20px 44px -20px rgba(0,0,0,0.95), 0 0 30px -16px ${accent}, inset 0 1px 0 ${accent}1f`,
          }}
        >
          {/* 上端に一本だけ光の線を通す */}
          <span
            aria-hidden
            className="block h-px w-full"
            style={{ background: `linear-gradient(90deg, transparent, ${accent}, transparent)` }}
          />
          <p className="px-3 pb-2 pt-2.5 text-[10px] leading-relaxed text-nebula-200/70">
            {data.blurb}
          </p>
          <div className="flex flex-col pb-1.5">
            {links.map((l) => (
              <LinkRow key={l.href} label={l.label} href={l.href} accent={accent} />
            ))}
          </div>
        </div>
      )}

      <button
        onClick={onSelect}
        aria-label={data.titleEn}
        aria-expanded={focused}
        className="relative isolate block px-4 py-1.5 text-center"
      >
        {/*
          可読性のための暗幕。四角い箱にならないよう、楕円のグラデーションを
          さらにぼかして輪郭を完全に消している（角が出ない）。
        */}
        <span
          aria-hidden
          className="pointer-events-none absolute left-1/2 top-1/2 -z-10 -translate-x-1/2 -translate-y-1/2 rounded-full transition-opacity duration-300"
          style={{
            background:
              "radial-gradient(closest-side, rgba(3,2,12,0.94), rgba(3,2,12,0.55) 58%, rgba(3,2,12,0) 100%)",
            width: "145%",
            height: "230%",
            filter: "blur(5px)",
            opacity: active ? 1 : 0.85,
          }}
        />
        {/* 選択中はその星の色がふわっと灯る */}
        <span
          aria-hidden
          className="pointer-events-none absolute left-1/2 top-1/2 -z-10 -translate-x-1/2 -translate-y-1/2 rounded-full transition-opacity duration-500"
          style={{
            background: `radial-gradient(closest-side, ${accent}2b, transparent 100%)`,
            width: "170%",
            height: "260%",
            filter: "blur(9px)",
            opacity: active ? 1 : 0,
          }}
        />

        {/* 上の目盛り線。ホバー・選択で伸びる */}
        <span
          aria-hidden
          className="mx-auto mb-[5px] block h-px transition-all duration-500"
          style={{
            width: active ? 34 : 16,
            background: `linear-gradient(90deg, transparent, ${accent}, transparent)`,
            opacity: active ? 0.95 : 0.4,
          }}
        />

        <span
          className="block whitespace-nowrap font-mono text-[11px] uppercase tracking-[0.3em] transition-colors duration-300"
          style={{
            color: active ? accent : "rgba(246,244,255,0.95)",
            textShadow: "0 1px 4px rgba(0,0,0,0.95), 0 0 14px rgba(0,0,0,0.9)",
          }}
        >
          {data.titleEn}
        </span>
      </button>

      {/* 惑星へ伸びる引き出し線 */}
      <span
        aria-hidden
        className="mt-1 block w-px transition-all duration-500"
        style={{
          height: active ? 24 : 14,
          background: `linear-gradient(180deg, ${accent}, transparent)`,
          opacity: active ? 0.9 : 0.45,
        }}
      />
    </div>
  );
}

/**
 * 中心星を「球」に見せるための周縁減光（limb darkening）。
 *
 * 自己発光だけの天体は全面が同じ明るさになり、Bloom も相まって
 * 真っ白な円盤に見えてしまう。実際の恒星と同じく、視線に対して斜めになる
 * 周縁ほど暗く・赤くすることで、初めて立体の球として読める。
 */
const limbDarkening = (shader: { fragmentShader: string }) => {
  shader.fragmentShader = shader.fragmentShader.replace(
    "#include <emissivemap_fragment>",
    `#include <emissivemap_fragment>
     float ndv = clamp(abs(dot(normalize(vNormal), normalize(vViewPosition))), 0.0, 1.0);
     float limb = pow(ndv, 0.8);
     totalEmissiveRadiance *= mix(0.20, 1.0, limb);
     totalEmissiveRadiance *= mix(vec3(1.0, 0.48, 0.14), vec3(1.0, 0.95, 0.84), limb);`
  );
};

/* ─────────────────────────────────────────────
   中心星（Connect = LINE）— GodRays の光源
   ───────────────────────────────────────────── */
function CoreStar({
  data,
  textures,
  reduced,
  onHover,
  onReady,
}: {
  data: Domain;
  textures: TextureSet | null;
  reduced: boolean;
  onHover: (v: boolean) => void;
  onReady: (m: THREE.Mesh | null) => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const coronaRef = useRef<THREE.Mesh>(null!);
  const labelRef = useRef<HTMLDivElement>(null);
  const [hovered, setHovered] = useState(false);
  const map = textures?.skins.plasma ?? null;

  useFrame((state, delta) => {
    if (!reduced) meshRef.current.rotation.y -= delta * 0.045; // 系全体と同じ左巻き
    if (labelRef.current) {
      // 惑星ラベルと同じく、寄ったときの巨大化を打ち消す
      const d = state.camera.position.length();
      labelRef.current.style.transform = `scale(${labelCounterScale(d).toFixed(3)})`;
    }
    const t = state.clock.getElapsedTime();
    coronaRef.current.scale.setScalar(1 + Math.sin(t * 0.7) * 0.04 + (hovered ? 0.08 : 0));
  });

  const open = () => {
    playSfx("chime");
    window.open(data.href, "_blank", "noopener,noreferrer");
  };

  return (
    <group
      onPointerOver={(e) => {
        e.stopPropagation();
        setHovered(true);
        onHover(true);
      }}
      onPointerOut={() => {
        setHovered(false);
        onHover(false);
      }}
      onClick={(e) => {
        e.stopPropagation();
        open();
      }}
    >
      <mesh
        ref={(m) => {
          meshRef.current = m!;
          onReady(m);
        }}
      >
        <sphereGeometry args={[CORE_SIZE, 64, 64]} />
        <meshStandardMaterial
          map={map}
          emissiveMap={map}
          emissive="#ffffff"
          emissiveIntensity={1.65}
          roughness={0.9}
          metalness={0.1}
          toneMapped={false}
          onBeforeCompile={limbDarkening}
        />
      </mesh>

      {/* コロナ（にじむ外殻） */}
      <mesh ref={coronaRef} scale={1.18}>
        <sphereGeometry args={[CORE_SIZE, 32, 32]} />
        <meshBasicMaterial
          color="#ffb347"
          transparent
          opacity={0.14}
          side={THREE.BackSide}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      <Html distanceFactor={HTML_DISTANCE_FACTOR} position={[0, CORE_SIZE + 1.1, 0]} center zIndexRange={[30, 0]}>
        {/* 中心星も惑星と同じ注記の作法に揃える（箱で囲わない） */}
        <button
          ref={labelRef as unknown as React.RefObject<HTMLButtonElement>}
          onClick={open}
          className="relative isolate flex select-none flex-col items-center px-4 py-1.5"
          style={{ transformOrigin: "50% 100%" }}
        >
          {/* 角の出ないぼかし楕円の暗幕 */}
          <span
            aria-hidden
            className="pointer-events-none absolute left-1/2 top-1/2 -z-10 -translate-x-1/2 -translate-y-1/2 rounded-full"
            style={{
              background:
                "radial-gradient(closest-side, rgba(12,6,2,0.94), rgba(12,6,2,0.55) 58%, rgba(12,6,2,0) 100%)",
              width: "150%",
              height: "230%",
              filter: "blur(5px)",
            }}
          />
          <span
            aria-hidden
            className="pointer-events-none absolute left-1/2 top-1/2 -z-10 -translate-x-1/2 -translate-y-1/2 rounded-full transition-opacity duration-500"
            style={{
              background: "radial-gradient(closest-side, rgba(255,190,90,0.22), transparent 100%)",
              width: "180%",
              height: "265%",
              filter: "blur(10px)",
              opacity: hovered ? 1 : 0.45,
            }}
          />
          <span
            aria-hidden
            className="mx-auto mb-[5px] block h-px transition-all duration-500"
            style={{
              width: hovered ? 38 : 20,
              background: "linear-gradient(90deg, transparent, #ffd77a, transparent)",
              opacity: hovered ? 0.95 : 0.5,
            }}
          />
          <span
            className="whitespace-nowrap font-mono text-[11.5px] uppercase tracking-[0.32em] transition-colors duration-300"
            style={{
              color: hovered ? "#fff6de" : "#ffe9b4",
              textShadow: "0 1px 4px rgba(0,0,0,0.95), 0 0 14px rgba(0,0,0,0.9)",
            }}
          >
            Connect
          </span>
          <span
            className="mt-[3px] font-mono text-[9px] tracking-[0.3em]"
            style={{ color: "rgba(255,231,175,0.62)", textShadow: "0 1px 3px rgba(0,0,0,0.95)" }}
          >
            LINE
          </span>
        </button>
      </Html>
    </group>
  );
}

/* ─────────────────────────────────────────────
   軌道リング
   ───────────────────────────────────────────── */
function OrbitTrack({ radius }: { radius: number }) {
  return (
    <mesh rotation-x={Math.PI / 2}>
      <ringGeometry args={[radius - 0.012, radius + 0.012, 160]} />
      <meshBasicMaterial
        color="#6f7bd0"
        transparent
        opacity={0.16}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
}

/* ─────────────────────────────────────────────
   公転する惑星
   ───────────────────────────────────────────── */
function Planet({
  data,
  cfg,
  radiusScale,
  sizeScale,
  textures,
  reduced,
  shadows,
  focused,
  onSelect,
  onHover,
  positions,
}: {
  data: Domain;
  cfg: PlanetConfig;
  /** 画面が縦長のときは軌道を詰める（引きすぎて全部小さくなるのを避ける） */
  radiusScale: number;
  /** 天体そのものの拡大率 */
  sizeScale: number;
  textures: TextureSet | null;
  reduced: boolean;
  shadows: boolean;
  focused: boolean;
  onSelect: (key: string | null) => void;
  onHover: (v: boolean) => void;
  positions: PositionMap;
}) {
  const groupRef = useRef<THREE.Group>(null!);
  const meshRef = useRef<THREE.Mesh>(null!);
  const angleRef = useRef(cfg.phase);
  const labelRef = useRef<HTMLDivElement>(null);
  const [hovered, setHovered] = useState(false);

  // ホバー中／フォーカス中は公転を止める（リンクを狙って押せるように）
  const frozen = hovered || focused || reduced;

  useFrame((state, delta) => {
    // 左巻き（上から見て反時計回り）に公転させる
    if (!frozen) angleRef.current -= delta * cfg.speed;
    const a = angleRef.current;
    const r = cfg.radius * radiusScale;
    const x = Math.cos(a) * r;
    const z = Math.sin(a) * r;
    groupRef.current.position.set(x, 0, z);
    positions.get(data.key)?.set(x, 0, z);
    if (!reduced) meshRef.current.rotation.y -= delta * 0.22; // 自転も公転と同じ向き

    // カメラが寄ってもラベルが巨大化しないよう、drei が掛けた倍率を打ち消す
    if (labelRef.current) {
      const d = state.camera.position.distanceTo(groupRef.current.position);
      labelRef.current.style.transform = `scale(${labelCounterScale(d).toFixed(3)})`;
    }
  });

  const setHover = (v: boolean) => {
    setHovered(v);
    onHover(v);
  };
  const select = () => {
    playSfx("chime");
    onSelect(focused ? null : data.key);
  };

  const size = cfg.size * sizeScale;
  const active = hovered || focused;
  const map = textures?.skins[cfg.skin] ?? null;
  const normalMap = cfg.normalScale > 0 ? (textures?.normal ?? null) : null;
  const normalScale = useMemo(
    () => new THREE.Vector2(cfg.normalScale, cfg.normalScale),
    [cfg.normalScale]
  );

  return (
    <group ref={groupRef}>
      <group
        onPointerOver={(e) => {
          e.stopPropagation();
          setHover(true);
        }}
        onPointerOut={() => setHover(false)}
        onClick={(e) => {
          e.stopPropagation();
          select();
        }}
      >
        {/* 当たり判定を広げる透明球 */}
        <mesh visible={false}>
          <sphereGeometry args={[size * 2.2, 8, 8]} />
        </mesh>

        <mesh ref={meshRef} castShadow={shadows} receiveShadow={shadows}>
          <sphereGeometry args={[size, 64, 64]} />
          <meshStandardMaterial
            map={map}
            color={map ? "#ffffff" : cfg.accent}
            normalMap={normalMap}
            normalScale={normalScale}
            roughness={cfg.roughness}
            metalness={cfg.metalness}
            emissive={cfg.emissive ?? "#000000"}
            emissiveIntensity={cfg.emissiveIntensity ?? 0}
          />
        </mesh>

        {/* 大気層（森と海の惑星のみ） */}
        {cfg.atmosphere && (
          <mesh scale={1.05}>
            <sphereGeometry args={[size, 32, 32]} />
            <meshStandardMaterial
              color={cfg.atmosphere}
              transparent
              opacity={0.3}
              blending={THREE.AdditiveBlending}
              depthWrite={false}
            />
          </mesh>
        )}

        {/* 選択・ホバー時に灯る輪郭光 */}
        <mesh scale={active ? 1.55 : 1.28} visible={active}>
          <sphereGeometry args={[size, 24, 24]} />
          <meshBasicMaterial
            color={cfg.accent}
            transparent
            opacity={0.18}
            side={THREE.BackSide}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
          />
        </mesh>
      </group>

      <Html
        distanceFactor={HTML_DISTANCE_FACTOR}
        position={[0, size + 0.7, 0]}
        center
        zIndexRange={[30, 0]}
        style={{ pointerEvents: "auto" }}
      >
        <PlanetLabel
          rootRef={labelRef}
          data={data}
          accent={cfg.accent}
          active={active}
          focused={focused}
          onSelect={select}
          onHover={setHover}
        />
      </Html>
    </group>
  );
}

/**
 * 俯瞰位置を枠の縦横比から決める。
 * 距離を固定にすると、横長では余白だらけ・縦長では軌道がはみ出す、が両立してしまう。
 * 「最外軌道が必ず入る距離」を横と縦の両方から求め、大きい方を採る。
 */
function ResponsiveHome({
  homeRef,
  rMax,
  elevation,
}: {
  homeRef: React.RefObject<[number, number, number]>;
  rMax: number;
  elevation: number;
}) {
  const size = useThree((st) => st.size);

  useEffect(() => {
    const halfAngle = Math.tan((CAMERA_FOV * Math.PI) / 360);
    const aspect = size.width / Math.max(1, size.height);
    const need = rMax * 1.12; // ラベル用の余白
    // 横方向：halfW = halfAngle * d * aspect ≥ need
    const dW = need / (halfAngle * Math.max(0.3, aspect));
    // 縦方向：円盤は傾いて見えるので、縦の広がりは rMax * sin(仰角)
    const dH = (need * Math.sin(elevation) + 1.6) / halfAngle;
    const d = Math.max(11, dW, dH);
    homeRef.current = [0, d * Math.sin(elevation), d * Math.cos(elevation)];
  }, [size.width, size.height, rMax, elevation, homeRef]);

  return null;
}

/**
 * スマホで天球図に触るとページが縦スクロールしなくなる問題への対処。
 * camera-controls はキャンバスに touch-action: none を書き込むため、
 * 1本指の操作を NONE にしていてもブラウザ側でスクロールが殺される。
 * CameraControls より後にマウントしてキャンバスへ pan-y を上書きする
 * （setupControls 側でも同じことをしているが、あちらは private フィールド
 *   参照なので、こちらを本命の保険として置いている）。
 */
function TouchScrollGuard() {
  const gl = useThree((st) => st.gl);
  useEffect(() => {
    const el = gl.domElement;
    el.style.touchAction = "pan-y";
    // 念のため、後から書き換えられても戻す
    const ob = new MutationObserver(() => {
      if (el.style.touchAction !== "pan-y") el.style.touchAction = "pan-y";
    });
    ob.observe(el, { attributes: true, attributeFilter: ["style"] });
    return () => ob.disconnect();
  }, [gl]);
  return null;
}

/* ─────────────────────────────────────────────
   カメラ追従：選ばれた星へ寄り、解除で俯瞰へ戻る
   ───────────────────────────────────────────── */
function FocusRig({
  controlsRef,
  focusedKey,
  positions,
  home,
}: {
  controlsRef: React.RefObject<CameraControlsImpl | null>;
  focusedKey: string | null;
  positions: PositionMap;
  home: React.RefObject<[number, number, number]>;
}) {
  const tmp = useRef(new THREE.Vector3());

  useFrame(() => {
    const c = controlsRef.current;
    if (!c) return;
    if (focusedKey) {
      const p = positions.get(focusedKey);
      if (!p) return;
      const size = PLANETS[focusedKey]?.size ?? 0.6;
      const dist = 2.2 + size * 4.5;
      const dir = tmp.current.copy(p).setY(0);
      const r = dir.length() || 1;
      dir.multiplyScalar((r + dist) / r);
      c.setLookAt(dir.x, size * 1.6, dir.z, p.x, p.y, p.z, true);
    } else {
      const h = home.current;
      c.setLookAt(h[0], h[1], h[2], 0, 0, 0, true);
    }
  });

  return null;
}

/* ─────────────────────────────────────────────
   シーン本体
   ───────────────────────────────────────────── */
export default function CelestialMap3D({ paused = false }: { paused?: boolean }) {
  const [focusedKey, setFocusedKey] = useState<string | null>(null);
  const [reduced, setReduced] = useState(false);
  const [compact, setCompact] = useState(false);
  const [textures, setTextures] = useState<TextureSet | null>(null);
  const [sun, setSun] = useState<THREE.Mesh | null>(null);
  const hoverCountRef = useRef(0);
  const controlsRef = useRef<CameraControlsImpl>(null);

  useEffect(() => {
    setReduced(window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    const mq = window.matchMedia("(max-width: 640px)");
    const sync = () => setCompact(mq.matches);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);

  // テクスチャを1枚ずつ焼く。間に setTimeout を挟んでメインスレッドを固めない
  useEffect(() => {
    let cancelled = false;
    const built: THREE.Texture[] = [];

    (async () => {
      const yieldToBrowser = () => new Promise((r) => setTimeout(r, 0));
      const t0 = performance.now();
      let bakeMs = 0;
      let worstMs = 0;
      const timed = <T,>(fn: () => T): T => {
        const s = performance.now();
        const out = fn();
        const d = performance.now() - s;
        bakeMs += d;
        worstMs = Math.max(worstMs, d);
        return out;
      };

      const normal = timed(() => makeCommonNormal(7));
      built.push(normal);
      await yieldToBrowser();
      if (cancelled) return;

      const skins: Partial<Record<PlanetSkin, THREE.Texture>> = {};
      const wanted: { skin: PlanetSkin; seed: number }[] = [
        { skin: "plasma", seed: 99 },
        ...Object.values(PLANETS).map((p) => ({ skin: p.skin, seed: p.seed })),
      ];
      for (const w of wanted) {
        if (cancelled) return;
        if (skins[w.skin]) continue;
        const tex = timed(() => makeDiffuse(w.skin, w.seed));
        skins[w.skin] = tex;
        built.push(tex);
        await yieldToBrowser();
      }

      if (cancelled) return;
      if (process.env.NODE_ENV !== "production") {
        console.info(
          `[celestial] textures: bake=${Math.round(bakeMs)}ms worstBlock=${Math.round(
            worstMs
          )}ms wall=${Math.round(performance.now() - t0)}ms`,
          Object.keys(skins)
        );
      }
      setTextures({ normal, skins });
    })();

    return () => {
      cancelled = true;
      built.forEach((t) => t.dispose());
    };
  }, []);

  const handleHover = (v: boolean) => {
    hoverCountRef.current = Math.max(0, hoverCountRef.current + (v ? 1 : -1));
    document.body.style.cursor = hoverCountRef.current > 0 ? "pointer" : "";
  };
  useEffect(
    () => () => {
      document.body.style.cursor = "";
    },
    []
  );

  const center = useMemo(() => DOMAINS.find((d) => d.key === "connect")!, []);
  const planets = useMemo(
    () => DOMAINS.filter((d) => d.key !== "connect" && !d.hidden && PLANETS[d.key]),
    []
  );
  const positions = useMemo<PositionMap>(
    () => new Map(planets.map((d) => [d.key, new THREE.Vector3()])),
    [planets]
  );

  /**
   * 縦長の画面では、直径24単位の円盤を横幅に収めるためにカメラを大きく引くしかなく、
   * 結果すべてが小さく窮屈になる。モバイルでは軌道自体を詰め、
   * さらに見下ろす角度を強めて縦の余白を使う。
   */
  const radiusScale = compact ? 0.62 : 1;
  /** 惑星が点にしか見えないのを避けるため、モバイルでは天体だけ大きくする */
  const sizeScale = compact ? 1.15 : 1;
  /** 縦長のモバイルは見下ろす角度を強めて、縦の余白を使う */
  const elevation = compact ? 0.55 : 0.4;
  const rMax = 12 * radiusScale;
  const homeRef = useRef<[number, number, number]>([0, 8, 21]);
  const shadows = !compact;

  /** 余白クリックで俯瞰へ戻す。ラベル（DOM オーバーレイ）上のクリックは対象外 */
  const handleMissed = (e: MouseEvent) => {
    const t = e.target as HTMLElement | null;
    if (t && t.tagName !== "CANVAS") return;
    setFocusedKey(null);
  };

  /**
   * 縦スクロールを奪わない入力設定。
   * ホイールはページのスクロールに委ね、スマホは1本指＝ページスクロール／2本指＝回転。
   */
  const setupControls = (c: CameraControlsImpl | null) => {
    controlsRef.current = c;
    if (!c) return;
    const A = (c.constructor as unknown as { ACTION?: Record<string, number> }).ACTION;
    if (!A) return;
    const mouse = c.mouseButtons as unknown as Record<string, number>;
    const touch = c.touches as unknown as Record<string, number>;
    mouse.wheel = A.NONE;
    mouse.middle = A.NONE;
    mouse.right = A.NONE;
    touch.one = A.NONE;
    touch.two = A.TOUCH_ZOOM_ROTATE;
    touch.three = A.NONE;

    /**
     * camera-controls はキャンバスに touch-action: none を書き込む。
     * 1本指の操作を NONE にしていても、ブラウザ側でスクロール自体が殺されるため、
     * スマホで天球図に触るとページが動かなくなる。
     * 縦スクロールだけ許す pan-y に上書きする（2本指の操作は従来どおり効く）。
     */
    const el = (c as unknown as { _domElement?: HTMLElement })._domElement;
    if (el) el.style.touchAction = "pan-y";
  };

  return (
    <Canvas
      shadows={shadows}
      camera={{ position: homeRef.current, fov: CAMERA_FOV, near: 0.1, far: 400 }}
      dpr={[1, 2]}
      frameloop={paused ? "demand" : "always"}
      onPointerMissed={handleMissed}
      gl={{ antialias: true, alpha: true }}
    >
      {/* 中心星だけが照らす、劇的なライティング */}
      <ambientLight intensity={0.05} color="#050510" />
      <pointLight
        position={[0, 0, 0]}
        intensity={4}
        decay={2}
        distance={0}
        color="#ffffff"
        castShadow={shadows}
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
        shadow-camera-near={0.5}
        shadow-camera-far={40}
        shadow-bias={-0.0015}
      />
      {/* 輪郭を浮かせる青いリムライト（画面奥から） */}
      <spotLight
        position={[-14, 10, -20]}
        angle={0.6}
        penumbra={1}
        intensity={22}
        color="#5f7bff"
        distance={90}
      />

      <Stars radius={140} depth={60} count={compact ? 2500 : 6000} factor={5} saturation={0} fade speed={reduced ? 0 : 1} />

      <CoreStar
        data={center}
        textures={textures}
        reduced={reduced}
        onHover={handleHover}
        onReady={setSun}
      />

      {planets.map((d) => (
        <OrbitTrack key={`ring-${d.key}`} radius={PLANETS[d.key].radius * radiusScale} />
      ))}

      {planets.map((d) => (
        <Planet
          key={d.key}
          data={d}
          cfg={PLANETS[d.key]}
          radiusScale={radiusScale}
          sizeScale={sizeScale}
          textures={textures}
          reduced={reduced}
          shadows={shadows}
          focused={focusedKey === d.key}
          onSelect={setFocusedKey}
          onHover={handleHover}
          positions={positions}
        />
      ))}

      <FocusRig
        controlsRef={controlsRef}
        focusedKey={focusedKey}
        positions={positions}
        home={homeRef}
      />
      <ResponsiveHome homeRef={homeRef} rMax={rMax} elevation={elevation} />

      <CameraControls
        ref={setupControls}
        minDistance={2}
        maxDistance={compact ? 44 : 36}
        maxPolarAngle={Math.PI / 2.1} // 地平線より下へ回り込ませない
        smoothTime={0.65}
      />
      <TouchScrollGuard />

      {/* 光の後処理：強い光沢＋中心星からの光芒 */}
      {sun ? (
        <EffectComposer>
          <Bloom intensity={1.5} luminanceThreshold={0.9} luminanceSmoothing={0.2} mipmapBlur />
          <GodRays
            sun={sun}
            density={0.92}
            decay={0.93}
            weight={0.42}
            exposure={0.3}
            samples={compact ? 30 : 60}
            blur
          />
        </EffectComposer>
      ) : (
        <EffectComposer>
          <Bloom intensity={1.5} luminanceThreshold={0.9} luminanceSmoothing={0.2} mipmapBlur />
        </EffectComposer>
      )}
    </Canvas>
  );
}
