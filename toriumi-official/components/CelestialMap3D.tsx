"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
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
 * ・星をクリック → カメラがその星まで飛び、ホログラム UI が開いて行き先が出る
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
  accent: string; // ホログラム UI とにじむ光の色
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
const HOLO = "#7cffb2"; // ホログラム UI の発光色

type TextureSet = { normal: THREE.Texture; skins: Partial<Record<PlanetSkin, THREE.Texture>> };
type PositionMap = Map<string, THREE.Vector3>;

/* ─────────────────────────────────────────────
   ホログラム UI ラベル
   ───────────────────────────────────────────── */
function HoloLabel({
  data,
  accent,
  active,
  focused,
  onSelect,
  onHover,
}: {
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
      className="select-none"
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
    >
      <button
        onClick={onSelect}
        aria-label={data.titleEn}
        aria-expanded={focused}
        className="relative block px-3 py-1.5 text-center transition-all duration-300"
        style={{
          background: active ? "rgba(6,20,14,0.82)" : "rgba(6,14,12,0.45)",
          border: `1px solid ${active ? HOLO : "rgba(124,255,178,0.35)"}`,
          boxShadow: active
            ? `0 0 18px ${HOLO}55, inset 0 0 12px ${HOLO}22`
            : `0 0 8px ${HOLO}22`,
          backdropFilter: "blur(6px)",
        }}
      >
        {/* SF ホログラムのコーナーブラケット */}
        {[
          "left-[-3px] top-[-3px] border-l border-t",
          "right-[-3px] top-[-3px] border-r border-t",
          "left-[-3px] bottom-[-3px] border-l border-b",
          "right-[-3px] bottom-[-3px] border-r border-b",
        ].map((pos) => (
          <span
            key={pos}
            aria-hidden
            className={`pointer-events-none absolute h-1.5 w-1.5 ${pos}`}
            style={{ borderColor: HOLO }}
          />
        ))}
        <span
          className="block font-mono text-[11px] uppercase tracking-[0.2em]"
          style={{ color: active ? HOLO : "rgba(203,255,226,0.85)" }}
        >
          {data.titleEn}
        </span>
        <span
          className="mt-0.5 block text-[9px] tracking-wide"
          style={{ color: active ? `${accent}` : "rgba(203,255,226,0.5)" }}
        >
          {data.titleJp}
        </span>
      </button>

      {focused && (
        <div
          className="mt-2 flex flex-col gap-1 p-1.5"
          style={{
            background: "rgba(6,20,14,0.9)",
            border: `1px solid ${HOLO}55`,
            boxShadow: `0 0 16px ${HOLO}33`,
            backdropFilter: "blur(8px)",
          }}
        >
          <span className="px-2 pb-1 font-mono text-[8px] uppercase tracking-[0.3em] text-[rgba(124,255,178,0.55)]">
            {data.blurb}
          </span>
          {links.map((l) => {
            const internal = l.href.startsWith("/");
            const cls =
              "whitespace-nowrap px-2.5 py-1 text-left font-mono text-[11px] tracking-wide transition-colors";
            const style = { color: "rgba(203,255,226,0.9)" };
            // 内部ページ（App / Website 一覧）は Next のクライアントルーティングで遷移
            return internal ? (
              <Link key={l.href} href={l.href} className={cls} style={style}>
                ▸ {l.label}
              </Link>
            ) : (
              <a
                key={l.href}
                href={l.href}
                target="_blank"
                rel="noopener noreferrer"
                className={cls}
                style={style}
              >
                ▸ {l.label}
              </a>
            );
          })}
        </div>
      )}
    </div>
  );
}

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
  const [hovered, setHovered] = useState(false);
  const map = textures?.skins.plasma ?? null;

  useFrame((state, delta) => {
    if (!reduced) meshRef.current.rotation.y += delta * 0.045;
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
          emissiveIntensity={2.5}
          roughness={0.9}
          metalness={0.1}
          toneMapped={false}
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

      <Html distanceFactor={16} position={[0, CORE_SIZE + 1.1, 0]} center zIndexRange={[30, 0]}>
        <button
          onClick={open}
          className="whitespace-nowrap px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.25em] backdrop-blur-sm transition-colors"
          style={{
            color: "#ffe6a8",
            background: "rgba(28,16,4,0.55)",
            border: "1px solid rgba(255,200,110,0.45)",
            boxShadow: "0 0 16px rgba(255,180,71,0.35)",
          }}
        >
          ✦ Connect
          <span className="ml-2 normal-case tracking-normal opacity-60">LINE</span>
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
  const [hovered, setHovered] = useState(false);

  // ホバー中／フォーカス中は公転を止める（リンクを狙って押せるように）
  const frozen = hovered || focused || reduced;

  useFrame((_, delta) => {
    if (!frozen) angleRef.current += delta * cfg.speed;
    const a = angleRef.current;
    const x = Math.cos(a) * cfg.radius;
    const z = Math.sin(a) * cfg.radius;
    groupRef.current.position.set(x, 0, z);
    positions.get(data.key)?.set(x, 0, z);
    if (!reduced) meshRef.current.rotation.y += delta * 0.22; // 自転
  });

  const setHover = (v: boolean) => {
    setHovered(v);
    onHover(v);
  };
  const select = () => {
    playSfx("chime");
    onSelect(focused ? null : data.key);
  };

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
          <sphereGeometry args={[cfg.size * 2.2, 8, 8]} />
        </mesh>

        <mesh ref={meshRef} castShadow={shadows} receiveShadow={shadows}>
          <sphereGeometry args={[cfg.size, 64, 64]} />
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
            <sphereGeometry args={[cfg.size, 32, 32]} />
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
          <sphereGeometry args={[cfg.size, 24, 24]} />
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
        distanceFactor={16}
        position={[0, cfg.size + 0.7, 0]}
        center
        zIndexRange={[30, 0]}
        style={{ pointerEvents: "auto" }}
      >
        <HoloLabel
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
  home: [number, number, number];
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
      c.setLookAt(home[0], home[1], home[2], 0, 0, 0, true);
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

  const home: [number, number, number] = compact ? [0, 10, 30] : [0, 8, 22];
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
  };

  return (
    <Canvas
      shadows={shadows}
      camera={{ position: home, fov: 45, near: 0.1, far: 400 }}
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
        <OrbitTrack key={`ring-${d.key}`} radius={PLANETS[d.key].radius} />
      ))}

      {planets.map((d) => (
        <Planet
          key={d.key}
          data={d}
          cfg={PLANETS[d.key]}
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
        home={home}
      />

      <CameraControls
        ref={setupControls}
        minDistance={2}
        maxDistance={compact ? 44 : 36}
        maxPolarAngle={Math.PI / 2.1} // 地平線より下へ回り込ませない
        smoothTime={0.65}
      />

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
