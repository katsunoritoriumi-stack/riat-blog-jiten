"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { DOMAINS, type Domain } from "@/lib/content";
import { playSfx } from "@/lib/sfx";
import {
  makeClouds,
  makeCommonNormal,
  makeDiffuse,
  makeRing,
  makeRoughness,
  type PlanetSkin,
} from "@/lib/planetTextures";
import { makeStarSprite } from "@/lib/spaceTextures";
import { STATIONS, STATION_BANDS } from "@/lib/stations";
import { depthOf, flight } from "./flightState";
import { makeRingGeometry } from "./ringGeometry";
import { bakeOnce, diffuseKey } from "./textureCache";

/**
 * 創造の座標軸 — 旅の途中の、ある領域に浮かんでいる太陽系。
 *
 * カードの中の 3D ではなく、背景の宇宙と同じ空間に置いてある。
 * スクロールで近づくと奥から現れ、滞在中は目の前に浮かび、やがて脇を通り過ぎる。
 *
 * 回廊は 1 画面あたり約 150 単位なので、太陽系もその尺度まで拡大する
 * （素の半径 12 のままだと一瞬で通り過ぎて何も見えない）。
 */

type Ring = { inner: number; outer: number; tilt: number };

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
  clouds?: boolean; // 独立した雲のシェルを纏う
  ring?: Ring;
  accent: string;
};

/** lib/content.ts の 6 ドメインに質感を割り当てる */
const PLANETS: Record<string, PlanetConfig> = {
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
  youtube: {
    skin: "gas",
    seed: 22,
    radius: 5.6,
    speed: 0.24,
    size: 0.95,
    phase: 2.1,
    roughness: 0.55,
    metalness: 0.05,
    normalScale: 0,
    ring: { inner: 1.45, outer: 2.35, tilt: 0.42 },
    accent: "#ff8a4c",
  },
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
  sns: {
    skin: "living",
    seed: 55,
    radius: 10.4,
    speed: 0.13,
    size: 0.78,
    phase: 3.1,
    roughness: 0.8,
    metalness: 0.1,
    normalScale: 1.2,
    clouds: true,
    accent: "#9ff0c0",
  },
  radio: {
    skin: "ice",
    seed: 66,
    radius: 12.0,
    speed: 0.11,
    size: 0.66,
    phase: 5.4,
    roughness: 0.7,
    metalness: 0.15,
    normalScale: 1.0,
    ring: { inner: 1.5, outer: 2.0, tilt: 1.32 },
    accent: "#9cc6ff",
  },
};

const CORE_SIZE = 1.7;
/** 回廊の尺度に合わせる拡大率。太陽系の外周（半径 12）が 108 単位になる */
const SYS_SCALE = 9;
/** 滞在の終わりでカメラがここまで近づく（ワールド単位） */
const HOLD_DIST = 200;
/** 太陽の正面に突っ込まないよう、系をわずかに下へずらす */
const SYS_OFFSET_Y = -14;
/**
 * 軌道面をま横から見ると、手前に来た惑星が必ず影側を向いて
 * 太陽の前の黒い塊になる（物理的には正しいが絵として損）。
 * 少し上から見下ろす角度にして、軌道が楕円に見えるようにする。
 */
const SYS_TILT: [number, number, number] = [-0.36, 0, 0.07];

const UNI_INDEX = STATIONS.findIndex((s) => s.id === "universe");

/** 太陽系の中心が置かれる深度 */
function systemDepth() {
  const b = STATION_BANDS[UNI_INDEX];
  return depthOf(b.start + b.span * 0.34) + HOLD_DIST;
}

/** リムを落として、明暗境界を締める（惑星の縁が硬く見えるのを防ぐ） */
function limbDarkening(shader: { fragmentShader: string }) {
  shader.fragmentShader = shader.fragmentShader.replace(
    "#include <emissivemap_fragment>",
    `#include <emissivemap_fragment>
     float ndv = clamp(abs(dot(normalize(vNormal), normalize(vViewPosition))), 0.0, 1.0);
     totalEmissiveRadiance *= mix(0.25, 1.0, pow(ndv, 0.8));`
  );
}

type TextureSet = {
  diffuse: Record<string, THREE.Texture>;
  normal: THREE.Texture;
  roughness: THREE.Texture;
  clouds: THREE.Texture;
  ring: THREE.Texture;
};

/**
 * テクスチャは 1 枚ずつ、間を空けて焼く。
 * まとめて焼くとメインスレッドが 1.6 秒固まり、スクロールが引っかかる。
 */
function useTextures(active: boolean): TextureSet | null {
  const [set, setSet] = useState<TextureSet | null>(null);
  const started = useRef(false);

  useEffect(() => {
    if (!active || started.current) return;
    started.current = true;
    let alive = true;

    (async () => {
      const diffuse: Record<string, THREE.Texture> = {};
      const skins = [...new Set(Object.values(PLANETS).map((p) => p.skin))];
      for (const skin of skins) {
        const seed = Object.values(PLANETS).find((p) => p.skin === skin)!.seed;
        diffuse[skin] = await bakeOnce(diffuseKey(skin, seed), () => makeDiffuse(skin, seed));
      }
      diffuse.plasma = await bakeOnce(diffuseKey("plasma", 3), () => makeDiffuse("plasma", 3));
      const normal = await bakeOnce("normal:101", () => makeCommonNormal(101));
      const roughness = await bakeOnce("rough:living:55", () => makeRoughness("living", 55));
      const clouds = await bakeOnce("clouds:154", () => makeClouds(154));
      const ring = await bakeOnce("ring:31", () => makeRing(31));
      if (alive) setSet({ diffuse, normal, roughness, clouds, ring });
    })();

    return () => {
      alive = false;
    };
  }, [active]);

  return set;
}

/* ── 中心星 ───────────────────────────────── */

function CoreStar({ map }: { map: THREE.Texture | undefined }) {
  const ref = useRef<THREE.Mesh>(null);
  // グロー用の柔らかい円。map を付けないと spriteMaterial はただの四角い板になる
  const glow = useMemo(() => makeStarSprite(false), []);
  useEffect(() => () => glow.dispose(), [glow]);

  useFrame((_, dt) => {
    if (ref.current) ref.current.rotation.y -= dt * 0.05;
  });
  return (
    <group>
      <mesh ref={ref}>
        <sphereGeometry args={[CORE_SIZE, 48, 48]} />
        <meshStandardMaterial
          map={map}
          emissiveMap={map}
          emissive="#ffb347"
          emissiveIntensity={2.6}
          color="#20100a"
          roughness={1}
          metalness={0}
        />
      </mesh>
      {/* にじむ光。板1枚で足りる（後処理は使わない＝モバイルでも軽い） */}
      <sprite scale={[CORE_SIZE * 11, CORE_SIZE * 11, 1]}>
        <spriteMaterial
          map={glow}
          color="#ffca7a"
          opacity={0.85}
          transparent
          depthWrite={false}
          depthTest={false}
          blending={THREE.AdditiveBlending}
        />
      </sprite>
      {/*
        群れごと SYS_SCALE 倍しているので、惑星までの実距離は 28〜84 単位になる。
        decay=2 のままだと届かず、惑星が真っ黒になる（実際になった）。
        decay=1 にして、内側ほど明るい自然な落ち方だけ残す。
      */}
      <pointLight position={[0, 0, 0]} intensity={110} decay={1} distance={0} color="#fff2e0" />
    </group>
  );
}

/* ── 惑星 ─────────────────────────────────── */

function Planet({
  domain,
  cfg,
  tex,
  onPick,
}: {
  domain: Domain;
  cfg: PlanetConfig;
  tex: TextureSet;
  onPick: (d: Domain) => void;
}) {
  const orbit = useRef<THREE.Group>(null);
  const body = useRef<THREE.Mesh>(null);
  const cloud = useRef<THREE.Mesh>(null);
  const angle = useRef(cfg.phase);
  const [hover, setHover] = useState(false);
  const ringGeo = useMemo(
    () =>
      cfg.ring ? makeRingGeometry(cfg.size * cfg.ring.inner, cfg.size * cfg.ring.outer, 96) : null,
    [cfg]
  );
  useEffect(() => () => ringGeo?.dispose(), [ringGeo]);

  useFrame((_, dt) => {
    // 左巻き（反時計回りではなく時計回り）に統一する
    angle.current -= dt * cfg.speed;
    const g = orbit.current;
    if (g) {
      g.position.x = Math.cos(angle.current) * cfg.radius;
      g.position.z = Math.sin(angle.current) * cfg.radius;
    }
    if (body.current) body.current.rotation.y -= dt * 0.25;
    if (cloud.current) cloud.current.rotation.y -= dt * 0.34;
  });

  const map = tex.diffuse[cfg.skin];

  return (
    <group ref={orbit}>
      <group
        onPointerOver={(e) => {
          e.stopPropagation();
          setHover(true);
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={() => {
          setHover(false);
          document.body.style.cursor = "";
        }}
        onClick={(e) => {
          e.stopPropagation();
          onPick(domain);
        }}
      >
        {/* 当たり判定を広げる透明球 */}
        <mesh visible={false}>
          <sphereGeometry args={[cfg.size * 2.4, 8, 8]} />
        </mesh>

        <mesh ref={body}>
          <sphereGeometry args={[cfg.size, 48, 48]} />
          <meshStandardMaterial
            map={map}
            normalMap={cfg.normalScale > 0 ? tex.normal : undefined}
            normalScale={
              cfg.normalScale > 0
                ? new THREE.Vector2(cfg.normalScale, cfg.normalScale)
                : undefined
            }
            roughnessMap={cfg.skin === "living" ? tex.roughness : undefined}
            roughness={cfg.roughness}
            metalness={cfg.metalness}
            emissive={cfg.emissive ?? "#000000"}
            emissiveIntensity={cfg.emissiveIntensity ?? 0}
            onBeforeCompile={limbDarkening}
          />
        </mesh>

        {/* 雲のシェル。本体と違う速度で回すと一気に生きている星に見える */}
        {cfg.clouds && (
          <mesh ref={cloud}>
            <sphereGeometry args={[cfg.size * 1.022, 40, 40]} />
            <meshStandardMaterial
              map={tex.clouds}
              transparent
              opacity={0.92}
              depthWrite={false}
              roughness={1}
              metalness={0}
            />
          </mesh>
        )}

        {/* 環 */}
        {cfg.ring && ringGeo && (
          <mesh geometry={ringGeo} rotation={[Math.PI / 2 - cfg.ring.tilt, 0, 0.3]}>
            <meshBasicMaterial
              map={tex.ring}
              side={THREE.DoubleSide}
              transparent
              depthWrite={false}
            />
          </mesh>
        )}
      </group>

      {/* 名前は画面上で一定の大きさに保つ（距離で拡大させない） */}
      <Html
        position={[0, cfg.size * 1.9, 0]}
        center
        zIndexRange={[8, 0]}
        style={{ pointerEvents: "none" }}
      >
        <div
          className="whitespace-nowrap font-mono uppercase"
          style={{
            fontSize: 10,
            letterSpacing: "0.24em",
            color: hover ? "#fff" : cfg.accent,
            textShadow: "0 1px 10px rgba(0,0,0,0.95)",
            opacity: hover ? 1 : 0.82,
            transition: "opacity 200ms, color 200ms",
          }}
        >
          {domain.titleEn}
        </div>
      </Html>
    </group>
  );
}

/** 軌道の輪 */
function OrbitRing({ radius }: { radius: number }) {
  const geo = useMemo(() => {
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i <= 128; i++) {
      const a = (i / 128) * Math.PI * 2;
      pts.push(new THREE.Vector3(Math.cos(a) * radius, 0, Math.sin(a) * radius));
    }
    return new THREE.BufferGeometry().setFromPoints(pts);
  }, [radius]);
  useEffect(() => () => geo.dispose(), [geo]);
  return (
    <primitive
      object={
        new THREE.Line(
          geo,
          new THREE.LineBasicMaterial({
            color: "#8b7cf0",
            transparent: true,
            opacity: 0.16,
            depthWrite: false,
          })
        )
      }
    />
  );
}

/* ── 本体 ─────────────────────────────────── */

export default function SolarSystem({ onPick }: { onPick: (d: Domain) => void }) {
  const root = useRef<THREE.Group>(null);
  const [near, setNear] = useState(false);
  const D = useMemo(systemDepth, []);
  const tex = useTextures(near);

  const domains = useMemo(() => DOMAINS.filter((d) => !d.hidden && PLANETS[d.key]), []);

  /**
   * 近づいたら出す・通り過ぎたら消す。
   * 太陽系は空間に固定されているので、見えている区間だけ描いて負荷を捨てる。
   */
  useFrame(() => {
    const g = root.current;
    if (!g) return;
    const dz = D - flight.depth; // カメラから見た前方距離
    // 遠すぎる／通り過ぎた区間は丸ごと描かない
    const on = dz < 1600 && dz > -300;
    if (on !== g.visible) g.visible = on;
    // 到達のかなり手前でテクスチャを焼き始める（間に合わせるため）
    if (!near && dz < 2600) setNear(true);
  });

  return (
    <group
      ref={root}
      position={[0, SYS_OFFSET_Y, -D]}
      rotation={SYS_TILT}
      scale={SYS_SCALE}
      visible={false}
    >
      {tex && (
        <>
          {/*
            手前側が潰れないようにする補助光は、シーン側の「遠い星明かり」
            （components/universe/Scene.tsx の directionalLight）が兼ねる。
            three.js のライトはカメラのレイヤーでしか絞れず、オブジェクト単位で
            当てるライトを分けられないため、飛来天体と共用にしてある。
          */}
          <CoreStar map={tex.diffuse.plasma} />
          {domains.map((d) => (
            <OrbitRing key={`o-${d.key}`} radius={PLANETS[d.key].radius} />
          ))}
          {domains.map((d) => (
            <Planet key={d.key} domain={d} cfg={PLANETS[d.key]} tex={tex} onPick={onPick} />
          ))}
        </>
      )}
    </group>
  );
}

/** 星をクリックしたときの既定の動き（外部リンクを開く） */
export function openDomain(d: Domain) {
  playSfx("chime");
  if (d.href) window.open(d.href, "_blank", "noopener,noreferrer");
}
