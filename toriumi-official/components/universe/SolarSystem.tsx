"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { DOMAINS, type Domain } from "@/lib/content";
import { setPickedDomain } from "@/lib/domainPick";
import { smoothstep } from "@/lib/flightMath";
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
import { CORRIDOR, depthOf, flight } from "./flightState";
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
  /**
   * 滞在中に着く定位置（60 度刻みのスロット番号 0-5）。
   * 自由に回らせたままだと、たまたま同じ方向に並んだ惑星が画面上で重なる。
   * 近づいたらこの並びへ寄せて、どの回転角でも重ならないようにする。
   * 割り当ては scratchpad の総当たり（全 720 通り × 全回転角）で決めた。
   */
  slot: number;
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
    slot: 1,
    roughness: 0.85,
    metalness: 0.05,
    normalScale: 2.5,
    accent: "#e8a06a",
  },
  youtube: {
    skin: "gas",
    seed: 22,
    radius: 6.0,
    speed: 0.24,
    size: 0.95,
    phase: 2.1,
    slot: 4,
    roughness: 0.55,
    metalness: 0.05,
    normalScale: 0,
    // 環は軌道の間隔より広くできない（隣の惑星に触れる）。外周を 2.35→2.05 に詰めた
    ring: { inner: 1.45, outer: 2.05, tilt: 0.42 },
    accent: "#ff8a4c",
  },
  fashion: {
    skin: "marble",
    seed: 33,
    radius: 8.0,
    speed: 0.19,
    size: 0.6,
    phase: 4.0,
    slot: 0,
    roughness: 0.9,
    metalness: 0.05,
    normalScale: 1.8,
    accent: "#d8c2cc",
  },
  produce: {
    skin: "cratered",
    seed: 44,
    radius: 9.6,
    speed: 0.155,
    size: 0.72,
    phase: 0.9,
    slot: 2,
    roughness: 0.95,
    metalness: 0.05,
    normalScale: 3.0,
    accent: "#bfae94",
  },
  sns: {
    skin: "living",
    seed: 55,
    radius: 11.2,
    speed: 0.13,
    size: 0.78,
    phase: 3.1,
    slot: 3,
    roughness: 0.8,
    metalness: 0.1,
    normalScale: 1.2,
    clouds: true,
    accent: "#9ff0c0",
  },
  radio: {
    skin: "ice",
    seed: 66,
    radius: 12.8,
    speed: 0.11,
    size: 0.66,
    phase: 5.4,
    slot: 5,
    roughness: 0.7,
    metalness: 0.15,
    normalScale: 1.0,
    ring: { inner: 1.5, outer: 2.0, tilt: 1.32 },
    accent: "#9cc6ff",
  },
};

const CORE_SIZE = 1.7;
/** 回廊の尺度に合わせる拡大率の上限。横に広い画面（PC）ではこの値のまま */
const SYS_SCALE = 9;
/** 滞在の 0.34 の地点でカメラがここまで近づく（ワールド単位） */
const HOLD_DIST = 200;
/** 系の中心を少し下げる量（局所単位。ワールドでは ×scale）。太陽の正面に突っ込まないため */
const OFFSET_Y = 1.5556;
/**
 * 軌道面をま横から見ると、手前に来た惑星が必ず影側を向いて
 * 太陽の前の黒い塊になる（物理的には正しいが絵として損）。
 * 少し上から見下ろす角度にして、軌道が楕円に見えるようにする。
 *
 * 縦長の画面（スマホ）では、この角度が浅いと軌道の奥半分が細い帯に潰れ、
 * 半径の違う惑星どうしが画面上で近づいてしまう。縦には余裕があるので、
 * 縦長になるほど見下ろす角度を強くする。
 */
const TILT_WIDE = -0.36;
const TILT_TALL = -0.78;
/** 系の見た目のいちばん外側（環を含む局所半径）。画面に収める計算に使う */
const R_OUTER = Math.max(
  ...Object.values(PLANETS).map((p) => p.radius + p.size * (p.ring?.outer ?? 1))
);

const UNI_INDEX = STATIONS.findIndex((s) => s.id === "universe");

/**
 * 毎フレーム読む共有値（React の再描画を挟みたくない）。
 * slow: 公転の遅さ。遠いと 1、目の前で 0.16 まで落ちる。
 *       回っている惑星は狙って押しにくく、スマホはホバーで止められないため。
 * pull: 定位置（slot）へ寄せる強さ 0-1。
 * spin: 系全体のゆっくりした回転。並びを保ったまま生きている感じを残す。
 * tilt: いまの見下ろし角。ラベルを画面の真上に出すために使う。
 */
const sysView = { slow: 1, pull: 0, spin: 0, tilt: TILT_WIDE };

/** 系全体の回転（rad/s）。並びは崩れないので、どの角度でも重ならない */
const SLOT_SPIN = 0.035;

/** 太陽系の中心が置かれる深度 */
function systemDepth() {
  const b = STATION_BANDS[UNI_INDEX];
  return depthOf(b.start + b.span * 0.34) + HOLD_DIST;
}

/** 太陽系が浮かんでいる深度（固定値）。通過天体側がここを避けるために参照する */
export const SYSTEM_DEPTH = systemDepth();

/**
 * 画面に収める基準の距離。
 * 滞在中はスクロールにつれて近づき、滞在の終わり（ZoomStage の HOLD_TO = 0.42）で
 * いちばん大きく見える。そこで収まっていれば滞在中ずっと収まる。
 */
function fitDistance() {
  const b = STATION_BANDS[UNI_INDEX];
  return HOLD_DIST - (0.42 - 0.34) * b.span * CORRIDOR;
}

/**
 * 画面の縦横比から、見下ろし角と拡大率を決める。
 *
 * 縦長の画面では、外周（半径 12.8＋環）がそのまま画面幅を食う。
 * PC の 9 倍のままだと外側の惑星が左右へはみ出して消えるので、
 * 収まるところまで縮める。横に広い画面では上限の SYS_SCALE で頭打ちになり、
 * これまでと同じ見え方のまま変わらない。
 */
export function viewFor(aspect: number, fov: number) {
  const tilt = TILT_WIDE + (TILT_TALL - TILT_WIDE) * smoothstep(1.15, 0.62, aspect);
  const halfV = Math.tan((fov * Math.PI) / 360) * fitDistance();
  const halfH = halfV * aspect;
  // 横：外周がそのまま効く。縦：傾けた分だけ潰れ、中心を下げた分を足す
  const byW = (halfH * 0.96) / R_OUTER;
  const byH = (halfV * 0.96) / (R_OUTER * Math.abs(Math.sin(tilt)) + OFFSET_Y);
  return { tilt, scale: Math.min(SYS_SCALE, byW, byH) };
}

/**
 * 当たり判定の半径（局所単位）。
 * 画面が狭いと系ごと縮むので、そのぶん判定を広げて指で押せる大きさを保つ。
 */
export function hitRadius(size: number, scale: number) {
  return Math.max(size * 3.4, Math.min(3.2, 2.1 * (SYS_SCALE / scale)));
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
      {/*
        にじむ光。板1枚で足りる（後処理は使わない＝モバイルでも軽い）。

        以前は半径 9.35（＝最内周の軌道 4.0 より大きい）を depthTest なしで
        描いていたため、いちばん内側の惑星が常に光に塗り潰され、
        太陽の縁にできた「こぶ」にしか見えなかった。
        大きさを内周に近いところまで絞り、深度判定も戻す。
        これで手前を通る惑星は光の上にはっきり出て、
        向こう側へ回った惑星だけがコロナに沈む（本来の見え方）。
      */}
      <sprite scale={[CORE_SIZE * 7, CORE_SIZE * 7, 1]}>
        <spriteMaterial
          map={glow}
          color="#ffca7a"
          opacity={0.85}
          transparent
          depthWrite={false}
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
  scale,
  onPick,
}: {
  domain: Domain;
  cfg: PlanetConfig;
  tex: TextureSet;
  scale: number;
  onPick: (d: Domain) => void;
}) {
  const orbit = useRef<THREE.Group>(null);
  const body = useRef<THREE.Mesh>(null);
  const cloud = useRef<THREE.Mesh>(null);
  const tag = useRef<THREE.Group>(null);
  const angle = useRef(cfg.phase);
  const [hover, setHover] = useState(false);
  const ringGeo = useMemo(
    () =>
      cfg.ring ? makeRingGeometry(cfg.size * cfg.ring.inner, cfg.size * cfg.ring.outer, 96) : null,
    [cfg]
  );
  useEffect(() => () => ringGeo?.dispose(), [ringGeo]);

  useFrame((_, dt) => {
    /**
     * 遠くにいる間は、それぞれの速さで自由に回る（生きている系に見せる）。
     * 近づくにつれて定位置（60 度刻みのスロット）へ寄せていき、
     * 見ている間は必ず 6 つが等間隔に開いた状態になる。
     * 系そのものはゆっくり回り続けるので、止まって見えることはない。
     */
    const pull = sysView.pull;
    angle.current -= dt * cfg.speed * (hover ? 0 : sysView.slow) * (1 - pull);
    if (pull > 0.001) {
      const target = sysView.spin + (cfg.slot * Math.PI) / 3;
      // 最短回りで寄せる（逆方向へ大回りしない）
      let d = (target - angle.current) % (Math.PI * 2);
      if (d > Math.PI) d -= Math.PI * 2;
      if (d < -Math.PI) d += Math.PI * 2;
      angle.current += d * (1 - Math.exp(-dt * 2.4 * pull));
    }

    const g = orbit.current;
    if (g) {
      g.position.x = Math.cos(angle.current) * cfg.radius;
      g.position.z = Math.sin(angle.current) * cfg.radius;
    }
    if (body.current) body.current.rotation.y -= dt * 0.25;
    if (cloud.current) cloud.current.rotation.y -= dt * 0.34;
    // 名前は必ず惑星の真上に出す（系の傾きを打ち消す）
    if (tag.current) tag.current.rotation.x = -sysView.tilt;
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
        {/* 当たり判定を広げる透明球。指で押すので本体よりかなり大きく取る */}
        <mesh visible={false}>
          <sphereGeometry args={[hitRadius(cfg.size, scale), 10, 10]} />
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

      {/*
        名前は画面上で一定の大きさに保つ（距離で拡大させない）。
        回っている球を指で狙うのは難しいので、このラベル自体も押せるようにして
        大きな的を用意する（スマホではこちらが主な入口になる）。
      */}
      <group ref={tag}>
        <Html position={[0, cfg.size * 1.9 + 0.35, 0]} center zIndexRange={[8, 0]}>
          <button
            type="button"
            aria-label={`${domain.titleEn} を開く`}
            onPointerEnter={() => setHover(true)}
            onPointerLeave={() => setHover(false)}
            onClick={(e) => {
              e.stopPropagation();
              onPick(domain);
            }}
            className="whitespace-nowrap font-mono uppercase"
            style={{
              // 指で押せる大きさの余白を持たせる（見た目は文字だけ）
              padding: "10px 14px",
              margin: "-10px -14px",
              background: "none",
              border: 0,
              cursor: "pointer",
              fontSize: 10,
              letterSpacing: "0.24em",
              color: hover ? "#fff" : cfg.accent,
              textShadow: "0 1px 10px rgba(0,0,0,0.95)",
              opacity: hover ? 1 : 0.82,
              transition: "opacity 200ms, color 200ms",
              touchAction: "manipulation",
            }}
          >
            {domain.titleEn}
          </button>
        </Html>
      </group>
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

export default function SolarSystem({
  fov,
  onPick,
}: {
  fov: number;
  onPick: (d: Domain) => void;
}) {
  const root = useRef<THREE.Group>(null);
  const [near, setNear] = useState(false);
  const D = useMemo(systemDepth, []);
  const tex = useTextures(near);
  const size = useThree((s) => s.size);

  // 画面の縦横比が変わったときだけ計算し直す（毎フレームは要らない）
  const view = useMemo(
    () => viewFor(size.width / Math.max(1, size.height), fov),
    [size.width, size.height, fov]
  );
  sysView.tilt = view.tilt;

  const domains = useMemo(() => DOMAINS.filter((d) => !d.hidden && PLANETS[d.key]), []);

  /**
   * 近づいたら出す・通り過ぎたら消す。
   * 太陽系は空間に固定されているので、見えている区間だけ描いて負荷を捨てる。
   */
  useFrame((_, dt) => {
    const g = root.current;
    if (!g) return;
    const dz = D - flight.depth; // カメラから見た前方距離
    // 遠すぎる／通り過ぎた区間は丸ごと描かない
    const on = dz < 1600 && dz > -300;
    if (on !== g.visible) g.visible = on;
    // 目の前に来たら公転をゆっくりにする（狙って押せるように）
    sysView.slow = 1 - 0.84 * smoothstep(760, 260, dz);
    // 近づくにつれて定位置へ寄せる（重ならない並びを保証する）
    sysView.pull = smoothstep(1150, 520, dz);
    sysView.spin -= dt * SLOT_SPIN;
    // 到達のかなり手前でテクスチャを焼き始める（間に合わせるため）
    if (!near && dz < 2600) setNear(true);
  });

  return (
    <group
      ref={root}
      position={[0, -OFFSET_Y * view.scale, -D]}
      rotation={[view.tilt, 0, 0.07]}
      scale={view.scale}
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
            <Planet
              key={d.key}
              domain={d}
              cfg={PLANETS[d.key]}
              tex={tex}
              scale={view.scale}
              onPick={onPick}
            />
          ))}
        </>
      )}
    </group>
  );
}

/**
 * 星をクリックしたときの動き。
 *
 * 行き先がひとつなら直接開く。複数ある星（Fashion / Work / SNS）は
 * 開きようがないので、DOM 側（ConstellationMap）に一覧を出してもらう。
 * ここを href だけ見る実装にしていたため、複数リンクの星が
 * 押しても何も起きない状態になっていた。
 */
export function openDomain(d: Domain) {
  playSfx("chime");
  const links = d.links ?? [];
  if (links.length > 1) {
    setPickedDomain(d.key);
    return;
  }
  const href = d.href ?? links[0]?.href;
  if (!href) return;
  setPickedDomain(null);
  if (href.startsWith("/") || href.startsWith("#")) window.location.href = href;
  else window.open(href, "_blank", "noopener,noreferrer");
}
