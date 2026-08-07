"use client";

import { useEffect, useMemo, useRef } from "react";
import { advance, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { clamp01, smoothstep } from "@/lib/flightMath";
import { STATIONS, STATION_BANDS } from "@/lib/stations";
import { makeMilkyBand, makeNebulaSprite, makeStarSprite, NEBULA_PALETTE } from "@/lib/spaceTextures";
import Flybys from "./Flybys";
import { CORRIDOR, depthOf, flight, RUSH_FULL } from "./flightState";
import SolarSystem, { openDomain } from "./SolarSystem";
import { makeStarLayer, pointSizeBase, TUBE_LEN } from "./starfield";

export type Quality = {
  coarse: boolean;
  stars: number;
  bright: number;
  nebulaLayers: number;
  fov: number;
};

/* ─────────────────────────────────────────────
   進行役：スクロール → 深度
   ───────────────────────────────────────────── */

/**
 * ページのスクロール量をそのまま「どこまで奥へ来たか」に変換する。
 *
 * scroll イベントにも lenis にも依存せず、毎フレーム window.scrollY を直に読む。
 * lenis は prefers-reduced-motion 下では存在しないため、購読すると環境によって止まる。
 */
function Driver({ fov }: { fov: number }) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;
  const prev = useRef(0);

  useFrame((_, dt) => {
    if (flight.locked) return; // 検証用スナップショット中
    const doc = document.documentElement;
    const max = doc.scrollHeight - window.innerHeight;
    const prog = max > 0 ? clamp01(window.scrollY / max) : 0;
    step(camera, prog, Math.min(dt, 0.1), prev, fov);
  });

  return null;
}

/** 1 フレーム進める。スナップショット（検証用）からも同じ関数を呼ぶ */
function step(
  camera: THREE.PerspectiveCamera,
  prog: number,
  dt: number,
  prev: { current: number },
  fov: number
) {
  flight.prog = prog;
  const target = depthOf(prog);
  // 臨界減衰ぎみに追従。止まればぴたりと目標に落ち着く
  const k = 1 - Math.exp(-dt * 9);
  flight.depth += (target - flight.depth) * k;

  const moved = flight.depth - prev.current;
  prev.current = flight.depth;
  const inst = dt > 0 ? Math.abs(moved) / dt : 0;
  // 速度は平滑化（1フレームだけ跳ねるとチカチカする）
  flight.speed += (inst - flight.speed) * Math.min(1, dt * 8);
  flight.rush = clamp01(flight.speed / RUSH_FULL);

  camera.position.z = -flight.depth;
  // 速いほど広角に。放射状に流れる感じが強くなる
  const wanted = fov * (1 + flight.rush * 0.16);
  if (Math.abs(camera.fov - wanted) > 0.02) {
    camera.fov = wanted;
    camera.updateProjectionMatrix();
  }
}

/* ─────────────────────────────────────────────
   星の回廊
   ───────────────────────────────────────────── */

function Stars({ quality }: { quality: Quality }) {
  const groupRef = useRef<THREE.Group>(null);
  const size = useThree((s) => s.size);

  const layers = useMemo(() => {
    const plain = makeStarSprite(false);
    const spiky = makeStarSprite(true);
    return [
      makeStarLayer({
        count: quality.stars,
        seed: 8123,
        sprite: plain,
        // 世界半径。画面上の大きさは uSize*aSize/距離 で、上限は maxSize で切る。
        // 小さくしすぎると遠くの星が 0.5px になって消える（実際に消えた）
        sizeRange: [0.35, 1.5],
        tempBias: 0.42,
        withStreaks: true,
        maxSize: 10,
      }),
      makeStarLayer({
        count: quality.bright,
        seed: 5501,
        sprite: spiky,
        sizeRange: [2.6, 7.5],
        tempBias: 0.5,
        withStreaks: false,
        maxSize: 62,
      }),
    ];
  }, [quality.stars, quality.bright]);

  useEffect(() => () => layers.forEach((l) => l.dispose()), [layers]);

  useFrame(() => {
    const g = groupRef.current;
    if (g) g.position.z = -flight.depth; // 筒はカメラに付いてくる
    const base = pointSizeBase(size.height, 60);
    // 速度に応じたストリーク長（ワールド単位）
    const streak = Math.min(70, flight.speed * 0.035);
    for (const l of layers) {
      const u = l.material.uniforms;
      u.uDepth.value = flight.depth;
      u.uSize.value = base;
      u.uStreak.value = streak;
      u.uStreakFade.value = smoothstep(2, 26, streak) * 0.55;
      l.lines.visible = streak > 2.5;
    }
  });

  return (
    <group ref={groupRef}>
      {layers.map((l, i) => (
        <group key={i}>
          <primitive object={l.points} />
          <primitive object={l.lines} />
        </group>
      ))}
    </group>
  );
}

/* ─────────────────────────────────────────────
   天の川（遠景）
   ───────────────────────────────────────────── */

function MilkyWay() {
  const ref = useRef<THREE.Mesh>(null);
  const tex = useMemo(() => makeMilkyBand(9), []);
  useEffect(() => () => tex.dispose(), [tex]);

  useFrame(() => {
    const m = ref.current;
    if (m) m.position.z = -flight.depth; // 遠景なので永遠に近づかない
  });

  return (
    <mesh ref={ref} rotation={[0.22, 0.6, 0.35]} renderOrder={-1}>
      <sphereGeometry args={[2600, 36, 24]} />
      <meshBasicMaterial
        map={tex}
        side={THREE.BackSide}
        transparent
        depthWrite={false}
        depthTest={false}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
}

/* ─────────────────────────────────────────────
   星雲：中を通り抜ける
   ───────────────────────────────────────────── */

type Sheet = { x: number; y: number; z: number; s: number; rot: number; op: number };
type Cloud = { depth: number; color: THREE.Color; sheets: Sheet[]; tex: number };

/**
 * 星雲は 1 枚の絵ではなく、奥行きにずらした薄い板を何枚も重ねて作る。
 * カメラが板の間を抜けていくので、近づくと視界いっぱいに広がって通り過ぎる。
 * 置き場所はステーションの「航行区間」（帯の 0.52〜0.82）の真ん中。
 * ここはセクションが何も出ない空白なので、宇宙の見せ場をここに寄せる。
 */
function useClouds(quality: Quality): Cloud[] {
  return useMemo(() => {
    const out: Cloud[] = [];
    const rnd = mulberryLite(4242);
    for (let i = 0; i < STATIONS.length - 1; i++) {
      const b = STATION_BANDS[i];
      const at = depthOf(b.start + b.span * 0.68);
      const pal = NEBULA_PALETTE[i % NEBULA_PALETTE.length];
      const sheets: Sheet[] = [];
      const n = quality.nebulaLayers;
      for (let k = 0; k < n; k++) {
        const t = k / (n - 1 || 1);
        sheets.push({
          x: (rnd() - 0.5) * 360,
          y: (rnd() - 0.5) * 260,
          z: (t - 0.5) * 520,
          s: 220 + rnd() * 300,
          rot: rnd() * Math.PI * 2,
          /**
           * 1枚あたりはごく薄く。最初 0.3〜0.75 にしたら重なりで
           * 画面全体が紫一色に染まり、星も天の川も見えなくなった。
           */
          op: 0.05 + rnd() * 0.11,
        });
      }
      out.push({
        depth: at,
        color: new THREE.Color(pal[0] / 255, pal[1] / 255, pal[2] / 255),
        sheets,
        tex: i % 3,
      });
    }
    return out;
  }, [quality.nebulaLayers]);
}

/** 種付き乱数（見た目を毎回同じにするためだけの軽いもの） */
function mulberryLite(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** この距離より手前に来た板は消していく（＝通り抜けたことになる） */
const SHEET_NEAR = 40;
const SHEET_SOLID = 320;
/** この距離より奥の板はまだ見えない */
const SHEET_FAR = 1500;

function Nebulae({ quality }: { quality: Quality }) {
  const clouds = useClouds(quality);
  const texes = useMemo(
    () =>
      [0, 1, 2].map((i) =>
        makeNebulaSprite(101 + i * 37, NEBULA_PALETTE[(i * 2) % NEBULA_PALETTE.length], 256)
      ),
    []
  );
  useEffect(() => () => texes.forEach((t) => t.dispose()), [texes]);

  const root = useRef<THREE.Group>(null);

  /**
   * 板ごとに距離で濃さを決める。
   * 近づくほど濃くなり、目の前まで来たら溶けて消える（いわゆるソフトパーティクル）。
   * これが無いと、板の中に居座って画面全体が一色に染まる。実際にそうなった。
   */
  useFrame(() => {
    const g = root.current;
    if (!g) return;
    const camZ = -flight.depth;
    g.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      const mat = m.material as THREE.MeshBasicMaterial & { userData: { base?: number } };
      const base = m.userData.base as number | undefined;
      if (base === undefined) return;
      // カメラから板までの前方距離
      const dz = camZ - m.getWorldPosition(TMP).z;
      const near = smoothstep(SHEET_NEAR, SHEET_SOLID, dz);
      const far = 1 - smoothstep(SHEET_FAR * 0.7, SHEET_FAR, dz);
      mat.opacity = base * near * far;
      m.visible = mat.opacity > 0.002;
    });
  });

  return (
    <group ref={root}>
      {clouds.map((c, ci) => (
        <group key={ci} position={[0, 0, -c.depth]}>
          {c.sheets.map((s, si) => (
            <mesh
              key={si}
              position={[s.x, s.y, s.z]}
              rotation={[0, 0, s.rot]}
              renderOrder={0}
              userData={{ base: s.op }}
            >
              <planeGeometry args={[s.s, s.s]} />
              <meshBasicMaterial
                map={texes[c.tex]}
                color={c.color}
                transparent
                opacity={0}
                depthWrite={false}
                depthTest={false}
                blending={THREE.AdditiveBlending}
                side={THREE.DoubleSide}
              />
            </mesh>
          ))}
        </group>
      ))}
    </group>
  );
}

const TMP = new THREE.Vector3();

/* ─────────────────────────────────────────────
   シーン本体
   ───────────────────────────────────────────── */

export default function Scene({ quality }: { quality: Quality }) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const prev = useRef(0);

  /**
   * 検証用の抜き出し口（開発時のみ）。
   * このリポジトリの作業環境ではブラウザペインが合成されず rAF が止まるので、
   * 指定したスクロール位置で 1 フレームだけ手で描画して画像を取り出す。
   */
  useEffect(() => {
    if (process.env.NODE_ENV === "production") return;
    const w = window as unknown as { __universeSnapshot?: (p: number) => string };
    w.__universeSnapshot = (p: number) => {
      /**
       * 深度を固定して 1 フレームだけ手で回す。
       * gl.render を直に呼ぶだけでは useFrame の中身（uniform の更新・
       * 星の筒の追従・星雲の距離フェード）が走らず、初期値のまま写ってしまう。
       * R3F の advance() を使えば購読済みの useFrame がすべて実行される。
       */
      flight.locked = true;
      flight.prog = p;
      flight.depth = depthOf(p);
      flight.speed = 0;
      flight.rush = 0;
      prev.current = flight.depth;
      camera.position.z = -flight.depth;
      camera.fov = quality.fov;
      camera.updateProjectionMatrix();
      advance(performance.now());
      const url = gl.domElement.toDataURL("image/png");
      flight.locked = false;
      return url;
    };
    return () => {
      delete w.__universeSnapshot;
    };
  }, [camera, gl, scene, quality.fov]);

  return (
    <>
      <Driver fov={quality.fov} />
      <MilkyWay />
      <Nebulae quality={quality} />
      <Stars quality={quality} />
      {/*
        遠い星明かり。飛来天体の明暗境界を作り、太陽系の手前側が
        真っ黒に潰れるのも防ぐ。three.js のライトはオブジェクト単位で
        分けられないので、両者で 1 つを共用している。
      */}
      <ambientLight intensity={0.09} color="#1a1830" />
      <directionalLight position={[-40, 32, 60]} intensity={1.5} color="#fff2e0" />

      {/* 航行区間ですれ違う天体と、到着の予兆 */}
      <Flybys enabled />

      {/* 旅の途中に浮かんでいる太陽系（創造の座標軸） */}
      <SolarSystem fov={quality.fov} onPick={openDomain} />
    </>
  );
}

export { CORRIDOR };
