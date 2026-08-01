"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { mulberry32 } from "@/lib/noise";
import { makeDiffuse, makeRing, type PlanetSkin } from "@/lib/planetTextures";
import { makeStarSprite } from "@/lib/spaceTextures";
import { STATIONS, STATION_BANDS } from "@/lib/stations";
import { depthOf, flight } from "./flightState";
import { makeRingGeometry } from "./ringGeometry";
import { bakeOnce, diffuseKey } from "./textureCache";

/**
 * 航行区間ですれ違う天体と、ステーション到着の予兆。
 *
 * セクションとセクションの間（帯の 0.52〜0.82）は何も表示されない空白で、
 * ここが「移動している時間」になる。そこに実際にライティングされた天体を置くと、
 * 近づくにつれて欠けた明暗境界が見え、脇を通り過ぎていく。
 * ビルボード（貼り絵）ではこの立体感は出ない。
 */

type Body = {
  depth: number;
  x: number;
  y: number;
  size: number;
  skin: PlanetSkin;
  seed: number;
  spin: number;
  tilt: [number, number, number];
  ring: boolean;
  ringTilt: number;
};

/** 飛来天体に使う質感。太陽系と同じ種を使い、テクスチャを共有する */
const FLYBY_SKINS: [PlanetSkin, number][] = [
  ["cratered", 44],
  ["gas", 22],
  ["ice", 66],
  ["desert", 11],
  ["marble", 33],
];

/** この距離まで来たら描き始める／描き終わる */
const SHOW_FAR = 1700;
const SHOW_NEAR = -260;

function useBodies(): Body[] {
  return useMemo(() => {
    const rnd = mulberry32(9182);
    const out: Body[] = [];
    for (let i = 0; i < STATIONS.length - 1; i++) {
      const b = STATION_BANDS[i];
      // 航行区間のなかに 1〜2 個。毎回同じ場所に出るよう種から決める
      const slots = i % 2 === 0 ? [0.58, 0.76] : [0.66];
      for (const q of slots) {
        const [skin, seed] = FLYBY_SKINS[out.length % FLYBY_SKINS.length];
        const th = rnd() * Math.PI * 2;
        // 軸からの距離。近すぎると視界を塞ぎ、遠すぎると気づかない
        const r = 150 + rnd() * 190;
        out.push({
          depth: depthOf(b.start + b.span * q),
          x: Math.cos(th) * r,
          y: Math.sin(th) * r * 0.7,
          size: 26 + rnd() * 62,
          skin,
          seed,
          spin: (0.02 + rnd() * 0.05) * (rnd() < 0.5 ? -1 : 1),
          tilt: [rnd() * 0.7 - 0.35, rnd() * Math.PI * 2, rnd() * 0.5 - 0.25],
          ring: rnd() < 0.35,
          /**
           * 環の傾き。mesh の rotation.x は (π/2 - ringTilt) なので、
           * ここを大きくすると正面向き（真上から見た輪）になってしまう。
           * 土星のように斜めから見た輪にしたいので小さめに取る。
           */
          ringTilt: 0.12 + rnd() * 0.26,
        });
      }
    }
    return out;
  }, []);
}

function Flyby({ body, tex, ring }: { body: Body; tex: THREE.Texture; ring: THREE.Texture }) {
  const group = useRef<THREE.Group>(null);
  const mesh = useRef<THREE.Mesh>(null);
  const ringGeo = useMemo(
    () => (body.ring ? makeRingGeometry(body.size * 1.45, body.size * 2.05, 84) : null),
    [body]
  );
  useEffect(() => () => ringGeo?.dispose(), [ringGeo]);

  useFrame((_, dt) => {
    const g = group.current;
    if (!g) return;
    const dz = body.depth - flight.depth;
    const on = dz < SHOW_FAR && dz > SHOW_NEAR;
    if (on !== g.visible) g.visible = on;
    if (on && mesh.current) mesh.current.rotation.y += dt * body.spin;
  });

  return (
    <group ref={group} position={[body.x, body.y, -body.depth]} visible={false}>
      <mesh ref={mesh} rotation={body.tilt}>
        <sphereGeometry args={[body.size, 40, 40]} />
        <meshStandardMaterial map={tex} roughness={0.92} metalness={0.04} />
      </mesh>
      {body.ring && ringGeo && (
        <mesh geometry={ringGeo} rotation={[Math.PI / 2 - body.ringTilt, 0, 0.4]}>
          <meshBasicMaterial
            map={ring}
            side={THREE.DoubleSide}
            transparent
            opacity={0.7}
            depthWrite={false}
          />
        </mesh>
      )}
    </group>
  );
}

/**
 * ステーション到着の予兆。
 * 次のセクションが現れる深度の奥に淡い光を置き、
 * セクションが「暗闇から光の中に浮かび上がる」ように見せる。
 */
function ArrivalGlow() {
  const glow = useMemo(() => makeStarSprite(false), []);
  useEffect(() => () => glow.dispose(), [glow]);
  const root = useRef<THREE.Group>(null);

  const spots = useMemo(
    () =>
      STATION_BANDS.map((b, i) => ({
        // 到着点の少し奥
        depth: depthOf(b.start) + 260,
        color: new THREE.Color().setHSL(0.09 + (i % 3) * 0.16, 0.55, 0.6),
      })),
    []
  );

  useFrame(() => {
    const g = root.current;
    if (!g) return;
    g.children.forEach((child, i) => {
      const dz = spots[i].depth - flight.depth;
      const sp = child as THREE.Sprite;
      const mat = sp.material as THREE.SpriteMaterial;
      // 近づくほど濃く、通り過ぎると消える
      const t = 1 - Math.min(1, Math.abs(dz) / 900);
      // 強くすると画面全体が霞む。あくまで「奥に何かある」と気づく程度に
      mat.opacity = 0.14 * t * t;
      sp.visible = mat.opacity > 0.004;
    });
  });

  return (
    <group ref={root}>
      {spots.map((s, i) => (
        <sprite key={i} position={[0, 0, -s.depth]} scale={[1500, 1100, 1]} visible={false}>
          <spriteMaterial
            map={glow}
            color={s.color}
            transparent
            opacity={0}
            depthWrite={false}
            depthTest={false}
            blending={THREE.AdditiveBlending}
          />
        </sprite>
      ))}
    </group>
  );
}

export default function Flybys({ enabled }: { enabled: boolean }) {
  const bodies = useBodies();
  const [tex, setTex] = useState<Record<string, THREE.Texture> | null>(null);
  const [ring, setRing] = useState<THREE.Texture | null>(null);
  const started = useRef(false);

  useEffect(() => {
    if (!enabled || started.current) return;
    started.current = true;
    let alive = true;
    (async () => {
      const out: Record<string, THREE.Texture> = {};
      for (const [skin, seed] of FLYBY_SKINS) {
        out[`${skin}:${seed}`] = await bakeOnce(diffuseKey(skin, seed), () =>
          makeDiffuse(skin, seed)
        );
      }
      const r = await bakeOnce("ring:31", () => makeRing(31));
      if (alive) {
        setTex(out);
        setRing(r);
      }
    })();
    return () => {
      alive = false;
    };
  }, [enabled]);

  return (
    <>
      <ArrivalGlow />
      {tex &&
        ring &&
        bodies.map((b, i) => (
          <Flyby key={i} body={b} tex={tex[`${b.skin}:${b.seed}`]} ring={ring} />
        ))}
    </>
  );
}
