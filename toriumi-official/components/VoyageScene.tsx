"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Stars, Html } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import * as THREE from "three";
import { DOMAINS, LINKS, MANIFESTO, SITE, YOUTUBE } from "@/lib/content";

/**
 * Voyage — Z軸フライスルー。
 * ページを縦にスクロールすると、カメラが宇宙の奥へ進み、
 * 一定の深さごとに置かれた「ステーション」を通過していく。
 *
 * 設計メモ：
 * - ScrollControls は使わない。あれは canvas 上にスクロール用オーバーレイを敷くため、
 *   canvas 内 Html のリンクがクリックできなくなる。ここではページ本来のスクロール量
 *   （window.scrollY）から直接カメラ Z を駆動している。
 * - カードは drei Html（transform なし＋distanceFactor）。3D 変形を掛けないので
 *   文字が滲まず、通常の CSS でモバイル対応もできる。
 * - 表示制御は React state ではなく DOM を直接触る（毎フレーム setState しない）。
 */

const HUD = "#6ff2d6";

/** カメラが進む総距離。ページ高さ（100vh × PAGES）をこの距離に写像する */
const TRAVEL = 150;
const START_Z = 5;

type Station = {
  id: string;
  tag: string;
  z: number;
  body: ReactNode;
};

/* ─────────────────────────────────────────────
   宇宙塵：カメラが突き抜けていく粒子
   ───────────────────────────────────────────── */
function SpaceDust({ count = 2200 }: { count?: number }) {
  const ref = useRef<THREE.Points>(null!);

  const positions = useMemo(() => {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 60;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 60;
      // カメラの通り道（+5 → -145）を覆う
      pos[i * 3 + 2] = 10 - Math.random() * (TRAVEL + 25);
    }
    return pos;
  }, [count]);

  useFrame((_, delta) => {
    // ごくゆっくり回して、静止した点描に見えないようにする
    ref.current.rotation.z += delta * 0.012;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.12}
        color="#88ccff"
        transparent
        opacity={0.6}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}

/* ─────────────────────────────────────────────
   ステーション：一定の深さに置かれた HUD カード
   ───────────────────────────────────────────── */
function StationCard({
  station,
  compact,
  children,
}: {
  station: Station;
  compact: boolean;
  children?: ReactNode;
}) {
  const cardRef = useRef<HTMLDivElement>(null);
  const lastOp = useRef(-1);

  useFrame((state) => {
    const el = cardRef.current;
    if (!el) return;
    // カメラは +z 側から近づいてくる。手前 24 で現れ、通過直前 4 で消える
    const d = state.camera.position.z - station.z;
    let op = 0;
    if (d > 4 && d < 26) {
      const fadeIn = THREE.MathUtils.smoothstep(d, 26, 15); // 遠 → 近
      const fadeOut = 1 - THREE.MathUtils.smoothstep(d, 8, 4); // 近すぎたら消す
      op = Math.min(fadeIn, fadeOut);
    }
    if (Math.abs(op - lastOp.current) < 0.01) return;
    lastOp.current = op;
    el.style.opacity = String(op);
    el.style.pointerEvents = op > 0.55 ? "auto" : "none";
  });

  return (
    <group position={[0, 0, station.z]}>
      {children}
      <Html
        center
        position={[0, 0, 0]}
        distanceFactor={compact ? 7 : 9}
        zIndexRange={[20, 0]}
        style={{ pointerEvents: "none" }}
      >
        <div
          ref={cardRef}
          style={{
            opacity: 0,
            width: compact ? 300 : 420,
            padding: compact ? "1.6rem 1.5rem" : "2.2rem 2.4rem",
            background: "rgba(8,15,30,0.78)",
            border: `1px solid ${HUD}4d`,
            boxShadow: `0 0 24px ${HUD}26, inset 0 0 18px ${HUD}0d`,
            backdropFilter: "blur(8px)",
            borderRadius: 8,
            textAlign: "left",
            transition: "opacity 120ms linear",
          }}
        >
          {/* コーナーブラケット */}
          {[
            { top: -4, left: -4, borderTop: 1, borderLeft: 1 },
            { top: -4, right: -4, borderTop: 1, borderRight: 1 },
            { bottom: -4, left: -4, borderBottom: 1, borderLeft: 1 },
            { bottom: -4, right: -4, borderBottom: 1, borderRight: 1 },
          ].map((p, i) => (
            <span
              key={i}
              aria-hidden
              style={{
                position: "absolute",
                width: 10,
                height: 10,
                borderStyle: "solid",
                borderColor: HUD,
                borderWidth: 0,
                borderTopWidth: p.borderTop ?? 0,
                borderLeftWidth: p.borderLeft ?? 0,
                borderRightWidth: p.borderRight ?? 0,
                borderBottomWidth: p.borderBottom ?? 0,
                top: p.top,
                bottom: p.bottom,
                left: p.left,
                right: p.right,
              }}
            />
          ))}

          <span
            className="font-mono"
            style={{
              display: "block",
              fontSize: 11,
              letterSpacing: "0.25em",
              color: HUD,
              marginBottom: "0.6rem",
            }}
          >
            {station.tag}
          </span>
          {station.body}
        </div>
      </Html>
    </group>
  );
}

/* ステーションに添える立体（ステーションごとの"目印"） */
function HeroOrb() {
  const ref = useRef<THREE.Mesh>(null!);
  useFrame((_, d) => {
    ref.current.rotation.y += d * 0.2;
  });
  return (
    <mesh ref={ref} position={[2.6, -0.8, -3.5]}>
      <sphereGeometry args={[0.9, 32, 32]} />
      <meshStandardMaterial color="#3366ff" roughness={0.3} emissive="#0a1a55" emissiveIntensity={1.2} />
    </mesh>
  );
}

function ManifestoShard() {
  const ref = useRef<THREE.Mesh>(null!);
  useFrame((_, d) => {
    ref.current.rotation.x += d * 0.25;
    ref.current.rotation.y += d * 0.18;
  });
  return (
    <mesh ref={ref} position={[-2.9, 1.2, -2]}>
      <octahedronGeometry args={[0.8]} />
      <meshStandardMaterial color="#00ffcc" wireframe emissive="#00ffcc" emissiveIntensity={0.6} />
    </mesh>
  );
}

/** 創造の座標軸のミニチュア：中心星のまわりを6ドメインが回る */
function MiniSystem() {
  const groupRef = useRef<THREE.Group>(null!);
  const items = useMemo(
    () =>
      DOMAINS.filter((d) => d.key !== "connect" && !d.hidden).map((d, i, arr) => ({
        key: d.key,
        color: ["#f0b429", "#f472b6", "#a78bfa", "#60a5fa", "#5eead4", "#7dd3fc"][i % 6],
        radius: 2.2 + i * 0.55,
        speed: 0.5 - i * 0.05,
        phase: (i / arr.length) * Math.PI * 2,
      })),
    []
  );
  const refs = useRef<THREE.Mesh[]>([]);

  useFrame((state, d) => {
    groupRef.current.rotation.y += d * 0.05;
    const t = state.clock.getElapsedTime();
    items.forEach((it, i) => {
      const m = refs.current[i];
      if (!m) return;
      const a = it.phase + t * it.speed;
      m.position.set(Math.cos(a) * it.radius, Math.sin(a * 0.6) * 0.4, Math.sin(a) * it.radius);
    });
  });

  return (
    <group ref={groupRef} position={[0, 0, -3]}>
      <mesh>
        <sphereGeometry args={[0.9, 32, 32]} />
        <meshStandardMaterial
          color="#ffaa22"
          emissive="#ff6600"
          emissiveIntensity={2.2}
          toneMapped={false}
        />
      </mesh>
      {items.map((it, i) => (
        <mesh
          key={it.key}
          ref={(m) => {
            if (m) refs.current[i] = m;
          }}
        >
          <sphereGeometry args={[0.24, 20, 20]} />
          <meshStandardMaterial color={it.color} emissive={it.color} emissiveIntensity={1.4} />
        </mesh>
      ))}
    </group>
  );
}

function SoundRings() {
  const ref = useRef<THREE.Group>(null!);
  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    ref.current.children.forEach((c, i) => {
      c.scale.setScalar(1 + Math.sin(t * 1.6 + i * 0.7) * 0.12);
    });
  });
  return (
    <group ref={ref} position={[0, 0, -3.2]}>
      {[1.6, 2.3, 3.0].map((r, i) => (
        <mesh key={r} rotation-x={Math.PI / 2.4}>
          <torusGeometry args={[r, 0.02, 12, 96]} />
          <meshStandardMaterial
            color="#f472b6"
            emissive="#f472b6"
            emissiveIntensity={1.2 - i * 0.3}
          />
        </mesh>
      ))}
    </group>
  );
}

function MakeCubes() {
  const ref = useRef<THREE.Group>(null!);
  useFrame((_, d) => {
    ref.current.rotation.y += d * 0.3;
  });
  return (
    <group ref={ref} position={[0, 0, -3.4]}>
      {[-2.2, 0, 2.2].map((x, i) => (
        <mesh key={x} position={[x, Math.sin(i) * 0.5, 0]}>
          <boxGeometry args={[0.7, 0.7, 0.7]} />
          <meshStandardMaterial
            color="#a78bfa"
            emissive="#7c3aed"
            emissiveIntensity={0.8}
            roughness={0.35}
            metalness={0.6}
            wireframe={i === 1}
          />
        </mesh>
      ))}
    </group>
  );
}

function FinalBeacon() {
  const ref = useRef<THREE.Mesh>(null!);
  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    const m = ref.current.material as THREE.MeshStandardMaterial;
    m.emissiveIntensity = 1.4 + Math.sin(t * 1.2) * 0.9;
  });
  return (
    <mesh ref={ref} position={[0, 0, -4]}>
      <sphereGeometry args={[0.35, 24, 24]} />
      <meshStandardMaterial
        color="#fceabb"
        emissive="#f0b429"
        emissiveIntensity={1.6}
        toneMapped={false}
      />
    </mesh>
  );
}

/* ─────────────────────────────────────────────
   カメラ：ページのスクロール量で Z を進める
   ───────────────────────────────────────────── */
function FlyCamera({ reduced }: { reduced: boolean }) {
  useFrame((state, delta) => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    const p = max > 0 ? window.scrollY / max : 0;
    const targetZ = START_Z - p * TRAVEL;

    state.camera.position.z = THREE.MathUtils.damp(
      state.camera.position.z,
      targetZ,
      2.6,
      delta
    );

    if (!reduced) {
      // 宇宙船に乗っているような微かな手ぶれ
      const t = state.clock.getElapsedTime();
      state.camera.position.x = Math.sin(t * 0.8) * 0.3;
      state.camera.position.y = Math.cos(t * 0.6) * 0.2;
    }
    state.camera.lookAt(
      state.camera.position.x * 0.5,
      state.camera.position.y * 0.5,
      state.camera.position.z - 10
    );
  });
  return null;
}

/* ─────────────────────────────────────────────
   本体
   ───────────────────────────────────────────── */
const linkStyle = {
  color: HUD,
  textDecoration: "none",
  fontSize: 13,
  lineHeight: 1.9,
  display: "block",
} as const;

const pStyle = { fontSize: 14, lineHeight: 1.85, color: "#a8bccb", margin: 0 } as const;
const h1Style = { fontSize: 26, margin: "0 0 0.4rem", color: "#fff", fontWeight: 600 } as const;
const h2Style = { fontSize: 21, margin: "0 0 0.6rem", color: "#e0f7fc", fontWeight: 600 } as const;

export default function VoyageScene() {
  const [reduced, setReduced] = useState(false);
  const [compact, setCompact] = useState(false);

  useEffect(() => {
    setReduced(window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    const mq = window.matchMedia("(max-width: 640px)");
    const sync = () => setCompact(mq.matches);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);

  const domains = useMemo(() => DOMAINS.filter((d) => d.key !== "connect" && !d.hidden), []);

  const stations: Station[] = [
    {
      id: "start",
      tag: "01 / VOYAGE START",
      z: 0,
      body: (
        <>
          <h1 style={h1Style}>{SITE.nameEn}</h1>
          <p style={pStyle}>
            a.k.a KIEJI — {SITE.taglineJp}
          </p>
          <p style={{ ...pStyle, marginTop: "1.2rem", color: HUD, fontSize: 12 }}>
            ↓ 下へスクロールして、宇宙の奥へ
          </p>
        </>
      ),
    },
    {
      id: "manifesto",
      tag: "02 / MANIFESTO",
      z: -26,
      body: (
        <>
          <h2 style={h2Style}>創造の意思</h2>
          <p style={pStyle}>{MANIFESTO.lines.join("")}</p>
          <p style={{ ...pStyle, marginTop: "0.9rem", fontSize: 12.5, color: "#8fa4b4" }}>
            アートも、音楽も、テクノロジーも——すべては同じひとつの創造のエネルギーが、
            異なる次元に立ち現れた姿にすぎない。
          </p>
        </>
      ),
    },
    {
      id: "universe",
      tag: "03 / UNIVERSE",
      z: -56,
      body: (
        <>
          <h2 style={h2Style}>創造の座標軸</h2>
          <p style={{ ...pStyle, marginBottom: "0.8rem" }}>
            ひとりの中に同時に存在する、六つの世界。
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", columnGap: 14 }}>
            {domains.map((d) => {
              const href = d.href ?? d.links?.[0]?.href ?? "#";
              const external = href.startsWith("http");
              return (
                <a
                  key={d.key}
                  href={href}
                  target={external ? "_blank" : undefined}
                  rel={external ? "noopener noreferrer" : undefined}
                  style={linkStyle}
                >
                  ▸ {d.titleEn}
                </a>
              );
            })}
          </div>
        </>
      ),
    },
    {
      id: "sound",
      tag: "04 / SOUND & VISION",
      z: -86,
      body: (
        <>
          <h2 style={h2Style}>EXODUS</h2>
          <p style={pStyle}>音楽と映像。声と光で綴る、もうひとつの次元。</p>
          <a
            href={YOUTUBE.channelUrl}
            target="_blank"
            rel="noopener noreferrer"
            style={{ ...linkStyle, marginTop: "0.9rem" }}
          >
            ▸ YouTube · @Exodus999
          </a>
          <a href={LINKS.radio} target="_blank" rel="noopener noreferrer" style={linkStyle}>
            ▸ Radio · stand.fm
          </a>
        </>
      ),
    },
    {
      id: "make",
      tag: "05 / MAKE",
      z: -112,
      body: (
        <>
          <h2 style={h2Style}>つくる</h2>
          <p style={pStyle}>アプリ開発 ／ Web制作 ／ 動画制作。イメージを、動く形へ。</p>
          <a
            href={LINKS.line}
            target="_blank"
            rel="noopener noreferrer"
            style={{ ...linkStyle, marginTop: "0.9rem" }}
          >
            ▸ ご依頼・ご相談（LINE）
          </a>
        </>
      ),
    },
    {
      id: "report",
      tag: "06 / FINAL REPORT",
      z: -138,
      body: (
        <>
          <h2 style={h2Style}>終活レポート</h2>
          <p style={pStyle}>
            このポータルサイトは、陽化（老化）を極めた惑星テラ（地球）でのミッションを
            締めくくるに当たり、Katsunori Toriumi が自己の記憶をまとめ上げる為に作成した
            終活レポートとして…
          </p>
          <p
            className="font-mono"
            style={{ marginTop: "1.2rem", fontSize: 10, letterSpacing: "0.35em", color: "#6d8394" }}
          >
            KATSUNORI TORIUMI a.k.a KIEJI
          </p>
          <a href="/" style={{ ...linkStyle, marginTop: "0.9rem" }}>
            ▸ サイトへ戻る
          </a>
        </>
      ),
    },
  ];

  const extras: Record<string, ReactNode> = {
    start: <HeroOrb />,
    manifesto: <ManifestoShard />,
    universe: <MiniSystem />,
    sound: <SoundRings />,
    make: <MakeCubes />,
    report: <FinalBeacon />,
  };

  return (
    <Canvas
      camera={{ position: [0, 0, START_Z], fov: 55, near: 0.1, far: 400 }}
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: false }}
      style={{ background: "#030509" }}
    >
      <ambientLight intensity={0.22} />
      <directionalLight position={[5, 10, 5]} intensity={1.4} color="#ffffff" />
      <pointLight position={[0, 0, -56]} intensity={40} decay={2} color="#ffaa44" />
      <pointLight position={[0, 0, -138]} intensity={16} decay={2} color="#f0b429" />

      <Stars radius={120} depth={60} count={compact ? 2000 : 4000} factor={4} saturation={0} fade />
      <SpaceDust count={compact ? 1200 : 2200} />

      {stations.map((s) => (
        <StationCard key={s.id} station={s} compact={compact}>
          {extras[s.id]}
        </StationCard>
      ))}

      <FlyCamera reduced={reduced} />

      <EffectComposer>
        <Bloom intensity={1.1} luminanceThreshold={0.75} luminanceSmoothing={0.2} mipmapBlur />
      </EffectComposer>
    </Canvas>
  );
}
