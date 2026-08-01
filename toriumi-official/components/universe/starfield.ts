/**
 * 星の回廊。
 *
 * 考え方：星は「筒」の中に一様に撒いてあり、カメラは筒の中を前へ進む。
 * 通り過ぎた星は、頂点シェーダの剰余（mod）で自動的に前方へ巻き戻る。
 * CPU 側は毎フレーム uniform を 1 つ書き換えるだけで、頂点バッファには一切触らない。
 *
 * ストリーク（速度線）は同じ属性を使う LineSegments で描く。
 * 頭は現在の深度、尾は uStreak ぶん奥の深度を見るので、
 * 速く動くほど自然に伸びる。
 */

import * as THREE from "three";
import { mulberry32, starColor } from "@/lib/noise";

/** 筒の長さ（この距離で 1 周して巻き戻る） */
export const TUBE_LEN = 780;
/** 筒の半径。大きいほど画面の外側まで星が散る */
export const TUBE_RADIUS = 300;

const VERT = /* glsl */ `
  uniform float uDepth;
  uniform float uLen;
  uniform float uSize;
  uniform float uMaxSize;
  uniform float uStreak;
  attribute float aOffset;
  attribute float aSize;
  attribute vec3 aColor;
  attribute float aSide;   // 0 = 頭 / 1 = 尾（点として使うときは常に 0）
  varying vec3 vColor;
  varying float vAlpha;

  void main() {
    // 尾は「少し前の深度」＝ぶんだけ奥にいたときの位置
    float dz = mod(aOffset - uDepth + aSide * uStreak, uLen);
    vec4 mv = modelViewMatrix * vec4(position.x, position.y, -dz, 1.0);
    gl_Position = projectionMatrix * mv;

    float d = max(dz, 1.0);
    // 上限を切らないと、すぐ手前を通る星が画面いっぱいの綿になる
    gl_PointSize = clamp(uSize * aSize / d, 0.5, uMaxSize);

    // 奥で湧き、手前で消える
    float fade = smoothstep(uLen, uLen * 0.7, dz) * smoothstep(1.5, 12.0, dz);
    vAlpha = fade * mix(1.0, 0.25, aSide);
    vColor = aColor;
  }
`;

const FRAG_POINT = /* glsl */ `
  uniform sampler2D uMap;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float a = texture2D(uMap, gl_PointCoord).a * vAlpha;
    if (a < 0.004) discard;
    gl_FragColor = vec4(vColor * a, a);
  }
`;

const FRAG_LINE = /* glsl */ `
  uniform float uStreakFade;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float a = vAlpha * uStreakFade;
    if (a < 0.004) discard;
    gl_FragColor = vec4(vColor * a, a);
  }
`;

export type StarLayer = {
  points: THREE.Points;
  lines: THREE.LineSegments;
  material: THREE.ShaderMaterial;
  lineMaterial: THREE.ShaderMaterial;
  dispose: () => void;
};

/**
 * 星の層をひとつ作る。
 * bright = true のときは数を絞り、大きく、光条つきのスプライトを使う。
 */
export function makeStarLayer(opts: {
  count: number;
  seed: number;
  sprite: THREE.Texture;
  sizeRange: [number, number];
  /** 色温度の下限（0=橙 1=青白）。参考画像に寄せて高温側へ偏らせる */
  tempBias: number;
  withStreaks: boolean;
  /** 画面上の最大直径(px) */
  maxSize: number;
}): StarLayer {
  const { count, seed, sprite, sizeRange, tempBias, withStreaks, maxSize } = opts;
  const rnd = mulberry32(seed);

  const pos = new Float32Array(count * 3);
  const off = new Float32Array(count);
  const size = new Float32Array(count);
  const col = new Float32Array(count * 3);

  for (let i = 0; i < count; i++) {
    // 円盤内に一様（sqrt を掛けないと中心に密集する）
    const r = TUBE_RADIUS * Math.sqrt(rnd());
    const th = rnd() * Math.PI * 2;
    pos[i * 3] = Math.cos(th) * r;
    pos[i * 3 + 1] = Math.sin(th) * r;
    pos[i * 3 + 2] = 0; // z はシェーダ側で決める
    off[i] = rnd() * TUBE_LEN;
    size[i] = sizeRange[0] + Math.pow(rnd(), 2.2) * (sizeRange[1] - sizeRange[0]);
    const t = tempBias + (1 - tempBias) * Math.pow(rnd(), 0.65);
    const c = starColor(t);
    col[i * 3] = c[0] / 255;
    col[i * 3 + 1] = c[1] / 255;
    col[i * 3 + 2] = c[2] / 255;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("aOffset", new THREE.BufferAttribute(off, 1));
  geo.setAttribute("aSize", new THREE.BufferAttribute(size, 1));
  geo.setAttribute("aColor", new THREE.BufferAttribute(col, 3));
  geo.setAttribute("aSide", new THREE.BufferAttribute(new Float32Array(count), 1));
  // 頂点位置をシェーダで動かすので、既定のバウンディングでは消される。無限にしておく
  geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Infinity);

  const uniforms = {
    uDepth: { value: 0 },
    uLen: { value: TUBE_LEN },
    uSize: { value: 1200 },
    uMaxSize: { value: maxSize },
    uStreak: { value: 0 },
    uMap: { value: sprite },
    uStreakFade: { value: 0 },
  };

  const material = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: VERT,
    fragmentShader: FRAG_POINT,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    depthTest: false,
  });
  const points = new THREE.Points(geo, material);
  points.frustumCulled = false;
  points.renderOrder = 2;

  /* ── ストリーク（頭と尾の 2 頂点を持つ線） ── */
  const lineGeo = new THREE.BufferGeometry();
  if (withStreaks) {
    const n = count * 2;
    const lpos = new Float32Array(n * 3);
    const loff = new Float32Array(n);
    const lsize = new Float32Array(n);
    const lcol = new Float32Array(n * 3);
    const lside = new Float32Array(n);
    for (let i = 0; i < count; i++) {
      for (let s = 0; s < 2; s++) {
        const j = i * 2 + s;
        lpos[j * 3] = pos[i * 3];
        lpos[j * 3 + 1] = pos[i * 3 + 1];
        lpos[j * 3 + 2] = 0;
        loff[j] = off[i];
        lsize[j] = size[i];
        lcol[j * 3] = col[i * 3];
        lcol[j * 3 + 1] = col[i * 3 + 1];
        lcol[j * 3 + 2] = col[i * 3 + 2];
        lside[j] = s;
      }
    }
    lineGeo.setAttribute("position", new THREE.BufferAttribute(lpos, 3));
    lineGeo.setAttribute("aOffset", new THREE.BufferAttribute(loff, 1));
    lineGeo.setAttribute("aSize", new THREE.BufferAttribute(lsize, 1));
    lineGeo.setAttribute("aColor", new THREE.BufferAttribute(lcol, 3));
    lineGeo.setAttribute("aSide", new THREE.BufferAttribute(lside, 1));
    lineGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Infinity);
  }

  const lineMaterial = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: VERT,
    fragmentShader: FRAG_LINE,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    depthTest: false,
  });
  const lines = new THREE.LineSegments(lineGeo, lineMaterial);
  lines.frustumCulled = false;
  lines.visible = false;
  lines.renderOrder = 1;

  return {
    points,
    lines,
    material,
    lineMaterial,
    dispose: () => {
      geo.dispose();
      lineGeo.dispose();
      material.dispose();
      lineMaterial.dispose();
    },
  };
}

/**
 * gl_PointSize の基準値。
 * 距離 d にある「世界半径 s」の点が画面上で何 px になるかは
 *   px = s * H / (tan(fov/2) * d)
 * なので、uSize = H / tan(fov/2) を渡しておけば aSize が世界半径になる。
 */
export function pointSizeBase(viewportHeightPx: number, fovDeg: number) {
  return viewportHeightPx / Math.tan((fovDeg * Math.PI) / 360);
}
