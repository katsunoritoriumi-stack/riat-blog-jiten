import * as THREE from "three";

/**
 * 環のジオメトリ。
 *
 * THREE.RingGeometry の UV は「平面写像」なので、そのまま帯テクスチャを貼ると
 * 同心円ではなく放射状の縞になる（実際にそうなった）。
 * u を半径方向・v を角度方向に貼り直して、内→外の濃淡が正しく出るようにする。
 */
export function makeRingGeometry(inner: number, outer: number, segments = 96) {
  const geo = new THREE.RingGeometry(inner, outer, segments, 1);
  const pos = geo.attributes.position;
  const uv = geo.attributes.uv;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    const r = Math.hypot(v.x, v.y);
    const u = (r - inner) / (outer - inner || 1);
    const a = Math.atan2(v.y, v.x) / (Math.PI * 2) + 0.5;
    uv.setXY(i, u, a);
  }
  uv.needsUpdate = true;
  return geo;
}
