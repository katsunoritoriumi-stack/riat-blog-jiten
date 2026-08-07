"use client";

import dynamic from "next/dynamic";
import { useBootDone } from "@/lib/bootGate";
import StaticSky from "./universe/StaticSky";

/**
 * 宇宙（WebGL）を、ブート演出が明けてからマウントする。
 *
 * three.js は初期チャンクの中でいちばん大きい。初回訪問では
 * BootSequence が約2.5秒のあいだ画面全体を覆っていて、その間 WebGL は
 * どのみち見えない。ここで待たせるぶん、最初の描画と Hero の動画に
 * 帯域と CPU を回せる（＝ LCP が改善する）。
 *
 * 見え方は変わらない。待っている間は WebGL 版と同じ静止星空を出しており、
 * 再訪（sessionStorage に boot-seen がある）や reduced-motion では
 * bootDone が即座に立つので、これまでと同じタイミングでマウントされる。
 */
const Universe = dynamic(() => import("./Universe"));

export default function UniverseGate() {
  const ready = useBootDone();
  return ready ? <Universe /> : <StaticSky />;
}
