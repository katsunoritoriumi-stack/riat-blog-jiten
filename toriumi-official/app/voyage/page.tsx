import type { Metadata } from "next";
import VoyageClient from "@/components/VoyageClient";

export const metadata: Metadata = {
  title: "Voyage — Katsunori Toriumi",
  description:
    "スクロールで宇宙の奥へ進む航行体験。鳥海勝稚（a.k.a KIEJI）の創造の座標軸を、深宇宙のステーションを辿りながら巡る。",
};

export default function VoyagePage() {
  return <VoyageClient />;
}
