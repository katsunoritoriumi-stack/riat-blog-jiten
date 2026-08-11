import type { ReactNode } from "react";
import JsonLd from "@/components/JsonLd";
import { homePageLd } from "@/lib/jsonLd";
import UniverseGate from "@/components/UniverseGate";
import SmoothScroll from "@/components/ui/SmoothScroll";
import CustomCursor from "@/components/ui/CustomCursor";
import ZoomStage from "@/components/ui/ZoomStage";
import BootSequence from "@/components/BootSequence";
import BackgroundMusic from "@/components/BackgroundMusic";
import WarpOverlay from "@/components/WarpOverlay";
import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Manifesto from "@/components/Manifesto";
import MarqueeDivider from "@/components/MarqueeDivider";
import ConstellationMap from "@/components/ConstellationMap";
import SoundVisionSection from "@/components/SoundVisionSection";
import AlbumSection from "@/components/AlbumSection";
import DigitalAISection from "@/components/DigitalAISection";
// import WorkSection from "@/components/WorkSection"; // Collection 一旦非公開
import ConnectSection from "@/components/ConnectSection";
import LegacyStatement from "@/components/LegacyStatement";
import SignalLost from "@/components/SignalLost";
import FinaleBackdrop from "@/components/FinaleBackdrop";
import FinaleBookLink from "@/components/FinaleBookLink";
import Footer from "@/components/Footer";
import { STATIONS } from "@/lib/stations";

/**
 * ホームは「宇宙の奥へ進む」1ページ体験。
 * Hero で UFO が出てタイトルが現れたあと、スクロールするたびにカメラが奥へ進み、
 * いま見ているセクションが拡大して通り過ぎ、次のセクションが奥から現れる。
 *
 * どのステーションをどれだけの距離で見せるか（id・scroll・予告編コピー）は
 * lib/stations.ts が唯一の出典。ここでは中身（node）だけを差し込む。
 */
const NODES: Record<string, ReactNode> = {
  home: <Hero />,
  manifesto: (
    <>
      <Manifesto />
      <MarqueeDivider />
    </>
  ),
  universe: <ConstellationMap />,
  sound: <SoundVisionSection />,
  album: <AlbumSection />,
  digital: <DigitalAISection />,
  connect: <ConnectSection />,
  report: <LegacyStatement />,
  // 終章は絵を画面いっぱいに敷き、その上に通信途絶とロゴの署名を重ねる。
  // min-h をステーション1枚ぶんにして、背景がフルサイズで広がるようにする。
  end: (
    <div className="relative isolate flex min-h-[100svh] flex-col justify-between pb-[2vh] pt-[14vh]">
      <FinaleBackdrop />
      <SignalLost />
      <Footer />
      {/* 絵の中の本へのリンク。文字より上に出さないと押せないので最後に置く */}
      <FinaleBookLink />
    </div>
  ),
};

export default function Home() {
  return (
    <SmoothScroll>
      <JsonLd data={homePageLd()} />
      {/* 背景の宇宙は WebGL。ブート演出が明けてからマウントする（UniverseGate） */}
      <UniverseGate />
      <CustomCursor />
      {/* サウンドボタンが押されたら、スクロールの開始に合わせて小さく流れ出す */}
      <BackgroundMusic />
      <WarpOverlay />
      <BootSequence />
      <Navbar />
      <ZoomStage stations={STATIONS.map((s) => ({ ...s, node: NODES[s.id] }))} />
    </SmoothScroll>
  );
}
