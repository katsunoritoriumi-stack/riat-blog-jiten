import SmoothScroll from "@/components/ui/SmoothScroll";
import CustomCursor from "@/components/ui/CustomCursor";
import ZoomStage from "@/components/ui/ZoomStage";
import BootSequence from "@/components/BootSequence";
import WarpOverlay from "@/components/WarpOverlay";
import GalaxyBackground from "@/components/GalaxyBackground";
import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Manifesto from "@/components/Manifesto";
import MarqueeDivider from "@/components/MarqueeDivider";
import ConstellationMap from "@/components/ConstellationMap";
import SoundVisionSection from "@/components/SoundVisionSection";
import DigitalAISection from "@/components/DigitalAISection";
// import WorkSection from "@/components/WorkSection"; // Collection 一旦非公開
import ConnectSection from "@/components/ConnectSection";
import LegacyStatement from "@/components/LegacyStatement";
import SignalLost from "@/components/SignalLost";
import FinaleBackdrop from "@/components/FinaleBackdrop";
import Footer from "@/components/Footer";

/**
 * ホームは「宇宙の奥へ進む」1ページ体験。
 * Hero で UFO が出てタイトルが現れたあと、スクロールするたびにカメラが奥へ進み、
 * いま見ているセクションが拡大して通り過ぎ、次のセクションが奥から現れる。
 * 各セクションの中身は従来のまま（ZoomStage は見せ方だけを担当する）。
 */
export default function Home() {
  return (
    <SmoothScroll>
      <GalaxyBackground />
      <CustomCursor />
      <WarpOverlay />
      <BootSequence />
      <Navbar />
      {/* scroll = そのステーションに割り当てる移動距離（画面高さの倍数）。
          大きいほど「そこへ向かう航行」と「滞在」が長くなる。 */}
      <ZoomStage
        stations={[
          { id: "home", node: <Hero />, scroll: 2.2 },
          {
            id: "manifesto",
            scroll: 2.9,
            node: (
              <>
                <Manifesto />
                <MarqueeDivider />
              </>
            ),
          },
          { id: "universe", node: <ConstellationMap />, scroll: 3.6 },
          { id: "sound", node: <SoundVisionSection />, scroll: 2.9 },
          { id: "digital", node: <DigitalAISection />, scroll: 2.9 },
          { id: "connect", node: <ConnectSection />, scroll: 3.4 },
          { id: "report", node: <LegacyStatement />, scroll: 3.4 },
          {
            id: "end",
            scroll: 2.6,
            node: (
              // 終章は絵を画面いっぱいに敷き、その上に通信途絶とロゴの署名を重ねる。
              // min-h をステーション1枚ぶんにして、背景がフルサイズで広がるようにする。
              <div className="relative isolate flex min-h-[100svh] flex-col justify-between pb-[2vh] pt-[14vh]">
                <FinaleBackdrop />
                <SignalLost />
                <Footer />
              </div>
            ),
          },
        ]}
      />
    </SmoothScroll>
  );
}
