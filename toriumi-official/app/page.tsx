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
      <ZoomStage
        stations={[
          { id: "home", node: <Hero />, scroll: 1.1 },
          {
            id: "manifesto",
            node: (
              <>
                <Manifesto />
                <MarqueeDivider />
              </>
            ),
          },
          { id: "universe", node: <ConstellationMap />, scroll: 1.5 },
          { id: "sound", node: <SoundVisionSection /> },
          { id: "digital", node: <DigitalAISection /> },
          { id: "connect", node: <ConnectSection />, scroll: 1.4 },
          { id: "report", node: <LegacyStatement />, scroll: 1.4 },
          {
            id: "end",
            node: (
              <>
                <SignalLost />
                <Footer />
              </>
            ),
          },
        ]}
      />
    </SmoothScroll>
  );
}
