import SmoothScroll from "@/components/ui/SmoothScroll";
import CustomCursor from "@/components/ui/CustomCursor";
import DepthArrival from "@/components/ui/DepthArrival";
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

export default function Home() {
  return (
    <SmoothScroll>
      <GalaxyBackground />
      <CustomCursor />
      <WarpOverlay />
      <BootSequence />
      <Navbar />
      <div className="relative z-10">
        {/* Hero でタイトルが出たあと、スクロールするほど宇宙の奥へ進み、
            各セクションが遠くから近づいてくる（DepthArrival）。
            Hero・マーキー・Footer は包まない（Hero は自前のスクロール演出を持ち、
            マーキーは装飾、Footer は最下部で到着が完了しないため）。 */}
        <main>
          <Hero />
          <DepthArrival>
            <Manifesto />
          </DepthArrival>
          <MarqueeDivider />
          <DepthArrival from={0.78}>
            <ConstellationMap />
          </DepthArrival>
          <DepthArrival>
            <SoundVisionSection />
          </DepthArrival>
          <DepthArrival>
            <DigitalAISection />
          </DepthArrival>
          {/* <WorkSection /> Collection 一旦非公開 */}
          {/* Connect は .glass（backdrop-filter）を8枚使うため変化量を小さく保つ */}
          <DepthArrival from={0.9}>
            <ConnectSection />
          </DepthArrival>
          <DepthArrival from={0.68}>
            <LegacyStatement />
          </DepthArrival>
          <DepthArrival from={0.68}>
            <SignalLost />
          </DepthArrival>
        </main>
        <Footer />
      </div>
    </SmoothScroll>
  );
}
