import SmoothScroll from "@/components/ui/SmoothScroll";
import CustomCursor from "@/components/ui/CustomCursor";
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
        <main>
          <Hero />
          <Manifesto />
          <MarqueeDivider />
          <ConstellationMap />
          <SoundVisionSection />
          <DigitalAISection />
          {/* <WorkSection /> Collection 一旦非公開 */}
          <ConnectSection />
          <LegacyStatement />
          <SignalLost />
        </main>
        <Footer />
      </div>
    </SmoothScroll>
  );
}
