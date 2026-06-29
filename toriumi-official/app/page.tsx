import SmoothScroll from "@/components/ui/SmoothScroll";
import CustomCursor from "@/components/ui/CustomCursor";
import GalaxyBackground from "@/components/GalaxyBackground";
import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Manifesto from "@/components/Manifesto";
import ConstellationMap from "@/components/ConstellationMap";
import QuantumArtGallery from "@/components/QuantumArtGallery";
import NiravSection from "@/components/NiravSection";
import SoundVisionSection from "@/components/SoundVisionSection";
import DigitalAISection from "@/components/DigitalAISection";
import WorkSection from "@/components/WorkSection";
import ConnectSection from "@/components/ConnectSection";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <SmoothScroll>
      <GalaxyBackground />
      <CustomCursor />
      <Navbar />
      <div className="relative z-10">
        <main>
          <Hero />
          <Manifesto />
          <ConstellationMap />
          <QuantumArtGallery />
          <NiravSection />
          <SoundVisionSection />
          <DigitalAISection />
          <WorkSection />
          <ConnectSection />
        </main>
        <Footer />
      </div>
    </SmoothScroll>
  );
}
