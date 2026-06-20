import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";

export default function Home() {
  return (
    <main className="min-h-screen bg-[#0d1510]">
      <Navbar />
      <Hero />

      {/* 次のステップで、この下にグリッドデザインのお知らせセクションを作ります */}
      <div className="h-96 w-full flex items-center justify-center text-white/20">
        [ここに次のセクションが入ります]
      </div>
    </main>
  );
}
