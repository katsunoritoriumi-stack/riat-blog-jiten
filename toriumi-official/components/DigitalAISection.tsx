"use client";

import { motion } from "framer-motion";
import { Code2, Globe, Film } from "lucide-react";
import SectionHeader from "./ui/SectionHeader";

const services = [
  {
    icon: Code2,
    en: "App Development",
    jp: "アプリ開発",
    note: "アイデアを、触れて動く形へ。設計から実装まで一気通貫で創る。",
  },
  {
    icon: Globe,
    en: "Web Production",
    jp: "Web制作",
    note: "世界観を宿したサイトを、デザインからコードまで仕立てる。",
  },
  {
    icon: Film,
    en: "Video Production",
    jp: "動画制作",
    note: "物語を映像に。撮影・編集・MVまで、イメージを動かす。",
  },
];

export default function DigitalAISection() {
  return (
    <section id="digital" className="relative mx-auto max-w-7xl px-6 py-28 sm:py-36">
      <div className="mb-16 max-w-2xl">
        <SectionHeader eyebrow="Job request" titleEn="Make" />
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {services.map((s, i) => (
          <motion.div
            key={s.en}
            initial={{ opacity: 0, y: 32 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-40px" }}
            transition={{ duration: 0.6, delay: i * 0.1 }}
            whileHover={{ y: -6 }}
            onMouseMove={(e) => {
              const r = e.currentTarget.getBoundingClientRect();
              e.currentTarget.style.setProperty("--mx", `${e.clientX - r.left}px`);
              e.currentTarget.style.setProperty("--my", `${e.clientY - r.top}px`);
            }}
            className="group relative overflow-hidden rounded-3xl glass p-8 transition-colors duration-500 hover:border-aurum-400/30"
          >
            {/* カーソルを追うスポットライト */}
            <div
              className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-500 group-hover:opacity-100"
              style={{
                background:
                  "radial-gradient(240px circle at var(--mx, 50%) var(--my, 50%), rgba(240,180,41,0.10), transparent 70%)",
              }}
            />
            <div className="pointer-events-none absolute -right-10 -top-10 h-32 w-32 rounded-full bg-nebula-500/20 blur-3xl transition-opacity duration-500 group-hover:opacity-100 opacity-40" />
            <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-nebula-500/40 to-aurum-400/20 text-aurum-200 transition-transform duration-500 group-hover:-rotate-6 group-hover:scale-110">
              <s.icon size={26} />
            </span>
            <p className="mt-6 font-display text-xs uppercase tracking-widest text-aurum-300/70">{s.en}</p>
            <h3 className="mt-1 font-serif text-2xl text-nebula-100">{s.jp}</h3>
            <p className="mt-4 text-sm leading-relaxed text-nebula-200/70">{s.note}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
