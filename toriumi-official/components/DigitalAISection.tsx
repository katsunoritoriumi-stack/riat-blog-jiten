"use client";

import { motion } from "framer-motion";
import { Code2, Globe, Film } from "lucide-react";
import SectionHeader from "./ui/SectionHeader";
import { LINKS } from "@/lib/content";

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
    <section data-section="digital" className="relative mx-auto max-w-7xl px-6 py-28 sm:py-36">
      <div className="mb-16 max-w-2xl">
        <SectionHeader eyebrow="Job request" titleEn="Make" />
      </div>

      {/* リンクではなく「制作メニューの紹介」。ボタンに見えないエディトリアル表現。 */}
      <div className="grid gap-x-10 gap-y-12 md:grid-cols-3">
        {services.map((s, i) => (
          <motion.div
            key={s.en}
            initial={{ opacity: 0, y: 28 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-40px" }}
            transition={{ duration: 0.6, delay: i * 0.1 }}
            className="relative border-t border-nebula-500/20 pt-6"
          >
            <span className="absolute right-0 top-6 font-mono text-xs text-nebula-400/40">
              0{i + 1}
            </span>
            <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-nebula-500/25 to-aurum-400/10 text-aurum-200/90 ring-1 ring-inset ring-nebula-400/15">
              <s.icon size={22} strokeWidth={1.6} />
            </span>
            <p className="mt-5 font-mono text-[11px] uppercase tracking-[0.25em] text-aurum-300/60">
              {s.en}
            </p>
            <h3 className="mt-1.5 font-serif text-2xl text-nebula-100">{s.jp}</h3>
            <p className="mt-3 text-sm leading-relaxed text-nebula-200/65">{s.note}</p>
          </motion.div>
        ))}
      </div>

      {/* 依頼はどこから？の案内（実際の導線は LINE へ直接） */}
      <p className="mt-14 text-center font-serif text-sm text-nebula-300/60">
        ご依頼・ご相談は
        <a
          href={LINKS.line}
          target="_blank"
          rel="noopener noreferrer"
          className="mx-1 text-aurum-200 underline decoration-aurum-400/40 underline-offset-4 transition-colors hover:decoration-aurum-300"
        >
          Connect
        </a>
        から。
      </p>
    </section>
  );
}
