import type { Metadata } from "next";
import ShowcaseGrid from "@/components/ShowcaseGrid";
import { APPS } from "@/lib/content";

export const metadata: Metadata = {
  title: "App — Katsunori Toriumi",
  description: "鳥海勝稚が制作したアプリ群。",
};

export default function AppsPage() {
  return (
    <ShowcaseGrid
      eyebrow="Work / App"
      titleEn="Apps"
      titleJp="制作したアプリケーション"
      items={APPS}
    />
  );
}
