import type { Metadata } from "next";
import ShowcaseGrid from "@/components/ShowcaseGrid";
import { WEBSITES } from "@/lib/content";

export const metadata: Metadata = {
  title: "Website — Katsunori Toriumi",
  description: "鳥海勝稚が制作したウェブサイト群。",
};

export default function WebsitesPage() {
  return (
    <ShowcaseGrid
      eyebrow="Work / Website"
      titleEn="Websites"
      titleJp="制作したウェブサイト"
      items={WEBSITES}
    />
  );
}
