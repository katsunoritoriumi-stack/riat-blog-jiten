import type { Metadata } from "next";
import JsonLd from "@/components/JsonLd";
import ShowcaseGrid from "@/components/ShowcaseGrid";
import { WEBSITES } from "@/lib/content";
import { websitesLd } from "@/lib/jsonLd";
import { OG_IMAGE } from "@/lib/site";

const DESCRIPTION =
  "鳥海勝稚（KIEJI）が制作したウェブサイトの一覧。デザインからコードまで一貫して仕立てています。ホームページ制作のご依頼も承ります。";

export const metadata: Metadata = {
  title: "制作したウェブサイト",
  description: DESCRIPTION,
  alternates: { canonical: "/websites/" },
  // 継承のままだとホームと同じ文言になり、共有したときに区別がつかない
  openGraph: {
    title: "制作したウェブサイト — Katsunori Toriumi",
    description: DESCRIPTION,
    url: "/websites/",
    type: "website",
    images: [OG_IMAGE],
  },
  twitter: {
    card: "summary_large_image",
    title: "制作したウェブサイト — Katsunori Toriumi",
    description: DESCRIPTION,
    images: [OG_IMAGE],
  },
};

export default function WebsitesPage() {
  return (
    <>
      <JsonLd data={websitesLd()} />
      <ShowcaseGrid
        eyebrow="Work / Website"
        titleEn="Websites"
        titleJp="制作したウェブサイト"
        items={WEBSITES}
      />
    </>
  );
}
