import type { Metadata } from "next";
import JsonLd from "@/components/JsonLd";
import ShowcaseGrid from "@/components/ShowcaseGrid";
import { APPS } from "@/lib/content";
import { appsLd } from "@/lib/jsonLd";
import { OG_IMAGE } from "@/lib/site";

const DESCRIPTION =
  "鳥海勝稚（KIEJI）が企画から開発まで手がけたWebアプリの一覧。3D地球儀・AI旅行プランナー・献立プランナーなど。アプリ開発のご依頼も承ります。";

export const metadata: Metadata = {
  title: "制作したアプリケーション",
  description: DESCRIPTION,
  alternates: { canonical: "/apps/" },
  // 継承のままだとホームと同じ文言になり、共有したときに区別がつかない
  openGraph: {
    title: "制作したアプリケーション — Katsunori Toriumi",
    description: DESCRIPTION,
    url: "/apps/",
    type: "website",
    images: [OG_IMAGE],
  },
  twitter: {
    card: "summary_large_image",
    title: "制作したアプリケーション — Katsunori Toriumi",
    description: DESCRIPTION,
    images: [OG_IMAGE],
  },
};

export default function AppsPage() {
  return (
    <>
      <JsonLd data={appsLd()} />
      <ShowcaseGrid
        eyebrow="Work / App"
        titleEn="Apps"
        titleJp="制作したアプリケーション"
        items={APPS}
      />
    </>
  );
}
