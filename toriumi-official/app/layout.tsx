import type { Metadata } from "next";
import { Sora, Space_Mono, Noto_Sans_JP, Noto_Serif_JP } from "next/font/google";
import JsonLd from "@/components/JsonLd";
import { siteLd } from "@/lib/jsonLd";
import { SITE_URL } from "@/lib/site";
import "./globals.css";

const sora = Sora({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
  weight: ["200", "300", "400", "500", "600", "700", "800"],
});

const spaceMono = Space_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
  weight: ["400", "700"],
});

const notoSans = Noto_Sans_JP({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
  weight: ["300", "400", "500", "700"],
});

const notoSerif = Noto_Serif_JP({
  subsets: ["latin"],
  variable: "--font-serif",
  display: "swap",
  weight: ["400", "600", "700"],
});

export const metadata: Metadata = {
  /**
   * 相対パスを絶対 URL に直すための基点。
   * OG 画像・canonical・sitemap はすべてここを見る。
   * 独自ドメインへ移すときは lib/site.ts の環境変数を設定するだけで済む。
   */
  metadataBase: new URL(SITE_URL),
  title: {
    default: "Katsunori Toriumi — Tricky Multi-Creator / 量子の海を渡る創造者",
    template: "%s — Katsunori Toriumi",
  },
  /**
   * 検索結果に出る一文。世界観だけだと何を頼めるのか伝わらないので、
   * 手がけている仕事を先に置いて、そのあとに世界観を残す。
   */
  description:
    "アプリ開発・Web制作・動画／MV制作を手がけるマルチクリエイター、鳥海勝稚（KIEJI）のオフィシャルサイト。アート・作詞作曲・AIセミナー・手作りグッズ・コンサルまで、領域を超えて創る。",
  applicationName: "Katsunori Toriumi",
  authors: [{ name: "鳥海 勝稚", url: SITE_URL }],
  creator: "鳥海 勝稚",
  alternates: { canonical: "/" },
  openGraph: {
    title: "Katsunori Toriumi — Tricky Multi-Creator",
    description:
      "量子の海を渡る創造者。アプリ開発・Web制作・動画／MV制作から、アート・音楽まで。",
    url: "/",
    siteName: "Katsunori Toriumi",
    locale: "ja_JP",
    type: "website",
  },
  twitter: {
    // 画像を大きく出す。既定の summary だと小さなサムネイルになる
    card: "summary_large_image",
    title: "Katsunori Toriumi — Tricky Multi-Creator",
    description:
      "量子の海を渡る創造者。アプリ開発・Web制作・動画／MV制作から、アート・音楽まで。",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: { index: true, follow: true, "max-image-preview": "large" },
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="ja"
      className={`${sora.variable} ${spaceMono.variable} ${notoSans.variable} ${notoSerif.variable}`}
    >
      <body className="font-sans antialiased">
        {/* 誰が・どんなサイトで・何を請け負うか（全ページ共通） */}
        <JsonLd data={siteLd()} />
        {/* global film grain */}
        <div className="grain-overlay" aria-hidden="true" />
        {children}
      </body>
    </html>
  );
}
