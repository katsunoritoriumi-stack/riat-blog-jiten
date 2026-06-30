import type { Metadata } from "next";
import { Sora, Space_Mono, Noto_Sans_JP, Noto_Serif_JP } from "next/font/google";
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
  title: "Katsunori Toriumi — Tricky Multi-Creator / 量子の海を渡る創造者",
  description:
    "アート・作詞作曲・アプリ／Web制作・MV／動画・AIセミナー・手作りグッズ・コンサル。領域を超えて創造するマルチクリエイター、鳥海勝稚のオフィシャルサイト。創造者＝ひとつの宇宙を描く。",
  openGraph: {
    title: "Katsunori Toriumi — Tricky Multi-Creator",
    description: "量子の海を渡る創造者。多次元の創造を巡る。",
    type: "website",
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
        {/* global film grain */}
        <div className="grain-overlay" aria-hidden="true" />
        {children}
      </body>
    </html>
  );
}
