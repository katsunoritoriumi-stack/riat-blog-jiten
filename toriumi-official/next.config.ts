import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 完全静的サイトのため静的エクスポート（out/）。Vercel のディレクトリ検出に依存せず確実に配信できる。
  output: "export",
  turbopack: {
    root: __dirname,
  },
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
