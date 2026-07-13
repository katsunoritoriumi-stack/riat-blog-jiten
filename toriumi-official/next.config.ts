import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 完全静的サイトのため静的エクスポート（out/）。Vercel のディレクトリ検出に依存せず確実に配信できる。
  output: "export",
  // /apps・/websites 等サブページを route/index.html 形式で出力し、
  // どの静的ホストでも確実にクリーンURLが解決できるようにする。
  trailingSlash: true,
  turbopack: {
    root: __dirname,
  },
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
