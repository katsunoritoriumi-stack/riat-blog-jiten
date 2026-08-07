import type { MetadataRoute } from "next";
import { abs, ROUTES } from "@/lib/site";

/**
 * out/sitemap.xml を生成する。
 * ページは 3 つだけなので lib/site.ts の ROUTES をそのまま並べる。
 * lastModified はビルド時刻＝デプロイした日時になる。
 */
export const dynamic = "force-static";

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();
  return ROUTES.map((path) => ({
    url: abs(path),
    lastModified: now,
    changeFrequency: "monthly" as const,
    // トップを主役に。制作物一覧はその次
    priority: path === "/" ? 1 : 0.8,
  }));
}
