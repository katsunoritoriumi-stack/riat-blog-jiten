import type { MetadataRoute } from "next";
import { abs } from "@/lib/site";

/**
 * out/robots.txt を生成する（output: "export" でもビルド時に静的出力される）。
 * 隠したいページは無いので全許可。sitemap の在り処だけ明示する。
 */
export const dynamic = "force-static";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [{ userAgent: "*", allow: "/" }],
    sitemap: abs("/sitemap.xml"),
    host: abs("/"),
  };
}
