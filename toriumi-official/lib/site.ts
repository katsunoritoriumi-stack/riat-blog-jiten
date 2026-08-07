/**
 * サイトの公開 URL。
 *
 * 独自ドメインへ移す日が来たら、Vercel の環境変数
 * NEXT_PUBLIC_SITE_URL を設定するだけで
 * canonical・sitemap・robots・OG 画像の絶対 URL がまとめて切り替わる。
 * （コード側は 1 か所も直さなくてよい）
 */
export const SITE_URL = (
  process.env.NEXT_PUBLIC_SITE_URL ?? "https://toriumi-official.vercel.app"
).replace(/\/$/, "");

/** 静的エクスポートは trailingSlash: true なので、パスは末尾スラッシュで揃える */
export const ROUTES = ["/", "/apps/", "/websites/"] as const;

export const abs = (path: string) => `${SITE_URL}${path}`;

/**
 * 共有カード用の画像。app/opengraph-image.jpg が実体。
 * ページ側で openGraph を上書きすると親から継承されなくなるので、
 * サブページではこれを明示的に渡す。
 */
export const OG_IMAGE = "/opengraph-image.jpg";

/**
 * Google Search Console の所有権確認トークン。
 * 「HTMLタグ」方式で出る <meta content="..."> の中身だけを、
 * Vercel の環境変数 NEXT_PUBLIC_GOOGLE_SITE_VERIFICATION に入れる。
 * 空なら meta タグ自体を出さない（未設定のタグを残さないため）。
 */
export const GOOGLE_SITE_VERIFICATION =
  process.env.NEXT_PUBLIC_GOOGLE_SITE_VERIFICATION?.trim() || "";
