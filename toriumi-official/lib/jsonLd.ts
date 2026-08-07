import { APPS, LINKS, SITE, WEBSITES, YOUTUBE, type WorkItem } from "./content";
import { abs, SITE_URL } from "./site";

/**
 * 構造化データ（JSON-LD）の組み立て。
 *
 * 検索エンジンに「これは誰で、何をやっていて、どこに他の顔があるのか」を
 * 機械可読で伝える。とくに sameAs は、散らばった SNS を同一人物として
 * 束ねる唯一の標準的な手段で、指名検索の見え方に効く。
 *
 * データは lib/content.ts の実データから組み立て、二重管理しない。
 * DOM にも React にも依存しないので node で検証できる。
 */

/** 事業として掲げているサービス。ページ本文（DigitalAISection / マーキー帯）と揃える */
const SERVICES = [
  { name: "アプリ開発", description: "アイデアを、触れて動く形へ。設計から実装まで一気通貫で創る。" },
  { name: "Web制作", description: "世界観を宿したサイトを、デザインからコードまで仕立てる。" },
  { name: "動画・MV制作", description: "物語を映像に。撮影・編集・MVまで、イメージを動かす。" },
  { name: "AIセミナー", description: "生成AIの使いどころを、実務の手触りで伝える。" },
  { name: "コンサルティング", description: "ブランドと体験の設計を伴走する。" },
  { name: "ハンドメイドグッズ", description: "身にまとう、聖なるアート。" },
] as const;

const PERSON_ID = abs("/#person");
const SITE_ID = abs("/#website");
const SERVICE_ID = abs("/#service");

/** 各SNS・ショップ。ここに並べたURLが「同じ人物の別の顔」として扱われる */
const sameAs = [
  LINKS.youtube,
  LINKS.instagram,
  LINKS.facebook,
  LINKS.radio,
  LINKS.baseToriumi,
  LINKS.baseQuantum,
  LINKS.nirav,
];

export function personLd() {
  return {
    "@type": "Person",
    "@id": PERSON_ID,
    name: SITE.nameJp,
    alternateName: [SITE.nameEn, "KIEJI", "K.TORIUMI"],
    jobTitle: SITE.roleEn,
    description:
      "アプリ開発・Web制作・動画／MV制作・アート・作詞作曲・AIセミナーを手がけるマルチクリエイター。",
    url: SITE_URL,
    image: abs("/logo-ktoriumi.webp"),
    sameAs,
    knowsAbout: SERVICES.map((s) => s.name),
  };
}

export function websiteLd() {
  return {
    "@type": "WebSite",
    "@id": SITE_ID,
    name: `${SITE.nameEn} — ${SITE.roleEn}`,
    url: SITE_URL,
    inLanguage: "ja",
    publisher: { "@id": PERSON_ID },
  };
}

/**
 * 仕事の受け口。
 * areaServed は「どこの依頼を受けるか」。住所は実在のものが無いと
 * ローカル検索の対象にならないため、あえて置いていない。
 */
export function serviceLd() {
  return {
    "@type": "ProfessionalService",
    "@id": SERVICE_ID,
    name: `${SITE.nameEn} — 制作・開発`,
    url: SITE_URL,
    image: abs("/opengraph-image.jpg"),
    founder: { "@id": PERSON_ID },
    provider: { "@id": PERSON_ID },
    areaServed: [
      { "@type": "AdministrativeArea", name: "長野県" },
      { "@type": "Country", name: "日本" },
    ],
    availableLanguage: ["ja"],
    hasOfferCatalog: {
      "@type": "OfferCatalog",
      name: "制作メニュー",
      itemListElement: SERVICES.map((s) => ({
        "@type": "Offer",
        itemOffered: { "@type": "Service", name: s.name, description: s.description },
      })),
    },
  };
}

/** テーマソングのMV。動画そのものをこのサイトで配信しているので明示する */
export function videoLd() {
  return {
    "@type": "VideoObject",
    name: "星の彼方へ / Beyond the Stars — オリジナルMV",
    description:
      "Exodus 名義のオリジナル楽曲「星の彼方へ」のミュージックビデオ。魂は、もっと自由になる。",
    thumbnailUrl: abs("/theme-mv-poster.webp"),
    contentUrl: abs("/theme-mv.mp4"),
    embedUrl: SITE_URL,
    // 実測 157.9 秒
    duration: "PT2M38S",
    uploadDate: "2026-08-01",
    creator: { "@id": PERSON_ID },
    inLanguage: "ja",
  };
}

/** 制作物の一覧ページ（/apps・/websites）用 */
export function worksLd(opts: {
  path: string;
  name: string;
  description: string;
  items: readonly WorkItem[];
}) {
  return {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "CollectionPage",
        "@id": abs(opts.path),
        name: opts.name,
        description: opts.description,
        url: abs(opts.path),
        inLanguage: "ja",
        isPartOf: { "@id": SITE_ID },
        about: { "@id": PERSON_ID },
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Home", item: abs("/") },
          { "@type": "ListItem", position: 2, name: opts.name, item: abs(opts.path) },
        ],
      },
      {
        "@type": "ItemList",
        name: opts.name,
        numberOfItems: opts.items.length,
        itemListElement: opts.items.map((w, i) => ({
          "@type": "ListItem",
          position: i + 1,
          name: w.title,
          description: w.note,
          ...(w.href ? { url: w.href } : {}),
        })),
      },
    ],
  };
}

/**
 * 全ページ共通（layout に置く）。
 * 「誰が」「どんなサイトで」「何を請け負うか」はどのページから来ても同じなので、
 * @id を振って一度だけ定義し、各ページからは参照するだけにする。
 */
export function siteLd() {
  return {
    "@context": "https://schema.org",
    "@graph": [personLd(), websiteLd(), serviceLd()],
  };
}

/** ホーム固有（MV はこのページにしか無いのでここに置く） */
export function homePageLd() {
  return {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebPage",
        "@id": abs("/"),
        url: abs("/"),
        name: `${SITE.nameEn} — ${SITE.roleEn}`,
        inLanguage: "ja",
        isPartOf: { "@id": SITE_ID },
        about: { "@id": PERSON_ID },
        primaryImageOfPage: abs("/opengraph-image.jpg"),
      },
      videoLd(),
    ],
  };
}

export const appsLd = () =>
  worksLd({
    path: "/apps/",
    name: "制作したアプリケーション",
    description: `${SITE.nameJp}が企画・開発したWebアプリ。${YOUTUBE.channelName}と同じ世界観で作っている。`,
    items: APPS,
  });

export const websitesLd = () =>
  worksLd({
    path: "/websites/",
    name: "制作したウェブサイト",
    description: `${SITE.nameJp}が制作したウェブサイトの一覧。`,
    items: WEBSITES,
  });
