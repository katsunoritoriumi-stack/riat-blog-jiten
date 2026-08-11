/**
 * Katsunori Toriumi オフィシャルサイト 全コンテンツ
 * ─────────────────────────────────────────────
 * 実画像・プロフィール文・経歴は氏からの正式提供後、
 * このファイルを編集するだけで差し替えできる設計。
 */

export const SITE = {
  nameEn: "Katsunori Toriumi",
  nameJp: "鳥海 勝稚",
  roleEn: "Tricky Multi-Creator",
  taglineJp: "銀河の海の渡り鳥",
} as const;

/** 外部リンク（SNS / ショップ） */
export const LINKS = {
  facebook: "https://www.facebook.com/toriumikatsunori",
  instagram: "https://www.instagram.com/katsunoritoriumi",
  youtube: "https://www.youtube.com/@Exodus999-j7b",
  line: "https://line.me/ti/p/pcon_XLogZ",
  baseQuantum: "https://katsunoritor.thebase.in/",
  baseToriumi: "https://toriumi.thebase.in/",
  produce: "https://10-xi-teal.vercel.app/",
  nirav: "https://nirav.base.shop/",
  radio: "https://stand.fm/channels/605df06b2b49b926c852b5f2",
} as const;

/** YouTube 埋め込み用（チャンネルの埋め込みは uploads プレイリストが安全） */
export const YOUTUBE = {
  channelUrl: LINKS.youtube,
  channelName: "Exodus チャンネル",
} as const;

export type Track = {
  /** ファイル名と DOM のキーに使う識別子。半角英数とハイフンだけ */
  id: string;
  /** 主題名（英語）。曲名の『』は外してある */
  title: string;
  /** 副題。原題の括弧の中身をそのまま置く（日本語のものはそのまま日本語） */
  sub?: string;
  /** 秒。すべて ffprobe の実測値（小数第一位を四捨五入） */
  duration: number;
};

/**
 * Exodus 名義のアルバム。
 *
 * 曲を足すときは tracks に 1 行足すだけでよい（プレイヤー側は触らない）。
 * 音源は public/music/<id>.mp3、曲ごとのアートは public/music/art/<id>.webp。
 * duration は必ず ffprobe の実測値を入れること。目分量で書くと
 * 構造化データ（MusicRecording.duration）に嘘が載る。
 */
export const ALBUM = {
  titleEn: "ANCIENT VOICES",
  titleJp: "古（いにしえ）の声",
  /** 何に対するオマージュなのか。見出しにも構造化データにも同じ値を使う */
  tribute: "Deep Forest",
  note: "深い森のざわめきと、遠い時代のうた。敬愛する Deep Forest へのオマージュとして編んだ11曲。",
  /**
   * ジャケット。MV とは別の作品なので、MV のキーアートは使わない。
   * 曲ごとのアートは元が 360px しかなく大きく出すと粗が見えるため、
   * 最も作品の質感に近い一枚を土台に、拡大・調整して起こしたもの。
   */
  cover: "/music/album-cover.webp",
  tracks: [
    { id: "symphony-of-the-wild", title: "SYMPHONY OF THE WILD", sub: "森のオーケストラ", duration: 211 },
    { id: "cosmic-celebration", title: "COSMIC CELEBRATION", sub: "Galactic Dance", duration: 286 },
    { id: "exodus", title: "EXODUS", sub: "Earth to the Stars", duration: 216 },
    { id: "wewe-ni-upendo", title: "WEWE NI UPENDO", sub: "あなたは愛", duration: 248 },
    { id: "genesis-prima", title: "GENESIS PRIMA", sub: "First Breath of Awakening", duration: 187 },
    { id: "velo-aether", title: "VELO AETHER", sub: "Message from the Air", duration: 223 },
    { id: "celestia", title: "CELÉSTIA", sub: "Ancient Echoes from the Edge", duration: 248 },
    { id: "gaia-primal", title: "GAIA PRIMAL", sub: "Explosion of Life", duration: 156 },
    { id: "lala-salama", title: "LALA SALAMA", sub: "Lullaby Under the Baobab", duration: 209 },
    { id: "miracle-eyes", title: "MIRACLE EYES", duration: 236 },
    { id: "ancient-voices", title: "Ancient Voices in a Digital Biosphere", duration: 249 },
  ] as Track[],
};

export const trackSrc = (t: Track) => `/music/${t.id}.mp3`;
/** 曲目一覧のサムネイル（160px） */
export const trackArt = (t: Track) => `/music/art/${t.id}.webp`;
/** 大ジャケットに出す絵（760px）。選ばれた曲の分だけ読み込む */
export const trackCover = (t: Track) => `/music/cover/${t.id}.webp`;
/** アルバムの総再生時間（秒） */
export const albumDuration = () => ALBUM.tracks.reduce((s, t) => s + t.duration, 0);

export type Domain = {
  key: string;
  titleEn: string;
  titleJp: string;
  blurb: string;
  /** 星座マップ上の相対座標 (0-100) */
  x: number;
  y: number;
  /** 単一リンク */
  href?: string;
  /** 複数リンク（クリックで選択肢を表示） */
  links?: { label: string; href: string }[];
  /** 一旦非公開（座標軸に表示しない） */
  hidden?: boolean;
};

/** 創造の座標軸：六つのカテゴリー（各星はリンク／中心は Connect = LINE） */
export const DOMAINS: Domain[] = [
  // 中心 Connect(50,50) を軸に半径32の正六角形（配列順＝外周のなぞり順）
  { key: "art", titleEn: "Art", titleJp: "アート", blurb: "神話と宇宙のシンボルを描く絵画群。", x: 22, y: 34, href: "https://toriumi.thebase.in/" },
  { key: "youtube", titleEn: "Youtube", titleJp: "映像・MV", blurb: "Exodus として映像世界を編む。", x: 50, y: 18, href: "https://www.youtube.com/@Exodus999-j7b" },
  { key: "fashion", titleEn: "Fashion", titleJp: "着るお守り", blurb: "身にまとう、聖なるアート。", x: 78, y: 34, links: [
    { label: "Original T-shirts", href: "https://katsunoritor.thebase.in/" },
    { label: "Produced Brand", href: "https://nirav.base.shop/" },
  ] },
  { key: "produce", titleEn: "Work", titleJp: "仕事・活動", blurb: "ブランドと体験を生み出す。", x: 78, y: 66, links: [
    { label: "Produce", href: "https://10-xi-teal.vercel.app/" },
    { label: "Message Video", href: "https://anemone-web.com/what-is-anemone/message-video/" },
    { label: "App", href: "/apps/" },
    { label: "Website", href: "/websites/" },
  ] },
  { key: "sns", titleEn: "SNS", titleJp: "発信", blurb: "日々の創造を発信する。", x: 50, y: 82, links: [
    { label: "Instagram", href: "https://www.instagram.com/katsunoritoriumi" },
    { label: "Facebook", href: "https://www.facebook.com/toriumikatsunori" },
  ] },
  { key: "work", titleEn: "Work", titleJp: "制作物", blurb: "世に放ったアプリとウェブサイト。", x: 24, y: 70, href: "#work", hidden: true },
  { key: "radio", titleEn: "Radio", titleJp: "音声配信", blurb: "声で綴る、日々の宇宙。stand.fm にて配信中。", x: 22, y: 66, href: "https://stand.fm/channels/605df06b2b49b926c852b5f2" },
  { key: "connect", titleEn: "Connect", titleJp: "つながる", blurb: "LINE でいつでも、創造の宇宙へ。", x: 50, y: 50, href: "https://line.me/ti/p/pcon_XLogZ" },
];

export type Artwork = {
  title: string;
  sub?: string;
  tier: "premium" | "digital";
  /** SVG シジル（量子モチーフ）の種別 */
  sigil: "sun" | "wind" | "goddess" | "kannon" | "crow" | "eye" | "scarab" | "quantum" | "spiral" | "lotus";
  hue: string;
};

/** Quantum Art 作品（参照ショップから抽出） */
export const ARTWORKS: Artwork[] = [
  { title: "To the Sky", sub: "天への上昇", tier: "premium", sigil: "wind", hue: "#60a5fa" },
  { title: "Winds of Change", sub: "変化の風", tier: "premium", sigil: "spiral", hue: "#5eead4" },
  { title: "木花咲耶姫", sub: "Konohanasakuyahime", tier: "premium", sigil: "lotus", hue: "#f472b6" },
  { title: "古代の女神", sub: "Ancient Goddess", tier: "premium", sigil: "goddess", hue: "#a78bfa" },
  { title: "Drops of the Sun", sub: "太陽の雫", tier: "premium", sigil: "sun", hue: "#f0b429" },
  { title: "観音", sub: "Kannon — Forgiveness", tier: "digital", sigil: "kannon", hue: "#a78bfa" },
  { title: "ヤタガラス", sub: "Three-legged Crow", tier: "digital", sigil: "crow", hue: "#5eead4" },
  { title: "ホルスの目", sub: "Eye of Horus", tier: "digital", sigil: "eye", hue: "#f0b429" },
  { title: "スカラベ", sub: "Scarab", tier: "digital", sigil: "scarab", hue: "#60a5fa" },
  { title: "クォンタム", sub: "Quantum", tier: "digital", sigil: "quantum", hue: "#f472b6" },
];

/** マーキー帯に流す肩書き群 */
export const ROLES: string[] = [
  "Art",
  "作詞作曲",
  "App / Web",
  "MV / Video",
  "Work",
  "AI Seminar",
  "Handmade Goods",
  "Consulting",
  "Produce",
];

export type NiravItem = { title: string; en: string; note: string };

/** Produce「着るお守り」Holy Art アパレル */
export const NIRAV_ITEMS: NiravItem[] = [
  { title: "天の鳥船", en: "Bird Ship in Heaven", note: "宇宙と創造性を呼ぶ" },
  { title: "愛の王", en: "King of Love", note: "真実の愛を深める" },
  { title: "光明", en: "Flower of Enlightenment", note: "内なる静けさへ" },
  { title: "みちびき", en: "Lead", note: "魂を導く力" },
  { title: "上昇", en: "Ascent", note: "高次の意識へ" },
  { title: "薬師如来", en: "Yakushi", note: "癒しと健やかさ" },
];

export type WorkItem = {
  title: string;
  type: string;
  note: string;
  href?: string; // 後でアプリ／ウェブサイトのURLを挿入
  hue: string;
};

/** Work：制作したアプリ／ウェブサイト（タイトル・説明はURLから推定。要確認） */
export const WORKS: WorkItem[] = [
  { title: "RIAT Quiz", type: "Web App", note: "知識クイズアプリ", href: "https://riat-quiz-frontend.vercel.app/", hue: "#f0b429" },
  { title: "Voyage", type: "AI Travel", note: "AI旅行プランナー", href: "https://voyage-ai-travel-planner-8gw80ve11.vercel.app/", hue: "#60a5fa" },
  { title: "献立マスター", type: "Web App", note: "献立プランナー", href: "https://kondate-master.vercel.app/", hue: "#5eead4" },
  { title: "数秘計算", type: "Web App", note: "数秘術カリキュレーター", href: "https://suuhi-keisan-v2.vercel.app/", hue: "#a78bfa" },
  { title: "タネと菜園", type: "Web App", note: "家庭菜園サポート", href: "https://tanetosaien.vercel.app/", hue: "#bef264" },
  { title: "Dragon Shrine Checker", type: "Web App", note: "龍神・神社チェッカー", href: "https://dragon-shrine-checker.vercel.app/", hue: "#f472b6" },
  { title: "静かな電子銀河", type: "Web", note: "アンビエント電子銀河", href: "https://shizuka-na-denshi-ginga.vercel.app/", hue: "#c4b5fd" },
  { title: "RIAT ブログ事典", type: "AI Search", note: "ブログ横断のAI検索辞典", href: "https://riat-blog-jiten-2.onrender.com/", hue: "#fb923c" },
];

/** Work（座標軸）→ App サブページに並べるアプリ群 */
export const APPS: WorkItem[] = [
  { title: "Earth Watch", type: "3D App", note: "Cesium製・地球可視化ダッシュボード", href: "https://earth-watch-gilt.vercel.app/", hue: "#60a5fa" },
  { title: "AI Atlas", type: "3D Graph", note: "AI勢力図を描く3Dグラフ", href: "https://ai-atlas-murex.vercel.app/", hue: "#a78bfa" },
  { title: "献立マスター", type: "Web App", note: "献立プランナー", href: "https://kondate-master.vercel.app/", hue: "#5eead4" },
  { title: "Voyage", type: "AI Travel", note: "AI旅行プランナー", href: "https://voyage-ai-travel-planner-8gw80ve11.vercel.app/", hue: "#f0b429" },
];

/** Work（座標軸）→ Website サブページに並べるサイト群 */
export const WEBSITES: WorkItem[] = [
  { title: "タネと菜園", type: "Website", note: "家庭菜園サポート", href: "https://tanetosaien.vercel.app/", hue: "#bef264" },
  { title: "Beautiful Aging", type: "Website", note: "荻原彩子のジェロントロジー講座", href: "https://beautiful-aging.vercel.app/index.html", hue: "#f472b6" },
];

/** Manifesto 本文 */
export const MANIFESTO = {
  eyebrow: "MANIFESTO",
  lines: [
    "ひとりの人間の中に、",
    "いくつもの宇宙が同時に存在している。",
  ],
  body:
    "絵を描き、歌を紡ぎ、コードを書き、形をつくる。鳥海勝稚にとって、表現の領域に境界はない。アートも、音楽も、テクノロジーも——すべては同じひとつの創造のエネルギーが、異なる次元に立ち現れた姿にすぎない。観測されるまで、すべての可能性は重なり合っている。彼の創造は、その重なりを一枚の絵に、一曲の歌に、ひとつの作品に収束させる行為だ。",
};

export const SECTIONS = [
  { id: "home", label: "Home" },
  { id: "universe", label: "Universe" },
  { id: "sound", label: "Sound & Vision" },
  // { id: "work", label: "Work" }, // Collection 一旦非公開
  { id: "connect", label: "Connect" },
] as const;
