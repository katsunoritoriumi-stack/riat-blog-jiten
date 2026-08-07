/**
 * ホームの「旅程」— 各ステーションの id・スクロール配分・予告編コピー。
 *
 * ここを唯一の出典にしている理由：
 * 見せ方（ZoomStage）と宇宙（Universe）が同じ配分を見ないと、
 * 「セクションが出る位置」と「宇宙で何かが起きる位置」がずれる。
 * どちらも計算式を持たず、この 1 ファイルから読む。
 *
 * DOM にも THREE にも依存しないので、帯の計算は node で検証できる。
 */

export type StationCaption = {
  en: string;
  jp: string;
  /** 予告編カードに出す数え上げ */
  counter?: { to: number; en: string; jp: string };
};

export type StationSpec = {
  id: string;
  /** そのステーションに割り当てる移動距離（画面高さの倍数）。大きいほど航行が長い */
  scroll: number;
  caption?: StationCaption;
  /**
   * true のとき、このステーションはクリックを受け取らず背後の宇宙へ通す。
   * 中身が「宇宙に浮かぶもの」自体で、DOM は見出しだけ——という場合に使う。
   */
  passthrough?: boolean;
};

export const STATIONS: StationSpec[] = [
  {
    id: "home",
    scroll: 2.2,
    caption: { en: "One human. Many universes.", jp: "ひとりの中に、いくつもの宇宙" },
  },
  {
    id: "manifesto",
    scroll: 2.9,
    caption: { en: "Six worlds orbit a single light.", jp: "六つの世界が、ひとつの光を巡る" },
  },
  {
    // 太陽系は背景の宇宙側にある。DOM は見出しだけなのでクリックは通す
    id: "universe",
    scroll: 3.6,
    caption: { en: "Some of them sing.", jp: "歌になるもの" },
    passthrough: true,
  },
  /**
   * 音楽と映像には独立した作品が 2 つある（MV とアルバム）。
   * 1 つのステーションに両方を積むと画面 2.6 枚ぶんの高さになり、
   * 中央寄せの結果どちらも上下が切れる。作品ごとに 1 画面を与える。
   */
  {
    // caption を持たせない。MV とアルバムは同じ「音楽と映像」の続きなので、
    // 間に予告編を挟むと別の章が始まったように見える
    id: "sound",
    scroll: 2.6,
  },
  {
    id: "album",
    scroll: 2.9,
    caption: { en: "Some of them are built.", jp: "形になるもの" },
  },
  {
    id: "digital",
    scroll: 2.9,
    caption: { en: "The shape of creation.", jp: "創造の形" },
  },
  {
    id: "connect",
    scroll: 3.4,
    caption: {
      en: "This is the final transmission.",
      jp: "これが、最後の通信",
      counter: { to: 7400, en: "Reincarnations on Earth", jp: "地球での転生回数" },
    },
  },
  {
    id: "report",
    scroll: 3.4,
    caption: { en: "A grand voyage begins.", jp: "壮大な旅の始まり" },
  },
  { id: "end", scroll: 2.6 },
];

export type Band = { start: number; span: number };

/** 重みを 0〜1 の帯に割り付ける */
export function computeBands(weights: number[]): Band[] {
  const total = weights.reduce((a, b) => a + b, 0);
  let acc = 0;
  return weights.map((w) => {
    const start = acc / total;
    acc += w;
    return { start, span: w / total };
  });
}

export const STATION_WEIGHTS = STATIONS.map((s) => s.scroll);
export const STATION_BANDS = computeBands(STATION_WEIGHTS);
/** 全体で何画面ぶんスクロールするか */
export const TOTAL_SCREENS = STATION_WEIGHTS.reduce((a, b) => a + b, 0);

/** id から帯を引く（見つからなければ null） */
export function bandOf(id: string): Band | null {
  const i = STATIONS.findIndex((s) => s.id === id);
  return i < 0 ? null : STATION_BANDS[i];
}

/**
 * ステーション帯の中の位置 q（0=到着, 0.34=滞在終わり, 0.52=通過完了）。
 * ZoomStage の定数と対応している。宇宙側はこれを見て
 * 「いま航行区間か、滞在中か」を判断する。
 */
export const Q_ARRIVE = 0;
export const Q_HOLD_END = 0.34;
export const Q_PASSED = 0.52;

/** 全体進捗 prog(0-1) を、ステーション index と帯内位置 q に分解する */
export function locate(prog: number): { index: number; q: number } {
  for (let i = STATION_BANDS.length - 1; i >= 0; i--) {
    const b = STATION_BANDS[i];
    if (prog >= b.start || i === 0) return { index: i, q: (prog - b.start) / b.span };
  }
  return { index: 0, q: 0 };
}
