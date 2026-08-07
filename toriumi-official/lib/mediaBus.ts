/**
 * ページ内で音が鳴るものを 1 つに絞るための調停役。
 *
 * SOUND & VISION には MV とアルバムの 2 つの音源が並ぶ。
 * 片方を鳴らしたらもう片方は必ず止めないと、音が重なって聴けたものではない。
 * かといって親コンポーネントに ref を持ち回らせると配線が増えるので、
 * 「鳴らす側が名乗り出たら、他は自分で止まる」という形にした。
 *
 * React の外（イベントハンドラの中）から同期的に呼べることが重要。
 * iOS では play() をユーザー操作と同じターンで呼ばないと弾かれるため、
 * ここに await や setState を挟んではいけない。
 */

type Stopper = () => void;

const stoppers = new Set<Stopper>();

/** 音源を登録する。戻り値を呼ぶと解除（useEffect のクリーンアップにそのまま渡せる） */
export function registerMedia(stop: Stopper): () => void {
  stoppers.add(stop);
  return () => {
    stoppers.delete(stop);
  };
}

/** これから鳴らす、と名乗り出る。自分以外はすべて止まる */
export function claimPlayback(self: Stopper) {
  for (const stop of stoppers) {
    if (stop !== self) stop();
  }
}

/** 全部止める（セクションから離れたときなど） */
export function stopAllMedia() {
  for (const stop of stoppers) stop();
}
