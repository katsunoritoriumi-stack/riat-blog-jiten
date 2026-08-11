"use client";

/**
 * 終章（SIGNAL LOST → ロゴの署名）の背景。
 * 画面いっぱいに絵を敷き、その上に文字とロゴが乗る。
 *
 * 絵は画面の形に合わせて 2 枚を出し分ける：
 *   ・横長（PC）… 人物が左端、右いっぱいに銀河と円盤が広がる構図
 *   ・縦長（スマホ）… 人物が中央にいる構図。横長の絵を縦に切ると人物が外れるため
 * <picture> の media で切り替えるので、読み込まれるのは片方だけ。
 *
 * 絵の中の本には FinaleBookLink がリンクを重ねる。位置合わせのため、
 * ここでは object-fit を使わず .finale-frame（絵の比率のまま画面を覆う箱）に
 * 収めている。寄りの動きも CSS 側（.finale-zoom）に持たせて、
 * リンク側の枠と必ず同じ進行で動くようにしてある。
 *
 * 暗幕は「上を軽く・下を強く」。上は円盤と銀河という見せ場なので沈めすぎず、
 * 下はロゴの署名の座を作るためにしっかり落とす。
 */
export default function FinaleBackdrop() {
  return (
    <div
      // overflow は clip。hidden だとスクロール可能な箱になり、リンク側の層とずれる
      className="pointer-events-none absolute inset-0 -z-10 overflow-clip"
      style={{ containerType: "size" }}
    >
      {/* 絵。ゆっくり寄っていく（宇宙の只中に留まっている感じ） */}
      <div className="finale-frame finale-zoom">
        <picture>
          <source media="(min-width: 768px)" srcSet="/finale-pc.webp" />
          <img
            src="/finale-mobile.webp"
            alt=""
            aria-hidden="true"
            width={1100}
            height={1310}
            loading="lazy"
            className="h-full w-full"
          />
        </picture>
      </div>

      {/* 上下から締める暗幕：上は見せ場なので軽く、下はロゴの座なので強く */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(180deg, rgba(3,2,10,0.72) 0%, rgba(3,2,10,0.24) 22%, rgba(3,2,10,0.18) 48%, rgba(3,2,10,0.72) 78%, rgba(3,2,10,0.96) 100%)",
        }}
      />
      {/* 四隅を落として画面の中心へ視線を集める */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at 52% 48%, transparent 32%, rgba(3,2,10,0.4) 80%, rgba(3,2,10,0.8) 100%)",
        }}
      />
      {/* サイト全体の紫を薄くかぶせて世界観に馴染ませる（絵の金と青は殺さない） */}
      <div
        className="absolute inset-0 mix-blend-soft-light"
        style={{ background: "linear-gradient(180deg, rgba(124,58,237,0.3), rgba(38,20,90,0.24))" }}
      />
    </div>
  );
}
