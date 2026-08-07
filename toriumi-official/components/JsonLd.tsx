/**
 * 構造化データを HTML に埋め込むだけの部品（サーバー側で描かれる）。
 *
 * dangerouslySetInnerHTML を使うのは、React が JSON 文字列内の記号を
 * エスケープしてしまい、検索エンジンが読めない形になるため。
 * 渡すのは自分たちが lib/jsonLd.ts で組み立てたオブジェクトだけで、
 * 外部入力は通さない。念のため `<` だけは実体参照に置き換えて、
 * script タグが途中で閉じられないようにしておく。
 */
export default function JsonLd({ data }: { data: object }) {
  const json = JSON.stringify(data).replace(/</g, "\\u003c");
  return (
    <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: json }} />
  );
}
