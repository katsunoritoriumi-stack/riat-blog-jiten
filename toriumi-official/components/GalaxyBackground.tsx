"use client";

import Galaxy from "./ui/Galaxy";

/**
 * サイト全体に固定表示する回転銀河の背景。スクロールしても常に背景に残る。
 * 可読性のため上にダークなヴィネット／オーバーレイを重ねる。
 */
export default function GalaxyBackground() {
  return (
    <div className="fixed inset-0 z-0 overflow-hidden bg-void-950" aria-hidden="true">
      {/* rotating galaxy */}
      <div className="absolute left-1/2 top-1/2 aspect-square w-[160vmin] -translate-x-1/2 -translate-y-1/2">
        <Galaxy density={0.6} />
      </div>
      {/* readability overlay */}
      <div className="absolute inset-0 bg-void-950/55" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_30%,rgba(3,2,10,0.85)_85%)]" />
    </div>
  );
}
