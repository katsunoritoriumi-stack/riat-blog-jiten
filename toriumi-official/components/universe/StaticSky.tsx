/**
 * WebGL が立ち上がる前（と、使えない環境）に出しておく静止した宇宙。
 *
 * three.js を含まない極小のファイルとして独立させてある。
 * Universe.tsx（three.js 入り）から export してしまうと、
 * これを参照するだけで three.js が初期バンドルに引きずり込まれるため。
 */
export default function StaticSky() {
  return (
    <div className="fixed inset-0 z-0 overflow-hidden bg-void-950" aria-hidden="true">
      <div className="absolute inset-0 universe-fallback" />
    </div>
  );
}
