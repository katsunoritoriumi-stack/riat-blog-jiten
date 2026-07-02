"use client";

import { ROLES } from "@/lib/content";

/**
 * セクション間を流れる低速マーキー帯。
 * 肩書き群を静かに循環させ、ページに呼吸のリズムを与える。
 * 左右はマスクでフェードアウトし、視界の端で消える。
 */
export default function MarqueeDivider() {
  // track は同一内容 2 連結（translateX(-50%) でシームレスにループ）
  const items = [...ROLES, ...ROLES];

  return (
    <div className="marquee-mask relative overflow-hidden py-4 opacity-70" aria-hidden="true">
      <div className="marquee-track animate-marquee">
        {[0, 1].map((copy) => (
          <span key={copy} className="inline-flex items-center">
            {items.map((role, i) => (
              <span
                key={`${copy}-${i}`}
                className="inline-flex items-center font-mono text-[11px] uppercase tracking-[0.35em] text-nebula-300/45"
              >
                <span className="px-6">{role}</span>
                <span className="text-aurum-400/50">✦</span>
              </span>
            ))}
          </span>
        ))}
      </div>
    </div>
  );
}
