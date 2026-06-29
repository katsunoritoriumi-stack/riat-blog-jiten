"use client";

import type { Artwork } from "@/lib/content";

/**
 * 各 Quantum Art 作品の量子モチーフを表す、オリジナルの SVG シジル。
 * 実画像が用意できるまでのプレースホルダー兼ビジュアル。
 * （既存作品の模写ではなく、抽象化した象徴的グリフ）
 */
export default function Sigil({
  type,
  hue,
  className = "",
}: {
  type: Artwork["sigil"];
  hue: string;
  className?: string;
}) {
  const stroke = hue;
  const common = {
    fill: "none",
    stroke,
    strokeWidth: 1.4,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
  };

  return (
    <svg
      viewBox="0 0 100 100"
      className={className}
      style={{ filter: `drop-shadow(0 0 6px ${hue}aa)` }}
      aria-hidden="true"
    >
      <g {...common}>
        {type === "sun" && (
          <>
            <circle cx="50" cy="50" r="16" />
            {Array.from({ length: 12 }).map((_, i) => {
              const a = (i / 12) * Math.PI * 2;
              const r = (v: number) => Math.round(v * 100) / 100;
              return (
                <line
                  key={i}
                  x1={r(50 + Math.cos(a) * 24)}
                  y1={r(50 + Math.sin(a) * 24)}
                  x2={r(50 + Math.cos(a) * 36)}
                  y2={r(50 + Math.sin(a) * 36)}
                />
              );
            })}
            <circle cx="50" cy="50" r="6" />
          </>
        )}

        {type === "wind" && (
          <>
            <path d="M20 40 q30 -18 50 0 t10 6" />
            <path d="M16 54 q34 -14 60 2" />
            <path d="M24 68 q26 -10 44 0" />
            <circle cx="78" cy="32" r="4" />
          </>
        )}

        {type === "goddess" && (
          <>
            <circle cx="50" cy="30" r="10" />
            <path d="M30 78 q20 -34 40 0" />
            <path d="M50 40 v22" />
            <path d="M34 86 q16 -10 32 0" />
            <circle cx="50" cy="14" r="2.5" />
          </>
        )}

        {type === "lotus" && (
          <>
            <path d="M50 74 C34 60 34 40 50 26 C66 40 66 60 50 74 Z" />
            <path d="M50 74 C30 66 22 50 26 36 C42 42 50 56 50 74 Z" />
            <path d="M50 74 C70 66 78 50 74 36 C58 42 50 56 50 74 Z" />
            <circle cx="50" cy="74" r="2.5" />
          </>
        )}

        {type === "kannon" && (
          <>
            <path d="M50 20 C40 30 40 44 50 50 C60 44 60 30 50 20 Z" />
            <path d="M32 80 q18 -40 36 0" />
            <ellipse cx="50" cy="52" rx="14" ry="6" />
            <path d="M38 86 q12 -8 24 0" />
          </>
        )}

        {type === "crow" && (
          <>
            <path d="M28 44 q22 -20 44 0" />
            <path d="M50 44 l-6 22" />
            <path d="M50 44 l4 22" />
            <path d="M50 44 l0 24" />
            <circle cx="50" cy="38" r="7" />
            <path d="M44 38 l-8 -4" />
          </>
        )}

        {type === "eye" && (
          <>
            <path d="M22 50 q28 -22 56 0 q-28 22 -56 0 Z" />
            <circle cx="50" cy="50" r="8" />
            <circle cx="50" cy="50" r="2.5" />
            <path d="M50 58 v14" />
            <path d="M50 72 q-10 4 -14 14" />
          </>
        )}

        {type === "scarab" && (
          <>
            <ellipse cx="50" cy="54" rx="16" ry="22" />
            <circle cx="50" cy="30" r="7" />
            <path d="M50 54 v22" />
            <path d="M34 48 l-12 -8" />
            <path d="M66 48 l12 -8" />
            <path d="M36 66 l-12 8" />
            <path d="M64 66 l12 8" />
          </>
        )}

        {type === "quantum" && (
          <>
            <circle cx="50" cy="50" r="6" />
            <ellipse cx="50" cy="50" rx="34" ry="13" />
            <ellipse cx="50" cy="50" rx="34" ry="13" transform="rotate(60 50 50)" />
            <ellipse cx="50" cy="50" rx="34" ry="13" transform="rotate(120 50 50)" />
          </>
        )}

        {type === "spiral" && (
          <>
            <path d="M50 50 m0 0 a4 4 0 1 1 -4 4 a8 8 0 1 1 12 -4 a13 13 0 1 1 -18 8 a20 20 0 1 1 28 -12" />
            <circle cx="50" cy="50" r="1.6" />
          </>
        )}
      </g>
    </svg>
  );
}
