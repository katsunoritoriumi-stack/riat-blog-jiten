import { SITE } from "@/lib/content";

export default function Footer() {
  return (
    <footer className="relative border-t border-nebula-500/15 px-6 py-12">
      <div className="mx-auto flex max-w-7xl flex-col items-center gap-4 text-center sm:flex-row sm:justify-between sm:text-left">
        <div>
          <p className="font-display text-xl font-bold tracking-tight text-aurum-200">Katsunori Toriumi</p>
          <p className="font-mono text-xs tracking-widest text-nebula-300/60">{SITE.roleEn} — {SITE.taglineJp}</p>
        </div>
        <p className="text-xs text-nebula-300/50">
          © {new Date().getFullYear()} {SITE.nameEn}. All works belong to the artist.
        </p>
      </div>
    </footer>
  );
}
