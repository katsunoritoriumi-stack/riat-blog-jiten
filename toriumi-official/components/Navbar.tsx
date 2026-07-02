"use client";

import { useEffect, useState } from "react";
import {
  motion,
  AnimatePresence,
  useScroll,
  useSpring,
} from "framer-motion";
import { Menu, X } from "lucide-react";
import { SECTIONS, SITE } from "@/lib/content";

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState<string>("home");

  // ページ全体のスクロール進捗（ナビ下端のヘアライン）
  const { scrollYProgress } = useScroll();
  const progress = useSpring(scrollYProgress, { stiffness: 140, damping: 30 });

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // scrollspy：ビューポート中央帯に入ったセクションを現在地とする
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) setActive(entry.target.id);
        }
      },
      { rootMargin: "-40% 0px -55% 0px" }
    );
    for (const s of SECTIONS) {
      const el = document.getElementById(s.id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, []);

  return (
    <motion.header
      initial={{ y: -40, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.8, delay: 0.3 }}
      className={`fixed top-0 inset-x-0 z-50 transition-all duration-500 ${
        scrolled ? "glass" : "bg-transparent"
      }`}
    >
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 sm:px-8">
        <a href="#home" className="group flex flex-col leading-none">
          <span className="font-display text-lg font-bold tracking-tight text-aurum-200">
            Katsunori&nbsp;Toriumi
          </span>
          <span className="font-mono text-[9px] tracking-[0.3em] text-nebula-300/70 uppercase">
            {SITE.roleEn}
          </span>
        </a>

        <ul className="hidden items-center gap-7 md:flex">
          {SECTIONS.map((s) => {
            const current = active === s.id;
            return (
              <li key={s.id} className="relative">
                <a
                  href={`#${s.id}`}
                  className={`link-sweep font-mono text-xs uppercase tracking-wider transition-colors hover:text-aurum-200 ${
                    current ? "text-aurum-200" : "text-nebula-300/80"
                  }`}
                >
                  {s.label}
                </a>
                {current && (
                  <motion.span
                    layoutId="nav-star"
                    transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    className="absolute -bottom-2.5 left-1/2 h-1 w-1 -translate-x-1/2 rounded-full bg-aurum-300 shadow-[0_0_8px_rgba(240,180,41,0.9)]"
                  />
                )}
              </li>
            );
          })}
        </ul>

        <button
          onClick={() => setOpen((v) => !v)}
          className="text-aurum-200 md:hidden"
          aria-label="メニュー"
        >
          {open ? <X size={24} /> : <Menu size={24} />}
        </button>
      </nav>

      {/* scroll progress hairline */}
      <motion.div
        style={{ scaleX: progress }}
        className="absolute bottom-0 left-0 h-px w-full origin-left bg-gradient-to-r from-aurum-400/0 via-aurum-300/80 to-iris-rose/70"
        aria-hidden="true"
      />

      <AnimatePresence>
        {open && (
          <motion.ul
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="glass overflow-hidden md:hidden"
          >
            {SECTIONS.map((s) => (
              <li key={s.id} className="border-t border-nebula-500/10">
                <a
                  href={`#${s.id}`}
                  onClick={() => setOpen(false)}
                  className={`block px-6 py-3 text-sm hover:text-aurum-200 ${
                    active === s.id ? "text-aurum-200" : "text-nebula-200"
                  }`}
                >
                  {s.label}
                </a>
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </motion.header>
  );
}
