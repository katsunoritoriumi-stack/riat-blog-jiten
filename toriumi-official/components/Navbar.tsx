"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X } from "lucide-react";
import { SECTIONS, SITE } from "@/lib/content";

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
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
          {SECTIONS.map((s) => (
            <li key={s.id}>
              <a
                href={`#${s.id}`}
                className="link-sweep font-mono text-xs uppercase tracking-wider text-nebula-300/80 transition-colors hover:text-aurum-200"
              >
                {s.label}
              </a>
            </li>
          ))}
        </ul>

        <button
          onClick={() => setOpen((v) => !v)}
          className="text-aurum-200 md:hidden"
          aria-label="メニュー"
        >
          {open ? <X size={24} /> : <Menu size={24} />}
        </button>
      </nav>

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
                  className="block px-6 py-3 text-sm text-nebula-200 hover:text-aurum-200"
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
