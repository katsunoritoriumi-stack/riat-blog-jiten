"use client";

import { useEffect, useState } from "react";
import { motion, useMotionValue, useSpring } from "framer-motion";

export default function CustomCursor() {
  const [hovered, setHovered] = useState(false);
  const [visible, setVisible] = useState(false);
  const [enabled, setEnabled] = useState(false);

  const cursorX = useMotionValue(-100);
  const cursorY = useMotionValue(-100);
  const springX = useSpring(cursorX, { damping: 25, stiffness: 250 });
  const springY = useSpring(cursorY, { damping: 25, stiffness: 250 });

  useEffect(() => {
    if (window.matchMedia("(pointer: coarse)").matches) return;
    setEnabled(true);

    const move = (e: MouseEvent) => {
      cursorX.set(e.clientX);
      cursorY.set(e.clientY);
      setVisible(true);
    };
    const leave = () => setVisible(false);

    window.addEventListener("mousemove", move);
    document.addEventListener("mouseleave", leave);

    const attach = () => {
      document.querySelectorAll("a, button, [data-cursor-hover]").forEach((el) => {
        el.addEventListener("mouseenter", () => setHovered(true));
        el.addEventListener("mouseleave", () => setHovered(false));
      });
    };
    attach();
    const observer = new MutationObserver(attach);
    observer.observe(document.body, { childList: true, subtree: true });

    return () => {
      window.removeEventListener("mousemove", move);
      document.removeEventListener("mouseleave", leave);
      observer.disconnect();
    };
  }, [cursorX, cursorY]);

  if (!enabled) return null;

  return (
    <>
      {/* core */}
      <motion.div
        className="fixed top-0 left-0 pointer-events-none z-[9999]"
        style={{ x: springX, y: springY }}
        animate={{
          width: hovered ? 54 : 10,
          height: hovered ? 54 : 10,
          opacity: visible ? 1 : 0,
        }}
        transition={{ type: "spring", damping: 20, stiffness: 300 }}
      >
        <div
          className="w-full h-full rounded-full -translate-x-1/2 -translate-y-1/2"
          style={{
            background: hovered
              ? "radial-gradient(circle, rgba(240,180,41,0.35), transparent 70%)"
              : "rgba(246,211,101,0.9)",
            boxShadow: "0 0 18px rgba(240,180,41,0.7)",
          }}
        />
      </motion.div>
      {/* ring */}
      <motion.div
        className="fixed top-0 left-0 pointer-events-none z-[9998]"
        style={{ x: cursorX, y: cursorY }}
        animate={{
          width: hovered ? 80 : 34,
          height: hovered ? 80 : 34,
          opacity: visible ? 0.5 : 0,
        }}
        transition={{ type: "spring", damping: 15, stiffness: 150 }}
      >
        <div className="w-full h-full rounded-full border border-[#a78bfa]/50 -translate-x-1/2 -translate-y-1/2" />
      </motion.div>
    </>
  );
}
