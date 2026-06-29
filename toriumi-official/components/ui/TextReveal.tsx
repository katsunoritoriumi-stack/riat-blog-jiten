"use client";

import { Fragment } from "react";
import { motion } from "framer-motion";

type Props = {
  children: string;
  className?: string;
  delay?: number;
  as?: "h1" | "h2" | "h3" | "p" | "span";
};

/**
 * 単語ごとにスライドアップで立ち上がるリビール。
 * 各単語を overflow-hidden でクリップし、単語間には実体スペースを入れて
 * 通常通り折り返せるようにする（長い見出しの溢れ対策）。
 */
export default function TextReveal({
  children,
  className = "",
  delay = 0,
  as: Tag = "h2",
}: Props) {
  const words = children.split(" ");

  return (
    <Tag className={className}>
      {words.map((word, i) => (
        <Fragment key={i}>
          <span className="inline-block overflow-hidden align-bottom">
            <motion.span
              className="inline-block"
              initial={{ y: "115%" }}
              whileInView={{ y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{
                duration: 0.7,
                delay: delay + i * 0.07,
                ease: [0.22, 1, 0.36, 1],
              }}
            >
              {word}
            </motion.span>
          </span>
          {i < words.length - 1 ? " " : null}
        </Fragment>
      ))}
    </Tag>
  );
}
