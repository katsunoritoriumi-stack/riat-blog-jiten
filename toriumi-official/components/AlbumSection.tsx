"use client";

import { motion } from "framer-motion";
import AlbumPlayer from "./AlbumPlayer";
import WorkLabel from "./ui/WorkLabel";
import { ALBUM } from "@/lib/content";

/**
 * 「音楽と映像」の 2 つめの作品。
 *
 * MV（SoundVisionSection）とは別のステーションに置いてある。
 * ひとつの画面に両方を積むと画面 2.6 枚ぶんの高さになり、
 * ZoomStage は中身を中央に置くので上下が同時に切れてしまうため。
 *
 * MV と同時に音が鳴らない仕組みは lib/mediaBus.ts が持っている。
 * ステーションが分かれてもモジュール変数なのでそのまま効く。
 */
export default function AlbumSection() {
  return (
    <section data-section="album" className="relative mx-auto max-w-5xl px-6 py-16 sm:py-20">
      {/* 題字（ANCIENT VOICES / 古の声）はジャケットの絵の中に置いてある */}
      <WorkLabel index={2} kind={`Music Album — Homage to ${ALBUM.tribute}`} />

      {/*
        説明文は PC では 1 行に収める。
        狭い画面では素直に折り返させる（無理に 1 行にすると字が小さくなりすぎる）。
      */}
      <motion.p
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.7, delay: 0.1 }}
        className="mb-6 max-w-xl font-serif text-sm leading-relaxed text-nebula-200/75 lg:max-w-none lg:whitespace-nowrap"
      >
        {ALBUM.note}
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-8%" }}
        transition={{ duration: 0.7 }}
      >
        <AlbumPlayer />
      </motion.div>
    </section>
  );
}
