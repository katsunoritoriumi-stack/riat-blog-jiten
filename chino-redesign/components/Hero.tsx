"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Search, ShieldAlert, Navigation, Calendar, CloudSun } from "lucide-react";

export default function Hero() {
  const [searchQuery, setSearchQuery] = useState("");

  // クイックアクセス用のモダンなタグ
  const quickLinks = [
    { label: "ごみ分別・収集日", icon: Calendar },
    { label: "防災・ハザードマップ", icon: ShieldAlert, highlight: true },
    { label: "移住・定住サポート", icon: Navigation },
    { label: "観光・八ヶ岳ライブカメラ", icon: CloudSun },
  ];

  return (
    <section className="relative w-full h-[90vh] min-h-[600px] flex items-center justify-center overflow-hidden bg-[#0d1510]">
      {/* 背景：八ヶ岳の深緑と澄んだ空をイメージしたグラデーションとオーバーレイ */}
      <div className="absolute inset-0 z-0">
        {/* 本番ではここに茅野市の美しい自然の画像や、シームレスなドローン映像の動画を配置します */}
        <div
          className="w-full h-full bg-cover bg-center scale-105 filter brightness-[0.4] contrast-[1.1]"
          style={{
            backgroundImage: `url('https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1920&q=80')`
          }}
        />
        {/* Strix風の洗練されたグラデーションオーバーレイ */}
        <div className="absolute inset-0 bg-gradient-to-t from-[#0d1510] via-transparent to-[#0d1510]/50" />
        <div className="absolute inset-0 bg-gradient-to-r from-[#0d1510]/80 via-transparent to-transparent" />
      </div>

      {/* コンテンツエリア */}
      <div className="relative z-10 w-full max-w-7xl mx-auto px-6 sm:px-8 lg:px-12 text-white flex flex-col items-start justify-center">

        {/* サブキャッチコピー：フェードイン＆下から湧き上がるアニメーション */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="flex items-center gap-3 mb-4"
        >
          <span className="w-8 h-[1px] bg-[#4ade80]" />
          <p className="text-xs sm:text-sm font-semibold tracking-[0.2em] text-[#4ade80] uppercase">
            Chino City Official Portal
          </p>
        </motion.div>

        {/* メインキャッチコピー：洗練された大きなタイポグラフィ */}
        <motion.h1
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1, ease: "easeOut" }}
          className="text-4xl sm:text-6xl lg:text-7xl font-light tracking-tight leading-[1.15] mb-8 max-w-3xl"
        >
          八ヶ岳の気候に寄り添い、
          <br />
          <span className="font-medium text-transparent bg-clip-text bg-gradient-to-r from-white via-white to-[#a7f3d0]">
            未来へ紡ぐスマートシティ
          </span>
        </motion.h1>

        {/* スマート検索バーセクション */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
          className="w-full max-w-2xl bg-white/10 backdrop-blur-md rounded-2xl p-2 border border-white/10 shadow-2xl mb-6"
        >
          <form onSubmit={(e) => e.preventDefault()} className="relative flex items-center">
            <div className="absolute left-4 text-white/50">
              <Search className="w-5 h-5" />
            </div>
            <input
              type="text"
              placeholder="例：住民票の写し、給付金、子育て支援..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-transparent pl-12 pr-24 py-4 text-white placeholder-white/40 focus:outline-none text-base sm:text-lg"
            />
            <button
              type="submit"
              className="absolute right-2 top-1/2 -translate-y-1/2 bg-[#10b981] hover:bg-[#059669] text-white font-medium px-6 py-2.5 rounded-xl transition-all duration-300 text-sm tracking-wider shadow-md hover:scale-[1.02]"
            >
              検索
            </button>
          </form>
        </motion.div>

        {/* クイックアクセスタグ */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.4 }}
          className="w-full max-w-3xl"
        >
          <p className="text-xs text-white/50 tracking-wider mb-3 pl-1">
            よく使われる手続き・情報
          </p>
          <div className="flex flex-wrap gap-2.5">
            {quickLinks.map((link, index) => {
              const Icon = link.icon;
              return (
                <motion.button
                  key={index}
                  whileHover={{ y: -3, backgroundColor: link.highlight ? "rgba(239, 68, 68, 0.2)" : "rgba(255, 255, 255, 0.15)" }}
                  className={`flex items-center gap-2 px-4 py-2.5 rounded-full text-xs font-medium tracking-wide border transition-all duration-200 backdrop-blur-sm
                    ${link.highlight
                      ? "bg-red-500/10 border-red-500/30 text-red-300"
                      : "bg-white/5 border-white/10 text-white/80 hover:text-white hover:border-white/30"
                    }
                  `}
                >
                  <Icon className={`w-3.5 h-3.5 ${link.highlight ? "text-red-400" : "text-[#4ade80]"}`} />
                  {link.label}
                </motion.button>
              );
            })}
          </div>
        </motion.div>

      </div>

      {/* スクロールを促すインジケーター（Strix風） */}
      <div className="absolute bottom-8 right-12 z-10 hidden sm:flex flex-col items-center gap-4">
        <span className="text-[10px] tracking-[0.3em] uppercase font-bold text-white/30 rotate-90 origin-bottom-right translate-y-[-20px]">
          Scroll
        </span>
        <div className="w-[1px] h-12 bg-white/20 relative overflow-hidden">
          <motion.div
            className="absolute top-0 left-0 right-0 bg-[#4ade80] w-full"
            animate={{
              top: ["0%", "100%"],
              height: ["0%", "100%", "0%"]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        </div>
      </div>
    </section>
  );
}
