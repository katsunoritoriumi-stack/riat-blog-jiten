"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X, ChevronDown, BookOpen, Users, Briefcase, Heart, Trees, Building } from "lucide-react";

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const [activeMenu, setActiveMenu] = useState<string | null>(null);

  // メガメニューのコンテンツ構造
  const menuItems = [
    {
      id: "kurashi",
      label: "くらし・手続き",
      icon: Users,
      submenu: [
        { title: "届出・証明", desc: "住民票、戸籍、マイナンバーカード", icon: BookOpen },
        { title: "税金・保険・年金", desc: "市民税、国民健康保険、各種年金", icon: Heart },
        { title: "子育て・教育", desc: "保育園・学校、手当、助成金制度", icon: Users },
        { title: "ごみ・環境", desc: "分別ルール、収集カレンダー、リサイクル", icon: Trees },
      ],
    },
    {
      id: "kanko",
      label: "観光・文化",
      icon: Trees,
      submenu: [
        { title: "八ヶ岳・自然", desc: "登山情報、トレッキングコース、ライブカメラ", icon: Trees },
        { title: "イベント・祭り", desc: "御柱祭、季節のイベント情報", icon: BookOpen },
        { title: "宿泊・温泉", desc: "市内の温泉施設、おすすめの宿", icon: Heart },
        { title: "移住・定住", desc: "ちの暮らしサポート、空き家バンク", icon: Users },
      ],
    },
    {
      id: "business",
      label: "市政・ビジネス",
      icon: Building,
      submenu: [
        { title: "事業者向け情報", desc: "入札・契約、企業誘致、産業振興", icon: Briefcase },
        { title: "ふるさと納税", desc: "茅野市への寄付、返礼品のご案内", icon: Heart },
        { title: "市長の部屋・方針", desc: "施政方針、記者会見、プロフィールの紹介", icon: Users },
        { title: "市議会・広報", desc: "議会中継、広報ちの、各種統計データ", icon: BookOpen },
      ],
    },
  ];

  return (
    <header className="fixed top-0 left-0 w-full z-50 bg-[#0d1510]/80 backdrop-blur-md border-b border-white/5 text-white">
      <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">

        {/* ロゴエリア */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex items-center gap-3 cursor-pointer"
        >
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-[#10b981] to-[#4ade80] flex items-center justify-center font-bold text-dark text-lg shadow-lg shadow-emerald-500/20">
            茅
          </div>
          <div>
            <span className="text-sm font-semibold tracking-[0.2em] block leading-none mb-1">CHINO CITY</span>
            <span className="text-[10px] text-white/50 tracking-wider block">茅野市ポータル</span>
          </div>
        </motion.div>

        {/* デスクトップ ナビゲーション */}
        <nav className="hidden lg:flex items-center gap-1 h-full">
          {menuItems.map((item) => (
            <div
              key={item.id}
              className="relative h-full flex items-center"
              onMouseEnter={() => setActiveMenu(item.id)}
              onMouseLeave={() => setActiveMenu(null)}
            >
              <button className={`px-5 h-full flex items-center gap-1.5 text-sm font-medium tracking-wide transition-colors relative
                ${activeMenu === item.id ? "text-[#4ade80]" : "text-white/80 hover:text-white"}`}
              >
                {item.label}
                <ChevronDown className={`w-4 h-4 transition-transform duration-300 ${activeMenu === item.id ? "rotate-180" : ""}`} />
                {activeMenu === item.id && (
                  <motion.div layoutId="navBorder" className="absolute bottom-0 left-0 right-0 h-[2px] bg-[#4ade80]" />
                )}
              </button>

              {/* メガメニュー (Strix風に滑らかにドロップダウン) */}
              <AnimatePresence>
                {activeMenu === item.id && (
                  <motion.div
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 10 }}
                    transition={{ duration: 0.25, ease: "easeOut" }}
                    className="absolute top-20 left-1/2 -translate-x-1/2 w-[600px] bg-[#121c16] border border-white/10 rounded-2xl p-6 shadow-2xl grid grid-cols-2 gap-4"
                  >
                    {item.submenu.map((sub, i) => {
                      const SubIcon = sub.icon;
                      return (
                        <motion.a
                          href="#"
                          key={i}
                          whileHover={{ x: 4, backgroundColor: "rgba(255,255,255,0.03)" }}
                          className="flex gap-4 p-3 rounded-xl transition-all border border-transparent hover:border-white/5"
                        >
                          <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center text-[#4ade80] shrink-0">
                            <SubIcon className="w-5 h-5" />
                          </div>
                          <div>
                            <h4 className="text-sm font-medium text-white mb-0.5">{sub.title}</h4>
                            <p className="text-xs text-white/40 leading-relaxed">{sub.desc}</p>
                          </div>
                        </motion.a>
                      );
                    })}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          ))}
        </nav>

        {/* 右側アクションボタン */}
        <div className="hidden lg:flex items-center gap-4">
          <button className="text-xs font-semibold tracking-wider text-white/60 hover:text-white transition-colors">
            ENGLISH
          </button>
          <button className="bg-white/5 hover:bg-white/10 border border-white/10 px-5 py-2.5 rounded-full text-xs font-medium tracking-wider transition-all duration-300">
            緊急連絡先
          </button>
        </div>

        {/* モバイルメニューボタン */}
        <button className="lg:hidden p-2 text-white/80 hover:text-white" onClick={() => setIsOpen(!isOpen)}>
          {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>
      </div>

      {/* モバイル用フルスクリーンナビゲーション */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="fixed inset-0 top-20 bg-[#0d1510] z-40 lg:hidden overflow-y-auto px-6 py-8"
          >
            <div className="flex flex-col gap-8">
              {menuItems.map((item) => (
                <div key={item.id} className="flex flex-col gap-3">
                  <div className="flex items-center gap-2 text-[#4ade80] font-medium text-sm tracking-wider border-b border-white/5 pb-2">
                    <item.icon className="w-4 h-4" />
                    {item.label}
                  </div>
                  <div className="grid gap-2 pl-2">
                    {item.submenu.map((sub, i) => (
                      <a key={i} href="#" className="py-2 flex flex-col">
                        <span className="text-sm text-white/90">{sub.title}</span>
                        <span className="text-xs text-white/40">{sub.desc}</span>
                      </a>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
