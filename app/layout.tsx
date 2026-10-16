import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = {
  title: 'LeRobot Argon — 世界就在手邊',
  description: '用 Google Search 自然簡潔介面指揮 LeRobot 101 單手自然動作與空間任務。由 Gemini 4 Argon · Gemini Robotics ER 2 · MotionGPT 驅動。',
};
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-Hant">
      <body>{children}</body>
    </html>
  );
}
