import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DataChat - AI Data Assistant",
  description: "Natural language interface for your data warehouse",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="font-sans">{children}</body>
    </html>
  );
}
