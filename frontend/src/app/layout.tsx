import type { Metadata } from "next";
import Script from "next/script";
import "./globals.css";
import { ReactQueryProvider } from "@/components/providers/ReactQueryProvider";

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
    <html lang="en" suppressHydrationWarning>
      <head>
        <Script id="datachat-theme-init" strategy="beforeInteractive">
          {`
            (function() {
              try {
                var key = "datachat.themeMode";
                var mode = window.localStorage.getItem(key) || "system";
                var prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
                var useDark = mode === "dark" || (mode === "system" && prefersDark);
                document.documentElement.classList.toggle("dark", useDark);
              } catch (_) {}
            })();
          `}
        </Script>
      </head>
      <body className="font-sans">
        <ReactQueryProvider>{children}</ReactQueryProvider>
      </body>
    </html>
  );
}
