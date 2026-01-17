/**
 * Main Chat Page
 *
 * Root page of the DataChat application.
 * Displays the chat interface in a full-height layout.
 */

import { ChatInterface } from "@/components/chat/ChatInterface";

export default function Home() {
  return (
    <main className="h-screen flex flex-col">
      <ChatInterface />
    </main>
  );
}
