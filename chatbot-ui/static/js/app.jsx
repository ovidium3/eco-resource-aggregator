function Chat() {
  const [messages, setMessages] = React.useState([]);
  const [conversations, setConversations] = React.useState([]);
  const [input, setInput] = React.useState("");
  const [waiting, setWaiting] = React.useState(false);
  const [activeChatIndex, setActiveChatIndex] = React.useState(null);
  const scrollRef = React.useRef(null);

  React.useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages]);

  function getFormattedTimestamp() {
    const now = new Date();
    return now.toLocaleString(undefined, {
      dateStyle: "short",
      timeStyle: "short",
    });
  }

  function createTitleFromFirstMessage(msgs) {
    const firstUserMsg = msgs.find((m) => m.from === "user");
    return firstUserMsg ? firstUserMsg.text.slice(0, 30) + "…" : "New Chat";
  }

  async function send(messageOverride = null) {
    const finalInput = messageOverride || input.trim();
    if (!finalInput || waiting) return;

    setMessages((prev) => [...prev, { from: "user", text: finalInput }]);
    setInput("");
    setWaiting(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: finalInput }),
      });
      const data = await res.json();
      const reply = data.reply || "⚠️ No response";

      setMessages((prev) => [...prev, { from: "bot", text: reply }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { from: "bot", text: "⚠️ Server error: " + err.message },
      ]);
    } finally {
      setWaiting(false);
    }
  }

  function startNewChat() {
    if (messages.length > 0) {
      setConversations((prev) => [
        ...prev,
        {
          title: createTitleFromFirstMessage(messages),
          timestamp: getFormattedTimestamp(),
          messages: messages,
        },
      ]);
    }
    setMessages([]);
    setActiveChatIndex(null);
  }

  function loadChat(index) {
    setMessages(conversations[index].messages);
    setActiveChatIndex(index);
  }

  const suggestions = [
    "What causes climate change?",
    "How can I reduce my carbon footprint?",
    "Tell me a fun fact about the planet",
  ];

  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "'Poppins', sans-serif" }}>
      {/* Sidebar */}
      <div style={{ width: "240px", background: "#f1f8e9", padding: "1rem", borderRight: "1px solid #c5e1a5", overflowY: "auto" }}>
        <h2 style={{ marginTop: 0, fontSize: "1.2rem" }}>💬 Chats</h2>
        <button
          onClick={startNewChat}
          style={{
            width: "100%",
            backgroundColor: "#81c784",
            color: "white",
            border: "none",
            borderRadius: "0.4rem",
            padding: "0.5rem",
            fontWeight: "500",
            cursor: "pointer",
            marginBottom: "1rem",
          }}
        >
          + New Chat
        </button>

        {conversations.map((chat, idx) => (
          <div
            key={idx}
            onClick={() => loadChat(idx)}
            style={{
              backgroundColor: activeChatIndex === idx ? "#c5e1a5" : "#ffffff",
              padding: "0.5rem",
              borderRadius: "0.4rem",
              marginBottom: "0.5rem",
              cursor: "pointer",
            }}
          >
            <div style={{ fontWeight: 600, fontSize: "0.95rem" }}>{chat.title}</div>
            <div style={{ fontSize: "0.8rem", color: "#555" }}>{chat.timestamp}</div>
          </div>
        ))}
      </div>

      {/* Main Chat Area */}
      <div style={{ flex: 1, padding: "1rem", overflowY: "auto", display: "flex", flexDirection: "column" }}>
        {/* Header */}
        <div style={{
          background: "linear-gradient(to right, #6aa84f, #38761d)",
          padding: "1rem",
          color: "white",
          borderRadius: "0.5rem",
          marginBottom: "1rem",
        }}>
          <h1 style={{ margin: 0, fontSize: "1.5rem", fontWeight: 600 }}>🌿 Climate Chatbot</h1>
        </div>

        {/* Message History */}
        <div
          ref={scrollRef}
          style={{
            flex: 1,
            overflowY: "auto",
            paddingRight: "6px",
            display: "flex",
            flexDirection: "column",
            gap: "0.5rem",
          }}
          className="scroll-area"
        >
          {messages.map((m, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                justifyContent: m.from === "user" ? "flex-end" : "flex-start",
              }}
            >
              <div className={m.from === "user" ? "bubble-user" : "bubble-bot"}>
                {m.text}
              </div>
            </div>
          ))}

          {waiting && (
            <div style={{ display: "flex", justifyContent: "flex-start" }}>
              <div style={{ padding: "0.4rem" }}>
                <img
                  src="https://upload.wikimedia.org/wikipedia/commons/6/6b/Rotating_globe.gif"
                  alt="Rotating Globe"
                  className="earth-spinner"
                />
              </div>
            </div>
          )}
        </div>

        {/* Input + Suggestions */}
        <div style={{ marginTop: "1rem" }}>
          <div style={{ display: "flex", gap: "0.75rem" }}>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send()}
              placeholder="Ask me about climate…"
              style={{
                flex: 1,
                padding: "0.75rem",
                borderRadius: "0.5rem",
                border: "1px solid #ccc",
              }}
              disabled={waiting}
            />
            <button
              onClick={() => send()}
              disabled={waiting}
              style={{
                backgroundColor: "#38761d",
                color: "white",
                padding: "0.75rem 1.25rem",
                borderRadius: "0.5rem",
                border: "none",
                fontWeight: "500",
                cursor: "pointer",
                opacity: waiting ? 0.6 : 1,
              }}
            >
              Send
            </button>
          </div>

          {/* Suggestions */}
          <div style={{ marginTop: "1rem", display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
            {suggestions.map((text, idx) => (
              <button
                key={idx}
                onClick={() => send(text)}
                disabled={waiting}
                style={{
                  padding: "0.5rem 0.75rem",
                  backgroundColor: "#e8f5e9",
                  border: "1px solid #c5e1a5",
                  borderRadius: "0.5rem",
                  cursor: "pointer",
                  fontSize: "0.9rem",
                  fontFamily: "'Poppins', sans-serif",
                }}
              >
                {text}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<Chat />);
