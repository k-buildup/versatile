"use client";

// Single Responsibility: 메시지 입력 UI만 담당

import { Send, MessageCircle, Wrench, Sparkle, Trash2 } from "lucide-react";
import { useState, useRef, KeyboardEvent } from "react";

import { ChatMode } from "@/shared/types";

interface ChatInputProps {
    currentMode: ChatMode;
    isProcessing: boolean;
    onSendMessage: (message: string) => void;
    onModeChange: (mode: ChatMode) => void;
    onClearChat: () => void;
}

export function ChatInput({
    currentMode,
    isProcessing,
    onSendMessage,
    onModeChange,
    onClearChat,
}: ChatInputProps) {
    const [message, setMessage] = useState("");
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const handleSubmit = () => {
        if (message.trim() && !isProcessing) {
            onSendMessage(message.trim());
            setMessage("");
            if (textareaRef.current) {
                textareaRef.current.style.height = "auto";
            }
        }
    };

    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey && !isProcessing) {
            e.preventDefault();
            handleSubmit();
        }
    };

    const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setMessage(e.target.value);
        // Auto-resize
        e.target.style.height = "auto";
        e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
    };

    const modes: Array<{ mode: ChatMode; icon: typeof MessageCircle }> = [
        { mode: "chat", icon: MessageCircle },
        { mode: "tool", icon: Wrench },
        { mode: "think", icon: Sparkle },
    ];

    const getModeButtonStyle = (mode: ChatMode) => ({
        backgroundColor:
            currentMode === mode
                ? "rgba(229, 229, 229, 0.1)"
                : "rgba(229, 229, 229, 0.05)",
    });

    return (
        <div
            className="border-t p-4"
            style={{ borderColor: "var(--color-dark-border)" }}
        >
            <div className="max-w-3xl mx-auto">
                <div className="flex gap-2">
                    {modes.map(({ mode, icon: Icon }) => (
                        <button
                            key={mode}
                            onClick={() => !isProcessing && onModeChange(mode)}
                            disabled={isProcessing}
                            className="size-[48px] px-4 py-2 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                            style={getModeButtonStyle(mode)}
                        >
                            <Icon className="w-4 h-4" />
                        </button>
                    ))}

                    <div className="flex-1 relative">
                        <textarea
                            ref={textareaRef}
                            value={message}
                            onChange={handleInput}
                            onKeyDown={handleKeyDown}
                            placeholder="메시지를 입력하세요..."
                            rows={1}
                            className="min-h-[48px] w-full px-4 py-3 border rounded-xl resize-none focus:outline-none focus:ring-2 text-sm"
                            style={{
                                backgroundColor: "var(--color-dark-hover)",
                                borderColor: "var(--color-dark-border)",
                                color: "var(--color-dark-text)",
                                maxHeight: "200px",
                            }}
                        />
                    </div>

                    <button
                        onClick={handleSubmit}
                        disabled={isProcessing || !message.trim()}
                        className="size-[48px] px-4 py-2 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                        style={{
                            backgroundColor: "var(--color-dark-text)",
                            color: "var(--color-dark-bg)",
                        }}
                    >
                        <Send className="w-4 h-4" />
                    </button>
                </div>

                <div className="flex items-center justify-between mt-2 px-2">
                    <div
                        className="text-xs"
                        style={{ color: "var(--color-dark-text-secondary)" }}
                    >
                        <span>{message.length}</span> characters
                    </div>
                    <button
                        onClick={onClearChat}
                        disabled={isProcessing}
                        className="text-xs transition-colors flex items-center gap-1 disabled:opacity-50"
                        style={{ color: "var(--color-dark-text-secondary)" }}
                    >
                        <Trash2 className="w-3 h-3" />
                        Clear chat
                    </button>
                </div>
            </div>
        </div>
    );
}
