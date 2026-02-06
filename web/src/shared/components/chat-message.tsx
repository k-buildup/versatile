"use client";

// Single Responsibility: 메시지 표시만 담당

import { UserIcon, BotIcon } from "lucide-react";

import { Message } from "@/shared/types";

interface ChatMessageProps {
    message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
    const isUser = message.role === "user";
    const isSystem = message.role === "system";

    if (isSystem) {
        return (
            <div className="flex items-center justify-center py-2">
                <div
                    className="text-xs px-3 py-1.5 rounded-full border"
                    style={{
                        color: "var(--color-dark-text-secondary)",
                        backgroundColor: "var(--color-dark-hover)",
                        borderColor: "var(--color-dark-border)",
                    }}
                >
                    {message.content}
                </div>
            </div>
        );
    }

    return (
        <div className="flex gap-4 max-w-3xl m-[0_auto]">
            <div
                className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                    isUser ? "bg-amber-700 text-white" : ""
                }`}
                style={
                    !isUser
                        ? { backgroundColor: "var(--color-dark-hover)" }
                        : undefined
                }
            >
                {isUser ? (
                    <UserIcon className="w-4 h-4" />
                ) : (
                    <BotIcon className="w-4 h-4" />
                )}
            </div>

            <div className="flex-1 space-y-2">
                <div
                    className={`prose prose-invert max-w-none ${
                        isUser
                            ? "px-4 py-3 rounded-2xl rounded-tl-sm"
                            : "markdown-content mt-[3px]"
                    }`}
                    style={
                        isUser
                            ? { backgroundColor: "var(--color-dark-hover)" }
                            : undefined
                    }
                >
                    {message.content}
                </div>
            </div>
        </div>
    );
}
