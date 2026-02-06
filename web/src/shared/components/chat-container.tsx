"use client";

// Single Responsibility: 채팅 메시지 리스트 표시만 담당

import { useEffect, useRef } from "react";
import { Sparkles } from "lucide-react";

import { ChatMessage } from "./chat-message";

import { Message } from "@/shared/types";

interface ChatContainerProps {
    messages: Message[];
    isProcessing: boolean;
}

export function ChatContainer({ messages, isProcessing }: ChatContainerProps) {
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (containerRef.current) {
            containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
    }, [messages, isProcessing]);

    const showWelcome = messages.length === 0 && !isProcessing;

    return (
        <div
            ref={containerRef}
            className="flex-1 overflow-y-auto px-6 py-6 space-y-6"
        >
            {showWelcome && (
                <div className="flex flex-col items-center justify-center h-full text-center">
                    <div className="max-w-md space-y-4">
                        <div
                            className="w-16 h-16 mx-auto rounded-full flex items-center justify-center"
                            style={{
                                backgroundColor: "var(--color-dark-hover)",
                            }}
                        >
                            <Sparkles
                                className="w-8 h-8"
                                style={{
                                    color: "var(--color-dark-text-secondary)",
                                }}
                            />
                        </div>
                        <h2 className="text-2xl font-semibold">
                            Versatile에 오신 것을 환영합니다
                        </h2>
                        <p
                            style={{
                                color: "var(--color-dark-text-secondary)",
                            }}
                        >
                            아래에 메시지를 입력하여 대화를 시작하세요.
                            <br />
                            Chat, Tool, Think 모드 중 하나를 선택할 수 있습니다.
                        </p>
                    </div>
                </div>
            )}

            {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
            ))}

            {isProcessing && (
                <div className="flex gap-4 max-w-3xl m-[0_auto]">
                    <div
                        className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
                        style={{ backgroundColor: "var(--color-dark-hover)" }}
                    >
                        <Sparkles className="w-4 h-4" />
                    </div>
                    <div className="flex items-center gap-1 py-3">
                        <div
                            className="typing-dot w-2 h-2 rounded-full animate-bounce"
                            style={{
                                backgroundColor:
                                    "var(--color-dark-text-secondary)",
                            }}
                        />
                        <div
                            className="typing-dot w-2 h-2 rounded-full animate-bounce delay-100"
                            style={{
                                backgroundColor:
                                    "var(--color-dark-text-secondary)",
                            }}
                        />
                        <div
                            className="typing-dot w-2 h-2 rounded-full animate-bounce delay-200"
                            style={{
                                backgroundColor:
                                    "var(--color-dark-text-secondary)",
                            }}
                        />
                    </div>
                </div>
            )}
        </div>
    );
}
