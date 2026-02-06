"use client";

// Single Responsibility: 채팅 상태 관리만 담당

import { useState, useCallback } from "react";

import { Message, ChatMode } from "@/shared/types";

export function useChat() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [currentMode, setCurrentMode] = useState<ChatMode>("chat");
    const [isProcessing, setIsProcessing] = useState(false);

    const addMessage = useCallback((message: Message) => {
        setMessages((prev) => [...prev, message]);
    }, []);

    const clearMessages = useCallback(() => {
        setMessages([]);
    }, []);

    const updateMode = useCallback((mode: ChatMode) => {
        setCurrentMode(mode);
    }, []);

    const startProcessing = useCallback(() => {
        setIsProcessing(true);
    }, []);

    const stopProcessing = useCallback(() => {
        setIsProcessing(false);
    }, []);

    return {
        messages,
        currentMode,
        isProcessing,
        addMessage,
        clearMessages,
        updateMode,
        startProcessing,
        stopProcessing,
    };
}
