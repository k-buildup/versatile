"use client";

// Interface Segregation & Dependency Inversion: Props를 통한 의존성 주입

import { useAuth, useChat, useSession } from "@/shared/hooks";

import { ChatContainer } from "@/shared/components/chat-container";
import { AuthModal } from "@/shared/components/auth-modal";
import { ChatInput } from "@/shared/components/chat-input";
import { Sidebar } from "@/shared/components/sidebar";
import { Header } from "@/shared/components/header";

export default function HomePage() {
    const { user, isAuthenticated, isLoading, logout } = useAuth();
    const {
        messages,
        currentMode,
        isProcessing,
        addMessage,
        clearMessages,
        updateMode,
        startProcessing,
        stopProcessing,
    } = useChat();
    const {
        sessions,
        currentSessionId,
        currentSession,
        createSession,
        switchSession,
    } = useSession();

    const handleLogin = (username: string, password: string) => {
        // TODO: API 연동
        console.log("Login:", username, password);
    };

    const handleRegister = (
        username: string,
        email: string,
        password: string,
        displayName: string,
    ) => {
        // TODO: API 연동
        console.log("Register:", username, email, password, displayName);
    };

    const handleSendMessage = (content: string) => {
        const userMessage: import("@/shared/types").UserMessage = {
            id: crypto.randomUUID(),
            role: "user",
            content,
            timestamp: new Date(),
        };
        addMessage(userMessage);

        if (!currentSessionId) {
            createSession(
                content.slice(0, 30) + (content.length > 30 ? "..." : ""),
            );
        }

        // TODO: API 연동
        startProcessing();
        setTimeout(() => {
            const assistantMessage: import("@/shared/types").AssistantMessage =
                {
                    id: crypto.randomUUID(),
                    role: "assistant",
                    content: "이것은 테스트 응답입니다.",
                    timestamp: new Date(),
                };
            addMessage(assistantMessage);
            stopProcessing();
        }, 1000);
    };

    const handleNewChat = () => {
        clearMessages();
        createSession();
    };

    const handleSessionClick = (sessionId: string) => {
        switchSession(sessionId);
        clearMessages();
        // TODO: 세션 히스토리 로드
    };

    const handleClearChat = () => {
        clearMessages();
    };

    if (isLoading) {
        return (
            <div
                className="flex items-center justify-center h-screen"
                style={{
                    backgroundColor: "var(--color-dark-bg)",
                    color: "var(--color-dark-text)",
                }}
            >
                Loading...
            </div>
        );
    }

    return (
        <>
            <AuthModal
                isOpen={!isAuthenticated}
                onLogin={handleLogin}
                onRegister={handleRegister}
            />

            {isAuthenticated && (
                <div
                    className="flex h-screen overflow-hidden"
                    style={{
                        backgroundColor: "var(--color-dark-bg)",
                        color: "var(--color-dark-text)",
                    }}
                >
                    <Sidebar
                        sessions={sessions}
                        currentSessionId={currentSessionId}
                        user={user}
                        onNewChat={handleNewChat}
                        onSessionClick={handleSessionClick}
                        onLogout={logout}
                    />

                    <div className="flex-1 flex flex-col">
                        <Header
                            title={currentSession?.title || "새 채팅"}
                            isConnected={true}
                        />

                        <ChatContainer
                            messages={messages}
                            isProcessing={isProcessing}
                        />

                        <ChatInput
                            currentMode={currentMode}
                            isProcessing={isProcessing}
                            onSendMessage={handleSendMessage}
                            onModeChange={updateMode}
                            onClearChat={handleClearChat}
                        />
                    </div>
                </div>
            )}
        </>
    );
}
