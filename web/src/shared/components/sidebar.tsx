"use client";

// Single Responsibility: 사이드바 UI만 담당

import { MessageCircle, Plus, User, LogOut } from "lucide-react";

import { Session, User as UserType } from "@/shared/types";

interface SidebarProps {
    sessions: Session[];
    currentSessionId: string | null;
    user: UserType | null;
    onNewChat: () => void;
    onSessionClick: (sessionId: string) => void;
    onLogout: () => void;
}

export function Sidebar({
    sessions,
    currentSessionId,
    user,
    onNewChat,
    onSessionClick,
    onLogout,
}: SidebarProps) {
    return (
        <div
            className="w-64 border-r flex flex-col"
            style={{
                backgroundColor: "var(--color-dark-sidebar)",
                borderColor: "var(--color-dark-border)",
            }}
        >
            <div
                className="p-4 border-b"
                style={{ borderColor: "var(--color-dark-border)" }}
            >
                <div className="flex items-center gap-2">
                    <MessageCircle className="w-5 h-5" />
                    <span className="text-lg font-semibold">Versatile</span>
                </div>
            </div>

            <div className="p-3">
                <button
                    onClick={onNewChat}
                    className="w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-colors hover:bg-[var(--color-dark-hover)]"
                >
                    <Plus className="w-4 h-4" />
                    <span className="text-sm">새 채팅</span>
                </button>
            </div>

            <div className="flex-1 overflow-y-auto px-3 space-y-1">
                <div
                    className="text-xs px-3 py-2 font-medium"
                    style={{ color: "var(--color-dark-text-secondary)" }}
                >
                    최근 항목
                </div>
                {sessions.map((session) => (
                    <button
                        key={session.id}
                        onClick={() => onSessionClick(session.id)}
                        className={`w-full text-left px-3 py-2 rounded-lg transition-colors text-sm truncate hover:bg-[var(--color-dark-hover)] ${
                            session.id === currentSessionId
                                ? "session-active"
                                : ""
                        }`}
                    >
                        {session.title}
                    </button>
                ))}
            </div>

            <div
                className="border-t p-3"
                style={{ borderColor: "var(--color-dark-border)" }}
            >
                <button className="w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-colors text-sm hover:bg-[var(--color-dark-hover)]">
                    <User className="w-4 h-4" />
                    <span>{user?.displayName || user?.username || "User"}</span>
                </button>
                <button
                    onClick={onLogout}
                    className="w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-colors text-sm mt-2 text-red-400 hover:bg-[var(--color-dark-hover)]"
                >
                    <LogOut className="w-4 h-4" />
                    <span>로그아웃</span>
                </button>
            </div>
        </div>
    );
}
