"use client";

// Single Responsibility: 세션 관리만 담당

import { useState, useCallback } from "react";

import { Session } from "@/shared/types";

export function useSession() {
    const [sessions, setSessions] = useState<Session[]>([]);
    const [currentSessionId, setCurrentSessionId] = useState<string | null>(
        null,
    );

    const createSession = useCallback((title: string = "새 채팅") => {
        const newSession: Session = {
            id: crypto.randomUUID(),
            title,
            createdAt: new Date(),
            updatedAt: new Date(),
        };
        setSessions((prev) => [newSession, ...prev]);
        setCurrentSessionId(newSession.id);
        return newSession;
    }, []);

    const switchSession = useCallback((sessionId: string) => {
        setCurrentSessionId(sessionId);
    }, []);

    const updateSessionTitle = useCallback(
        (sessionId: string, title: string) => {
            setSessions((prev) =>
                prev.map((session) =>
                    session.id === sessionId
                        ? { ...session, title, updatedAt: new Date() }
                        : session,
                ),
            );
        },
        [],
    );

    const deleteSession = useCallback(
        (sessionId: string) => {
            setSessions((prev) =>
                prev.filter((session) => session.id !== sessionId),
            );
            if (currentSessionId === sessionId) {
                setCurrentSessionId(null);
            }
        },
        [currentSessionId],
    );

    const currentSession = sessions.find((s) => s.id === currentSessionId);

    return {
        sessions,
        currentSessionId,
        currentSession,
        createSession,
        switchSession,
        updateSessionTitle,
        deleteSession,
    };
}
