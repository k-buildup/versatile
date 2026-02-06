"use client";

// Single Responsibility: 인증 상태 관리만 담당

import { useState, useEffect } from "react";

import { User, AuthState } from "@/shared/types";

const AUTH_TOKEN_KEY = "authToken";
const CURRENT_USER_KEY = "currentUser";

export function useAuth() {
    const [authState, setAuthState] = useState<AuthState>({
        user: null,
        token: null,
        isAuthenticated: false,
    });
    const [isLoading, setIsLoading] = useState(true);

    const login = (user: User, token: string) => {
        localStorage.setItem(AUTH_TOKEN_KEY, token);
        localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(user));
        setAuthState({
            user,
            token,
            isAuthenticated: true,
        });
    };

    const logout = () => {
        localStorage.removeItem(AUTH_TOKEN_KEY);
        localStorage.removeItem(CURRENT_USER_KEY);
        setAuthState({
            user: null,
            token: null,
            isAuthenticated: false,
        });
    };

    useEffect(() => {
        const token = localStorage.getItem(AUTH_TOKEN_KEY);
        const userJson = localStorage.getItem(CURRENT_USER_KEY);

        if (token && userJson) {
            try {
                const user = JSON.parse(userJson) as User;
                // eslint-disable-next-line react-hooks/set-state-in-effect
                setAuthState({
                    user,
                    token,
                    isAuthenticated: true,
                });
            } catch (error) {
                console.error("Failed to parse user data:", error);
                logout();
            }
        }
        setIsLoading(false);
    }, []);

    return {
        ...authState,
        isLoading,
        login,
        logout,
    };
}
