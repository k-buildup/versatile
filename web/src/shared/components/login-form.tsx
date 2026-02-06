"use client";

// Single Responsibility: 로그인 UI만 담당

import { useState } from "react";

interface LoginFormProps {
    onLogin: (username: string, password: string) => void;
    onSwitchToRegister: () => void;
    error?: string;
}

export function LoginForm({
    onLogin,
    onSwitchToRegister,
    error,
}: LoginFormProps) {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (username.trim() && password) {
            onLogin(username.trim(), password);
        }
    };

    return (
        <form onSubmit={handleSubmit} className="space-y-4">
            <div>
                <label
                    className="block text-sm mb-2"
                    style={{ color: "var(--color-dark-text-secondary)" }}
                >
                    사용자명
                </label>
                <input
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2"
                    style={{
                        backgroundColor: "var(--color-dark-hover)",
                        borderColor: "var(--color-dark-border)",
                        color: "var(--color-dark-text)",
                    }}
                    placeholder="username"
                    required
                />
            </div>
            <div>
                <label
                    className="block text-sm mb-2"
                    style={{ color: "var(--color-dark-text-secondary)" }}
                >
                    비밀번호
                </label>
                <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2"
                    style={{
                        backgroundColor: "var(--color-dark-hover)",
                        borderColor: "var(--color-dark-border)",
                        color: "var(--color-dark-text)",
                    }}
                    placeholder="••••••••"
                    required
                />
            </div>
            <button
                type="submit"
                className="w-full px-4 py-2 rounded-lg transition-colors font-medium"
                style={{
                    backgroundColor: "var(--color-dark-text)",
                    color: "var(--color-dark-bg)",
                }}
            >
                로그인
            </button>
            {error && (
                <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
                    {error}
                </div>
            )}
            <div className="mt-4 text-center text-sm">
                <span style={{ color: "var(--color-dark-text-secondary)" }}>
                    계정이 없으신가요?
                </span>
                <button
                    type="button"
                    onClick={onSwitchToRegister}
                    className="hover:underline ml-1"
                    style={{ color: "var(--color-dark-text)" }}
                >
                    회원가입
                </button>
            </div>
        </form>
    );
}
