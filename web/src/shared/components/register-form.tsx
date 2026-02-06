"use client";

// Single Responsibility: 회원가입 UI만 담당

import { useState } from "react";

interface RegisterFormProps {
    onRegister: (
        username: string,
        email: string,
        password: string,
        displayName: string,
    ) => void;
    onSwitchToLogin: () => void;
    error?: string;
}

export function RegisterForm({
    onRegister,
    onSwitchToLogin,
    error,
}: RegisterFormProps) {
    const [username, setUsername] = useState("");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [displayName, setDisplayName] = useState("");

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (username.trim() && email.trim() && password) {
            onRegister(
                username.trim(),
                email.trim(),
                password,
                displayName.trim(),
            );
        }
    };

    const inputStyle = {
        backgroundColor: "var(--color-dark-hover)",
        borderColor: "var(--color-dark-border)",
        color: "var(--color-dark-text)",
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
                    style={inputStyle}
                    placeholder="username"
                    required
                />
            </div>
            <div>
                <label
                    className="block text-sm mb-2"
                    style={{ color: "var(--color-dark-text-secondary)" }}
                >
                    이메일
                </label>
                <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2"
                    style={inputStyle}
                    placeholder="email@example.com"
                    required
                />
            </div>
            <div>
                <label
                    className="block text-sm mb-2"
                    style={{ color: "var(--color-dark-text-secondary)" }}
                >
                    표시 이름
                </label>
                <input
                    type="text"
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2"
                    style={inputStyle}
                    placeholder="표시될 이름"
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
                    style={inputStyle}
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
                회원가입
            </button>
            {error && (
                <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
                    {error}
                </div>
            )}
            <div className="mt-4 text-center text-sm">
                <span style={{ color: "var(--color-dark-text-secondary)" }}>
                    이미 계정이 있으신가요?
                </span>
                <button
                    type="button"
                    onClick={onSwitchToLogin}
                    className="hover:underline ml-1"
                    style={{ color: "var(--color-dark-text)" }}
                >
                    로그인
                </button>
            </div>
        </form>
    );
}
