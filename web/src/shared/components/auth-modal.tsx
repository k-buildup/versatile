"use client";

// Single Responsibility: 인증 모달 컨테이너만 담당

import { MessageCircle } from "lucide-react";
import { useState } from "react";

import { RegisterForm } from "./register-form";
import { LoginForm } from "./login-form";

interface AuthModalProps {
    isOpen: boolean;
    onLogin: (username: string, password: string) => void;
    onRegister: (
        username: string,
        email: string,
        password: string,
        displayName: string,
    ) => void;
    error?: string;
}

export function AuthModal({
    isOpen,
    onLogin,
    onRegister,
    error,
}: AuthModalProps) {
    const [isLoginMode, setIsLoginMode] = useState(true);

    if (!isOpen) return null;

    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center"
            style={{
                backgroundColor: "var(--color-dark-bg)",
            }}
        >
            <div
                className="border rounded-xl p-8 max-w-md w-full mx-4"
                style={{
                    backgroundColor: "var(--color-dark-sidebar)",
                    borderColor: "var(--color-dark-border)",
                }}
            >
                <div className="text-center mb-6">
                    <div
                        className="w-16 h-16 mx-auto rounded-full flex items-center justify-center mb-4"
                        style={{ backgroundColor: "var(--color-dark-hover)" }}
                    >
                        <MessageCircle className="w-8 h-8 stroke-white" />
                    </div>
                    <h2
                        className="text-2xl font-semibold mb-2"
                        style={{
                            color: "var(--color-dark-text)",
                        }}
                    >
                        Versatile
                    </h2>
                    <p style={{ color: "var(--color-dark-text-secondary)" }}>
                        로그인하여 시작하세요
                    </p>
                </div>

                {isLoginMode ? (
                    <LoginForm
                        onLogin={onLogin}
                        onSwitchToRegister={() => setIsLoginMode(false)}
                        error={error}
                    />
                ) : (
                    <RegisterForm
                        onRegister={onRegister}
                        onSwitchToLogin={() => setIsLoginMode(true)}
                        error={error}
                    />
                )}
            </div>
        </div>
    );
}
