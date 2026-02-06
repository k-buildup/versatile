"use client";

// Single Responsibility: 헤더 UI만 담당

import { MenuIcon } from "lucide-react";

interface HeaderProps {
    title: string;
    isConnected: boolean;
}

export function Header({ title, isConnected }: HeaderProps) {
    return (
        <div
            className="border-b p-4"
            style={{ borderColor: "var(--color-dark-border)" }}
        >
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <button className="lg:hidden p-2 rounded-lg hover:bg-[var(--color-dark-hover)]">
                        <MenuIcon className="w-5 h-5" />
                    </button>
                    <h1 className="text-lg font-semibold">{title}</h1>
                </div>
                <div className="flex items-center gap-2">
                    <div
                        className="flex items-center gap-2 px-3 py-1.5 border rounded-full"
                        style={{
                            backgroundColor: "var(--color-dark-hover)",
                            borderColor: "var(--color-dark-border)",
                        }}
                    >
                        <div
                            className={`w-2 h-2 rounded-full ${
                                isConnected ? "bg-green-500" : "bg-red-500"
                            }`}
                        />
                        <span
                            className="text-xs"
                            style={{
                                color: "var(--color-dark-text-secondary)",
                            }}
                        >
                            {isConnected ? "Connected" : "Offline"}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
}
