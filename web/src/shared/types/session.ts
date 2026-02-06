// Single Responsibility: 세션 관련 타입만 정의

import { Message, ChatMode } from "./message";

export interface Session {
    id: string;
    title: string;
    createdAt: Date;
    updatedAt: Date;
}

export interface ChatSession extends Session {
    messages: Message[];
    mode: ChatMode;
}
