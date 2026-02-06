// Single Responsibility: 메시지 관련 타입만 정의

export type MessageRole = "user" | "assistant" | "system";

export type ChatMode = "chat" | "tool" | "think";

export interface BaseMessage {
    id: string;
    role: MessageRole;
    content: string;
    timestamp: Date;
}

export interface UserMessage extends BaseMessage {
    role: "user";
}

export interface AssistantMessage extends BaseMessage {
    role: "assistant";
    thinkingProcess?: ThinkingStep[];
    toolUsage?: ToolUsage[];
}

export interface SystemMessage extends BaseMessage {
    role: "system";
}

export interface ThinkingStep {
    id: string;
    todo: string;
    content: string;
    isCompleted: boolean;
    isExpanded: boolean;
}

export interface ToolUsage {
    id: string;
    toolName: string;
    toolInput: string;
    toolOutput: string;
}

export type Message = UserMessage | AssistantMessage | SystemMessage;
