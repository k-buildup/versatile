// Single Responsibility: 사용자 관련 타입만 정의

export interface User {
    id: string;
    username: string;
    email: string;
    displayName: string;
}

export interface AuthState {
    user: User | null;
    token: string | null;
    isAuthenticated: boolean;
}
