const KEY = 'webkiln.threadId'

// sessionStorage can be missing or blocked, so every access is guarded.
export function readThreadId(): string | null {
    try {
        return sessionStorage.getItem(KEY)
    } catch {
        return null
    }
}

export function saveThreadId(threadId: string): void {
    try {
        sessionStorage.setItem(KEY, threadId)
    } catch {
        // Not remembering the run is fine. The page still works.
    }
}

export function clearThreadId(): void {
    try {
        sessionStorage.removeItem(KEY)
    } catch {
        // Nothing to clear when storage is unavailable.
    }
}
