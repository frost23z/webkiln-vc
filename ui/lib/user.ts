const KEY = 'webkiln.userId'

// Used when localStorage is blocked: one id for as long as the page stays open.
let fallback: string | null = null

function newId(): string {
    return crypto.randomUUID()
}

/**
 * The id that says whose long-term memory a run uses. It is made once and kept in localStorage,
 * so the same browser gets the same memory in every session. It is not a login: anyone who
 * copies the id gets that memory, and clearing the browser's data starts a new one.
 */
export function getUserId(): string {
    try {
        const saved = localStorage.getItem(KEY)
        if (saved) return saved

        const id = newId()
        localStorage.setItem(KEY, id)
        return id
    } catch {
        // Storage can be missing or blocked. Memory then lasts only until the page is closed.
        fallback ??= newId()
        return fallback
    }
}
