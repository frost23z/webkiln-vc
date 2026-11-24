import type { ErrorDetail } from './types'

export class ApiError extends Error {
    constructor(
        readonly code: string,
        message: string,
        readonly details: ErrorDetail[] = []
    ) {
        super(message)
        this.name = 'ApiError'
    }
}
