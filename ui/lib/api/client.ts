import createClient from 'openapi-fetch'
import type { paths } from '../generated/schema'

import { ApiError } from './errors'
import { readEvents, type SseEvent } from './sse'
import type { Brief, ErrorResponse, ReviewDecision } from './types'

const baseUrl = process.env.NEXT_PUBLIC_API_URL

if (!baseUrl) {
    throw new ApiError('config_error', 'NEXT_PUBLIC_API_URL is not set. See .env.example.')
}

const client = createClient<paths>({ baseUrl })

function unwrap<T>(result: { data?: T; error?: unknown }): T {
    if (result.data !== undefined) return result.data

    const body = (result.error as Partial<ErrorResponse> | undefined)?.error

    throw new ApiError(
        body?.code ?? 'unknown_error',
        body?.message ?? 'The API returned an unexpected response.',
        body?.details ?? []
    )
}

async function call<T>(request: () => Promise<{ data?: T; error?: unknown }>): Promise<T> {
    try {
        return unwrap(await request())
    } catch (err) {
        if (err instanceof ApiError) throw err

        throw new ApiError('network_error', 'Could not reach the API. Is it running?')
    }
}

export { call }

export function startRun(brief: Brief) {
    return call(() => client.POST('/runs', { body: brief }))
}

export function listRuns() {
    return call(() => client.GET('/runs'))
}

export function getRun(threadId: string) {
    return call(() =>
        client.GET('/runs/{thread_id}', {
            params: { path: { thread_id: threadId } },
        })
    )
}

export function retryRun(threadId: string) {
    return call(() =>
        client.POST('/runs/{thread_id}/retry', {
            params: { path: { thread_id: threadId } },
        })
    )
}

/**
 * Answers the brand review. The key makes a repeat of the same click harmless: the API runs the
 * answer once and replies to a repeat with the run as it stands.
 */
export function resumeRun(threadId: string, decision: ReviewDecision, key: string) {
    return call(() =>
        client.POST('/runs/{thread_id}/resume', {
            params: { path: { thread_id: threadId }, header: { 'idempotency-key': key } },
            body: decision,
        })
    )
}

export function getTaskReport(threadId: string, taskId: string) {
    return call(() =>
        client.GET('/runs/{thread_id}/tasks/{task_id}', {
            params: { path: { thread_id: threadId, task_id: taskId } },
        })
    )
}

/**
 * Follows the run and yields its server-sent events until the stream ends.
 * A run that is already going first replays its events so far. `fetch` is used instead of `EventSource` so the request can be aborted.
 */
export async function* streamRun(threadId: string, signal?: AbortSignal): AsyncGenerator<SseEvent> {
    let response: Response
    try {
        response = await fetch(`${baseUrl}/runs/${encodeURIComponent(threadId)}/stream`, {
            headers: { Accept: 'text/event-stream' },
            signal,
        })
    } catch (err) {
        if (signal?.aborted) throw err
        throw new ApiError('network_error', 'Could not reach the API. Is it running?')
    }

    if (!response.ok || !response.body) {
        const body = (await response.json().catch(() => null)) as ErrorResponse | null
        throw new ApiError(
            body?.error?.code ?? 'unknown_error',
            body?.error?.message ?? 'The API returned an unexpected response.',
            body?.error?.details ?? []
        )
    }

    yield* readEvents(response.body)
}
