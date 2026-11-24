import createClient from 'openapi-fetch'
import type { paths } from '../generated/schema'

import { ApiError } from './errors'
import type { Brief, ErrorResponse } from './types'

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

export function getRun(threadId: string) {
    return call(() =>
        client.GET('/runs/{thread_id}', {
            params: { path: { thread_id: threadId } },
        })
    )
}
