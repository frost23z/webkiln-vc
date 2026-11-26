import type { SseEvent } from './api'

export function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Names of the nodes that finished in an `updates` event. Empty for any other event. */
export function finishedNodes(event: SseEvent): string[] {
    return event.event === 'updates' && isRecord(event.data) ? Object.keys(event.data) : []
}

/** One status line for a stream event, or null when there is nothing to show. */
export function describeEvent(event: SseEvent): string | null {
    const { data } = event
    switch (event.event) {
        case 'updates': {
            const nodes = finishedNodes(event).map((name) => name.replaceAll('_', ' '))
            return nodes.length > 0 ? `Finished ${nodes.join(', ')}` : null
        }
        case 'custom':
            return describeCustom(data)
        case 'error':
            return isRecord(data) && typeof data.message === 'string'
                ? data.message
                : 'The run failed.'
        case 'done':
            return isRecord(data) && typeof data.status === 'string'
                ? `Run ${data.status}`
                : 'Run finished'
        default:
            return null
    }
}

const WORKER_EVENTS: Record<string, string> = {
    worker_started: 'started',
    worker_done: 'finished',
    worker_failed: 'failed',
}

const DECISIONS: Record<string, string> = {
    retry: 'will run again',
    skip: 'is skipped',
    ask_user: 'needs your decision',
}

const MEMORY_KINDS: Record<string, [one: string, many: string]> = {
    brand: ['the brand guideline', 'brand guidelines'],
    preference: ['a preference', 'preferences'],
    decision: ['a decision', 'decisions'],
    feedback: ['your feedback', 'feedback notes'],
    business_fact: ['a fact about the business', 'facts about the business'],
    seo: ['the SEO summary', 'SEO summaries'],
}

/** "the brand guideline, 2 facts about the business and the SEO summary", or null when empty. */
function describeKinds(kinds: unknown): string | null {
    if (!Array.isArray(kinds)) return null

    const counts = new Map<string, number>()
    for (const kind of kinds) {
        if (typeof kind === 'string') counts.set(kind, (counts.get(kind) ?? 0) + 1)
    }

    const parts = [...counts].map(([kind, count]) => {
        const [one, many] = MEMORY_KINDS[kind] ?? [kind, kind]
        return count === 1 ? one : `${count} ${many}`
    })
    if (parts.length === 0) return null
    if (parts.length === 1) return parts[0]
    return `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`
}

function describeCustom(data: unknown): string | null {
    if (typeof data === 'string') return data
    if (isRecord(data) && data.event === 'memory_saved') {
        const saved = describeKinds(data.kinds)
        return saved ? `Remembered for next time: ${saved}` : null
    }
    if (isRecord(data) && data.event === 'failure_decision' && typeof data.action === 'string') {
        const outcome = DECISIONS[data.action] ?? data.action
        const reason = typeof data.reason === 'string' ? `: ${data.reason}` : ''
        return `${String(data.agent)} ${outcome}${reason}`
    }
    if (isRecord(data) && typeof data.event === 'string' && data.event in WORKER_EVENTS) {
        return `${String(data.agent)} ${WORKER_EVENTS[data.event]}`
    }
    return JSON.stringify(data)
}
