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
            return typeof data === 'string' ? data : JSON.stringify(data)
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
