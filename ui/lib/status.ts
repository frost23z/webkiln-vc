import type { RunStatus, SseEvent } from './api'
import { isRecord } from './progress'

export type Banner = { label: string; tone: 'working' | 'done' | 'failed' }

// Phase names for the nodes that exist so far. A node that is not listed shows as "Working".
const PHASES: Record<string, string> = {
    process_initial_input: 'Planning',
    seed_tasks: 'Planning',
    dispatch: 'Researching',
    run_worker: 'Researching',
    brand_review: 'Waiting for approval',
    edit_router: 'Applying modifications',
}

/** `node` is the last node that finished, so its phase shows until the next update arrives. */
export function bannerFor(status: RunStatus, node: string | null): Banner {
    switch (status) {
        case 'running':
            return { label: (node && PHASES[node]) || 'Working', tone: 'working' }
        case 'completed':
            return { label: 'Finished', tone: 'done' }
        case 'failed':
            return { label: 'Run failed', tone: 'failed' }
    }
}

/** The final status carried by the `done` event, or null for any other event. */
export function doneStatus(event: SseEvent): RunStatus | null {
    if (event.event !== 'done' || !isRecord(event.data)) return null
    const { status } = event.data
    return status === 'completed' || status === 'failed' ? status : null
}
