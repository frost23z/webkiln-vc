import { z } from 'zod'
import type { SseEvent } from './api'
import { isRecord } from './progress'

const taskSchema = z.object({
    id: z.string(),
    agent: z.string(),
    instruction: z.string().default(''),
    depends_on: z.array(z.string()).default([]),
})

const planSchema = z.object({
    rationale: z.string(),
    tasks: z.array(taskSchema),
})

export type Task = z.infer<typeof taskSchema>
export type Plan = z.infer<typeof planSchema>

/** A plan from untrusted data, or null if the data is not a plan. */
export function parsePlan(value: unknown): Plan | null {
    const parsed = planSchema.safeParse(value)
    return parsed.success ? parsed.data : null
}

/**
 * The plan carried by an `updates` event, or null for any other event. Every run starts from the
 * default plan, which `seed_tasks` writes. Later plan changes will arrive from the Manager nodes.
 */
export function planFromEvent(event: SseEvent): Plan | null {
    if (event.event !== 'updates' || !isRecord(event.data)) return null
    const update = event.data['seed_tasks']
    return isRecord(update) && 'plan' in update ? parsePlan(update.plan) : null
}
