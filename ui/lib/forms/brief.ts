import { z } from 'zod'

import type { Brief } from '../api/types'

export const briefSchema = z.object({
    brand_name: z
        .string()
        .trim()
        .min(2, 'Brand name needs at least 2 characters.')
        .max(80, 'Brand name can have at most 80 characters.'),

    prompt: z
        .string()
        .trim()
        .min(20, 'Prompt needs at least 20 characters.')
        .max(2000, 'Prompt can have at most 2000 characters.'),

    niche: z
        .string()
        .trim()
        .min(3, 'Niche needs at least 3 characters.')
        .max(80, 'Niche can have at most 80 characters.'),
})

export type BriefInput = z.input<typeof briefSchema>
export type ValidatedBrief = z.output<typeof briefSchema>

export type BriefErrors = Partial<Record<keyof Brief, string>>

export function validateBrief(values: BriefInput): {
    brief: Brief | null
    errors: BriefErrors
} {
    const result = briefSchema.safeParse(values)

    if (result.success) {
        return {
            brief: result.data,
            errors: {},
        }
    }

    const errors: BriefErrors = {}

    for (const issue of result.error.issues) {
        const field = issue.path[0] as keyof Brief

        if (field in briefSchema.shape && !errors[field]) {
            errors[field] = issue.message
        }
    }

    return {
        brief: null,
        errors,
    }
}
