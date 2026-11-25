import { z } from 'zod'
import type { DirectEdits } from './api'

const strings = z.array(z.string()).default([])

const guidelineSchema = z.object({
    summary: z.string(),
    positioning: z.string(),
    personality_and_tone: z.object({
        traits: strings,
        voice: z.string(),
        do: strings,
        dont: strings,
    }),
    messaging: z.object({
        tagline: z.string(),
        value_proposition: z.string(),
        key_messages: strings,
    }),
    palette: z.object({
        primary: z.string(),
        secondary: z.string(),
        accent: z.string(),
        background: z.string(),
    }),
    typography: z.object({
        heading_font: z.string(),
        body_font: z.string(),
        notes: z.string(),
    }),
    logo_direction: z.string(),
    image_guidelines: strings,
    ui_ux_direction: strings,
    inspiration: strings,
})

export type Guideline = z.infer<typeof guidelineSchema>

const reviewSchema = z.object({
    type: z.literal('brand_review'),
    version: z.number(),
    revisions_left: z.number(),
    guideline: z.unknown(),
    error: z.string().nullish(),
})

export type BrandReview = {
    version: number
    revisionsLeft: number
    /** Why the last answer was refused, if it was. */
    error: string | null
    /** The guideline in full, or null when it does not have the expected shape. */
    guideline: Guideline | null
}

/** The question a waiting run asks, from untrusted data. Null when it is not a brand review. */
export function parseReview(value: unknown): BrandReview | null {
    const review = reviewSchema.safeParse(value)
    if (!review.success) return null
    const guideline = guidelineSchema.safeParse(review.data.guideline)
    return {
        version: review.data.version,
        revisionsLeft: review.data.revisions_left,
        error: review.data.error ?? null,
        guideline: guideline.success ? guideline.data : null,
    }
}

/** The fields the user can change by hand, as the form holds them. */
export type EditableFields = {
    primary: string
    secondary: string
    accent: string
    background: string
    heading_font: string
    body_font: string
    positioning: string
    tagline: string
}

export function editableFields(guideline: Guideline): EditableFields {
    return {
        ...guideline.palette,
        heading_font: guideline.typography.heading_font,
        body_font: guideline.typography.body_font,
        positioning: guideline.positioning,
        tagline: guideline.messaging.tagline,
    }
}

/** Only what the user changed, so an untouched field is never sent back. */
export function changedEdits(original: EditableFields, current: EditableFields): DirectEdits {
    const edits: DirectEdits = {}
    for (const key of Object.keys(original) as (keyof EditableFields)[]) {
        const now = current[key].trim()
        if (now.toLowerCase() !== original[key].toLowerCase()) edits[key] = now
    }
    return edits
}
