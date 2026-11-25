'use client'

import type { ReviewDecision } from '@/lib/api'
import {
    changedEdits,
    editableFields,
    type BrandReview,
    type EditableFields,
    type Guideline,
} from '@/lib/review'
import { useState } from 'react'

const inputClass =
    'w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 shadow-sm focus:border-zinc-900 focus:outline-none dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-50 dark:focus:border-zinc-300'
const buttonClass =
    'rounded-md border border-zinc-300 px-4 py-2 text-sm font-medium hover:bg-zinc-100 disabled:opacity-50 dark:border-zinc-700 dark:hover:bg-zinc-900'

const COLORS = ['primary', 'secondary', 'accent', 'background'] as const

type Props = {
    review: BrandReview
    busy: boolean
    onDecide: (decision: ReviewDecision) => void
}

/**
 * The brand guideline, for the user to approve or change. The run is paused until it is answered.
 * Colours, fonts, positioning and tagline can be edited by hand: the API applies those in code,
 * with no model call. Anything else goes in the feedback, which the Manager reads.
 */
export function ApprovalCard({ review, busy, onDecide }: Props) {
    const { guideline } = review
    // The card is remounted for every new version (see the `key` in the workspace), so the form
    // starts from the guideline on screen.
    const original = guideline ? editableFields(guideline) : null
    const [fields, setFields] = useState<EditableFields | null>(original)
    const [feedback, setFeedback] = useState('')

    const edits = original && fields ? changedEdits(original, fields) : {}
    const changed = Object.keys(edits).length > 0 || feedback.trim().length > 0
    const canRevise = review.revisionsLeft > 0

    function set(field: keyof EditableFields, value: string) {
        setFields((current) => (current ? { ...current, [field]: value } : current))
    }

    function requestChanges() {
        const text = feedback.trim()
        onDecide({
            action: 'revise',
            ...(text ? { feedback: text } : {}),
            ...(Object.keys(edits).length > 0 ? { edits } : {}),
        })
    }

    return (
        <section
            aria-label='Brand guideline review'
            className='flex flex-col gap-5 rounded-md border border-amber-300 p-4 text-sm dark:border-amber-800'
        >
            <div className='flex flex-col gap-1'>
                <h2 className='text-base font-medium'>Brand guideline, version {review.version}</h2>
                <p className='text-zinc-600 dark:text-zinc-400'>
                    The wireframe and the website are not built until you approve this.{' '}
                    {canRevise
                        ? `You can ask for changes ${review.revisionsLeft} more ${review.revisionsLeft === 1 ? 'time' : 'times'}.`
                        : 'You have used all your rounds of changes, so only Approve is left.'}
                </p>
            </div>
            {review.error && (
                <p
                    role='alert'
                    className='rounded-md border border-red-300 bg-red-50 px-4 py-2 text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100'
                >
                    That answer was not accepted: {review.error}
                </p>
            )}
            {guideline && fields ? (
                <>
                    <GuidelineView guideline={guideline} />
                    <fieldset
                        disabled={busy || !canRevise}
                        className='flex flex-col gap-4 rounded-md border border-zinc-200 p-4 dark:border-zinc-800'
                    >
                        <legend className='px-1 font-medium'>Change it yourself</legend>
                        <div className='grid grid-cols-2 gap-4 sm:grid-cols-4'>
                            {COLORS.map((name) => (
                                <label key={name} className='flex flex-col gap-1'>
                                    <span className='capitalize'>{name} colour</span>
                                    <input
                                        type='color'
                                        value={fields[name]}
                                        onChange={(e) => set(name, e.target.value)}
                                        className='h-10 w-full cursor-pointer rounded-md border border-zinc-300 dark:border-zinc-700'
                                    />
                                </label>
                            ))}
                        </div>
                        <div className='grid gap-4 sm:grid-cols-2'>
                            <label className='flex flex-col gap-1'>
                                <span>Heading font</span>
                                <input
                                    className={inputClass}
                                    value={fields.heading_font}
                                    onChange={(e) => set('heading_font', e.target.value)}
                                />
                            </label>
                            <label className='flex flex-col gap-1'>
                                <span>Body font</span>
                                <input
                                    className={inputClass}
                                    value={fields.body_font}
                                    onChange={(e) => set('body_font', e.target.value)}
                                />
                            </label>
                        </div>
                        <label className='flex flex-col gap-1'>
                            <span>Tagline</span>
                            <input
                                className={inputClass}
                                value={fields.tagline}
                                onChange={(e) => set('tagline', e.target.value)}
                            />
                        </label>
                        <label className='flex flex-col gap-1'>
                            <span>Positioning</span>
                            <textarea
                                rows={3}
                                className={inputClass}
                                value={fields.positioning}
                                onChange={(e) => set('positioning', e.target.value)}
                            />
                        </label>
                    </fieldset>
                </>
            ) : (
                <p className='text-zinc-600 dark:text-zinc-400'>
                    The guideline could not be shown here. You can still approve it, or ask for
                    changes in words.
                </p>
            )}
            <label className='flex flex-col gap-1'>
                <span className='font-medium'>What should change?</span>
                <textarea
                    rows={3}
                    className={inputClass}
                    value={feedback}
                    disabled={busy || !canRevise}
                    maxLength={2000}
                    placeholder='For example: more premium, with dark green and gold. Or: we sell to business travellers.'
                    onChange={(e) => setFeedback(e.target.value)}
                />
            </label>
            <div className='flex flex-wrap gap-3'>
                <button
                    type='button'
                    disabled={busy}
                    onClick={() => onDecide({ action: 'approve' })}
                    className='rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-700 disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300'
                >
                    {busy ? 'Sending…' : 'Approve'}
                </button>
                <button
                    type='button'
                    disabled={busy || !canRevise || !changed}
                    onClick={requestChanges}
                    className={buttonClass}
                >
                    Request changes
                </button>
            </div>
        </section>
    )
}

function GuidelineView({ guideline }: { guideline: Guideline }) {
    const { palette, typography, personality_and_tone: tone, messaging } = guideline
    return (
        <div className='flex flex-col gap-4'>
            <p>{guideline.summary}</p>
            <div className='flex flex-wrap gap-3' aria-label='Palette'>
                {COLORS.map((name) => (
                    <div key={name} className='flex items-center gap-2'>
                        <span
                            aria-hidden
                            className='h-6 w-6 rounded-full border border-zinc-300 dark:border-zinc-700'
                            style={{ backgroundColor: palette[name] }}
                        />
                        <span className='capitalize'>
                            {name} <code>{palette[name]}</code>
                        </span>
                    </div>
                ))}
            </div>
            <Section title='Positioning'>
                <p>{guideline.positioning}</p>
            </Section>
            <Section title='Messaging'>
                <p>
                    <span className='font-medium'>Tagline:</span> {messaging.tagline}
                </p>
                <p>{messaging.value_proposition}</p>
                <List items={messaging.key_messages} />
            </Section>
            <Section title='Personality and tone'>
                <p>{tone.traits.join(', ')}</p>
                <p>{tone.voice}</p>
                <p className='font-medium'>Do</p>
                <List items={tone.do} />
                <p className='font-medium'>Do not</p>
                <List items={tone.dont} />
            </Section>
            <Section title='Typography'>
                <p>
                    {typography.heading_font} for headings, {typography.body_font} for text.{' '}
                    {typography.notes}
                </p>
            </Section>
            <Section title='Logo'>
                <p>{guideline.logo_direction}</p>
            </Section>
            <Section title='Images'>
                <List items={guideline.image_guidelines} />
            </Section>
            <Section title='UI and UX'>
                <List items={guideline.ui_ux_direction} />
            </Section>
            <Section title='Inspiration'>
                <List items={guideline.inspiration} />
            </Section>
        </div>
    )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
    return (
        <section className='flex flex-col gap-1'>
            <h3 className='font-medium'>{title}</h3>
            {children}
        </section>
    )
}

function List({ items }: { items: string[] }) {
    return (
        <ul className='list-disc pl-5'>
            {items.map((item) => (
                <li key={item}>{item}</li>
            ))}
        </ul>
    )
}
