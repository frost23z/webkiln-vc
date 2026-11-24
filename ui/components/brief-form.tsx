'use client'

import type { Brief } from '@/lib/api'
import { validateBrief, type BriefErrors } from '@/lib/forms/brief'
import { useState } from 'react'

const inputClass =
    'w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 shadow-sm focus:border-zinc-900 focus:outline-none dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-50 dark:focus:border-zinc-300'

export function BriefForm({ onSubmit, busy }: { onSubmit: (brief: Brief) => void; busy: boolean }) {
    const [values, setValues] = useState({ brand_name: '', niche: '', prompt: '' })
    const [errors, setErrors] = useState<BriefErrors>({})

    function set(field: keyof Brief, value: string) {
        setValues((current) => ({ ...current, [field]: value }))
    }

    function handleSubmit(event: React.SyntheticEvent<HTMLFormElement>) {
        event.preventDefault()
        const result = validateBrief(values)
        setErrors(result.errors)
        if (result.brief) onSubmit(result.brief)
    }

    return (
        <form onSubmit={handleSubmit} noValidate className='flex flex-col gap-5'>
            <Field label='Brand name' id='brand_name' error={errors.brand_name}>
                <input
                    id='brand_name'
                    className={inputClass}
                    value={values.brand_name}
                    onChange={(e) => set('brand_name', e.target.value)}
                    aria-invalid={errors.brand_name ? true : undefined}
                />
            </Field>
            <Field label='Niche' id='niche' error={errors.niche}>
                <input
                    id='niche'
                    className={inputClass}
                    value={values.niche}
                    onChange={(e) => set('niche', e.target.value)}
                    aria-invalid={errors.niche ? true : undefined}
                />
            </Field>
            <Field label='Prompt' id='prompt' error={errors.prompt}>
                <textarea
                    id='prompt'
                    rows={6}
                    className={inputClass}
                    value={values.prompt}
                    onChange={(e) => set('prompt', e.target.value)}
                    aria-invalid={errors.prompt ? true : undefined}
                />
            </Field>
            <button
                type='submit'
                disabled={busy}
                className='self-start rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-700 disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300'
            >
                {busy ? 'Starting…' : 'Build website'}
            </button>
        </form>
    )
}

function Field({
    label,
    id,
    error,
    children,
}: {
    label: string
    id: string
    error?: string
    children: React.ReactNode
}) {
    return (
        <div className='flex flex-col gap-1.5'>
            <label htmlFor={id} className='text-sm font-medium'>
                {label}
            </label>
            {children}
            {error && (
                <p role='alert' className='text-sm text-red-600 dark:text-red-400'>
                    {error}
                </p>
            )}
        </div>
    )
}
