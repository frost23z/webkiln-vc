'use client'

import { ApiError, getRun, startRun, type Brief, type RunDetail } from '@/lib/api'
import { useState } from 'react'
import { BriefForm } from './brief-form'

/** The message of an API error plus its field details, or a generic text for anything else. */
function errorText(err: unknown): string {
    if (!(err instanceof ApiError)) return 'Something went wrong.'
    const details = err.details.map((detail) => `${detail.field}: ${detail.message}`)
    return [err.message, ...details].join(' ')
}

export function Workspace() {
    const [run, setRun] = useState<RunDetail | null>(null)
    const [error, setError] = useState<string | null>(null)
    // True from pressing the submit button until the run is shown.
    const [starting, setStarting] = useState(false)

    async function submit(brief: Brief) {
        setError(null)
        setStarting(true)
        try {
            const { thread_id } = await startRun(brief)
            setRun(await getRun(thread_id))
        } catch (err) {
            setError(errorText(err))
        } finally {
            setStarting(false)
        }
    }

    return (
        <div className='flex flex-col gap-8'>
            <BriefForm onSubmit={submit} busy={starting} />
            {error && (
                <p
                    role='alert'
                    className='rounded-md border border-red-300 bg-red-50 px-4 py-2 text-sm text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100'
                >
                    {error}
                </p>
            )}
            {run && (
                <pre className='overflow-x-auto rounded-md bg-zinc-100 p-4 text-xs dark:bg-zinc-900'>
                    {JSON.stringify(run, null, 2)}
                </pre>
            )}
        </div>
    )
}
