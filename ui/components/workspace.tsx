'use client'

import { ApiError, getRun, startRun, type Brief, type RunDetail } from '@/lib/api'
import { clearThreadId, readThreadId, saveThreadId } from '@/lib/thread'
import { useEffect, useState } from 'react'
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
    // True until the saved thread id has been looked up, so the form does not flash on refresh.
    const [restoring, setRestoring] = useState(true)
    // True from pressing the submit button until the run is shown.
    const [starting, setStarting] = useState(false)

    // On page load, show the run saved in this tab, if there is one.
    useEffect(() => {
        const threadId = readThreadId()
        if (!threadId) {
            queueMicrotask(() => setRestoring(false))
            return
        }
        let cancelled = false
        getRun(threadId)
            .then((saved) => {
                if (!cancelled) setRun(saved)
            })
            .catch((err) => {
                if (cancelled) return
                if (err instanceof ApiError && err.code === 'not_found') {
                    clearThreadId() // the API forgot this run (runs are in memory)
                } else {
                    // The API may be down only for a moment, so the saved run is kept.
                    setError(`Could not load the saved run. ${errorText(err)}`)
                }
            })
            .finally(() => {
                if (!cancelled) setRestoring(false)
            })
        return () => {
            cancelled = true
        }
    }, [])

    async function submit(brief: Brief) {
        setError(null)
        setStarting(true)
        try {
            const { thread_id } = await startRun(brief)
            saveThreadId(thread_id)
            setRun(await getRun(thread_id))
        } catch (err) {
            setError(errorText(err))
        } finally {
            setStarting(false)
        }
    }

    function newRun() {
        clearThreadId()
        setRun(null)
        setError(null)
    }

    if (restoring) {
        return (
            <p role='status' className='text-sm text-zinc-600 dark:text-zinc-400'>
                Loading the saved run…
            </p>
        )
    }

    return (
        <div className='flex flex-col gap-8'>
            {run ? (
                <button
                    type='button'
                    onClick={newRun}
                    className='self-start rounded-md border border-zinc-300 px-4 py-2 text-sm font-medium hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-900'
                >
                    New run
                </button>
            ) : (
                <BriefForm onSubmit={submit} busy={starting} />
            )}
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
