'use client'

import {
    ApiError,
    getRun,
    startRun,
    streamRun,
    type Brief,
    type RunDetail,
    type RunStatus,
} from '@/lib/api'
import { describeEvent, finishedNodes } from '@/lib/progress'
import { bannerFor, doneStatus } from '@/lib/status'
import { clearThreadId, readThreadId, saveThreadId } from '@/lib/thread'
import { useEffect, useRef, useState } from 'react'
import { BriefForm } from './brief-form'
import { ProgressLines } from './progress-lines'
import { StatusBanner } from './status-banner'

/** The message of an API error plus its field details, or a generic text for anything else. */
function errorText(err: unknown): string {
    if (!(err instanceof ApiError)) return 'Something went wrong.'
    const details = err.details.map((detail) => `${detail.field}: ${detail.message}`)
    return [err.message, ...details].join(' ')
}

export function Workspace() {
    const [run, setRun] = useState<RunDetail | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [lines, setLines] = useState<string[]>([])
    // The last node that finished. The banner turns it into a phase name.
    const [node, setNode] = useState<string | null>(null)
    // The live status. `run.status` only changes when the run is loaded again.
    const [status, setStatus] = useState<RunStatus | null>(null)
    // Lets "New run" stop a stream that is still open.
    const streamAbort = useRef<AbortController | null>(null)
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
                if (cancelled) return
                setRun(saved)
                setStatus(saved.status)
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
        setLines([])
        setNode(null)
        setStatus(null)
        setStarting(true)
        const abort = new AbortController()
        streamAbort.current = abort
        try {
            const { thread_id } = await startRun(brief)
            saveThreadId(thread_id)
            setRun(await getRun(thread_id))
            setStarting(false)
            setStatus('running')
            for await (const event of streamRun(thread_id, abort.signal)) {
                const line = describeEvent(event)
                if (line) setLines((current) => [...current, line])
                const last = finishedNodes(event).at(-1)
                if (last) setNode(last)
                const final = doneStatus(event)
                if (final) setStatus(final)
            }
            setRun(await getRun(thread_id)) // the state after the run
        } catch (err) {
            if (abort.signal.aborted) return // the user started a new run
            setStatus((current) => (current === 'running' ? 'failed' : current))
            setError(errorText(err))
        } finally {
            setStarting(false)
        }
    }

    function newRun() {
        streamAbort.current?.abort()
        clearThreadId()
        setRun(null)
        setError(null)
        setLines([])
        setNode(null)
        setStatus(null)
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
            {status && <StatusBanner banner={bannerFor(status, node)} />}
            <ProgressLines lines={lines} />
            {run && (
                <pre className='overflow-x-auto rounded-md bg-zinc-100 p-4 text-xs dark:bg-zinc-900'>
                    {JSON.stringify(run, null, 2)}
                </pre>
            )}
        </div>
    )
}
