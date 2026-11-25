'use client'

import {
    ApiError,
    getRun,
    listRuns,
    retryRun,
    startRun,
    streamRun,
    type Brief,
    type RunDetail,
    type RunStatus,
    type RunSummary,
} from '@/lib/api'
import { parsePlan, parseTaskStates, planFromEvent, type Plan } from '@/lib/plan'
import { describeEvent, finishedNodes } from '@/lib/progress'
import { bannerFor, doneStatus } from '@/lib/status'
import { clearThreadId, readThreadId, saveThreadId } from '@/lib/thread'
import { useCallback, useEffect, useRef, useState } from 'react'
import { BriefForm } from './brief-form'
import { ProgressLines } from './progress-lines'
import { RunHistory } from './run-history'
import { StatusBanner } from './status-banner'
import { TaskList } from './task-list'

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
    // The plan as soon as seed_tasks finishes. A saved run gets its plan from `run.state`.
    const [livePlan, setLivePlan] = useState<Plan | null>(null)
    // Lets "New run" stop a stream that is still open.
    const streamAbort = useRef<AbortController | null>(null)
    // True until the saved thread id has been looked up, so the form does not flash on refresh.
    const [restoring, setRestoring] = useState(true)
    // True from pressing the submit button until the run is shown.
    const [starting, setStarting] = useState(false)
    // The saved runs, listed under the form.
    const [history, setHistory] = useState<RunSummary[]>([])

    /**
     * Follows a run until it ends. The run goes on in the API whether or not this page is open,
     * so a connection made late first replays the events so far. After every update the run is
     * loaded again, so the task statuses follow the graph state.
     */
    const follow = useCallback(async (threadId: string) => {
        streamAbort.current?.abort()
        const abort = new AbortController()
        streamAbort.current = abort
        setLines([])
        setLivePlan(null)
        setNode(null)
        setStatus('running')

        let loading = false
        let again = false
        async function refresh() {
            if (loading) {
                again = true // one more load after this one, so the last update is never missed
                return
            }
            loading = true
            try {
                do {
                    again = false
                    const loaded = await getRun(threadId)
                    if (!abort.signal.aborted) setRun(loaded)
                } while (again)
            } catch {
                // The next update loads it again.
            } finally {
                loading = false
            }
        }

        try {
            for await (const event of streamRun(threadId, abort.signal)) {
                const line = describeEvent(event)
                if (line) setLines((current) => [...current, line])
                const planned = planFromEvent(event)
                if (planned) setLivePlan(planned)
                const last = finishedNodes(event).at(-1)
                if (last) setNode(last)
                const final = doneStatus(event)
                if (final) setStatus(final)
                if (event.event === 'updates') void refresh()
            }
            const loaded = await getRun(threadId) // the state after the run
            if (abort.signal.aborted) return
            setRun(loaded)
            setStatus(loaded.status)
        } catch (err) {
            if (abort.signal.aborted) return // the user started a new run
            // Losing the connection does not stop the run, so show what the API says it is now.
            setError(errorText(err))
            try {
                const loaded = await getRun(threadId)
                setRun(loaded)
                setStatus(loaded.status)
            } catch {
                // The API is down. The banner keeps what it showed.
            }
        }
    }, [])

    // On page load, show the run saved in this tab, if there is one. A run that is still going
    // is followed again.
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
                if (saved.status === 'running') void follow(threadId)
            })
            .catch((err) => {
                if (cancelled) return
                if (err instanceof ApiError && err.code === 'not_found') {
                    clearThreadId() // the API has no run with this id
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
            streamAbort.current?.abort()
        }
    }, [follow])

    // Load the saved runs whenever the form is on screen: on page load and after "New run".
    const formShown = !restoring && run === null
    useEffect(() => {
        if (!formShown) return
        let cancelled = false
        listRuns()
            .then((runs) => {
                if (!cancelled) setHistory(runs)
            })
            .catch(() => {
                // The list is only a shortcut, so the form works without it.
            })
        return () => {
            cancelled = true
        }
    }, [formShown])

    async function openRun(threadId: string) {
        setError(null)
        try {
            const saved = await getRun(threadId)
            saveThreadId(threadId)
            setRun(saved)
            setStatus(saved.status)
            setLines([])
            setLivePlan(null)
            setNode(null)
            if (saved.status === 'running') void follow(threadId)
        } catch (err) {
            setError(errorText(err))
        }
    }

    async function submit(brief: Brief) {
        setError(null)
        setLines([])
        setLivePlan(null)
        setNode(null)
        setStatus(null)
        setStarting(true)
        try {
            const { thread_id } = await startRun(brief)
            saveThreadId(thread_id)
            setRun(await getRun(thread_id))
            setStarting(false)
            await follow(thread_id)
        } catch (err) {
            setError(errorText(err))
        } finally {
            setStarting(false)
        }
    }

    /** Run the failed tasks again. The ones that finished are not repeated. */
    async function retry() {
        if (!run) return
        setError(null)
        try {
            await retryRun(run.thread_id)
            await follow(run.thread_id)
        } catch (err) {
            setError(errorText(err))
        }
    }

    function newRun() {
        streamAbort.current?.abort()
        clearThreadId()
        setRun(null)
        setError(null)
        setLines([])
        setLivePlan(null)
        setNode(null)
        setStatus(null)
    }

    const plan = livePlan ?? parsePlan(run?.state?.plan)

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
                <>
                    <BriefForm onSubmit={submit} busy={starting} />
                    <RunHistory runs={history} onOpen={openRun} />
                </>
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
            {status === 'failed' && run && (
                <button
                    type='button'
                    onClick={retry}
                    className='self-start rounded-md border border-zinc-300 px-4 py-2 text-sm font-medium hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-900'
                >
                    Retry failed tasks
                </button>
            )}
            {plan && (
                <TaskList
                    plan={plan}
                    states={parseTaskStates(run?.state)}
                    threadId={run?.thread_id}
                />
            )}
            {run && (
                <pre className='overflow-x-auto rounded-md bg-zinc-100 p-4 text-xs dark:bg-zinc-900'>
                    {JSON.stringify(run, null, 2)}
                </pre>
            )}
        </div>
    )
}
