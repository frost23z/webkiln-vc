'use client'

import {
    ApiError,
    getChat,
    getRun,
    listRuns,
    listVersions,
    restoreVersion,
    resumeRun,
    retryRun,
    sendMessage,
    startRun,
    streamRun,
    type Brief,
    type ChatEntry,
    type ReviewDecision,
    type RunDetail,
    type RunStatus,
    type RunSummary,
    type VersionInfo,
} from '@/lib/api'
import { parseAnalysis, parsePlan, parseTaskStates, planFromEvent, type Plan } from '@/lib/plan'
import { describeEvent, finishedNodes } from '@/lib/progress'
import { parseReview } from '@/lib/review'
import { parseSite } from '@/lib/site'
import { bannerFor, doneStatus } from '@/lib/status'
import { clearThreadId, readThreadId, saveThreadId } from '@/lib/thread'
import { useCallback, useEffect, useRef, useState } from 'react'
import { AnalysisCard } from './analysis-card'
import { ApprovalCard } from './approval-card'
import { BriefForm } from './brief-form'
import { ChatPanel } from './chat-panel'
import { ProgressLines } from './progress-lines'
import { RunHistory } from './run-history'
import { SitePanel } from './site-panel'
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
    // True while an answer to the brand review is on its way to the API.
    const [answering, setAnswering] = useState(false)
    // The chat about the finished site and its saved versions, as the API has them.
    const [chat, setChat] = useState<ChatEntry[]>([])
    const [versions, setVersions] = useState<VersionInfo[]>([])
    // What the user just sent, shown in the chat until the saved chat has it.
    const [pending, setPending] = useState<string | null>(null)
    // An older version the user is looking at. `null` follows the newest one.
    const [pinned, setPinned] = useState<number | null>(null)
    const [page, setPage] = useState('index.html')

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

    const threadId = run?.thread_id
    const site = parseSite(run?.state)
    const latest = site?.version ?? null

    /** Load the chat and the versions of the site again. A failure leaves what is shown. */
    const reloadSite = useCallback(async (id: string) => {
        try {
            const [saved, listed] = await Promise.all([getChat(id), listVersions(id)])
            setChat(saved)
            setVersions(listed)
        } catch {
            // The next change loads them again.
        }
    }, [])

    // Once the site is finished, and again after every change to it (the run is `completed`
    // between messages), load its chat and versions.
    useEffect(() => {
        if (!threadId || latest === null || status !== 'completed') return
        let cancelled = false
        Promise.all([getChat(threadId), listVersions(threadId)])
            .then(([saved, listed]) => {
                if (cancelled) return
                setChat(saved)
                setVersions(listed)
            })
            .catch(() => {
                // The next change loads them again.
            })
        return () => {
            cancelled = true
        }
    }, [threadId, latest, status])

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
            setChat([])
            setVersions([])
            setPinned(null)
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

    /**
     * Answer the brand review. The run carries on in the API from where it paused, so the page
     * follows it again. The key is new for every click, and a repeat of one is harmless.
     */
    async function decide(decision: ReviewDecision) {
        if (!run) return
        setError(null)
        setAnswering(true)
        try {
            await resumeRun(run.thread_id, decision, crypto.randomUUID())
            await follow(run.thread_id)
        } catch (err) {
            setError(errorText(err))
        } finally {
            setAnswering(false)
        }
    }

    /**
     * Send a chat message about the site. The API refuses it when the site is being changed or
     * has a newer version than the one on screen, and says why. Otherwise the run handles it in
     * the background, and the page follows it like any run: the preview moves to the new version.
     */
    async function sendEdit(text: string) {
        if (!run || latest === null) return
        if (pinned !== null && pinned !== latest) {
            // Changes always go on from the newest version, so say so instead of sending a
            // message that the API would refuse as out of date.
            setError(
                `You are looking at version ${pinned}. Go back to version ${latest}, or restore ` +
                    `version ${pinned}, before asking for a change.`
            )
            return
        }
        setError(null)
        setPending(text)
        try {
            await sendMessage(run.thread_id, text, latest)
            setPinned(null)
            await follow(run.thread_id)
            await reloadSite(run.thread_id)
        } catch (err) {
            setError(errorText(err))
        } finally {
            setPending(null)
        }
    }

    /** Make an older version the newest one again. History stays, so this can be undone. */
    async function restore(version: number) {
        if (!run) return
        setError(null)
        try {
            await restoreVersion(run.thread_id, version)
            setPinned(null)
            setRun(await getRun(run.thread_id))
            await reloadSite(run.thread_id)
        } catch (err) {
            setError(errorText(err))
        }
    }

    function newRun() {
        streamAbort.current?.abort()
        clearThreadId()
        setRun(null)
        setChat([])
        setVersions([])
        setPinned(null)
        setPending(null)
        setError(null)
        setLines([])
        setLivePlan(null)
        setNode(null)
        setStatus(null)
    }

    const plan = livePlan ?? parsePlan(run?.state?.plan)
    const analysis = parseAnalysis(run?.state)
    const review = status === 'waiting' ? parseReview(run?.review) : null

    if (restoring) {
        return (
            <p role='status' className='text-sm text-zinc-600 dark:text-zinc-400'>
                Loading the saved run…
            </p>
        )
    }

    const viewing = pinned ?? latest

    return (
        <div className={`flex w-full flex-col gap-8 ${site ? '' : 'max-w-2xl'}`}>
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
            {run && site && viewing !== null && (
                <div className='grid gap-6 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]'>
                    <ChatPanel
                        messages={chat}
                        pending={pending}
                        busy={status === 'running'}
                        onSend={sendEdit}
                    />
                    <SitePanel
                        threadId={run.thread_id}
                        version={viewing}
                        latest={latest ?? viewing}
                        pages={site.pages}
                        page={page}
                        versions={versions}
                        busy={status === 'running'}
                        onPage={setPage}
                        onVersion={setPinned}
                        onRestore={restore}
                    />
                </div>
            )}
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
            {review && (
                <ApprovalCard
                    key={review.version}
                    review={review}
                    busy={answering}
                    onDecide={decide}
                />
            )}
            {analysis && <AnalysisCard analysis={analysis} />}
            {plan && (
                <TaskList
                    plan={plan}
                    states={parseTaskStates(run?.state)}
                    threadId={run?.thread_id}
                />
            )}
        </div>
    )
}
