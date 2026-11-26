'use client'

import { getTaskReport, type TaskReport } from '@/lib/api'
import { parseWireframe } from '@/lib/wireframe'
import { useEffect, useState } from 'react'
import { WireframeOutline } from './wireframe-outline'

/** A finished worker's result, read from the saved output: its summary and the full data. */
export function TaskReportView({ threadId, taskId }: { threadId: string; taskId: string }) {
    const [report, setReport] = useState<TaskReport | null>(null)
    const [failed, setFailed] = useState(false)

    useEffect(() => {
        let cancelled = false
        getTaskReport(threadId, taskId)
            .then((loaded) => {
                if (!cancelled) setReport(loaded)
            })
            .catch(() => {
                if (!cancelled) setFailed(true)
            })
        return () => {
            cancelled = true
        }
    }, [threadId, taskId])

    if (failed) return <p className='text-xs text-red-700 dark:text-red-300'>Could not load it.</p>
    if (!report) return <p className='text-xs text-zinc-600 dark:text-zinc-400'>Loading…</p>
    const wireframe = parseWireframe(report.output)
    return (
        <div className='flex flex-col gap-2 rounded-md bg-zinc-100 p-3 dark:bg-zinc-900'>
            <p>{report.summary}</p>
            {wireframe && <WireframeOutline wireframe={wireframe} />}
            <details>
                <summary className='cursor-pointer text-xs font-medium'>Full output</summary>
                <pre className='mt-2 overflow-x-auto text-xs'>
                    {JSON.stringify(report.output, null, 2)}
                </pre>
            </details>
        </div>
    )
}
