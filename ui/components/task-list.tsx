'use client'

import type { Plan, TaskState } from '@/lib/plan'
import { useState } from 'react'
import { TaskReportView } from './task-report'

const STATUS_LABEL: Record<string, string> = {
    pending: 'Waiting',
    running: 'Running',
    done: 'Done',
    failed: 'Failed',
    needs_rework: 'Needs rework',
    skipped: 'Skipped',
    needs_user: 'Needs you',
}

const STATUS_CLASS: Record<string, string> = {
    running: 'bg-blue-100 text-blue-900 dark:bg-blue-950 dark:text-blue-100',
    done: 'bg-green-100 text-green-900 dark:bg-green-950 dark:text-green-100',
    failed: 'bg-red-100 text-red-900 dark:bg-red-950 dark:text-red-100',
}

type Props = {
    plan: Plan
    /** Live state by task id. A task with no entry has not started. */
    states: Record<string, TaskState>
    /** Needed to open a report. Without it the reports are not offered. */
    threadId?: string
}

export function TaskList({ plan, states, threadId }: Props) {
    const [open, setOpen] = useState<string | null>(null)
    return (
        <section aria-label='Plan' className='flex flex-col gap-3'>
            <p className='text-sm text-zinc-600 dark:text-zinc-400'>{plan.rationale}</p>
            <ol className='flex flex-col gap-2'>
                {plan.tasks.map((task) => {
                    const status = states[task.id]?.status ?? 'pending'
                    const error = states[task.id]?.error
                    return (
                        <li
                            key={task.id}
                            className='flex flex-col gap-1 rounded-md border border-zinc-200 p-3 text-sm dark:border-zinc-800'
                        >
                            <span className='flex items-center gap-2 font-medium'>
                                <span>
                                    {task.id} <span className='font-normal'>({task.agent})</span>
                                </span>
                                <span
                                    className={`rounded px-2 py-0.5 text-xs font-medium ${STATUS_CLASS[status] ?? 'bg-zinc-100 text-zinc-800 dark:bg-zinc-900 dark:text-zinc-200'}`}
                                >
                                    {STATUS_LABEL[status] ?? status}
                                </span>
                            </span>
                            {task.instruction && <p>{task.instruction}</p>}
                            {task.depends_on.length > 0 && (
                                <p className='text-xs text-zinc-600 dark:text-zinc-400'>
                                    After {task.depends_on.join(', ')}
                                </p>
                            )}
                            {error && (
                                <p className='text-xs text-red-700 dark:text-red-300'>{error}</p>
                            )}
                            {status === 'done' && threadId && (
                                <>
                                    <button
                                        type='button'
                                        onClick={() => setOpen(open === task.id ? null : task.id)}
                                        aria-expanded={open === task.id}
                                        className='self-start text-xs font-medium underline'
                                    >
                                        {open === task.id ? 'Hide report' : 'Show report'}
                                    </button>
                                    {open === task.id && (
                                        <TaskReportView threadId={threadId} taskId={task.id} />
                                    )}
                                </>
                            )}
                        </li>
                    )
                })}
            </ol>
        </section>
    )
}
