import type { RunSummary } from '@/lib/api'

const statusLabel: Record<RunSummary['status'], string> = {
    running: 'Running',
    completed: 'Finished',
    failed: 'Failed',
}

type Props = {
    runs: RunSummary[]
    onOpen: (threadId: string) => void
}

export function RunHistory({ runs, onOpen }: Props) {
    if (runs.length === 0) return null
    return (
        <section className='flex flex-col gap-2'>
            <h2 className='text-sm font-medium text-zinc-600 dark:text-zinc-400'>Earlier runs</h2>
            <ul className='flex flex-col divide-y divide-zinc-200 rounded-md border border-zinc-200 dark:divide-zinc-800 dark:border-zinc-800'>
                {runs.map((run) => (
                    <li key={run.thread_id}>
                        <button
                            type='button'
                            onClick={() => onOpen(run.thread_id)}
                            className='flex w-full items-center justify-between gap-4 px-4 py-3 text-left text-sm hover:bg-zinc-100 dark:hover:bg-zinc-900'
                        >
                            <span className='font-medium'>{run.brief.brand_name}</span>
                            <span className='text-zinc-600 dark:text-zinc-400'>
                                {statusLabel[run.status]} ·{' '}
                                {new Date(run.updated_at).toLocaleString()}
                            </span>
                        </button>
                    </li>
                ))}
            </ul>
        </section>
    )
}
