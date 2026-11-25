import type { Plan } from '@/lib/plan'

export function TaskList({ plan }: { plan: Plan }) {
    return (
        <section aria-label='Plan' className='flex flex-col gap-3'>
            <p className='text-sm text-zinc-600 dark:text-zinc-400'>{plan.rationale}</p>
            <ol className='flex flex-col gap-2'>
                {plan.tasks.map((task) => (
                    <li
                        key={task.id}
                        className='flex flex-col gap-1 rounded-md border border-zinc-200 p-3 text-sm dark:border-zinc-800'
                    >
                        <span className='font-medium'>
                            {task.id} <span className='font-normal'>({task.agent})</span>
                        </span>
                        {task.instruction && <p>{task.instruction}</p>}
                        {task.depends_on.length > 0 && (
                            <p className='text-xs text-zinc-600 dark:text-zinc-400'>
                                After {task.depends_on.join(', ')}
                            </p>
                        )}
                    </li>
                ))}
            </ol>
        </section>
    )
}
