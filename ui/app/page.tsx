import { Workspace } from '@/components/workspace'

export default function Home() {
    return (
        <main className='mx-auto flex w-full max-w-2xl flex-1 flex-col gap-8 px-6 py-16'>
            <header className='flex flex-col gap-2'>
                <h1 className='text-3xl font-semibold tracking-tight'>Webkiln</h1>
                <p className='text-zinc-600 dark:text-zinc-400'>
                    Describe the brand and the niche. The agents plan, research and build the site.
                </p>
            </header>
            <Workspace />
        </main>
    )
}
