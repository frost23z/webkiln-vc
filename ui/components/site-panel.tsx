'use client'

import { previewUrl, type VersionInfo } from '@/lib/api'
import { pageLabel } from '@/lib/site'

type Props = {
    threadId: string
    /** The version on screen. */
    version: number
    /** The newest version. */
    latest: number
    pages: string[]
    page: string
    versions: VersionInfo[]
    busy: boolean
    onPage: (path: string) => void
    /** `null` goes back to the newest version. */
    onVersion: (version: number | null) => void
    onRestore: (version: number) => void
}

/**
 * The generated site in a frame. The pages are untrusted, so the frame has `sandbox` with no
 * flags: no script, no forms, no same-origin access. The API also serves them with a policy
 * that allows no script and no network.
 */
export function SitePanel(props: Props) {
    const { threadId, version, latest, pages, page, versions, busy } = props
    const known = pages.includes(page) ? page : (pages[0] ?? 'index.html')
    return (
        <section aria-label='Preview' className='flex min-w-0 flex-col gap-3'>
            <div className='flex flex-wrap items-center justify-between gap-2'>
                <h2 className='text-sm font-medium text-zinc-600 dark:text-zinc-400'>Preview</h2>
                <div className='flex items-center gap-2 text-sm'>
                    <label htmlFor='site-version' className='text-zinc-600 dark:text-zinc-400'>
                        Version
                    </label>
                    <select
                        id='site-version'
                        value={version}
                        onChange={(event) => {
                            const picked = Number(event.target.value)
                            props.onVersion(picked === latest ? null : picked)
                        }}
                        className='rounded-md border border-zinc-300 bg-transparent px-2 py-1 dark:border-zinc-700'
                    >
                        {versions.map((v) => (
                            <option key={v.version} value={v.version} className='text-black'>
                                {v.version}: {v.summary}
                            </option>
                        ))}
                        {versions.length === 0 && <option value={version}>{version}</option>}
                    </select>
                    {version !== latest && (
                        <button
                            type='button'
                            disabled={busy}
                            onClick={() => props.onRestore(version)}
                            className='rounded-md border border-zinc-300 px-3 py-1 font-medium hover:bg-zinc-100 disabled:opacity-50 dark:border-zinc-700 dark:hover:bg-zinc-900'
                        >
                            Restore this version
                        </button>
                    )}
                </div>
            </div>
            {pages.length > 1 && (
                <nav aria-label='Pages' className='flex flex-wrap gap-1'>
                    {pages.map((path) => (
                        <button
                            key={path}
                            type='button'
                            aria-current={path === known ? 'page' : undefined}
                            onClick={() => props.onPage(path)}
                            className={`rounded-md px-3 py-1 text-sm ${
                                path === known
                                    ? 'bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900'
                                    : 'hover:bg-zinc-100 dark:hover:bg-zinc-900'
                            }`}
                        >
                            {pageLabel(path)}
                        </button>
                    ))}
                </nav>
            )}
            <iframe
                key={`${threadId}-${version}-${known}`}
                title={`Preview of version ${version}`}
                src={previewUrl(threadId, version, known)}
                sandbox=''
                referrerPolicy='no-referrer'
                className='h-[36rem] w-full rounded-md border border-zinc-200 bg-white dark:border-zinc-800'
            />
        </section>
    )
}
