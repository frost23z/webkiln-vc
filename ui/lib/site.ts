import { isRecord } from './progress'

export type Site = {
    /** The newest saved version: an edit's, or else the build's version 1. */
    version: number
    /** The pages of the site as file paths, the home page first. */
    pages: string[]
}

function pathsOf(value: unknown): string[] | null {
    return Array.isArray(value) && value.every((v) => typeof v === 'string')
        ? (value as string[])
        : null
}

function pagesOf(paths: string[]): string[] {
    const pages = paths.filter((path) => path.endsWith('.html') && !path.includes('/'))
    return pages.sort((a, b) => Number(b === 'index.html') - Number(a === 'index.html'))
}

/**
 * The site in a run's graph state, or null before the build has finished. An edit saves a newer
 * head in `site`, and until then the build in `outputs.generate` is version 1. The API sends
 * paths and never the files, so this reads only the paths.
 */
export function parseSite(state: unknown): Site | null {
    if (!isRecord(state)) return null
    const { site, outputs } = state
    if (isRecord(site) && typeof site.version === 'number') {
        const paths = pathsOf(site.file_paths)
        if (paths) return { version: site.version, pages: pagesOf(paths) }
    }
    const build = isRecord(outputs) ? outputs.generate : null
    const paths = isRecord(build) ? pathsOf(build.file_paths) : null
    return paths ? { version: 1, pages: pagesOf(paths) } : null
}

/** The label of a page tab: "index.html" is "Home" and "rooms.html" is "Rooms". */
export function pageLabel(path: string): string {
    const name = path.replace(/\.html$/, '')
    if (name === 'index') return 'Home'
    return name.replaceAll('-', ' ').replace(/^./, (c) => c.toUpperCase())
}
