import { z } from 'zod'

const sectionSchema = z.object({
    id: z.string(),
    type: z.string(),
    title: z.string(),
})

const pageSchema = z.object({
    slug: z.string(),
    name: z.string(),
    parent: z.string().nullish(),
    seo: z.object({ title: z.string() }),
    sections: z.array(sectionSchema),
})

const wireframeSchema = z.object({
    pages: z.array(pageSchema).min(1),
    navigation: z.array(z.string()).default([]),
    functional_requirements: z
        .array(z.object({ kind: z.string(), page: z.string(), description: z.string() }))
        .default([]),
})

export type Wireframe = z.infer<typeof wireframeSchema>
type Page = Wireframe['pages'][number]

/** A page and the pages below it, in wireframe order. */
export type SitemapNode = {
    page: Page
    inMenu: boolean
    children: SitemapNode[]
}

/** The wireframe in a saved output, from untrusted data. Null when it is not a wireframe. */
export function parseWireframe(output: unknown): Wireframe | null {
    const parsed = wireframeSchema.safeParse(output)
    return parsed.success ? parsed.data : null
}

/**
 * The sitemap as a tree. A page whose parent does not exist is shown at the top level, so no page
 * is ever hidden, and a page is shown once even when the parents form a loop.
 */
export function sitemap(wireframe: Wireframe): SitemapNode[] {
    const slugs = new Set(wireframe.pages.map((page) => page.slug))
    const menu = new Set(wireframe.navigation)
    const shown = new Set<string>()

    const below = (parent: string | null): SitemapNode[] => {
        const nodes: SitemapNode[] = []
        for (const page of wireframe.pages) {
            const root = page.parent == null || !slugs.has(page.parent)
            const belongs = parent === null ? root : page.parent === parent
            if (!belongs || shown.has(page.slug)) continue
            shown.add(page.slug)
            nodes.push({ page, inMenu: menu.has(page.slug), children: below(page.slug) })
        }
        return nodes
    }

    const tree = below(null)
    // Pages in a loop have no top-level entry. Show each as its own top-level page.
    for (const page of wireframe.pages) {
        if (shown.has(page.slug)) continue
        shown.add(page.slug)
        tree.push({ page, inMenu: menu.has(page.slug), children: below(page.slug) })
    }
    return tree
}
