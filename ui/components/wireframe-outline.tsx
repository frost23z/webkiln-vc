import { sitemap, type SitemapNode, type Wireframe } from '@/lib/wireframe'

function PageNode({ node }: { node: SitemapNode }) {
    const { page } = node
    return (
        <li className='flex flex-col gap-1'>
            <span className='font-medium'>
                {page.name} <span className='font-normal'>({page.slug})</span>
                {node.inMenu && (
                    <span className='ml-2 rounded bg-zinc-200 px-1.5 py-0.5 text-xs font-medium dark:bg-zinc-800'>
                        Menu
                    </span>
                )}
            </span>
            <ul className='list-disc pl-5 text-xs text-zinc-700 dark:text-zinc-300'>
                {page.sections.map((section) => (
                    <li key={section.id}>
                        {section.type}: {section.title}
                    </li>
                ))}
            </ul>
            {node.children.length > 0 && (
                <ul className='ml-3 flex flex-col gap-2 border-l border-zinc-300 pl-3 dark:border-zinc-700'>
                    {node.children.map((child) => (
                        <PageNode key={child.page.slug} node={child} />
                    ))}
                </ul>
            )}
        </li>
    )
}

/** The sitemap of a wireframe with the sections of every page, and what the site must do. */
export function WireframeOutline({ wireframe }: { wireframe: Wireframe }) {
    return (
        <div className='flex flex-col gap-2 text-sm'>
            <ul aria-label='Sitemap' className='flex flex-col gap-2'>
                {sitemap(wireframe).map((node) => (
                    <PageNode key={node.page.slug} node={node} />
                ))}
            </ul>
            {wireframe.functional_requirements.length > 0 && (
                <div>
                    <p className='font-medium'>Functional requirements</p>
                    <ul className='list-disc pl-5 text-xs'>
                        {wireframe.functional_requirements.map((need) => (
                            <li key={`${need.kind}-${need.page}`}>
                                {need.kind} on {need.page}: {need.description}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    )
}
