export function ProgressLines({ lines }: { lines: string[] }) {
    if (lines.length === 0) return null
    return (
        <ol
            aria-live='polite'
            className='flex flex-col gap-1 rounded-md border border-zinc-200 p-4 text-sm dark:border-zinc-800'
        >
            {lines.map((line, index) => (
                <li key={index}>{line}</li>
            ))}
        </ol>
    )
}
