import type { Banner } from '@/lib/status'

const toneClass: Record<Banner['tone'], string> = {
    working:
        'border-blue-300 bg-blue-50 text-blue-900 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-100',
    waiting:
        'border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100',
    done: 'border-green-300 bg-green-50 text-green-900 dark:border-green-800 dark:bg-green-950 dark:text-green-100',
    failed: 'border-red-300 bg-red-50 text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100',
}

export function StatusBanner({ banner }: { banner: Banner }) {
    return (
        <div
            role='status'
            className={`rounded-md border px-4 py-2 text-sm font-medium ${toneClass[banner.tone]}`}
        >
            {banner.label}
        </div>
    )
}
