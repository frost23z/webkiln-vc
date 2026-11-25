import type { BriefAnalysis } from '@/lib/plan'

/** What the Manager understood from the brief, so the user can see how the workers read it. */
export function AnalysisCard({ analysis }: { analysis: BriefAnalysis }) {
    return (
        <section
            aria-label='Brief analysis'
            className='flex flex-col gap-2 rounded-md border border-zinc-200 p-4 text-sm dark:border-zinc-800'
        >
            <h2 className='font-medium'>How the brief was read</h2>
            <p>
                <span className='font-medium'>Purpose:</span> {analysis.purpose}
            </p>
            <p>
                <span className='font-medium'>Business model:</span> {analysis.business_model}
            </p>
            <ul className='list-disc pl-5'>
                {analysis.expected_functionality.map((item) => (
                    <li key={item}>{item}</li>
                ))}
            </ul>
        </section>
    )
}
