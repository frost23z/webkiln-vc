'use client'

import type { ChatEntry } from '@/lib/api'
import { useEffect, useRef, useState } from 'react'

type Props = {
    messages: ChatEntry[]
    /** What the user just sent, shown until the saved chat has it. */
    pending: string | null
    /** True while the run handles a message: the site is locked until it finishes. */
    busy: boolean
    onSend: (text: string) => void
}

const bubble = {
    user: 'self-end bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900',
    assistant: 'self-start bg-zinc-100 dark:bg-zinc-900',
}

export function ChatPanel({ messages, pending, busy, onSend }: Props) {
    const [text, setText] = useState('')
    const end = useRef<HTMLDivElement>(null)
    const shown = pending ? [...messages, { role: 'user' as const, content: pending }] : messages

    useEffect(() => {
        end.current?.scrollIntoView({ block: 'nearest' })
    }, [shown.length])

    function submit(event: React.SubmitEvent<HTMLFormElement>) {
        event.preventDefault()
        const message = text.trim()
        if (!message || busy) return
        setText('')
        onSend(message)
    }

    return (
        <section aria-label='Chat' className='flex min-h-96 flex-col gap-3'>
            <h2 className='text-sm font-medium text-zinc-600 dark:text-zinc-400'>Edit the site</h2>
            <div
                role='log'
                aria-live='polite'
                className='flex max-h-[32rem] flex-1 flex-col gap-2 overflow-y-auto rounded-md border border-zinc-200 p-3 dark:border-zinc-800'
            >
                {shown.length === 0 && (
                    <p className='text-sm text-zinc-600 dark:text-zinc-400'>
                        Ask for a change, such as “make the homepage heading warmer”, “use a darker
                        green” or “add a page about us”.
                    </p>
                )}
                {shown.map((message, index) => (
                    <p
                        key={index}
                        className={`max-w-[85%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${bubble[message.role]}`}
                    >
                        {message.content}
                    </p>
                ))}
                {busy && (
                    <p className='self-start text-sm text-zinc-600 dark:text-zinc-400'>
                        Working on it…
                    </p>
                )}
                <div ref={end} />
            </div>
            <form onSubmit={submit} className='flex gap-2'>
                <input
                    value={text}
                    onChange={(event) => setText(event.target.value)}
                    maxLength={2000}
                    disabled={busy}
                    placeholder={busy ? 'Wait for the change to finish' : 'What should change?'}
                    aria-label='Message'
                    className='min-w-0 flex-1 rounded-md border border-zinc-300 bg-transparent px-3 py-2 text-sm disabled:opacity-60 dark:border-zinc-700'
                />
                <button
                    type='submit'
                    disabled={busy || text.trim() === ''}
                    className='rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900'
                >
                    Send
                </button>
            </form>
        </section>
    )
}
