export type SseEvent = { event: string; data: unknown }

/**
 * Reads a server-sent event body. Events end at a blank line; each has an
 * `event:` name and one or more `data:` lines holding JSON.
 */
export async function* readEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<SseEvent> {
    const reader = body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    try {
        while (true) {
            const { done, value } = await reader.read()
            if (done) break
            buffer += decoder.decode(value, { stream: true }).replaceAll('\r\n', '\n')
            let end = buffer.indexOf('\n\n')
            while (end !== -1) {
                const event = parseBlock(buffer.slice(0, end))
                buffer = buffer.slice(end + 2)
                if (event) yield event
                end = buffer.indexOf('\n\n')
            }
        }
    } finally {
        await reader.cancel().catch(() => undefined)
    }
}

function parseBlock(block: string): SseEvent | null {
    let event = 'message'
    const data: string[] = []
    for (const line of block.split('\n')) {
        if (line === '' || line.startsWith(':')) continue // blank or comment
        const colon = line.indexOf(':')
        const field = colon === -1 ? line : line.slice(0, colon)
        const value = colon === -1 ? '' : line.slice(colon + 1).replace(/^ /, '')
        if (field === 'event') event = value
        else if (field === 'data') data.push(value)
    }
    if (data.length === 0) return null
    const text = data.join('\n')
    try {
        return { event, data: JSON.parse(text) }
    } catch {
        return { event, data: text }
    }
}
