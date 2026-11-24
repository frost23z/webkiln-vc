import type { components } from '../generated/schema'

type Schemas = components['schemas']

export type Brief = Schemas['Input']
export type RunStatus = Schemas['RunResponse']['status']
export type RunResponse = Schemas['RunResponse']
export type RunDetail = Schemas['RunDetail']
