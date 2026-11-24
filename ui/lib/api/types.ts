import type { components } from '../generated/schema'

type Schemas = components['schemas']

export type Brief = Schemas['Input']
export type RunStatus = Schemas['RunResponse']['status']
export type RunResponse = Schemas['RunResponse']
export type RunDetail = Schemas['RunDetail']
export type ErrorResponse = Schemas['ErrorResponse']
export type ErrorDetail = Schemas['ErrorDetail']
