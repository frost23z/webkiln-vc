import type { components } from '../generated/schema'

type Schemas = components['schemas']

export type Brief = Schemas['Input']
export type RunStatus = Schemas['RunResponse']['status']
export type RunResponse = Schemas['RunResponse']
export type RunDetail = Schemas['RunDetail']
export type RunSummary = Schemas['RunSummary']
export type ErrorResponse = Schemas['ErrorResponse']
export type ErrorDetail = Schemas['ErrorDetail']
export type TaskReport = Schemas['TaskReport']
export type ReviewDecision = Schemas['ReviewDecision']
export type DirectEdits = Schemas['DirectEdits']
export type ChatEntry = Schemas['ChatEntry']
export type VersionInfo = Schemas['VersionInfo']
export type RestoreResponse = Schemas['RestoreResponse']
