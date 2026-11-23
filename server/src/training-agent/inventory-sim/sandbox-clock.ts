/**
 * The `sandbox_clock` controller scenario (#9025 part A), served by the
 * inventory tenants:
 *
 *   { "scenario": "sandbox_clock", "params": { "operation": "get" } }
 *   { "scenario": "sandbox_clock", "params": { "operation": "start", "to": "2026-12-06T00:00:00Z" } }
 *   { "scenario": "sandbox_clock", "params": { "operation": "advance_time", "to": "2027-01-15T00:00:00Z" } }
 *   { "scenario": "sandbox_clock", "params": { "operation": "advance_time", "by": "P30D" } }
 *
 * It moves the caller's one sandbox clock (`simulated_now` in
 * `account-sandbox.ts`), the same clock `reporting_subscription_probe`
 * `advance_time` moves. `start` pins an unused sandbox at any instant, past or
 * future; after that the clock only moves forward, and `reset_state` unpins it
 * and makes `start` possible again.
 * Advancing runs the day steps for each day crossed in the market of the
 * tenant it is called on; the caller's other markets run the same days on
 * their next read, with the same results. Writing only that tenant's market
 * keeps the write under the router's lock for that tenant.
 */
import { loadSandbox, sandboxNowIso, sandboxNowMs, sandboxWallClockMs, type LoadedSandbox } from '../account-sandbox.js';
import { runSandboxClockAdvanceHooks } from '../subscription-probe.js';
import type { TrainingContext } from '../types.js';
import { DAY_MS, type SeasonSeed } from './engine/index.js';
import { advanceMarket } from './market-store.js';

export const SANDBOX_CLOCK_SCENARIO = 'sandbox_clock';

type Json = Record<string, unknown>;

/** A full date, `T`, a time with optional fractional seconds, and a `Z` or
 * ±hh:mm offset. */
const RFC_3339_RE = /^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})$/;
const DURATION_RE = /^P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$/;
/** Ten years: far past any season, short of overflowing a Date. */
const MAX_ADVANCE_MS = 10 * 366 * 24 * 60 * 60 * 1000;

/** An ISO 8601 duration in weeks, days, hours, minutes and seconds, in
 * milliseconds. Months and years are refused: their length depends on the
 * date. */
export function parseDuration(value: string): number | undefined {
  const match = DURATION_RE.exec(value);
  if (!match || value === 'P' || value.endsWith('T')) return undefined;
  const [weeks, days, hours, minutes, seconds] = match.slice(1).map(part => (part ? Number(part) : 0));
  return ((((weeks * 7 + days) * 24 + hours) * 60 + minutes) * 60 + seconds) * 1000;
}

function failure(error: string, detail: string): Json {
  return { success: false, error, error_detail: detail };
}

/** Claims the caller's sandbox for a start: returns why it is not unused, or
 * undefined after removing its untouched grants and empty markets, which are
 * rebuilt at the started clock. */
export type UnusedSandboxClaim = (sandbox: LoadedSandbox) => Promise<string | undefined>;

async function startClock(
  params: Json,
  ctx: TrainingContext,
  sandbox: LoadedSandbox,
  nowMs: number,
  claimUnusedSandbox: UnusedSandboxClaim,
): Promise<Json> {
  if (params.by !== undefined) return failure('INVALID_PARAMS', 'start takes params.to (an RFC 3339 date-time), not params.by');
  const target = typeof params.to === 'string' && RFC_3339_RE.test(params.to) ? Date.parse(params.to) : Number.NaN;
  if (!Number.isFinite(target)) return failure('INVALID_PARAMS', 'start requires params.to, an RFC 3339 date-time');
  if (Math.abs(target - sandboxWallClockMs()) > MAX_ADVANCE_MS) return failure('INVALID_PARAMS', 'start sets the clock at most ten years from now');
  const reason = await claimUnusedSandbox(sandbox);
  if (reason) {
    return failure('INVALID_TRANSITION', `start sets the clock of an unused sandbox only: ${reason}. Call reset_state on an inventory tenant, then start.`);
  }
  sandbox.doc.simulated_now = new Date(target).toISOString();
  sandbox.save();
  if (target > nowMs) await runSandboxClockAdvanceHooks(ctx, target);
  return {
    success: true,
    simulated: { simulated_now: sandbox.doc.simulated_now, previous_simulated_now: new Date(nowMs).toISOString(), pinned: true },
    message: `Sandbox clock started at ${sandbox.doc.simulated_now}`,
  };
}

export async function handleSandboxClock(
  params: Json,
  ctx: TrainingContext,
  market: { tenantId: string; seed: SeasonSeed },
  claimUnusedSandbox: UnusedSandboxClaim,
): Promise<Json> {
  const operation = params.operation;
  const sandbox = await loadSandbox(ctx.principal);
  const nowMs = sandboxNowMs(sandbox.doc);
  if (operation === 'get') {
    return {
      success: true,
      simulated: { simulated_now: sandboxNowIso(sandbox.doc), pinned: sandbox.doc.simulated_now !== undefined },
      message: `simulated_now is ${sandboxNowIso(sandbox.doc)}`,
    };
  }
  if (operation === 'start') return startClock(params, ctx, sandbox, nowMs, claimUnusedSandbox);
  if (operation !== 'advance_time') {
    return failure('INVALID_PARAMS', 'sandbox_clock requires params.operation: get, start or advance_time');
  }
  const hasTo = params.to !== undefined;
  const hasBy = params.by !== undefined;
  if (hasTo === hasBy) return failure('INVALID_PARAMS', 'advance_time takes exactly one of params.to (an RFC 3339 date-time) or params.by (an ISO 8601 duration)');
  let target: number;
  if (hasTo) {
    target = typeof params.to === 'string' ? Date.parse(params.to) : Number.NaN;
    if (!Number.isFinite(target)) return failure('INVALID_PARAMS', 'params.to must be an RFC 3339 date-time');
  } else {
    const duration = typeof params.by === 'string' ? parseDuration(params.by) : undefined;
    if (duration === undefined) return failure('INVALID_PARAMS', 'params.by must be an ISO 8601 duration in weeks, days, hours, minutes or seconds, such as P30D or PT6H');
    target = nowMs + duration;
  }
  if (target <= nowMs) {
    return failure('INVALID_TRANSITION', `The sandbox clock only moves forward; simulated_now is ${new Date(nowMs).toISOString()}`);
  }
  if (target - nowMs > MAX_ADVANCE_MS) return failure('INVALID_PARAMS', 'advance_time moves the clock at most ten years at a time');
  sandbox.doc.simulated_now = new Date(target).toISOString();
  sandbox.save();
  await advanceMarket(market.tenantId, market.seed, ctx.principal);
  const daysCrossed = Math.floor(target / DAY_MS) - Math.floor(nowMs / DAY_MS);
  await runSandboxClockAdvanceHooks(ctx, target);
  return {
    success: true,
    simulated: { simulated_now: sandbox.doc.simulated_now, previous_simulated_now: new Date(nowMs).toISOString(), days_crossed: daysCrossed },
    message: `Sandbox clock advanced to ${sandbox.doc.simulated_now}`,
  };
}
