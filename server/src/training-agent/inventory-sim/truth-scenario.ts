/**
 * The `simulation_truth` controller scenario, served by the inventory
 * tenants. It reads what the caller's simulated market actually did, per
 * package and per cell, up to the last processed day:
 *
 *   { "scenario": "simulation_truth", "params": { "media_buy_ids": ["mb_0001"], "include": ["cell_daily_actuals"], "start_date": "2026-12-01", "end_date": "2026-12-31" } }
 *
 * It is sandbox-only and read-only: the market is opened caught up to the
 * clock and never saved, and the clock is untouched.
 */
import { simulationTruth, type SeasonSeed } from './engine/index.js';
import { openMarket } from './market-store.js';

export const SIMULATION_TRUTH_SCENARIO = 'simulation_truth';

/** Most packages one response carries. */
export const MAX_TRUTH_PACKAGES = 200;
/** Longest window of daily actuals, in days. */
export const MAX_TRUTH_WINDOW_DAYS = 92;

type Json = Record<string, unknown>;

function failure(error: string, detail: string): Json {
  return { success: false, error, error_detail: detail };
}

function isCalendarDate(value: string): boolean {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return false;
  return new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]))).toISOString().slice(0, 10) === value;
}

function stringList(value: unknown): string[] | undefined {
  return Array.isArray(value) && value.every(item => typeof item === 'string') ? value as string[] : undefined;
}

export async function handleSimulationTruth(
  params: Json,
  market: { tenantId: string; seed: SeasonSeed },
  principal: string | undefined,
): Promise<Json> {
  let mediaBuyIds: string[] | undefined;
  if (params.media_buy_ids !== undefined) {
    mediaBuyIds = stringList(params.media_buy_ids);
    if (!mediaBuyIds) return failure('INVALID_PARAMS', 'params.media_buy_ids must be an array of media buy ids');
  }
  let include: string[] = [];
  if (params.include !== undefined) {
    const listed = stringList(params.include);
    if (!listed || listed.some(item => item !== 'cell_daily_actuals')) {
      return failure('INVALID_PARAMS', 'params.include may only list cell_daily_actuals');
    }
    include = listed;
  }
  for (const key of ['start_date', 'end_date']) {
    if (params[key] !== undefined && (typeof params[key] !== 'string' || !isCalendarDate(params[key] as string))) {
      return failure('INVALID_PARAMS', `params.${key} must be a YYYY-MM-DD date`);
    }
  }
  if (typeof params.start_date === 'string' && typeof params.end_date === 'string' && params.start_date > params.end_date) {
    return failure('INVALID_PARAMS', 'params.start_date must not be after params.end_date');
  }

  const open = await openMarket(market.tenantId, market.seed, principal);
  const { calendar } = open.sim.traffic;
  const processed = open.sim.state.processed_through_day;

  const owner = new Map<string, { media_buy_id: string; account_id: string }>();
  for (const buy of open.doc.media_buys) {
    for (const pkg of buy.packages) owner.set(pkg.package_id, { media_buy_id: buy.media_buy_id, account_id: buy.account_id });
  }

  let bookingIds: string[];
  if (mediaBuyIds) {
    const known = new Set(open.doc.media_buys.map(buy => buy.media_buy_id));
    const missing = mediaBuyIds.find(id => !known.has(id));
    if (missing !== undefined) return failure('NOT_FOUND', `No media buy ${missing} in this market`);
    const wanted = new Set(mediaBuyIds);
    bookingIds = [...owner].filter(([, who]) => wanted.has(who.media_buy_id)).map(([id]) => id);
  } else {
    bookingIds = [...owner.keys()];
  }
  if (bookingIds.length > MAX_TRUTH_PACKAGES) {
    return failure('INVALID_PARAMS', `${bookingIds.length} packages exceed the limit of ${MAX_TRUTH_PACKAGES}; pass media_buy_ids to narrow the response`);
  }

  let dailyActuals: { startDay: number; endDay: number } | undefined;
  if (include.includes('cell_daily_actuals')) {
    const ceiling = Math.min(processed, calendar.days);
    const endDay = Math.max(0, Math.min(
      typeof params.end_date === 'string' ? calendar.dayOfDate(params.end_date) + 1 : ceiling,
      ceiling,
    ));
    const startDay = Math.max(0, Math.min(
      typeof params.start_date === 'string'
        ? calendar.dayOfDate(params.start_date)
        : Math.max(0, (typeof params.end_date === 'string' ? endDay : processed) - MAX_TRUTH_WINDOW_DAYS),
      endDay,
    ));
    if (endDay - startDay > MAX_TRUTH_WINDOW_DAYS) {
      return failure('INVALID_PARAMS', `cell_daily_actuals covers at most ${MAX_TRUTH_WINDOW_DAYS} days; narrow start_date and end_date`);
    }
    dailyActuals = { startDay, endDay };
  }

  const truth = simulationTruth(open.sim, { bookingIds, ...(dailyActuals && { dailyActuals }) });
  const packages = truth.bookings.flatMap(({ booking_id, ...rest }) => {
    const who = owner.get(booking_id);
    return who ? [{ package_id: booking_id, media_buy_id: who.media_buy_id, account_id: who.account_id, ...rest }] : [];
  });
  return {
    success: true,
    seed_id: truth.seed_id,
    simulated_now: open.nowIso,
    ...(truth.processed_through_day > 0 && { processed_through_date: calendar.dateOf(truth.processed_through_day - 1) }),
    world: truth.world,
    packages,
    ...(truth.dailyActuals && {
      cell_index: truth.dailyActuals.cell_index,
      cell_daily_actuals: truth.dailyActuals.cell_daily_actuals,
    }),
  };
}
