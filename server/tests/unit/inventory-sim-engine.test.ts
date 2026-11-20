/**
 * #9025 part B: the deterministic inventory delivery engine behind the
 * training agent's inventory tenants. Allocation primitives, traffic
 * generation, the day steps, forecasts, installments and reporting delay.
 */
import { describe, expect, it } from 'vitest';
import {
  ADSERVER_SEASON,
  CONTENT_SEASON,
  EngineError,
  InventorySimulation,
  SeasonTraffic,
  allocateAuction,
  allocateHighWaterMark,
  audienceCurve,
  takeEqualShare,
  standardNormal,
  takeShareOfVoice,
  toAdcpForecast,
  uniform,
  type CellCounts,
  type SeasonSeed,
} from '../../src/training-agent/inventory-sim/engine/index.js';

const at = (iso: string) => Date.parse(iso);
const counts = (entries: Array<[number, number]>): CellCounts => new Map(entries);
const total = (taken: CellCounts | undefined) => [...(taken ?? new Map()).values()].reduce((sum, value) => sum + value, 0);

function expectEngineError(fn: () => unknown, code: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(EngineError);
  expect((caught as EngineError).code).toBe(code);
}

const ZERO_FORECAST_ERROR: SeasonSeed['forecast_error'] = {
  channel_sigma: { display: 0 },
  ar1_phi: 0,
  lead_time_tau_days: 14,
  lead_time_horizon_days: 60,
  log_clip: 0.6931471805599453,
  cell_kappa: 0,
  cell_min_impressions: 100,
};
const NOISY_FORECAST_ERROR: SeasonSeed['forecast_error'] = {
  ...ZERO_FORECAST_ERROR,
  channel_sigma: { display: 0.1 },
  cell_kappa: 1,
};

/** One placement, one region, one profile: 10,000 impressions a day, with a
 * five-day outage that cuts traffic to 20%. */
function scarceSeed(overrides: Partial<SeasonSeed> = {}): SeasonSeed {
  return {
    seed_id: 'scarce-test',
    rng_seed: 'scarce-test-v1',
    start_date: '2027-01-01',
    days: 60,
    currency: 'USD',
    regions: [{ region_id: 'r1', country: 'US', name: 'Region one', share: 1 }],
    audience_profiles: [
      { profile_id: 'p1', name: 'Profile one', share: 0.6 },
      { profile_id: 'p2', name: 'Profile two', share: 0.4 },
    ],
    segments: [
      { segment_id: 's1', name: 'Segment one', profile_ids: ['p1'] },
      { segment_id: 's12', name: 'Segment one and two', profile_ids: ['p1', 'p2'] },
    ],
    properties: [{ property_id: 'site', name: 'Site', domain: 'site.example' }],
    placements: [{
      placement_id: 'banner',
      name: 'Banner',
      channel: 'display',
      property_ids: ['site'],
      formats: ['display_300x250'],
      base_daily_impressions: 10_000,
      floor_cpm: 2,
      rates: { viewability: 0.7, click_rate: 0.001, completion_rate: 0, conversion_rate: 0.02 },
      reporting_lag_days: 0,
    }],
    products: [
      { product_id: 'banner_guaranteed', name: 'Banner', description: 'Guaranteed banner', delivery_type: 'guaranteed', placement_ids: ['banner'], rate_cpm: 10 },
      { product_id: 'banner_auction', name: 'Banner auction', description: 'Auction banner', delivery_type: 'non_guaranteed', placement_ids: ['banner'] },
    ],
    traffic: { weekday_shape: [1, 1, 1, 1, 1, 1, 1] },
    forecast_error: ZERO_FORECAST_ERROR,
    events: [{
      type: 'traffic_multiplier',
      event_id: 'outage',
      description: 'Outage',
      start_date: '2027-01-06',
      end_date: '2027-01-10',
      multiplier: 0.2,
    }],
    ...overrides,
  };
}

describe('allocation primitives', () => {
  it('serves the most constrained booking first: B takes the first cell, A the second', () => {
    const remaining = counts([[1, 100], [2, 100]]);
    const result = allocateHighWaterMark(remaining, [
      { id: 'A', cells: [1, 2], target: 100 },
      { id: 'B', cells: [1], target: 100 },
    ]);
    expect(Object.fromEntries(result.get('B')!)).toEqual({ 1: 100 });
    expect(Object.fromEntries(result.get('A')!)).toEqual({ 2: 100 });
    expect(total(result.get('A'))).toBe(100);
    expect(total(result.get('B'))).toBe(100);
  });

  it('takes the same share of every usable cell and never more than the target', () => {
    const remaining = counts([[1, 300], [2, 100], [3, 600]]);
    const taken = takeEqualShare(remaining, [1, 2, 3], 500);
    expect(total(taken)).toBe(500);
    expect(Object.fromEntries(taken)).toEqual({ 1: 150, 2: 50, 3: 300 });
    expect(Object.fromEntries(remaining)).toEqual({ 1: 150, 2: 50, 3: 300 });
  });

  it('hands rounding leftovers to the largest remainders and keeps totals exact', () => {
    const remaining = counts([[1, 7], [2, 7], [3, 7]]);
    const taken = takeEqualShare(remaining, [1, 2, 3], 10);
    expect(total(taken)).toBe(10);
    for (const value of remaining.values()) expect(value).toBeGreaterThanOrEqual(0);
    expect([...taken.values()].sort()).toEqual([3, 3, 4]);
  });

  it('gives everything left when the cells cannot cover the target', () => {
    const remaining = counts([[1, 40], [2, 30]]);
    const result = allocateHighWaterMark(remaining, [{ id: 'A', cells: [1, 2], target: 100 }]);
    expect(total(result.get('A'))).toBe(70);
    expect(total(remaining)).toBe(0);
  });

  it('breaks supply ties by booking id', () => {
    const remaining = counts([[1, 100]]);
    const result = allocateHighWaterMark(remaining, [
      { id: 'b', cells: [1], target: 80 },
      { id: 'a', cells: [1], target: 80 },
    ]);
    expect(total(result.get('a'))).toBe(80);
    expect(total(result.get('b'))).toBe(20);
  });

  it('serves the highest bid first, respects floors and stops at the daily budget', () => {
    const remaining = counts([[1, 1000], [2, 1000]]);
    const floors: Record<number, number> = { 1: 1, 2: 5 };
    const result = allocateAuction(remaining, [
      { id: 'low', cells: [1, 2], bid_cpm: 2, daily_budget: 100 },
      { id: 'high', cells: [1, 2], bid_cpm: 6, daily_budget: 6 },
    ], cell => floors[cell]);
    // high: $6 at $6 CPM buys 1,000, spread over both cells.
    expect(total(result.get('high'))).toBe(1000);
    // low: below the floor of cell 2, so only what high left of cell 1.
    expect([...result.get('low')!.keys()]).toEqual([1]);
    expect(total(result.get('low'))).toBe(500);
  });

  it('takes a share of each cell for share-of-voice', () => {
    const actual = counts([[1, 100], [2, 51]]);
    const remaining = new Map(actual);
    expect(Object.fromEntries(takeShareOfVoice(remaining, actual, [1, 2], 0.5))).toEqual({ 1: 50, 2: 25 });
    expect(Object.fromEntries(takeShareOfVoice(remaining, actual, [1, 2], 0.5))).toEqual({ 1: 50, 2: 25 });
    expect(Object.fromEntries(remaining)).toEqual({ 2: 1, 1: 0 });
  });
});

describe('traffic', () => {
  it('generates the same tables from the same seed, and different ones from another rng_seed', () => {
    const noisy = scarceSeed({
      forecast_error: NOISY_FORECAST_ERROR,
      traffic: { weekday_shape: [1, 1, 1, 1, 1, 1, 1], truth_daily_sigma: 0.05 },
    });
    const a = new SeasonTraffic(noisy);
    const b = new SeasonTraffic(noisy);
    const c = new SeasonTraffic({ ...noisy, rng_seed: 'other' });
    for (let day = 0; day < 60; day++) {
      expect([...a.day(day).actual]).toEqual([...b.day(day).actual]);
      expect(a.rates(day, 0)).toEqual(b.rates(day, 0));
      for (const asOf of [0, 30]) {
        for (let cell = 0; cell < a.slots.length; cell++) {
          expect(a.forecastValue(day, cell, asOf)).toBe(b.forecastValue(day, cell, asOf));
        }
      }
    }
    expect([...a.day(3).actual]).not.toEqual([...c.day(3).actual]);
    expect(a.forecastValue(3, 0, 0)).not.toBe(c.forecastValue(3, 0, 0));
    expect(a.forecastValue(40, 1, 30)).not.toBe(c.forecastValue(40, 1, 30));
  });

  it('keeps expected and actual apart by the daily wobble and events only', () => {
    const traffic = new SeasonTraffic(ADSERVER_SEASON);
    let expected = 0;
    let actual = 0;
    let differing = 0;
    for (let day = 60; day < 90; day++) {
      const tables = traffic.day(day);
      for (const [cell, value] of tables.expected) {
        expected += value;
        actual += tables.actual.get(cell) ?? 0;
        if ((tables.actual.get(cell) ?? 0) !== value) differing += 1;
      }
    }
    expect(actual).not.toBe(expected);
    expect(differing).toBeGreaterThan(100);
    expect(Math.abs(actual / expected - 1)).toBeLessThan(0.15);

    // Forecast error never reaches the truth: with no wobble, spikes or events, actual is expected.
    const quiet = new SeasonTraffic(scarceSeed({ events: [], forecast_error: { ...NOISY_FORECAST_ERROR, channel_sigma: { display: 0.5 }, cell_kappa: 5 } }));
    for (let day = 0; day < 60; day++) expect([...quiet.day(day).actual]).toEqual([...quiet.day(day).expected]);
  });

  it('counts a profile once when two targeted segments overlap', () => {
    const traffic = new SeasonTraffic(scarceSeed());
    const both = traffic.eligibleCells('banner_guaranteed', { segment_ids: ['s1', 's12'] });
    expect(both).toEqual(traffic.eligibleCells('banner_guaranteed', { segment_ids: ['s12'] }));
    expect(new Set(both).size).toBe(both.length);
    expect(traffic.eligibleCells('banner_guaranteed', { segment_ids: ['s1'] })).toHaveLength(1);
  });

  it('builds installment audience over the channel curve', () => {
    expect(audienceCurve('newsletter')).toHaveLength(3);
    expect(audienceCurve('podcast')).toHaveLength(30);
    expect(audienceCurve('creator_video').length).toBeGreaterThanOrEqual(28);
    for (const channel of ['newsletter', 'podcast', 'creator_video'] as const) {
      expect(audienceCurve(channel).reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 9);
    }
  });

  it('refuses an inconsistent seed', () => {
    expect(() => new SeasonTraffic(scarceSeed({ regions: [{ region_id: 'r1', country: 'US', name: 'R', share: 0.5 }] })))
      .toThrow(EngineError);
    expect(() => new SeasonTraffic(scarceSeed({
      products: [{ product_id: 'x', name: 'x', description: 'x', delivery_type: 'guaranteed', placement_ids: ['missing'], rate_cpm: 1 }],
    }))).toThrow(/unknown placement/);
  });
});

describe('simulation', () => {
  const now = at('2026-12-31T12:00:00Z');
  const flight = { flight_start_ms: at('2027-01-01T00:00:00Z'), flight_end_ms: at('2027-01-31T00:00:00Z') };

  it('serves a booking from the day after it is confirmed', () => {
    const sim = InventorySimulation.create(scarceSeed(), at('2027-01-03T09:00:00Z'));
    sim.book({ booking_id: 'g', product_id: 'banner_guaranteed', ...flight, budget: 10_000, impression_goal: 50_000 }, at('2027-01-03T09:00:00Z'));
    sim.advanceTo(at('2027-01-06T00:00:00Z'));
    expect(sim.booking('g').days.map(day => day.date)).toEqual(['2027-01-04', '2027-01-05']);
  });

  it('under-delivers a guarantee through an outage, then catches up', () => {
    const sim = InventorySimulation.create(scarceSeed(), now);
    // 30 days at 10,000 a day, minus 40,000 lost in the outage, leaves
    // 260,000 of supply for a 200,000 goal.
    sim.book({ booking_id: 'g', product_id: 'banner_guaranteed', ...flight, budget: 10_000, impression_goal: 200_000 }, now);
    sim.advanceTo(at('2027-01-31T00:00:00Z'));
    const days = sim.booking('g').days;
    const outage = days.filter(day => day.date >= '2027-01-06' && day.date <= '2027-01-10');
    expect(outage.every(day => day.impressions < day.target!)).toBe(true);
    const before = days.find(day => day.date === '2027-01-05')!.target!;
    const after = days.find(day => day.date === '2027-01-11')!.target!;
    expect(after).toBeGreaterThan(before);
    expect(days.every(day => day.impressions <= day.target!)).toBe(true);
    expect(sim.booking('g').delivered.impressions).toBe(200_000);
    expect(sim.booking('g').delivered.spend).toBe(2000);
  });

  it('bills spend from cumulative impressions, so daily rounding never drifts the total', () => {
    const seed = scarceSeed();
    seed.products[0].rate_cpm = 14;
    const sim = InventorySimulation.create(seed, now);
    sim.book({ booking_id: 'g', product_id: 'banner_guaranteed', ...flight, budget: 10_000, impression_goal: 180_000 }, now);
    sim.book({ booking_id: 'n', product_id: 'banner_auction', ...flight, budget: 1_000_000, bid_cpm: 3.333 }, now);
    sim.advanceTo(at('2027-01-15T06:00:00Z'));
    sim.update('n', { bid_cpm: 2.777 }, at('2027-01-15T06:00:00Z'));
    sim.advanceTo(at('2027-01-31T00:00:00Z'));
    const g = sim.booking('g');
    expect(g.delivered.impressions).toBe(180_000);
    expect(g.delivered.spend).toBe(2520);
    for (const booking of [g, sim.booking('n')]) {
      const daily = booking.days.reduce((sum, day) => sum + Math.round(day.spend * 100), 0);
      expect(daily).toBe(Math.round(booking.delivered.spend * 100));
    }
    const n = sim.booking('n');
    const before = n.days.filter(day => day.date < '2027-01-15').reduce((sum, day) => sum + day.impressions, 0);
    const after = n.delivered.impressions - before;
    expect(n.delivered.spend).toBe(Math.round(((before * 3.333) / 1000 + (after * 2.777) / 1000) * 100) / 100);
  });

  it('gives non-guaranteed bookings only what guarantees leave, first-price', () => {
    const sim = InventorySimulation.create(scarceSeed(), now);
    sim.book({ booking_id: 'g', product_id: 'banner_guaranteed', ...flight, budget: 10_000, impression_goal: 150_000 }, now);
    sim.book({ booking_id: 'n', product_id: 'banner_auction', ...flight, budget: 1_000_000, bid_cpm: 3 }, now);
    sim.advanceTo(at('2027-01-31T00:00:00Z'));
    const g = sim.booking('g');
    const n = sim.booking('n');
    expect(g.delivered.impressions).toBe(150_000);
    for (const day of n.days) {
      const guaranteed = g.days.find(other => other.day === day.day)!.impressions;
      const supply = sim.traffic.day(day.day).actual.get(0)! + sim.traffic.day(day.day).actual.get(1)!;
      expect(day.impressions).toBe(supply - guaranteed);
      expect(day.spend).toBeCloseTo((day.impressions * 3) / 1000, 2);
    }
  });

  it('subtracts guaranteed bookings from the forecast and returns the AdCP shape', () => {
    const sim = InventorySimulation.create(scarceSeed(), now);
    const before = sim.forecast({ product_id: 'banner_guaranteed', ...flight }, now);
    expect(before.available).toBe(300_000);
    sim.book({ booking_id: 'g', product_id: 'banner_guaranteed', ...flight, budget: 10_000, impression_goal: 100_000 }, now);
    const after = sim.forecast({ product_id: 'banner_guaranteed', ...flight }, now);
    expect(after.booked).toBe(100_000);
    expect(after.available).toBe(200_000);
    const shaped = toAdcpForecast(after, { guaranteed: true, generatedAt: '2026-12-31T12:00:00.000Z', validUntil: '2027-01-01T00:00:00.000Z' });
    expect(shaped.points[0].metrics.impressions.mid).toBe(200_000);
    expect(shaped.method).toBe('guaranteed');
    expect(shaped.forecast_range_unit).toBe('availability');
  });

  it('releases a day of delivery only after the day closes and the lag passes', () => {
    const sim = InventorySimulation.create(scarceSeed(), now);
    sim.book({ booking_id: 'g', product_id: 'banner_guaranteed', ...flight, budget: 10_000, impression_goal: 100_000 }, now);
    expect(sim.delivery('g', at('2027-01-01T23:00:00Z')).days).toHaveLength(0);
    const report = sim.delivery('g', at('2027-01-02T01:00:00Z'));
    expect(report.days.map(day => day.date)).toEqual(['2027-01-01']);
    expect(report.reported_through_day).toBe(0);
  });

  it('is deterministic: the same calls give the same state', () => {
    const run = () => {
      const sim = InventorySimulation.create(ADSERVER_SEASON, now);
      sim.book({ booking_id: 'g', product_id: 'ros_display_guaranteed', ...flight, budget: 50_000, impression_goal: 3_000_000 }, now);
      sim.book({ booking_id: 't', product_id: 'homepage_takeover_daily', flight_start_ms: at('2027-01-10'), flight_end_ms: at('2027-01-11'), budget: 2000 }, now);
      sim.book({ booking_id: 'n', product_id: 'ros_display_auction', ...flight, budget: 20_000, bid_cpm: 3.5 }, now);
      sim.advanceTo(at('2027-01-15T06:00:00Z'));
      sim.update('n', { bid_cpm: 5 }, at('2027-01-15T06:00:00Z'));
      sim.advanceTo(at('2027-02-02T00:00:00Z'));
      return sim.state;
    };
    expect(run()).toEqual(run());
  });

  it('refuses a second share-of-voice booking on the same cells', () => {
    const sim = InventorySimulation.create(ADSERVER_SEASON, now);
    const day = { flight_start_ms: at('2027-01-10'), flight_end_ms: at('2027-01-11') };
    sim.book({ booking_id: 't1', product_id: 'homepage_takeover_daily', ...day, budget: 2000 }, now);
    expectEngineError(() => sim.book({ booking_id: 't2', product_id: 'homepage_takeover_daily', ...day, budget: 2000 }, now), 'SOLD_OUT');
  });

  it('pauses and cancels', () => {
    const sim = InventorySimulation.create(scarceSeed(), now);
    sim.book({ booking_id: 'g', product_id: 'banner_guaranteed', ...flight, budget: 10_000, impression_goal: 100_000 }, now);
    sim.advanceTo(at('2027-01-03T00:00:00Z'));
    sim.update('g', { status: 'paused' }, at('2027-01-03T00:00:00Z'));
    sim.advanceTo(at('2027-01-05T00:00:00Z'));
    expect(sim.booking('g').days).toHaveLength(2);
    sim.update('g', { status: 'canceled' }, at('2027-01-05T00:00:00Z'));
    expect(() => sim.update('g', { status: 'active' }, at('2027-01-05T00:00:00Z'))).toThrow(/canceled/);
  });
});

describe('installments', () => {
  const now = at('2026-12-31T12:00:00Z');

  it('delivers nothing for a cancelled episode and reports it once its date passes', () => {
    const sim = InventorySimulation.create(CONTENT_SEASON, now);
    const flight = { flight_start_ms: at('2027-01-01'), flight_end_ms: at('2027-03-01') };
    sim.book({ booking_id: 'ran', product_id: 'morning_brief_midroll', targeting: { installment_ids: ['morning_brief_011'] }, ...flight, budget: 5000 }, now);
    sim.book({ booking_id: 'cut', product_id: 'morning_brief_midroll', targeting: { installment_ids: ['morning_brief_012'] }, ...flight, budget: 5000 }, now);
    const forecast = sim.forecast({ product_id: 'morning_brief_midroll', targeting: { installment_ids: ['morning_brief_013'] }, ...flight }, now);
    expect(forecast.available).toBeGreaterThan(35_000);

    const statusOn = (iso: string) => sim.installmentStatuses(at(iso), 'morning_brief_midroll')
      .find(installment => installment.installment_id === 'morning_brief_012')!.state;
    expect(statusOn('2027-01-17T12:00:00Z')).toBe('scheduled');
    expect(statusOn('2027-01-18T12:00:00Z')).toBe('cancelled');

    const end = at('2027-03-01T12:00:00Z');
    expect(sim.delivery('cut', end).totals.impressions).toBe(0);
    const ran = sim.delivery('ran', end);
    expect(ran.totals.impressions).toBeGreaterThan(30_000);
    // Listens arrive over 30 days from 2027-01-11.
    const listened = ran.days.filter(day => day.impressions > 0);
    expect(listened).toHaveLength(30);
    expect(listened[0].date).toBe('2027-01-11');
    expect(listened[0].impressions).toBeGreaterThan(listened[10].impressions);
    expect(sim.state.log.find(entry => entry.date === '2027-01-18')!.installments)
      .toContainEqual({ installment_id: 'morning_brief_012', outcome: 'cancelled' });
  });

  it('moves a postponed installment to its new date', () => {
    const sim = InventorySimulation.create(CONTENT_SEASON, now);
    const statuses = sim.installmentStatuses(at('2027-01-20T00:00:00Z'), 'maker_studio_integration');
    const moved = statuses.find(installment => installment.installment_id === 'maker_studio_006')!;
    expect(moved.state).toBe('postponed');
    expect(moved.rescheduled_date).toBe('2027-01-23');
    const cells = sim.traffic.eligibleCells('maker_studio_integration', { installment_ids: ['maker_studio_006'] });
    const actualOn = (date: string) => cells.reduce((sum, cell) => sum + (sim.traffic.day(sim.traffic.calendar.dayOfDate(date)).actual.get(cell) ?? 0), 0);
    const expectedOn = (date: string) => cells.reduce((sum, cell) => sum + (sim.traffic.day(sim.traffic.calendar.dayOfDate(date)).expected.get(cell) ?? 0), 0);
    expect(expectedOn('2027-01-18')).toBeGreaterThan(0);
    expect(actualOn('2027-01-18')).toBe(0);
    expect(actualOn('2027-01-23')).toBeGreaterThan(0);
  });
});

describe('lead-time forecast error', () => {
  const DAYS = 1100;
  const FLAT = [1, 1, 1, 1, 1, 1, 1];
  const NO_ERROR: SeasonSeed['forecast_error'] = {
    channel_sigma: { display: 0, olv: 0 },
    ar1_phi: 0.85,
    lead_time_tau_days: 14,
    lead_time_horizon_days: 60,
    log_clip: 50,
    cell_kappa: 0,
    cell_min_impressions: 100,
  };

  interface SynthOptions {
    publishers?: number;
    base?: number;
    error?: Partial<SeasonSeed['forecast_error']>;
    traffic?: Partial<SeasonSeed['traffic']>;
    events?: SeasonSeed['events'];
    sharedProperty?: boolean;
  }

  /** Twenty single-property publishers, each with one display placement, one region and one
   * profile, so cell `i` is publisher `i` and its expected traffic is `base` every day. With
   * `sharedProperty`, one property carries a display placement (cell 0) and an olv placement (cell 1). */
  function synthSeed(options: SynthOptions = {}): SeasonSeed {
    const { publishers = 20, base = 10_000, error = {}, traffic = {}, events } = options;
    const ids = options.sharedProperty ? ['site'] : Array.from({ length: publishers }, (_, i) => `site_${i}`);
    const placement = (id: string, channel: 'display' | 'olv', propertyId: string) => ({
      placement_id: id,
      name: id,
      channel,
      property_ids: [propertyId],
      formats: ['display_300x250'],
      base_daily_impressions: base,
      floor_cpm: 1,
      rates: { viewability: 0.7, click_rate: 0.001, completion_rate: 0, conversion_rate: 0.02 },
      reporting_lag_days: 0,
    });
    return {
      seed_id: 'synth',
      rng_seed: 'synth-v1',
      start_date: '2027-01-01',
      days: DAYS,
      currency: 'USD',
      regions: [{ region_id: 'r1', country: 'US', name: 'Region one', share: 1 }],
      audience_profiles: [{ profile_id: 'p1', name: 'Profile one', share: 1 }],
      segments: [],
      properties: ids.map(id => ({ property_id: id, name: id, domain: `${id}.example` })),
      placements: options.sharedProperty
        ? [placement('pl_display', 'display', 'site'), placement('pl_olv', 'olv', 'site')]
        : ids.map(id => placement(`pl_${id}`, 'display', id)),
      products: [],
      traffic: { weekday_shape: FLAT, ...traffic },
      forecast_error: { ...NO_ERROR, ...error },
      ...(events ? { events } : {}),
    };
  }

  const zeroErrors = (seed: SeasonSeed): SeasonSeed => ({
    ...seed,
    forecast_error: {
      ...seed.forecast_error,
      channel_sigma: Object.fromEntries(Object.keys(seed.forecast_error.channel_sigma).map(channel => [channel, 0])),
      cell_kappa: 0,
      publishers: undefined,
    },
  });

  /** ln(F / reference) where the reference is the same world without error: B × e^u × Ŝ. */
  function logErrorOf(seed: SeasonSeed) {
    const traffic = new SeasonTraffic(seed);
    const reference = new SeasonTraffic(zeroErrors(seed));
    return {
      traffic,
      at: (day: number, cell: number, asOf: number) =>
        Math.log(traffic.forecastValue(day, cell, asOf) / reference.forecastValue(day, cell, asOf)),
    };
  }

  const sd = (values: number[]) => {
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    return Math.sqrt(values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length);
  };

  it('keeps every forecast within the log clip of the error-free value', () => {
    const clip = 0.3;
    const error = {
      channel_sigma: { display: 0.4 },
      cell_kappa: 4,
      log_clip: clip,
      publishers: [{ publisher_id: 'site_0', bias: -0.1, bias_sigma: 0.2 }],
      spikes: { per_publisher_min: 1, per_publisher_max: 2, size_min: 0.2, size_max: 1, decay: 0.9, decay_forecast_sigma: 0.3 },
    };
    for (const traffic of [{}, { truth_daily_sigma: 0.05 }]) {
      const { traffic: t, at } = logErrorOf(synthSeed({ base: 100, error, traffic }));
      let widest = 0;
      for (const asOf of [0, 200, 700]) {
        for (let day = asOf; day < DAYS; day += 3) {
          for (let cell = 0; cell < t.slots.length; cell++) {
            const logError = Math.abs(at(day, cell, asOf));
            widest = Math.max(widest, logError);
            expect(logError).toBeLessThanOrEqual(clip + 1e-9);
          }
        }
      }
      // The sigmas are large enough that the clip binds, so the bound is not vacuous.
      expect(widest).toBeGreaterThan(clip - 1e-9);
    }
  });

  it('revises a day forecast toward the same publisher draw as the as-of day approaches', () => {
    const { at } = logErrorOf(synthSeed({ error: { channel_sigma: { display: 0.02 }, ar1_phi: 0.5 } }));
    for (const [day, cell] of [[500, 0], [640, 3], [900, 11]]) {
      const lead = (s: number) => Math.sqrt(1 + (day - s) / 14);
      const reference = at(day, cell, day - 30) / lead(day - 30);
      expect(Math.abs(reference)).toBeGreaterThan(1e-4);
      let previous = Infinity;
      for (let asOf = day - 30; asOf <= day; asOf++) {
        expect(at(day, cell, asOf) / lead(asOf)).toBeCloseTo(reference, 9);
        const size = Math.abs(at(day, cell, asOf));
        expect(size).toBeLessThan(previous);
        previous = size;
      }
    }
  });

  it('draws the publisher term as an AR(1) series with unit variance', () => {
    const { at, traffic } = logErrorOf(synthSeed({ error: { channel_sigma: { display: 1 }, log_clip: 50 } }));
    let lagged = 0;
    let squares = 0;
    let count = 0;
    for (let cell = 0; cell < traffic.slots.length; cell++) {
      for (let day = 0; day < DAYS - 1; day++) {
        const x = at(day, cell, day);
        lagged += x * at(day + 1, cell, day + 1);
        squares += x * x;
        count += 1;
      }
    }
    expect(lagged / squares).toBeGreaterThan(0.82);
    expect(lagged / squares).toBeLessThan(0.88);
    expect(squares / count).toBeGreaterThan(0.9);
    expect(squares / count).toBeLessThan(1.1);
  });

  it('shares the publisher term across channels, scaled by each channel sigma', () => {
    const { at } = logErrorOf(synthSeed({
      sharedProperty: true,
      error: { channel_sigma: { display: 0.03, olv: 0.08 }, ar1_phi: 0.5 },
    }));
    let compared = 0;
    for (let day = 10; day < DAYS; day += 7) {
      const display = at(day, 0, day - 5);
      if (Math.abs(display) < 1e-6) continue;
      expect(at(day, 1, day - 5) / display).toBeCloseTo(0.08 / 0.03, 9);
      compared += 1;
    }
    expect(compared).toBeGreaterThan(100);
  });

  it('gives small cells more cell noise, with a floor on tiny ones', () => {
    const observed = (base: number) => {
      const { traffic, at } = logErrorOf(synthSeed({ base, error: { cell_kappa: 1 } }));
      const values: number[] = [];
      for (let cell = 0; cell < traffic.slots.length; cell++) for (let day = 0; day < DAYS; day++) values.push(at(day, cell, 0));
      return sd(values);
    };
    expect(observed(100)).toBeGreaterThan(0.075);
    expect(observed(100)).toBeLessThan(0.125);
    expect(observed(10_000)).toBeGreaterThan(0.0075);
    expect(observed(10_000)).toBeLessThan(0.0125);
    expect(observed(20)).toBeGreaterThan(0.075);
    expect(observed(20)).toBeLessThan(0.125);
    const traffic = new SeasonTraffic(synthSeed({ base: 20, error: { cell_kappa: 1 } }));
    expect(traffic.errorSigmas(5, 0, 0).cell).toBeCloseTo(0.1, 12);
    expect(new SeasonTraffic(synthSeed({ base: 10_000, error: { cell_kappa: 1 } })).errorSigmas(5, 0, 0).cell).toBeCloseTo(0.01, 12);
  });

  it('keeps scripted events in the truth only', () => {
    const noisy = { channel_sigma: { display: 0.1 }, cell_kappa: 1 };
    const plain = new SeasonTraffic(synthSeed({ error: noisy, traffic: { truth_daily_sigma: 0.03 } }));
    const withEvent = new SeasonTraffic(synthSeed({
      error: noisy,
      traffic: { truth_daily_sigma: 0.03 },
      events: [{ type: 'traffic_multiplier', event_id: 'surge', description: 'Surge', start_date: '2027-03-01', end_date: '2027-03-10', multiplier: 3, property_ids: ['site_0'] }],
    }));
    const surgeDay = withEvent.calendar.dayOfDate('2027-03-05');
    for (const asOf of [0, surgeDay - 3, surgeDay]) {
      for (let day = 0; day < DAYS; day += 4) {
        for (let cell = 0; cell < plain.slots.length; cell++) {
          expect(withEvent.forecastValue(day, cell, asOf)).toBe(plain.forecastValue(day, cell, asOf));
        }
      }
    }
    expect(withEvent.day(surgeDay).actual.get(0)).toBeGreaterThan((plain.day(surgeDay).actual.get(0) ?? 0) * 2.9);
    expect(withEvent.day(surgeDay).actual.get(1)).toBe(plain.day(surgeDay).actual.get(1));

    const content = new SeasonTraffic(CONTENT_SEASON);
    const quiet = new SeasonTraffic({ ...CONTENT_SEASON, events: [] });
    const cancelled = content.slots.filter(slot => slot.installment_id === 'morning_brief_012').map(slot => slot.index);
    for (const asOf of [0, 11, 40]) {
      for (let day = 0; day < 120; day += 3) {
        for (let cell = 0; cell < content.slots.length; cell++) {
          expect(content.forecastValue(day, cell, asOf)).toBe(quiet.forecastValue(day, cell, asOf));
        }
      }
    }
    const actualOf = (traffic: SeasonTraffic) => cancelled.reduce((sum, cell) => {
      for (let day = 0; day < 120; day++) sum += traffic.day(day).actual.get(cell) ?? 0;
      return sum;
    }, 0);
    expect(actualOf(content)).toBe(0);
    expect(actualOf(quiet)).toBeGreaterThan(10_000);
  });

  describe('spikes', () => {
    const spikes = { per_publisher_min: 2, per_publisher_max: 2, size_min: 0.5, size_max: 1.5, decay: 0.9, decay_forecast_sigma: 0.3 };
    const rng = 'synth-v1';
    /** The spikes of a publisher, redrawn here from the model's keys. */
    const spikesOf = (publisher: string, config: typeof spikes) => [0, 1].map(i => ({
      day: Math.floor(uniform(`${rng}|spike-day|${publisher}|${i}`) * DAYS),
      size: config.size_min + uniform(`${rng}|spike-size|${publisher}|${i}`) * (config.size_max - config.size_min),
      estimated: config.decay ** Math.exp(config.decay_forecast_sigma * standardNormal(`${rng}|spike-decay-est|${publisher}|${i}`)),
    }));
    const tail = (list: ReturnType<typeof spikesOf>, day: number, asOf: number, decayOf: (spike: ReturnType<typeof spikesOf>[number]) => number) =>
      list.reduce((product, spike) => (spike.day <= Math.min(asOf, day) ? product * (1 + spike.size * decayOf(spike) ** (day - spike.day)) : product), 1);

    it('forecasts a spike only from the day it starts', () => {
      const without = new SeasonTraffic(synthSeed({ traffic: { truth_daily_sigma: 0.02 } }));
      const withSpikes = new SeasonTraffic(synthSeed({ traffic: { truth_daily_sigma: 0.02 }, error: { spikes } }));
      for (let cell = 0; cell < 20; cell++) {
        const list = spikesOf(`site_${cell}`, spikes);
        const first = Math.min(...list.map(spike => spike.day));
        for (let day = 0; day < DAYS; day += 5) {
          if (first > 0) expect(withSpikes.forecastValue(day, cell, first - 1)).toBe(without.forecastValue(day, cell, first - 1));
          for (const asOf of [first, first + 40, 900]) {
            if (asOf >= DAYS || day < asOf) continue;
            const ratio = withSpikes.forecastValue(day, cell, asOf) / without.forecastValue(day, cell, asOf);
            expect(ratio).toBeCloseTo(tail(list, day, asOf, spike => spike.estimated), 9);
          }
        }
      }
    });

    it('matches the true tail when the forecaster knows the decay', () => {
      const exact = { ...spikes, decay_forecast_sigma: 0 };
      const without = new SeasonTraffic(synthSeed());
      const withSpikes = new SeasonTraffic(synthSeed({ error: { spikes: exact } }));
      let compared = 0;
      for (let cell = 0; cell < 20; cell++) {
        const last = Math.max(...spikesOf(`site_${cell}`, exact).map(spike => spike.day));
        for (let day = last; day < Math.min(DAYS, last + 40); day++) {
          const forecastRatio = withSpikes.forecastValue(day, cell, last) / without.forecastValue(day, cell, last);
          const trueRatio = withSpikes.day(day).actual.get(cell)! / without.day(day).actual.get(cell)!;
          expect(forecastRatio).toBeCloseTo(trueRatio, 3);
          compared += 1;
        }
      }
      expect(compared).toBeGreaterThan(400);
    });

    it('reports no tail spread before a spike starts and a positive one after', () => {
      const traffic = new SeasonTraffic(synthSeed({ error: { spikes } }));
      const first = Math.min(...spikesOf('site_0', spikes).map(spike => spike.day));
      expect(traffic.spikeTailSpread(first + 5, 0, Math.max(0, first - 1))).toBe(0);
      expect(traffic.spikeTailSpread(first + 5, 0, first)).toBeGreaterThan(0);
      expect(new SeasonTraffic(synthSeed()).spikeTailSpread(10, 0, 10)).toBe(0);
    });

    it('centres the tail spread on each spike own decay estimate', () => {
      const traffic = new SeasonTraffic(synthSeed({ error: { spikes } }));
      const list = spikesOf('site_0', spikes);
      const asOf = Math.max(...list.map(spike => spike.day));
      const day = Math.min(DAYS - 1, asOf + 10);
      const around = (sign: number) => Math.log(tail(list, day, asOf, spike => spike.estimated ** Math.exp(sign * spikes.decay_forecast_sigma)));
      expect(traffic.spikeTailSpread(day, 0, asOf)).toBeCloseTo(Math.abs(around(-1) - around(1)) / 2, 12);
    });

    it('ignores a spike that starts after the day when the as-of day is past it', () => {
      const without = new SeasonTraffic(synthSeed({ traffic: { truth_daily_sigma: 0.02 } }));
      const withSpikes = new SeasonTraffic(synthSeed({ traffic: { truth_daily_sigma: 0.02 }, error: { spikes } }));
      let checked = 0;
      for (let cell = 0; cell < 20; cell++) {
        for (const spike of spikesOf(`site_${cell}`, spikes)) {
          if (spike.day < 2) continue;
          const day = spike.day - 1;
          const others = spikesOf(`site_${cell}`, spikes).filter(other => other !== spike && other.day <= day);
          if (others.length > 0) continue;
          for (const asOf of [spike.day, Math.min(DAYS - 1, spike.day + 30)]) {
            expect(withSpikes.forecastValue(day, cell, asOf)).toBe(without.forecastValue(day, cell, asOf));
            checked += 1;
          }
        }
      }
      expect(checked).toBeGreaterThan(20);
    });
  });

  it('forecasts above the truth when the bias is negative', () => {
    const traffic = new SeasonTraffic(synthSeed({
      traffic: { truth_daily_sigma: 0.03 },
      error: { publishers: [{ publisher_id: 'site_0', bias: -0.02 }] },
    }));
    const reference = new SeasonTraffic(zeroErrors(synthSeed({ traffic: { truth_daily_sigma: 0.03 } })));
    for (const [day, asOf] of [[100, 0], [100, 100], [700, 650]]) {
      expect(traffic.forecastValue(day, 0, asOf) / reference.forecastValue(day, 0, asOf)).toBeCloseTo(Math.exp(0.02), 12);
      expect(traffic.forecastValue(day, 1, asOf)).toBe(reference.forecastValue(day, 1, asOf));
    }
  });

  it('puts the month of the year in expected traffic', () => {
    const month_shape = [1, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    const traffic = new SeasonTraffic(synthSeed({ traffic: { weekday_shape: [1, 2, 3, 4, 5, 6, 7], month_shape } }));
    const january = traffic.calendar.dayOfDate('2027-01-08');
    const february = traffic.calendar.dayOfDate('2027-02-05');
    expect(traffic.calendar.weekday(february)).toBe(traffic.calendar.weekday(january));
    expect(traffic.day(february).expected.get(0)).toBe(traffic.day(january).expected.get(0)! / 2);
    for (const asOf of [0, 20, february]) {
      expect(traffic.forecastValue(february, 0, asOf) / traffic.forecastValue(january, 0, asOf)).toBeCloseTo(0.5, 12);
    }
  });

  it('varies the truth day to day and forecasts it', () => {
    const traffic = new SeasonTraffic(synthSeed({ traffic: { truth_daily_sigma: 0.05 } }));
    const ratios: number[] = [];
    for (let cell = 0; cell < traffic.slots.length; cell++) {
      for (let day = 0; day < DAYS; day++) {
        const actual = traffic.day(day).actual.get(cell)!;
        ratios.push(actual / traffic.day(day).expected.get(cell)!);
        for (const asOf of [0, 500]) expect(Math.abs(traffic.forecastValue(day, cell, asOf) - actual)).toBeLessThanOrEqual(1);
      }
    }
    expect(sd(ratios)).toBeGreaterThan(0.04);
    expect(sd(ratios)).toBeLessThan(0.06);
  });

  it('stops widening after the lead-time horizon', () => {
    const { at } = logErrorOf(synthSeed({ error: { channel_sigma: { display: 0.1 }, lead_time_horizon_days: 60 } }));
    for (const cell of [0, 5, 9]) {
      const atHorizon = at(300, cell, 240);
      expect(Math.abs(atHorizon)).toBeGreaterThan(1e-4);
      expect(at(300, cell, 100)).toBeCloseTo(atHorizon, 12);
    }
  });

  it('refuses non-finite error values', () => {
    const cases: Array<[string, Partial<SeasonSeed['forecast_error']>, Partial<SeasonSeed['traffic']>]> = [
      ['channel_sigma', { channel_sigma: { display: Infinity } }, {}],
      ['cell_kappa', { cell_kappa: Infinity }, {}],
      ['log_clip', { log_clip: Infinity }, {}],
      ['lead_time_tau_days', { lead_time_tau_days: Infinity }, {}],
      ['cell_min_impressions', { cell_min_impressions: Infinity }, {}],
      ['truth_daily_sigma', {}, { truth_daily_sigma: Infinity }],
      ['month_shape', {}, { month_shape: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, Infinity] }],
      ['decay_forecast_sigma', { spikes: { per_publisher_min: 0, per_publisher_max: 1, size_min: 0, size_max: 1, decay: 0.5, decay_forecast_sigma: Infinity } }, {}],
      ['size_max', { spikes: { per_publisher_min: 0, per_publisher_max: 1, size_min: 0, size_max: Infinity, decay: 0.5, decay_forecast_sigma: 0 } }, {}],
      ['bias_sigma', { publishers: [{ publisher_id: 'site_0', bias: 0, bias_sigma: Infinity }] }, {}],
      ['error_scale', { publishers: [{ publisher_id: 'site_0', bias: 0, error_scale: Infinity }] }, {}],
    ];
    for (const [name, error, traffic] of cases) {
      expect(() => new SeasonTraffic(synthSeed({ error, traffic })), name).toThrow(EngineError);
    }
  });
});

describe('published range', () => {
  const DAY_MS = 86_400_000;
  const start = at('2027-01-01T00:00:00Z');
  const dayStart = (day: number) => start + day * DAY_MS;
  const noonOf = (day: number) => dayStart(day) + 12 * 3_600_000;
  const FLAT = [1, 1, 1, 1, 1, 1, 1];

  /** One display placement per publisher, one region and one profile, so cell `i` is publisher `i`
   * and expects `base` impressions every day; `prod_i` sells that cell alone. */
  function rangeSeed(options: { publishers?: number; days?: number; error?: Partial<SeasonSeed['forecast_error']>; traffic?: Partial<SeasonSeed['traffic']> } = {}): SeasonSeed {
    const { publishers = 1, days = 1100, error = {}, traffic = {} } = options;
    const ids = Array.from({ length: publishers }, (_, i) => `site_${i}`);
    return {
      seed_id: 'range',
      rng_seed: 'range-v1',
      start_date: '2027-01-01',
      days,
      currency: 'USD',
      regions: [{ region_id: 'r1', country: 'US', name: 'Region one', share: 1 }],
      audience_profiles: [{ profile_id: 'p1', name: 'Profile one', share: 1 }],
      segments: [],
      properties: ids.map(id => ({ property_id: id, name: id, domain: `${id}.example` })),
      placements: ids.map(id => ({
        placement_id: `pl_${id}`,
        name: id,
        channel: 'display' as const,
        property_ids: [id],
        formats: ['display_300x250'],
        base_daily_impressions: 10_000,
        floor_cpm: 1,
        rates: { viewability: 0.7, click_rate: 0.001, completion_rate: 0, conversion_rate: 0.02 },
        reporting_lag_days: 0,
      })),
      products: ids.map((id, i) => ({
        product_id: `prod_${i}`,
        name: id,
        description: id,
        delivery_type: 'guaranteed' as const,
        placement_ids: [`pl_${id}`],
        rate_cpm: 10,
      })),
      traffic: { weekday_shape: FLAT, ...traffic },
      forecast_error: {
        channel_sigma: { display: 0 },
        ar1_phi: 0.5,
        lead_time_tau_days: 14,
        lead_time_horizon_days: 60,
        log_clip: 50,
        cell_kappa: 0,
        cell_min_impressions: 100,
        ...error,
      },
    };
  }

  const flightOf = (first: number, last: number) => ({ flight_start_ms: dayStart(first), flight_end_ms: dayStart(last) });

  it('forecasts the same flight differently as the clock moves and narrows the range', () => {
    const sim = InventorySimulation.create(rangeSeed({ error: { channel_sigma: { display: 0.2 } } }), noonOf(0));
    const query = { product_id: 'prod_0', ...flightOf(20, 35) };
    const early = sim.forecast(query, noonOf(0));
    const later = sim.forecast(query, noonOf(7));
    expect(later.impressions).not.toBe(early.impressions);
    const width = (result: typeof early) => (result.high - result.low) / result.impressions;
    expect(width(early)).toBeGreaterThan(0);
    expect(width(later)).toBeLessThan(width(early));
  });

  it('gives the same forecast to two markets at the same simulated time', () => {
    const seed = rangeSeed({ publishers: 3, error: { channel_sigma: { display: 0.2 }, cell_kappa: 1 }, traffic: { truth_daily_sigma: 0.03 } });
    const walked = InventorySimulation.create(seed, noonOf(0));
    for (const day of [3, 9, 17]) walked.forecast({ product_id: 'prod_1', ...flightOf(day + 1, day + 20) }, noonOf(day));
    const fresh = InventorySimulation.create(seed, noonOf(17));
    for (const product_id of ['prod_0', 'prod_1', 'prod_2']) {
      const query = { product_id, ...flightOf(18, 60), budget: 4000 };
      expect(walked.forecast(query, noonOf(17))).toEqual(fresh.forecast(query, noonOf(17)));
    }
  });

  describe('closed form', () => {
    const PHI = 0.5;
    const SIGMA = 0.1;
    const KAPPA = 2;
    const growth = (h: number) => Math.sqrt(1 + h / 14);

    it('adds the publisher term with the AR correlation between the two days', () => {
      const sim = InventorySimulation.create(rangeSeed({ error: { channel_sigma: { display: SIGMA }, ar1_phi: PHI } }), noonOf(0));
      const [f1, f2] = [1, 2].map(day => sim.traffic.forecastValue(day, 0, 0));
      const [s1, s2] = [SIGMA * growth(1), SIGMA * growth(2)];
      const numerator = (f1 * s1) ** 2 + (f2 * s2) ** 2 + 2 * PHI * f1 * s1 * f2 * s2;
      const result = sim.forecast({ product_id: 'prod_0', ...flightOf(1, 3) }, noonOf(0));
      expect(result.sigma ** 2).toBeCloseTo(numerator / (f1 + f2) ** 2, 12);
    });

    it('adds the publisher term with the AR correlation at every lag across four days', () => {
      const sim = InventorySimulation.create(rangeSeed({ error: { channel_sigma: { display: SIGMA }, ar1_phi: PHI } }), noonOf(0));
      const days = [1, 2, 3, 4];
      const f = days.map(day => sim.traffic.forecastValue(day, 0, 0));
      const s = days.map(day => SIGMA * growth(day));
      let numerator = 0;
      for (let i = 0; i < days.length; i++) {
        for (let j = 0; j < days.length; j++) numerator += f[i] * f[j] * PHI ** Math.abs(i - j) * s[i] * s[j];
      }
      const total = f.reduce((a, b) => a + b, 0);
      const result = sim.forecast({ product_id: 'prod_0', ...flightOf(1, 5) }, noonOf(0));
      expect(result.sigma ** 2).toBeCloseTo(numerator / total ** 2, 12);
    });

    it('adds the cell term as independent per cell and day', () => {
      const sim = InventorySimulation.create(rangeSeed({ error: { cell_kappa: KAPPA } }), noonOf(0));
      const [f1, f2] = [1, 2].map(day => sim.traffic.forecastValue(day, 0, 0));
      const sigmaCell = KAPPA / Math.sqrt(10_000);
      const result = sim.forecast({ product_id: 'prod_0', ...flightOf(1, 3) }, noonOf(0));
      expect(result.sigma ** 2).toBeCloseTo((sigmaCell ** 2 * (f1 ** 2 + f2 ** 2)) / (f1 + f2) ** 2, 12);
    });

    it('floors the cell term at the minimum impressions', () => {
      const sim = InventorySimulation.create(rangeSeed({ error: { cell_kappa: KAPPA, cell_min_impressions: 40_000 } }), noonOf(0));
      const [f1, f2] = [1, 2].map(day => sim.traffic.forecastValue(day, 0, 0));
      const sigmaCell = KAPPA / Math.sqrt(40_000);
      const result = sim.forecast({ product_id: 'prod_0', ...flightOf(1, 3) }, noonOf(0));
      expect(result.sigma ** 2).toBeCloseTo((sigmaCell ** 2 * (f1 ** 2 + f2 ** 2)) / (f1 + f2) ** 2, 12);
    });

    it('adds the audience term per installment, fully correlated across its days', () => {
      const seed: SeasonSeed = { ...CONTENT_SEASON, forecast_error: { ...CONTENT_SEASON.forecast_error, cell_kappa: 0, log_clip: 50 } };
      const sim = InventorySimulation.create(seed, noonOf(0));
      const sigma = seed.collections!.find(collection => collection.collection_id === 'morning_brief')!.audience_sigma;
      const ids = ['morning_brief_011', 'morning_brief_013'];
      const [fa, fb] = ids.map(id => {
        let sum = 0;
        for (const cell of sim.traffic.eligibleCells('morning_brief_midroll', { installment_ids: [id] })) {
          for (let day = 1; day < 120; day++) sum += sim.traffic.forecastValue(day, cell, 0);
        }
        return sum;
      });
      const result = sim.forecast({ product_id: 'morning_brief_midroll', targeting: { installment_ids: ids }, ...flightOf(1, 120) }, noonOf(0));
      expect(result.sigma ** 2).toBeCloseTo((sigma ** 2 * (fa ** 2 + fb ** 2)) / (fa + fb) ** 2, 12);
    });
  });

  it('adds the spike term as one spread fully correlated across cells and days', () => {
    const rng = 'range-v1';
    const spikes = { per_publisher_min: 1, per_publisher_max: 1, size_min: 1, size_max: 1, decay: 0.95, decay_forecast_sigma: 0.5 };
    const base = rangeSeed({ publishers: 2, days: 60, error: { spikes } });
    const seed: SeasonSeed = {
      ...base,
      products: [{ product_id: 'both', name: 'both', description: 'both', delivery_type: 'guaranteed', placement_ids: ['pl_site_0', 'pl_site_1'], rate_cpm: 10 }],
    };
    const startDays = ['site_0', 'site_1'].map(id => Math.floor(uniform(`${rng}|spike-day|${id}|0`) * 60));
    const asOf = Math.max(...startDays);
    const sim = InventorySimulation.create(seed, noonOf(asOf));
    const days = [asOf + 1, asOf + 2];
    const spread = (day: number, cell: number) => sim.traffic.spikeTailSpread(day, cell, asOf);
    const forecast = (day: number, cell: number) => sim.traffic.forecastValue(day, cell, asOf);
    for (const day of days) for (const cell of [0, 1]) expect(spread(day, cell)).toBeGreaterThan(0);
    let numerator = 0;
    let total = 0;
    for (const day of days) for (const cell of [0, 1]) {
      numerator += forecast(day, cell) * spread(day, cell);
      total += forecast(day, cell);
    }
    const result = sim.forecast({ product_id: 'both', ...flightOf(asOf + 1, asOf + 3) }, noonOf(asOf));
    expect(result.sigma ** 2).toBeCloseTo(numerator ** 2 / total ** 2, 12);
  });

  it('covers the truth about 80% of the time', () => {
    const seed = rangeSeed({ publishers: 20, error: { channel_sigma: { display: 0.1 } } });
    const sim = InventorySimulation.create(seed, noonOf(0));
    let covered = 0;
    let samples = 0;
    for (let asOf = 0; asOf < 1000; asOf++) {
      const target = asOf + 7;
      const actuals = sim.traffic.day(target).actual;
      for (let cell = 0; cell < 20; cell++) {
        const result = sim.forecast({ product_id: `prod_${cell}`, ...flightOf(target, target + 1) }, noonOf(asOf));
        const actual = actuals.get(cell)!;
        if (actual >= result.low && actual <= result.high) covered++;
        samples++;
      }
    }
    const share = covered / samples;
    expect(share).toBeGreaterThan(0.75);
    expect(share).toBeLessThan(0.85);
  });

  it('caps the band at the clip', () => {
    const sim = InventorySimulation.create(rangeSeed({ error: { channel_sigma: { display: 3 }, lead_time_horizon_days: 365, log_clip: 0.5 } }), noonOf(0));
    const result = sim.forecast({ product_id: 'prod_0', ...flightOf(400, 401) }, noonOf(0));
    expect(1.2816 * result.sigma).toBeGreaterThan(0.5);
    // low, high and impressions are each rounded to whole impressions of about 10,000, so the ratio is good to 1e-4.
    expect(result.high / result.impressions).toBeLessThanOrEqual(Math.exp(0.5) * (1 + 1e-4));
    expect(result.high / result.impressions).toBeCloseTo(Math.exp(0.5), 3);
    expect(result.low / result.impressions).toBeCloseTo(Math.exp(-0.5), 3);
  });

  it('biases forecasts by the publisher bias', () => {
    const seed = rangeSeed({
      publishers: 5,
      error: { channel_sigma: { display: 0.05 }, publishers: Array.from({ length: 5 }, (_, i) => ({ publisher_id: `site_${i}`, bias: 0.1, bias_sigma: 0 })) },
      traffic: { truth_daily_sigma: 0.03 },
    });
    const traffic = new SeasonTraffic(seed);
    let sum = 0;
    let n = 0;
    for (let day = 7; day < seed.days; day++) {
      for (let cell = 0; cell < 5; cell++) {
        sum += Math.log(traffic.forecastValue(day, cell, day - 7) / traffic.day(day).actual.get(cell)!);
        n++;
      }
    }
    expect(Math.abs(sum / n + 0.1)).toBeLessThan(0.02);
  });

  it('nets guaranteed bookings out of the revised forecast', () => {
    const seed = rangeSeed({ days: 90, error: { channel_sigma: { display: 0.15 }, cell_kappa: 1 }, traffic: { truth_daily_sigma: 0.03 } });
    const sim = InventorySimulation.create(seed, noonOf(0));
    sim.book({ booking_id: 'g', product_id: 'prod_0', ...flightOf(1, 31), budget: 5000, impression_goal: 100_000 }, noonOf(0));
    for (const asOf of [0, 12]) {
      const result = sim.forecast({ product_id: 'prod_0', ...flightOf(1, 61), budget: 2500 }, noonOf(asOf));
      const affordable = 250_000;
      expect(Math.abs(result.available + result.booked - result.capacity)).toBeLessThanOrEqual(1);
      expect(Math.abs(result.booked - (100_000 - sim.booking('g').delivered.impressions))).toBeLessThanOrEqual(1);
      expect(result.low).toBeLessThanOrEqual(result.impressions);
      expect(result.impressions).toBeLessThanOrEqual(result.high);
      expect(result.high).toBeLessThanOrEqual(affordable);
    }
  });

  it('scales share-of-voice reservations with the revised forecast', () => {
    const base = rangeSeed({ days: 90, error: { channel_sigma: { display: 0.15 }, cell_kappa: 1 }, traffic: { truth_daily_sigma: 0.03 } });
    const seed: SeasonSeed = {
      ...base,
      products: [...base.products, { product_id: 'sov', name: 'sov', description: 'sov', delivery_type: 'guaranteed', placement_ids: ['pl_site_0'], rate_cpm: 10, share_of_voice: 0.4 }],
    };
    const sim = InventorySimulation.create(seed, noonOf(0));
    sim.book({ booking_id: 's', product_id: 'sov', ...flightOf(1, 31), budget: 100_000 }, noonOf(0));
    const booked: number[] = [];
    for (const asOf of [0, 12]) {
      const result = sim.forecast({ product_id: 'prod_0', ...flightOf(1, 61) }, noonOf(asOf));
      let held = 0;
      for (let day = asOf + 1; day < 31; day++) held += 0.4 * sim.traffic.forecastValue(day, 0, asOf);
      expect(Math.abs(result.booked - held)).toBeLessThanOrEqual(1);
      expect(Math.abs(result.available + result.booked - result.capacity)).toBeLessThanOrEqual(1);
      booked.push(result.booked);
    }
    expect(booked[1]).not.toBe(booked[0]);
  });
});
