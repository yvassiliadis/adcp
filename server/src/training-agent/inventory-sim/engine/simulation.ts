/**
 * One simulated market: a season's traffic, the bookings made against it,
 * and the delivery recorded day by day. All times are milliseconds on the
 * host's clock; the engine never reads the wall clock.
 *
 * Each simulated day runs, in order:
 *  1. Bookings confirmed by the end of day N serve from day N+1, or from
 *     their flight start if later.
 *  2. Read the day's actual traffic (events are already applied to it).
 *  3. Guaranteed first: share-of-voice bookings take their cells, then
 *     impression-goal bookings are served by the high-water mark method.
 *  4. Non-guaranteed bookings bid for what is left, first-price, above the
 *     placement floor, until each one's daily budget is spent.
 *  5. Installments scheduled for the day run, or are postponed or
 *     cancelled. One that runs builds its audience over its channel's
 *     curve, which the traffic tables already hold.
 *  6. Record delivery per booking and per cell.
 *
 * The same seed and the same calls always give the same results.
 */
import {
  allocateAuction,
  allocateFractionalHighWaterMark,
  allocateHighWaterMark,
  supplyOf,
  takeFractionalShareOfVoice,
  takeShareOfVoice,
  type BidRequest,
  type CellCounts,
  type FractionalGoalRequest,
  type GoalRequest,
} from './allocation.js';
import { EngineError } from './errors.js';
import { trafficFor, type SeasonTraffic } from './traffic.js';
import type {
  Booking,
  BookingDay,
  BookingStatus,
  BookingTargeting,
  CellRates,
  DayLog,
  DeliveryMetrics,
  Installment,
  SeasonSeed,
  SeedProduct,
  SimulationState,
} from './types.js';

/** z for a 10th–90th percentile forecast range. */
const RANGE_Z = 1.2816;
/** A dry run counts a booking as harmed when its undelivered impressions grow by more than this. */
const DRY_TOLERANCE = 0.5;
const DRY_EPSILON = 1e-9;
const MAX_BISECTION_STEPS = 32;
/** The bisection stops when the bracket is this narrow: 1 impression, or 0.05% of the upper end. */
const BISECTION_TOLERANCE = 0.0005;
const bisectionTolerance = (high: number): number => Math.max(1, Math.floor(BISECTION_TOLERANCE * high));
/** Sorts after every real booking id. */
const CANDIDATE_ID = '\uffffcandidate';

/** The forecast one call reads. Only the default reader, which is a pure function of the day, lets the base run be memoised. */
interface Reader {
  read: ForecastReader;
  memoise: boolean;
}

/** A booking as the dry run sees it. */
interface DryBooking {
  id: string;
  kind: 'goal' | 'share_of_voice';
  cells: readonly number[];
  cellSet: ReadonlySet<number>;
  /** First serving day, and one past the last inside the season. */
  first: number;
  last: number;
  /** The flight's last day, exclusive: the pacing horizon. */
  end: number;
  /** Goals: impressions still to deliver when the booking enters. */
  left: number;
  /** Goals: running totals of expected traffic over the cells, as `expectedTotals` holds them. Pacing reads these, as `runDay` does. */
  expected?: Float64Array;
  share: number;
  /** Sorts after bookings with an equal supply. */
  tail?: boolean;
}

interface BaseRun {
  books: DryBooking[];
  from: number;
  to: number;
  /** Each goal's undelivered impressions after the last day. */
  left: Float64Array;
  /** Undelivered impressions at the start of each day from `from`. */
  snapshots: Float64Array[];
  /** Impressions every booking takes, per day and cell. */
  held: Map<number, CellCounts>;
}

interface GoalAvailability {
  available: number;
  /** Forecast over the candidate's cells and days, before any booking. */
  gross: number;
  /** The part of `gross` that is not available, unrounded. */
  held: number;
  /** Calls `onTake` with the candidate's per-day, per-cell take when it holds `x`. */
  takes: (x: number, onTake: (day: number, cell: number, count: number) => void) => void;
  /** No booking meets the candidate, so `takes` reports its cells' forecast and sums to `available` to rounding. */
  untouched: boolean;
}

export interface BookRequest {
  booking_id: string;
  product_id: string;
  targeting?: BookingTargeting;
  flight_start_ms: number;
  flight_end_ms: number;
  budget: number;
  /** Guaranteed products: the rate of the pricing option the buyer chose.
   * Defaults to the product's rate. */
  rate_cpm?: number;
  /** Guaranteed impression-goal products. Defaults to what the budget buys
   * at the booking's rate. */
  impression_goal?: number;
  /** Non-guaranteed products. */
  bid_cpm?: number;
}

export interface BookingPatch {
  status?: BookingStatus;
  budget?: number;
  impression_goal?: number;
  bid_cpm?: number;
  flight_end_ms?: number;
}

export interface ForecastQuery {
  product_id: string;
  targeting?: BookingTargeting;
  flight_start_ms: number;
  flight_end_ms: number;
  budget?: number;
  /** Guaranteed products: the rate of the pricing option forecast. Defaults
   * to the product's rate. */
  rate_cpm?: number;
  bid_cpm?: number;
}

export interface ForecastResult extends DeliveryMetrics {
  product_id: string;
  currency: string;
  /** First day a booking made now could serve. */
  start_day: number;
  /** Exclusive. */
  end_day: number;
  /** Forecast impressions over the covered cells and days. */
  capacity: number;
  /** Of those, reserved by guaranteed bookings. */
  booked: number;
  available: number;
  /** Lower end of the band: built on capacity, net of bookings, and capped at
   * what the budget buys. */
  low: number;
  /** Upper end of the band, built the same way as `low`. */
  high: number;
  /** Price used for spend: the product's rate, or the bid. */
  cpm?: number;
  /** Standard deviation of the log error, `ln(actual / forecast)`; low and high sit `RANGE_Z` of
   * these either side of capacity, capped at twice the season's `log_clip`, the widest error the truth
   * and the forecast's known part can open between them. */
  sigma: number;
}

export interface DeliveryReport {
  booking: Omit<Booking, 'days'>;
  /** Last day whose delivery is reported, inclusive. */
  reported_through_day?: number;
  totals: DeliveryMetrics;
  days: BookingDay[];
  /** Goal bookings: impressions an even pace against expected traffic would
   * have delivered by the reported day. */
  expected_impressions?: number;
}

/** Reads the forecast impressions of one cell on one day, as known on a fixed day. */
export type ForecastReader = (day: number, cell: number) => number;

export type InstallmentState = 'scheduled' | 'published' | 'postponed' | 'cancelled';

export interface InstallmentStatus extends Installment {
  state: InstallmentState;
  /** Date it now runs on, once a postponement is known. */
  rescheduled_date?: string;
}

function zeroMetrics(): DeliveryMetrics {
  return { impressions: 0, spend: 0, clicks: 0, viewable_impressions: 0, completed_views: 0, conversions: 0 };
}

function roundMoney(value: number): number {
  return Math.round(value * 100) / 100;
}

function addMetrics(into: DeliveryMetrics, from: DeliveryMetrics): void {
  into.impressions += from.impressions;
  into.spend = roundMoney(into.spend + from.spend);
  into.clicks += from.clicks;
  into.viewable_impressions += from.viewable_impressions;
  into.completed_views += from.completed_views;
  into.conversions += from.conversions;
}

function positive(value: number | undefined, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new EngineError('INVALID_BOOKING', `${label} must be a positive number`);
  }
  return value;
}

/**
 * What a goal serves today out of its `left` undelivered impressions: its
 * share of the expected traffic for the rest of the window, or an even split
 * over the days left when the window expects none. That fallback is
 * defensive: no seed reaches it, but installment cells can have none on days
 * they serve. `runDay` rounds the result up; the dry run keeps it fractional.
 */
function pacedTarget(left: number, expectedToday: number, expectedWindow: number, daysLeft: number): number {
  return expectedWindow > 0 ? (left * expectedToday) / expectedWindow : left / daysLeft;
}

function overlaps(a: readonly number[], b: ReadonlySet<number>): boolean {
  return a.some(cell => b.has(cell));
}

export class InventorySimulation {
  readonly traffic: SeasonTraffic;
  private readonly cellCache = new Map<string, number[]>();
  /** Per booking id, running totals of expected traffic (the rounded `expected`
   * table every forecast shares) over the booking's cells: entry d is the sum
   * over days [0, d). Cells never change after booking, so only the flight end
   * is read live. Pacing reads expected traffic, not an as-of forecast, so
   * the totals never depend on the day they are read. */
  private readonly expectedPrefix = new Map<string, Float64Array>();
  private readonly cellSets = new WeakMap<readonly number[], ReadonlySet<number>>();
  private readonly memo = new Map<string, BaseRun>();
  private memoStamp = '';
  private revision = 0;
  private dryRunCount = 0;
  private bookingsRun = 0;

  /** `products` are sold beside the seed's own, on its placements. */
  constructor(readonly seed: SeasonSeed, readonly state: SimulationState, products: readonly SeedProduct[] = []) {
    if (state.seed_id !== seed.seed_id) {
      throw new EngineError('INVALID_SEED', `State belongs to seed ${state.seed_id}, not ${seed.seed_id}`);
    }
    this.traffic = trafficFor(seed).withProducts(products);
  }

  /** A fresh market whose first unprocessed day is the day `nowMs` falls on. */
  static create(seed: SeasonSeed, nowMs: number, products: readonly SeedProduct[] = []): InventorySimulation {
    const traffic = trafficFor(seed);
    return new InventorySimulation(seed, {
      seed_id: seed.seed_id,
      processed_through_day: traffic.calendar.dayOf(nowMs),
      bookings: [],
      log: [],
    }, products);
  }

  today(nowMs: number): number {
    return this.traffic.calendar.dayOf(nowMs);
  }

  /** Run every day that has closed by `nowMs`. Earlier instants are a no-op. */
  advanceTo(nowMs: number): number[] {
    const today = this.today(nowMs);
    const ran: number[] = [];
    while (this.state.processed_through_day < today) {
      this.runDay(this.state.processed_through_day);
      ran.push(this.state.processed_through_day);
      this.state.processed_through_day += 1;
    }
    return ran;
  }

  booking(bookingId: string): Booking {
    const booking = this.state.bookings.find(candidate => candidate.booking_id === bookingId);
    if (!booking) throw new EngineError('BOOKING_NOT_FOUND', `Unknown booking ${bookingId}`);
    return booking;
  }

  private cellsFor(booking: Pick<Booking, 'booking_id' | 'product_id' | 'targeting'>): number[] {
    let cells = this.cellCache.get(booking.booking_id);
    if (!cells) {
      cells = this.traffic.eligibleCells(booking.product_id, booking.targeting);
      this.cellCache.set(booking.booking_id, cells);
    }
    return cells;
  }

  private flightDays(startMs: number, endMs: number): { start: number; end: number } {
    if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) {
      throw new EngineError('INVALID_BOOKING', 'The flight must end after it starts');
    }
    const { calendar } = this.traffic;
    return { start: calendar.dayOf(startMs), end: Math.ceil((endMs - calendar.startMs) / 86_400_000) };
  }

  book(request: BookRequest, nowMs: number): Booking {
    this.advanceTo(nowMs);
    this.revision += 1;
    try {
      return this.place(request, nowMs);
    } finally {
      this.revision += 1;
    }
  }

  private place(request: BookRequest, nowMs: number): Booking {
    if (this.state.bookings.some(booking => booking.booking_id === request.booking_id)) {
      throw new EngineError('DUPLICATE_BOOKING', `Booking ${request.booking_id} already exists`);
    }
    const product = this.traffic.requireProduct(request.product_id);
    const { start, end } = this.flightDays(request.flight_start_ms, request.flight_end_ms);
    const budget = positive(request.budget, 'budget');
    const targeting: BookingTargeting = structuredClone(request.targeting ?? {});
    const cells = this.traffic.eligibleCells(product.product_id, targeting);
    if (cells.length === 0) throw new EngineError('INVALID_TARGETING', 'The targeting leaves no inventory in this product');
    const booking: Booking = {
      booking_id: request.booking_id,
      product_id: product.product_id,
      kind: product.delivery_type === 'non_guaranteed' ? 'auction' : product.share_of_voice ? 'share_of_voice' : 'goal',
      status: 'active',
      targeting,
      flight_start_day: start,
      flight_end_day: end,
      confirmed_day: this.today(nowMs),
      cpm: 0,
      budget,
      delivered: zeroMetrics(),
      days: [],
    };
    if (booking.kind === 'auction') {
      booking.cpm = positive(request.bid_cpm, 'bid_cpm');
    } else {
      booking.cpm = request.rate_cpm !== undefined ? positive(request.rate_cpm, 'rate_cpm') : product.rate_cpm!;
    }
    if (booking.kind === 'goal') {
      booking.impression_goal = Math.floor(request.impression_goal !== undefined
        ? positive(request.impression_goal, 'impression_goal')
        : (budget * 1000) / booking.cpm);
      if (booking.impression_goal <= 0) throw new EngineError('INVALID_BOOKING', 'The budget buys no impressions at this rate');
      this.requireAvailable(booking, cells, booking.impression_goal, nowMs);
    }
    if (booking.kind === 'share_of_voice') {
      booking.share_of_voice = product.share_of_voice!;
      const mine = new Set(cells);
      const sold = this.state.bookings
        .filter(other => other.kind === 'share_of_voice' && other.status !== 'canceled'
          && other.flight_start_day < end && start < other.flight_end_day
          && overlaps(this.cellsFor(other), mine))
        .reduce((sum, other) => sum + (other.share_of_voice ?? 0), 0);
      if (sold + booking.share_of_voice > 1 + 1e-9) {
        throw new EngineError('SOLD_OUT', `Product ${product.product_id} is already sold for part of this flight`);
      }
      const today = this.today(nowMs);
      const { first, last } = this.servingDays(booking, today);
      const candidate = this.candidateOf(cells, first, last, end, 'share_of_voice', booking.share_of_voice, booking.booking_id);
      if (this.harmsGoals(candidate, today, this.readerFor(today))) {
        throw new EngineError('INSUFFICIENT_INVENTORY',
          `Product ${product.product_id} would leave other guaranteed bookings short of their impression goals for this flight and targeting`);
      }
    }
    this.cellCache.set(booking.booking_id, cells);
    this.state.bookings.push(booking);
    return booking;
  }

  /**
   * Change a booking. Every field is validated first. A goal booking whose
   * goal rose or whose last day moved is then checked once, as it would be
   * after the change, so a refusal leaves it untouched.
   */
  update(bookingId: string, patch: BookingPatch, nowMs: number): Booking {
    this.advanceTo(nowMs);
    const booking = this.booking(bookingId);
    if (booking.status === 'canceled') throw new EngineError('INVALID_BOOKING', `Booking ${bookingId} is canceled`);
    this.revision += 1;
    try {
      let goal: number | undefined;
      if (patch.impression_goal !== undefined) {
        if (booking.kind !== 'goal') throw new EngineError('INVALID_BOOKING', 'Only goal bookings have an impression goal');
        goal = Math.floor(positive(patch.impression_goal, 'impression_goal'));
      }
      const budget = patch.budget !== undefined ? positive(patch.budget, 'budget') : undefined;
      let bid: number | undefined;
      if (patch.bid_cpm !== undefined) {
        if (booking.kind !== 'auction') throw new EngineError('INVALID_BOOKING', 'Only non-guaranteed bookings bid');
        bid = positive(patch.bid_cpm, 'bid_cpm');
      }
      let endDay: number | undefined;
      if (patch.flight_end_ms !== undefined) {
        endDay = this.flightDays(this.traffic.calendar.startMs + booking.flight_start_day * 86_400_000, patch.flight_end_ms).end;
      }
      if (booking.kind === 'goal') {
        const raised = goal !== undefined && goal > (booking.impression_goal ?? 0);
        const moved = endDay !== undefined && endDay !== booking.flight_end_day;
        if (raised || moved) {
          this.requireAvailable({ ...booking, flight_end_day: endDay ?? booking.flight_end_day }, this.cellsFor(booking), goal ?? booking.impression_goal ?? 0, nowMs);
        }
      }
      if (booking.kind === 'share_of_voice' && endDay !== undefined && endDay > booking.flight_end_day && patch.status !== 'canceled') {
        this.requireExtensionFits(booking, endDay, nowMs);
      }
      if (goal !== undefined) booking.impression_goal = goal;
      if (budget !== undefined) booking.budget = budget;
      if (bid !== undefined) booking.cpm = bid;
      if (endDay !== undefined) booking.flight_end_day = endDay;
      if (patch.status !== undefined) booking.status = patch.status;
      return booking;
    } finally {
      this.revision += 1;
    }
  }

  /**
   * Refuse a share-of-voice booking whose end moves later when the added days
   * are already sold to another share-of-voice booking on the same cells, or
   * when holding them would leave a goal short. Only the added days are
   * weighed: the booking is already in the market for the days it has, and days
   * before today are past selling. The sold-share range is clamped to the
   * season while book() counts past the season end, which only matters for
   * flights that run past the season end.
   */
  private requireExtensionFits(booking: Booking, endDay: number, nowMs: number): void {
    const addedFrom = booking.flight_end_day;
    const addedTo = Math.min(endDay, this.traffic.calendar.days);
    if (addedFrom >= addedTo) return;
    const cells = this.cellsFor(booking);
    const share = booking.share_of_voice ?? 0;
    const today = this.today(nowMs);
    const soldFrom = Math.max(addedFrom, today);
    if (soldFrom < addedTo) {
      const mine = new Set(cells);
      const sold = this.state.bookings
        .filter(other => other.kind === 'share_of_voice' && other.status !== 'canceled' && other.booking_id !== booking.booking_id
          && other.flight_start_day < addedTo && soldFrom < other.flight_end_day
          && overlaps(this.cellsFor(other), mine))
        .reduce((sum, other) => sum + (other.share_of_voice ?? 0), 0);
      if (sold + share > 1 + 1e-9) {
        throw new EngineError('SOLD_OUT', `Product ${booking.product_id} is already sold for part of the added days`);
      }
    }
    const { first, last } = this.servingDays({ ...booking, flight_end_day: endDay }, today);
    const from = Math.max(first, addedFrom);
    if (from >= last) return;
    const candidate = this.candidateOf(cells, from, last, endDay, 'share_of_voice', share);
    if (this.harmsGoals(candidate, today, this.readerFor(today))) {
      throw new EngineError('INSUFFICIENT_INVENTORY',
        `Extending ${booking.booking_id} would leave other guaranteed bookings short of their impression goals`);
    }
  }

  private expectedPrefixOf(cells: readonly number[]): Float64Array {
    const { days } = this.traffic.calendar;
    const prefix = new Float64Array(days + 1);
    for (let d = 0; d < days; d++) prefix[d + 1] = prefix[d] + supplyOf(this.traffic.day(d).expected, cells);
    return prefix;
  }

  private expectedTotals(booking: Pick<Booking, 'booking_id' | 'product_id' | 'targeting'>): Float64Array {
    let prefix = this.expectedPrefix.get(booking.booking_id);
    if (!prefix) {
      prefix = this.expectedPrefixOf(this.cellsFor(booking));
      this.expectedPrefix.set(booking.booking_id, prefix);
    }
    return prefix;
  }

  private runDay(day: number): void {
    const { calendar } = this.traffic;
    const tables = this.traffic.day(day);
    const remaining: CellCounts = new Map(tables.actual);
    const serving = this.state.bookings.filter(booking => booking.status === 'active'
      && day >= Math.max(booking.confirmed_day + 1, booking.flight_start_day)
      && day < booking.flight_end_day);
    const allocations = new Map<string, CellCounts>();
    const targets = new Map<string, number>();

    for (const booking of serving.filter(b => b.kind === 'share_of_voice').sort((a, b) => (a.booking_id < b.booking_id ? -1 : 1))) {
      allocations.set(booking.booking_id, takeShareOfVoice(remaining, tables.actual, this.cellsFor(booking), booking.share_of_voice ?? 0));
    }

    const goals: GoalRequest[] = [];
    for (const booking of serving.filter(b => b.kind === 'goal')) {
      const left = Math.max(0, (booking.impression_goal ?? 0) - booking.delivered.impressions);
      const affordable = Math.floor(((booking.budget - booking.delivered.spend) * 1000) / booking.cpm + 1e-9);
      // What is left is spread over the days [day, end) in proportion to expected
      // traffic.
      const prefix = this.expectedTotals(booking);
      const end = Math.min(booking.flight_end_day, calendar.days);
      const expectedToday = prefix[day + 1] - prefix[day];
      const sumF = prefix[end] - prefix[day];
      const paced = pacedTarget(left, expectedToday, sumF, booking.flight_end_day - day);
      const planned = Math.ceil(sumF > 0 ? paced - 1e-9 : paced);
      const target = Math.max(0, Math.min(left, planned, affordable));
      targets.set(booking.booking_id, target);
      goals.push({ id: booking.booking_id, cells: this.cellsFor(booking), target });
    }
    for (const [id, taken] of allocateHighWaterMark(remaining, goals)) allocations.set(id, taken);

    const bids: BidRequest[] = serving.filter(b => b.kind === 'auction').map(booking => ({
      id: booking.booking_id,
      cells: this.cellsFor(booking),
      bid_cpm: booking.cpm,
      daily_budget: Math.max(0, booking.budget - booking.delivered.spend) / (booking.flight_end_day - day),
    }));
    for (const [id, taken] of allocateAuction(remaining, bids, cell => this.traffic.floorOf(cell))) allocations.set(id, taken);

    let sold = 0;
    for (const booking of serving) {
      const taken = allocations.get(booking.booking_id) ?? new Map<number, number>();
      const record = this.recordDay(booking, day, taken, targets.get(booking.booking_id));
      sold += record.impressions;
    }

    const installments: DayLog['installments'] = [];
    for (const installment of this.traffic.installments) {
      if (installment.scheduled_day === day && installment.outcome === 'cancelled') {
        installments.push({ installment_id: installment.installment_id, outcome: 'cancelled' });
      } else if (installment.scheduled_day === day && installment.outcome === 'postponed') {
        installments.push({ installment_id: installment.installment_id, outcome: 'postponed', to_date: calendar.dateOf(installment.actual_day!) });
      }
      if (installment.actual_day === day) installments.push({ installment_id: installment.installment_id, outcome: 'ran' });
    }
    this.state.log.push({
      day,
      date: calendar.dateOf(day),
      actual_impressions: supplyOf(tables.actual, [...tables.actual.keys()]),
      sold_impressions: sold,
      installments,
    });
  }

  private recordDay(booking: Booking, day: number, taken: CellCounts, target: number | undefined): BookingDay {
    let clicks = 0;
    let viewable = 0;
    let completed = 0;
    let conversions = 0;
    let impressions = 0;
    const cells: Array<[number, number]> = [];
    for (const [cell, count] of [...taken].sort((a, b) => a[0] - b[0])) {
      if (count <= 0) continue;
      const rates = this.traffic.rates(day, cell);
      impressions += count;
      clicks += count * rates.click_rate;
      viewable += count * rates.viewability;
      completed += count * rates.completion_rate;
      conversions += count * rates.click_rate * rates.conversion_rate;
      cells.push([cell, count]);
    }
    // Spend is the change in rounded cumulative spend, so the total is
    // always exact: 180,000 impressions at $14 CPM come to $2,520.00 however
    // they split across days, and a bid change mid-flight is priced from then.
    const exactBefore = booking.spend_exact ?? booking.delivered.spend;
    let exactAfter = exactBefore + (impressions * booking.cpm) / 1000;
    if (booking.kind === 'share_of_voice') exactAfter = Math.min(exactAfter, Math.max(exactBefore, booking.budget));
    booking.spend_exact = exactAfter;
    const spend = roundMoney(roundMoney(exactAfter) - booking.delivered.spend);
    const record: BookingDay = {
      day,
      date: this.traffic.calendar.dateOf(day),
      impressions,
      spend,
      clicks: Math.round(clicks),
      viewable_impressions: Math.round(viewable),
      completed_views: Math.round(completed),
      conversions: Math.round(conversions),
      ...(target !== undefined && { target }),
      cells,
    };
    booking.days.push(record);
    addMetrics(booking.delivered, record);
    return record;
  }

  /** The days a booking still serves, inside the season. `today` has not run
   * yet, so a booking confirmed before it serves it too. */
  private servingDays(booking: Pick<Booking, 'confirmed_day' | 'flight_start_day' | 'flight_end_day'>, today: number): { first: number; last: number } {
    return {
      first: Math.max(today, booking.confirmed_day + 1, booking.flight_start_day, 0),
      last: Math.min(booking.flight_end_day, this.traffic.calendar.days),
    };
  }

  /** `F(cell, day)` as known on `asOfDay`, computed once per cell and day for the life of the reader. */
  private forecastReader(asOfDay: number): ForecastReader {
    const memo = new Map<number, number>();
    const stride = this.traffic.slots.length;
    return (day, cell) => {
      const key = day * stride + cell;
      let value = memo.get(key);
      if (value === undefined) {
        value = this.traffic.forecastValue(day, cell, asOfDay);
        memo.set(key, value);
      }
      return value;
    };
  }

  /** The forecast one call reads: `forecastOf` when given, else the default as known on `today`. Only the default lets the base run be memoised. */
  private readerFor(today: number, forecastOf?: ForecastReader): Reader {
    return forecastOf ? { read: forecastOf, memoise: false } : { read: this.forecastReader(today), memoise: true };
  }

  private cellSetOf(cells: readonly number[]): ReadonlySet<number> {
    let set = this.cellSets.get(cells);
    if (!set) this.cellSets.set(cells, (set = new Set(cells)));
    return set;
  }

  private candidateOf(
    cells: readonly number[],
    first: number,
    last: number,
    end: number,
    kind: DryBooking['kind'],
    share = 0,
    id = CANDIDATE_ID,
  ): DryBooking {
    // Built fresh each time: the candidate's id is shared by every check, so the per-booking cache cannot hold its prefix.
    const expected = kind === 'goal' ? this.expectedPrefixOf(cells) : undefined;
    return { id, kind, cells, cellSet: this.cellSetOf(cells), first, last, end, left: 0, expected, share, tail: true };
  }

  /**
   * The guaranteed bookings a dry run models, each entering on its first
   * serving day from `today`. Paused bookings count as active because they
   * may resume; canceled and auction bookings reserve nothing.
   */
  private dryBookings(today: number, except?: string): DryBooking[] {
    const books: DryBooking[] = [];
    for (const booking of this.state.bookings) {
      if (booking.status === 'canceled' || booking.kind === 'auction' || booking.booking_id === except) continue;
      const { first, last } = this.servingDays(booking, today);
      if (first >= last) continue;
      const cells = this.cellsFor(booking);
      const common = { id: booking.booking_id, cells, cellSet: this.cellSetOf(cells), first, last, end: booking.flight_end_day };
      if (booking.kind === 'share_of_voice') {
        books.push({ ...common, kind: 'share_of_voice', left: 0, share: booking.share_of_voice ?? 0 });
        continue;
      }
      const left = Math.max(0, (booking.impression_goal ?? 0) - booking.delivered.impressions);
      if (left > DRY_EPSILON) books.push({ ...common, kind: 'goal', left, expected: this.expectedTotals(booking), share: 0 });
    }
    return books;
  }

  /** `except`, or undefined when it names no booking a dry run models. */
  private liveExcept(except: string | undefined): string | undefined {
    if (except === undefined) return undefined;
    const live = this.state.bookings.some(booking => booking.booking_id === except
      && booking.status !== 'canceled' && booking.kind !== 'auction');
    return live ? except : undefined;
  }

  private interacts(a: DryBooking, b: DryBooking): boolean {
    return a.first < b.last && b.first < a.last && a.cells.some(cell => b.cellSet.has(cell));
  }

  /** Indices of the bookings that share a cell and a day with `candidate`, directly or through each other. */
  private componentOf(books: readonly DryBooking[], candidate: DryBooking): number[] {
    const members: number[] = [];
    const seen = new Set<number>();
    const queue: DryBooking[] = [candidate];
    for (let next = queue.pop(); next; next = queue.pop()) {
      books.forEach((book, index) => {
        if (seen.has(index) || !this.interacts(next!, book)) return;
        seen.add(index);
        members.push(index);
        queue.push(book);
      });
    }
    return members.sort((a, b) => a - b);
  }

  /**
   * The cells a dry run allocates over, as units with their forecast for each
   * day in [from, to). With `merge` set, cells held by exactly the same
   * bookings form one unit: every allocation takes the same share of each of
   * them, so a unit behaves as the sum of its cells. Without it each cell is
   * its own unit, which is what a run that reports per-cell takes needs.
   */
  private unitsFor(
    books: readonly DryBooking[],
    from: number,
    to: number,
    read: ForecastReader,
    merge: boolean,
  ): { unitsOf: number[][]; values: Map<number, Float64Array> } {
    const span = Math.max(0, to - from);
    const unitsOf: number[][] = books.map(() => []);
    const values = new Map<number, Float64Array>();
    const sum = (cells: readonly number[]) => {
      const perDay = new Float64Array(span);
      for (let d = 0; d < span; d++) for (const cell of cells) perDay[d] += read(from + d, cell);
      return perDay;
    };
    if (!merge) {
      books.forEach((book, i) => {
        unitsOf[i] = book.cells as number[];
        for (const cell of book.cells) if (!values.has(cell)) values.set(cell, sum([cell]));
      });
      return { unitsOf, values };
    }
    const owners = new Map<number, number[]>();
    books.forEach((book, i) => {
      for (const cell of book.cells) {
        const held = owners.get(cell);
        if (held) held.push(i);
        else owners.set(cell, [i]);
      }
    });
    const bySignature = new Map<string, { unit: number; cells: number[] }>();
    for (const [cell, held] of owners) {
      const signature = held.join(',');
      let group = bySignature.get(signature);
      if (!group) {
        group = { unit: bySignature.size, cells: [] };
        bySignature.set(signature, group);
        for (const i of held) unitsOf[i].push(group.unit);
      }
      group.cells.push(cell);
    }
    for (const { unit, cells } of bySignature.values()) values.set(unit, sum(cells));
    return { unitsOf, values };
  }

  /**
   * A dry run of `runDay` over the days [from, to), into the supply `read`
   * returns: share-of-voice bookings take their share of each cell by id, then
   * goal bookings are served by the high-water mark method, each paced to
   * expected traffic as `runDay` paces it.
   * Impressions are fractional, budgets are not applied, and nothing is written to the state or the traffic tables.
   * `left` holds each goal's undelivered impressions and is updated.
   */
  private simulate(
    books: readonly DryBooking[],
    from: number,
    to: number,
    left: Float64Array,
    read: ForecastReader,
    hooks: { snapshots?: Float64Array[]; observe?: (index: number, day: number, cell: number, count: number) => void } = {},
  ): void {
    this.dryRunCount += 1;
    const indexOf = new Map(books.map((book, index) => [book.id, index]));
    const indices = books.map((_, index) => index);
    const shares = indices.filter(i => books[i].kind === 'share_of_voice').sort((a, b) => (books[a].id < books[b].id ? -1 : 1));
    const goals = indices.filter(i => books[i].kind === 'goal');
    const { unitsOf, values } = this.unitsFor(books, from, to, read, hooks.observe === undefined);
    const { calendar } = this.traffic;
    const expected = (i: number, day: number) => books[i].expected![day];
    for (let day = from; day < to; day++) {
      hooks.snapshots?.push(left.slice());
      const sharesToday = shares.filter(i => books[i].first <= day && day < books[i].last);
      const goalsToday = goals.filter(i => books[i].first <= day && day < books[i].last && left[i] > DRY_EPSILON);
      if (sharesToday.length + goalsToday.length === 0) continue;
      const actual: CellCounts = new Map();
      for (const i of [...sharesToday, ...goalsToday]) {
        for (const unit of unitsOf[i]) if (!actual.has(unit)) actual.set(unit, values.get(unit)![day - from]);
      }
      const remaining: CellCounts = new Map(actual);
      for (const i of sharesToday) {
        const taken = takeFractionalShareOfVoice(remaining, actual, unitsOf[i], books[i].share);
        if (hooks.observe) for (const [cell, count] of taken) hooks.observe(i, day, cell, count);
      }
      const requests: FractionalGoalRequest[] = goalsToday.map(i => ({
        id: books[i].id,
        cells: unitsOf[i],
        last: books[i].tail === true,
        target: pacedTarget(left[i], expected(i, day + 1) - expected(i, day), expected(i, Math.min(books[i].end, calendar.days)) - expected(i, day), books[i].end - day),
      }));
      for (const [id, taken] of allocateFractionalHighWaterMark(remaining, requests)) {
        const index = indexOf.get(id)!;
        let total = 0;
        for (const [cell, count] of taken) {
          total += count;
          hooks.observe?.(index, day, cell, count);
        }
        left[index] = Math.max(0, left[index] - total);
      }
    }
  }

  /**
   * The dry run with no candidate, memoised per (today, processed day,
   * booking revision, `except`). Only the default reader is memoised: a
   * caller-supplied `forecastOf` is never.
   */
  private baseRun(today: number, except: string | undefined, reader: Reader): BaseRun {
    const stamp = `${today}|${this.state.processed_through_day}|${this.revision}`;
    if (stamp !== this.memoStamp) {
      this.memo.clear();
      this.memoStamp = stamp;
    }
    const key = except ?? '';
    const cached = reader.memoise ? this.memo.get(key) : undefined;
    if (cached) return cached;
    const books = this.dryBookings(today, except);
    const from = books.length > 0 ? Math.min(...books.map(book => book.first)) : today;
    const to = books.length > 0 ? Math.max(...books.map(book => book.last)) : today;
    const left = Float64Array.from(books, book => book.left);
    const snapshots: Float64Array[] = [];
    const held = new Map<number, CellCounts>();
    this.simulate(books, from, to, left, reader.read, {
      snapshots,
      observe: (_index, day, cell, count) => {
        let counts = held.get(day);
        if (!counts) held.set(day, (counts = new Map()));
        counts.set(cell, (counts.get(cell) ?? 0) + count);
      },
    });
    const run: BaseRun = { books, from, to, left, snapshots, held };
    if (reader.memoise) this.memo.set(key, run);
    return run;
  }

  /** What booking `index` still has to deliver at the start of `day` in the base run. */
  private leftAt(run: BaseRun, day: number, index: number): number {
    if (day <= run.from) return run.books[index].left;
    if (day >= run.to) return run.left[index];
    return run.snapshots[day - run.from][index];
  }

  /** The base run's bookings that can meet the candidate, with what each still has to deliver on its first day. */
  private neighbours(run: BaseRun, candidate: DryBooking): { members: DryBooking[]; indices: number[]; start: number[] } {
    const indices = this.componentOf(run.books, candidate);
    return {
      members: indices.map(index => run.books[index]),
      indices,
      start: indices.map(index => this.leftAt(run, candidate.first, index)),
    };
  }

  /** Runs the members and the candidate from the candidate's first day, the candidate needing `x` impressions. */
  private runWith(
    members: readonly DryBooking[],
    start: readonly number[],
    candidate: DryBooking,
    x: number,
    read: ForecastReader,
    observe?: (index: number, day: number, cell: number, count: number) => void,
  ): Float64Array {
    const books = [...members, candidate];
    this.bookingsRun += books.length;
    const left = Float64Array.from([...start, x]);
    this.simulate(books, candidate.first, Math.max(...books.map(book => book.last)), left, read, { observe });
    return left;
  }

  /** Forecast impressions over the candidate's cells and days, and the part share-of-voice bookings leave. */
  private forecastOver(candidate: DryBooking, read: ForecastReader, shares: readonly DryBooking[]): { gross: number; net: number } {
    let gross = 0;
    let net = 0;
    for (let day = candidate.first; day < candidate.last; day++) {
      for (const cell of candidate.cells) {
        const value = read(day, cell);
        if (value <= 0) continue;
        let share = 0;
        for (const other of shares) if (other.first <= day && day < other.last && other.cellSet.has(cell)) share += other.share;
        gross += value;
        net += value * (1 - Math.min(1, share));
      }
    }
    return { gross, net };
  }

  /**
   * Whether the candidate can hold `x` impressions: its dry-run shortfall
   * grows by no more than the tolerance over its shortfall running alone with
   * the share-of-voice bookings, and no other goal booking's shortfall grows
   * by more than the tolerance over the base run's.
   */
  private fits(
    run: BaseRun,
    near: { members: DryBooking[]; indices: number[]; start: number[] },
    candidate: DryBooking,
    x: number,
    read: ForecastReader,
  ): boolean {
    const { members, indices, start } = near;
    const left = this.runWith(members, start, candidate, x, read);
    for (let k = 0; k < members.length; k++) {
      if (members[k].kind === 'goal' && left[k] - run.left[indices[k]] > DRY_TOLERANCE) return false;
    }
    const keep = members.map((_, k) => k).filter(k => members[k].kind === 'share_of_voice');
    const alone = this.runWith(keep.map(k => members[k]), keep.map(() => 0), candidate, x, read);
    return left[members.length] - alone[keep.length] <= DRY_TOLERANCE;
  }

  /**
   * The most a goal booking with the candidate's cells and days can still
   * hold: the largest whole number of impressions such that, entering with
   * that many left, it delivers as well as it would alone and leaves every
   * other goal as well served as before. `except` is a booking to leave out.
   * With no booking sharing a cell and day it is the forecast the
   * share-of-voice bookings leave. Otherwise it is found by bisection, at
   * most 32 dry runs from the candidate's first day. It stops when the bracket is
   * within max(1, 0.05% of the upper end) and returns the feasible lower end,
   * so the number is conservative by at most that much.
   */
  private goalAvailable(candidate: DryBooking, today: number, except: string | undefined, reader: Reader): GoalAvailability {
    const leave = this.liveExcept(except);
    const direct = this.dryBookings(today, leave).filter(book => this.interacts(candidate, book));
    const { gross, net } = this.forecastOver(candidate, reader.read, direct.filter(book => book.kind === 'share_of_voice'));
    const upper = Math.round(net);
    if (upper <= 0) return { available: 0, gross, held: gross - Math.max(0, net), takes: () => {}, untouched: false };
    const takesWith = (near: { members: DryBooking[]; start: number[] }) =>
      (x: number, onTake: (day: number, cell: number, count: number) => void) => {
        if (x <= 0) return;
        const last = near.members.length;
        this.runWith(near.members, near.start, candidate, x, reader.read, (index, day, cell, count) => {
          if (index === last) onTake(day, cell, count);
        });
      };
    if (direct.length === 0) {
      const forecastTakes = (_x: number, onTake: (day: number, cell: number, count: number) => void) => {
        for (let day = candidate.first; day < candidate.last; day++) {
          for (const cell of candidate.cells) {
            const value = reader.read(day, cell);
            if (value > 0) onTake(day, cell, value);
          }
        }
      };
      return { available: upper, gross, held: gross - net, takes: forecastTakes, untouched: true };
    }
    const run = this.baseRun(today, leave, reader);
    const near = this.neighbours(run, candidate);
    let low = 0;
    let high = upper;
    let available = upper;
    if (!this.fits(run, near, candidate, upper, reader.read)) {
      for (let step = 0; step < MAX_BISECTION_STEPS && high - low > bisectionTolerance(high); step++) {
        const mid = Math.floor((low + high) / 2);
        if (this.fits(run, near, candidate, mid, reader.read)) low = mid;
        else high = mid;
      }
      available = low;
    }
    return { available, gross, held: gross - (available === upper ? net : available), takes: takesWith(near), untouched: false };
  }

  /** Whether a share-of-voice booking would leave a goal booking worse served than it is. */
  private harmsGoals(candidate: DryBooking, today: number, reader: Reader): boolean {
    const direct = this.componentOf(this.dryBookings(today), candidate);
    if (direct.length === 0) return false;
    const run = this.baseRun(today, undefined, reader);
    const near = this.neighbours(run, candidate);
    if (!near.members.some(book => book.kind === 'goal')) return false;
    const left = this.runWith(near.members, near.start, candidate, 0, reader.read);
    return near.members.some((book, k) => book.kind === 'goal' && left[k] - run.left[near.indices[k]] > DRY_TOLERANCE);
  }

  /**
   * Refuse a goal whose undelivered part is more than the booking can hold
   * over its serving days, net of every other guaranteed booking: the
   * `available` a forecast for the same cells and flight reports. A
   * guarantee the market cannot cover is refused, never capped.
   */
  private requireAvailable(
    booking: Booking,
    cells: readonly number[],
    goal: number,
    nowMs: number,
    forecastOf?: ForecastReader,
  ): void {
    const today = this.today(nowMs);
    const left = goal - booking.delivered.impressions;
    if (left <= 0) return;
    const { first, last } = this.servingDays(booking, today);
    const candidate = this.candidateOf(cells, first, last, booking.flight_end_day, 'goal');
    const { available } = this.goalAvailable(candidate, today, booking.booking_id, this.readerFor(today, forecastOf));
    if (left <= available) return;
    const delivered = booking.delivered.impressions > 0 ? ` (${left} still to deliver)` : '';
    throw new EngineError('INSUFFICIENT_INVENTORY',
      `Requested ${goal} impressions${delivered}, but only ${available} are available on ${booking.product_id} for this flight and targeting`);
  }

  /** Refuse a goal booking whose undelivered goal the market no longer
   * covers: the check `update` runs on a raise or a moved end. A host that
   * applies several changes as one runs it on each goal they raised or
   * shortened, once all are in. */
  requireGoalAvailable(bookingId: string, nowMs: number): void {
    this.advanceTo(nowMs);
    const booking = this.booking(bookingId);
    if (booking.kind !== 'goal' || booking.status === 'canceled') return;
    this.requireAvailable(booking, this.cellsFor(booking), booking.impression_goal ?? 0, nowMs);
  }

  /**
   * The impressions a goal booking could still hold over its serving days
   * with its own reservation set aside: what `update` weighs a goal against.
   * A booking holding `delivered + bookingAvailable` is the largest goal an
   * update accepts.
   *
   * @internal Exists for tests and measurement.
   */
  bookingAvailable(bookingId: string, nowMs: number): number {
    this.advanceTo(nowMs);
    const booking = this.booking(bookingId);
    if (booking.kind !== 'goal' || booking.status === 'canceled') {
      throw new EngineError('INVALID_BOOKING', `Booking ${bookingId} is not an active goal booking`);
    }
    const today = this.today(nowMs);
    const { first, last } = this.servingDays(booking, today);
    const candidate = this.candidateOf(this.cellsFor(booking), first, last, booking.flight_end_day, 'goal');
    return this.goalAvailable(candidate, today, booking.booking_id, this.readerFor(today)).available;
  }

  /**
   * Dry runs run so far: a measure of what the checks and forecasts cost.
   * @internal Exists for tests and measurement.
   */
  get dryRuns(): number {
    return this.dryRunCount;
  }

  /**
   * Bookings simulated across the candidate dry runs so far: what component pruning keeps small.
   * @internal Exists for tests and measurement.
   */
  get candidateBookingsRun(): number {
    return this.bookingsRun;
  }

  /**
   * Variance of the log error, `ln(actual / forecast)`, over `cells` and the days `first` to `last`
   * (exclusive), as seen on `asOfDay`: the variance of the log of actual over
   * forecast traffic for the whole flight, a first-order approximation that
   * ignores the clip. With `F` a cell-day's forecast, `a(c)` the cell's error
   * scale, `A_{p,t} = Σ_{c of p} F·a(c)` a publisher's weighted forecast on a
   * day and `W = Σ F`, the variance is the sum of these terms over `W²`:
   *  - the same-day shock of each publisher, an AR(1) in days with the
   *    season's `ar1_phi`. The forecast knows every shock before `asOfDay`,
   *    so the part left is `x(t) − φ^(h+1)·x(asOfDay−1)` at lead `h = t − asOfDay`:
   *    `Σ_t Σ_t' A_t A_t' φ^|t−t'| − (Σ_t A_t φ^(h+1))²` (the square is
   *    absent when no day has closed);
   *  - the revision shocks of each publisher that are not yet revealed, each
   *    revealed on day `r` and spread over the days `r+1` to `r+horizon` by an
   *    AR(1) in days: `τ⁻¹ Σ_r Σ_{t,t' ∈ T(r)} φ^|t−t'| A_t A_t'`;
   *  - the cell shock, independent per cell and day: `Σ (F·σ_cell)²`;
   *  - the daily wobble of the truth, shared by the cells of a placement:
   *    `Σ_t Σ_placement (Σ_{c in placement} F·σ_u)²`;
   *  - the audience error of each installment, fully correlated across its days;
   *  - the spread of the spike tails already started, taken as fully
   *    correlated across both cells and days (the conservative reading).
   * Publishers' shocks are independent of one another, so their terms add.
   * Each AR(1) sum is one pass: `carry` holds `Σ_{t' ≤ t} A_t' φ^(t−t')`.
   */
  private forecastLogVariance(
    cells: readonly number[],
    first: number,
    last: number,
    asOfDay: number,
    forecastOf: ForecastReader,
  ): number {
    const { traffic } = this;
    const { ar1_phi: phi, lead_time_tau_days: tau, lead_time_horizon_days: horizon } = this.seed.forecast_error;
    const dailySigma = this.seed.traffic.truth_daily_sigma ?? 0;
    const length = last - first;
    if (length <= 0) return 0;
    const publisherIndex = new Map<string, number>();
    const cellPublisher = cells.map(cell => {
      const id = traffic.publisherOf(cell);
      let index = publisherIndex.get(id);
      if (index === undefined) publisherIndex.set(id, (index = publisherIndex.size));
      return index;
    });
    const publishers = publisherIndex.size;
    const weighted = new Float64Array(length * publishers);
    const audience = new Map<string, number>();
    let total = 0;
    let perCell = 0;
    let wobble = 0;
    let spiked = 0;
    for (let day = first; day < last; day++) {
      const row = (day - first) * publishers;
      const byPlacement = new Map<string, number>();
      let daySpiked = 0;
      cells.forEach((cell, i) => {
        const value = forecastOf(day, cell);
        if (value === 0) return;
        total += value;
        weighted[row + cellPublisher[i]] += value * traffic.publisherScale(cell);
        perCell += (value * traffic.cellSigma(day, cell)) ** 2;
        daySpiked += value * traffic.spikeTailSpread(day, cell, asOfDay);
        if (dailySigma > 0) {
          const placement = traffic.placementOf(cell);
          byPlacement.set(placement, (byPlacement.get(placement) ?? 0) + value);
        }
        const installmentId = traffic.slots[cell].installment_id;
        if (installmentId !== undefined) audience.set(installmentId, (audience.get(installmentId) ?? 0) + value);
      });
      for (const value of byPlacement.values()) wobble += (value * dailySigma) ** 2;
      spiked += daySpiked;
    }
    if (total === 0) return 0;

    let sameDay = 0;
    for (let p = 0; p < publishers; p++) {
      let carry = 0;
      let sum = 0;
      let known = 0;
      let decay = phi ** (first - asOfDay + 1);
      for (let t = 0; t < length; t++) {
        const value = weighted[t * publishers + p];
        sum += value * (value + 2 * phi * carry);
        carry = phi * carry + value;
        known += value * decay;
        decay *= phi;
      }
      sameDay += sum - (asOfDay >= 1 ? known * known : 0);
    }

    let revisions = 0;
    for (let r = Math.max(asOfDay, first - horizon); r <= last - 2; r++) {
      const from = Math.max(first, r + 1);
      const to = Math.min(last - 1, r + horizon);
      for (let p = 0; p < publishers; p++) {
        let carry = 0;
        for (let t = from; t <= to; t++) {
          const value = weighted[(t - first) * publishers + p];
          revisions += value * (value + 2 * phi * carry);
          carry = phi * carry + value;
        }
      }
    }
    revisions /= tau;

    let audienceTerm = 0;
    for (const [installmentId, value] of audience) {
      const collection = traffic.collections.get(traffic.installment(installmentId)!.collection_id)!;
      audienceTerm += (value * collection.audience_sigma) ** 2;
    }
    return (sameDay + revisions + perCell + wobble + audienceTerm + spiked ** 2) / (total * total);
  }

  /**
   * Forecast traffic over the cells a product covers, minus what guaranteed
   * bookings hold. Days start no earlier than tomorrow, the first day a
   * booking made now could serve. Forecasts are read as of today, so the same
   * flight narrows and shifts as the clock moves.
   *
   * A goal product's `available` is what a goal booking could hold, the same
   * number a booking is checked against, and `free` is the candidate's own
   * per-cell take at that goal. A share-of-voice or auction product's is the
   * forecast less what a dry run of the bookings in place takes; a
   * share-of-voice product shows 0 when a booking would leave a goal short.
   *
   * @param forecastOf @internal Reads the forecast instead of the default, as known on today. Never memoised.
   */
  forecast(query: ForecastQuery, nowMs: number, forecastOf?: ForecastReader): ForecastResult {
    this.advanceTo(nowMs);
    const product = this.traffic.requireProduct(query.product_id);
    const { start, end } = this.flightDays(query.flight_start_ms, query.flight_end_ms);
    const cells = this.traffic.eligibleCells(product.product_id, query.targeting);
    const today = this.today(nowMs);
    const reader = this.readerFor(today, forecastOf);
    // The days a booking confirmed now would serve.
    const { first, last } = this.servingDays({ confirmed_day: today, flight_start_day: start, flight_end_day: end }, today);
    const goalProduct = product.delivery_type === 'guaranteed' && !product.share_of_voice;
    const candidate = this.candidateOf(cells, first, last, end, goalProduct ? 'goal' : 'share_of_voice', product.share_of_voice ?? 0);
    let clicks = 0;
    let viewable = 0;
    let completed = 0;
    let conversions = 0;
    let freeTotal = 0;
    let untouched = false;
    const expectedRates = new Map<number, CellRates>();
    const tally = (_day: number, cell: number, free: number) => {
      let rates = expectedRates.get(cell);
      if (!rates) expectedRates.set(cell, (rates = this.traffic.expectedRates(cell)));
      clicks += free * rates.click_rate;
      viewable += free * rates.viewability;
      completed += free * rates.completion_rate;
      conversions += free * rates.click_rate * rates.conversion_rate;
      freeTotal += free;
    };
    let capacity = 0;
    let booked = 0;
    let available = 0;
    let refused = false;
    if (goalProduct) {
      const goal = this.goalAvailable(candidate, today, undefined, reader);
      capacity = goal.gross;
      available = goal.available;
      booked = Math.max(0, goal.held);
      goal.takes(available, tally);
      untouched = goal.untouched;
    } else {
      refused = product.share_of_voice !== undefined && this.harmsGoals(candidate, today, reader);
      const held = this.baseRun(today, undefined, reader).held;
      let taken = 0;
      for (let day = first; day < last; day++) {
        for (const cell of cells) {
          const value = reader.read(day, cell);
          if (value === 0) continue;
          const part = Math.min(value, held.get(day)?.get(cell) ?? 0);
          capacity += value;
          taken += part;
          if (!refused && value > part) tally(day, cell, value - part);
        }
      }
      available = refused ? 0 : Math.max(0, Math.round(capacity - taken));
      booked = refused ? capacity : taken;
    }
    const cpm = product.delivery_type === 'guaranteed' ? query.rate_cpm ?? product.rate_cpm : query.bid_cpm;
    const affordable = query.budget !== undefined && cpm ? Math.floor((query.budget * 1000) / cpm) : Number.POSITIVE_INFINITY;
    const impressions = Math.min(available, affordable);
    const scale = goalProduct && !untouched ? (freeTotal > 0 ? impressions / freeTotal : 0) : (available > 0 ? impressions / available : 0);
    const sigma = Math.sqrt(this.forecastLogVariance(cells, first, last, today, reader.read));
    const halfWidth = Math.min(RANGE_Z * sigma, 2 * this.seed.forecast_error.log_clip);
    const bound = (factor: number) => (refused ? 0 : Math.round(Math.min(affordable, Math.max(0, capacity * factor - booked))));
    return {
      product_id: product.product_id,
      currency: this.seed.currency,
      start_day: first,
      end_day: Math.max(first, end),
      capacity: Math.round(capacity),
      booked: Math.round(booked),
      available,
      impressions,
      low: bound(Math.exp(-halfWidth)),
      high: bound(Math.exp(halfWidth)),
      spend: cpm ? roundMoney((impressions * cpm) / 1000) : 0,
      clicks: Math.round(clicks * scale),
      viewable_impressions: Math.round(viewable * scale),
      completed_views: Math.round(completed * scale),
      conversions: Math.round(conversions * scale),
      ...(cpm !== undefined && { cpm }),
      sigma,
    };
  }

  /**
   * Delivery as reported at `nowMs`: each day is released once it has
   * closed and the product's reporting lag has passed.
   */
  delivery(bookingId: string, nowMs: number): DeliveryReport {
    this.advanceTo(nowMs);
    const booking = this.booking(bookingId);
    const lag = this.traffic.reportingLagDays(booking.product_id);
    const today = this.today(nowMs);
    const days = booking.days.filter(record => record.day + lag < today);
    const totals = zeroMetrics();
    for (const record of days) addMetrics(totals, record);
    const { days: _all, ...summary } = booking;
    const reportedThrough = days.length > 0 ? days[days.length - 1].day : undefined;
    let expected: number | undefined;
    if (booking.kind === 'goal' && reportedThrough !== undefined) {
      const firstServe = Math.max(booking.confirmed_day + 1, booking.flight_start_day);
      const span = booking.flight_end_day - firstServe;
      const prefix = this.expectedTotals(booking);
      const end = Math.min(booking.flight_end_day, this.traffic.calendar.days);
      const from = Math.min(Math.max(0, firstServe), end);
      const planned = prefix[end] - prefix[from];
      const through = prefix[Math.min(reportedThrough + 1, end)] - prefix[from];
      const fraction = planned > 0 ? through / planned : (reportedThrough + 1 - firstServe) / Math.max(1, span);
      expected = Math.round((booking.impression_goal ?? 0) * Math.min(1, fraction));
    }
    return {
      booking: structuredClone(summary),
      ...(reportedThrough !== undefined && { reported_through_day: reportedThrough }),
      totals,
      days: structuredClone(days),
      ...(expected !== undefined && { expected_impressions: expected }),
    };
  }

  /** What is known about each installment at `nowMs`: an outcome is known
   * from its scheduled day. */
  installmentStatuses(nowMs: number, productId?: string): InstallmentStatus[] {
    this.advanceTo(nowMs);
    const today = this.today(nowMs);
    const { calendar } = this.traffic;
    const installments = productId ? this.traffic.productInstallments(productId) : this.traffic.installments;
    return installments.map(installment => {
      const known = installment.scheduled_day <= today;
      let state: InstallmentState = 'scheduled';
      if (known && installment.outcome === 'cancelled') state = 'cancelled';
      else if (installment.actual_day !== undefined && installment.actual_day <= today) state = 'published';
      else if (known && installment.outcome === 'postponed') state = 'postponed';
      return {
        ...installment,
        state,
        ...(known && installment.outcome === 'postponed' && { rescheduled_date: calendar.dateOf(installment.actual_day!) }),
      };
    });
  }
}
