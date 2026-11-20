/**
 * The season's traffic, generated once from the seed: for every cell (one
 * day × region × property × placement × audience profile, and installment
 * for content placements), the impressions expected and the impressions that
 * actually arrive. Actual differs from expected by a daily wobble, spikes and
 * scripted events. A forecast is the expected traffic seen through a
 * lead-time error: it is revised as its as-of day approaches and knows
 * neither scripted events nor a spike before it starts. Every draw is a pure
 * function of its key, precomputed here so a forecast is arithmetic only.
 * Rates per cell are drawn on demand from the same keys.
 */
import type { CellCounts } from './allocation.js';
import { SeasonCalendar } from './calendar.js';
import { audienceCurve } from './curves.js';
import { EngineError } from './errors.js';
import { standardNormal, uniform } from './rng.js';
import type {
  BookingTargeting,
  CellRates,
  CellSlot,
  Installment,
  InstallmentChannel,
  SeasonSeed,
  SeedAudienceProfile,
  SeedCollection,
  SeedPlacement,
  SeedProduct,
  SeedProperty,
  SeedRegion,
  TrafficMultiplierEvent,
} from './types.js';

const INSTALLMENT_CHANNELS: readonly string[] = ['podcast', 'newsletter', 'creator_video'];
const SHARE_TOLERANCE = 1e-6;
/** Per-cell spread of the drawn rates around the placement's base rates. */
const RATE_SIGMA = 0.05;

export function isInstallmentChannel(channel: string): channel is InstallmentChannel {
  return INSTALLMENT_CHANNELS.includes(channel);
}

export interface DayTables {
  /** Expected impressions per cell: the unrounded base traffic, rounded. */
  expected: CellCounts;
  /** Impressions that actually arrive per cell. */
  actual: CellCounts;
}

/** A spike in a publisher's traffic: a surge or slump that decays over days. */
interface Spike {
  day: number;
  size: number;
  decay: number;
  /** The decay the forecaster believes in once the spike has started. */
  estimatedDecay: number;
}

function invalid(message: string): never {
  throw new EngineError('INVALID_SEED', message);
}

function indexBy<T, K extends keyof T>(items: readonly T[], key: K, label: string): Map<T[K], T> {
  const map = new Map<T[K], T>();
  for (const item of items) {
    if (map.has(item[key])) invalid(`Duplicate ${label} ${String(item[key])}`);
    map.set(item[key], item);
  }
  return map;
}

function checkShares(items: ReadonlyArray<{ share: number }>, label: string): void {
  if (items.length === 0) invalid(`A seed needs at least one ${label}`);
  if (items.some(item => !(item.share > 0))) invalid(`Every ${label} share must be positive`);
  const total = items.reduce((sum, item) => sum + item.share, 0);
  if (Math.abs(total - 1) > SHARE_TOLERANCE) invalid(`${label} shares sum to ${total}, not 1`);
}

/** The first problem with the traffic shape and forecast error of a seed, if any.
 * Each check is positive, so a missing or NaN value fails instead of producing NaN forecasts. */
function forecastErrorProblem(seed: SeasonSeed): string | undefined {
  const { traffic, forecast_error: error } = seed;
  if (!error || typeof error !== 'object') return 'forecast_error is required';
  const finite = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
  const wholeAtLeastZero = (x: unknown) => typeof x === 'number' && Number.isInteger(x) && x >= 0;
  if (traffic.month_shape !== undefined) {
    const shape = traffic.month_shape;
    if (!Array.isArray(shape) || shape.length !== 12 || !shape.every(x => finite(x) && x > 0)) {
      return 'traffic.month_shape needs twelve values above 0, January first';
    }
  }
  if (traffic.truth_daily_sigma !== undefined && !(finite(traffic.truth_daily_sigma) && traffic.truth_daily_sigma >= 0)) {
    return 'traffic.truth_daily_sigma must be 0 or more';
  }
  const sigmas = error.channel_sigma;
  if (!sigmas || typeof sigmas !== 'object') return 'forecast_error.channel_sigma is required';
  for (const [name, sigma] of Object.entries(sigmas)) {
    if (!(finite(sigma) && sigma >= 0)) return `forecast_error.channel_sigma.${name} must be 0 or more`;
  }
  for (const placement of seed.placements) {
    if (sigmas[placement.channel] === undefined) {
      return `forecast_error.channel_sigma.${placement.channel} is needed by placement ${placement.placement_id}`;
    }
  }
  if (!(finite(error.ar1_phi) && error.ar1_phi >= 0 && error.ar1_phi < 1)) return 'forecast_error.ar1_phi must be 0 or more and below 1';
  if (!(finite(error.lead_time_tau_days) && error.lead_time_tau_days > 0)) return 'forecast_error.lead_time_tau_days must be above 0';
  if (!wholeAtLeastZero(error.lead_time_horizon_days)) return 'forecast_error.lead_time_horizon_days must be a whole number, 0 or more';
  if (!(finite(error.log_clip) && error.log_clip > 0)) return 'forecast_error.log_clip must be above 0';
  if (!(finite(error.cell_kappa) && error.cell_kappa >= 0)) return 'forecast_error.cell_kappa must be 0 or more';
  if (!(finite(error.cell_min_impressions) && error.cell_min_impressions >= 1)) return 'forecast_error.cell_min_impressions must be 1 or more';
  const publisherIds = new Set(seed.properties.map(p => p.publisher_id ?? p.property_id));
  const listed = new Set<string>();
  for (const [i, entry] of (error.publishers ?? []).entries()) {
    const at = `forecast_error.publishers[${i}]`;
    if (!publisherIds.has(entry.publisher_id)) return `${at}.publisher_id is unknown publisher ${entry.publisher_id}`;
    if (listed.has(entry.publisher_id)) return `${at}.publisher_id lists publisher ${entry.publisher_id} twice`;
    listed.add(entry.publisher_id);
    if (!finite(entry.bias)) return `${at}.bias must be a number`;
    if (entry.bias_sigma !== undefined && !(finite(entry.bias_sigma) && entry.bias_sigma >= 0)) return `${at}.bias_sigma must be 0 or more`;
    if (entry.error_scale !== undefined && !(finite(entry.error_scale) && entry.error_scale >= 0)) return `${at}.error_scale must be 0 or more`;
  }
  const spikes = error.spikes;
  if (spikes !== undefined) {
    if (!wholeAtLeastZero(spikes.per_publisher_min)) return 'forecast_error.spikes.per_publisher_min must be a whole number, 0 or more';
    if (!wholeAtLeastZero(spikes.per_publisher_max) || spikes.per_publisher_max < spikes.per_publisher_min) {
      return 'forecast_error.spikes.per_publisher_max must be a whole number, at least per_publisher_min';
    }
    if (spikes.per_publisher_max > seed.days) return 'forecast_error.spikes.per_publisher_max must be at most the number of days in the season';
    if (!(finite(spikes.size_min) && spikes.size_min > -1)) return 'forecast_error.spikes.size_min must be above -1';
    if (!(finite(spikes.size_max) && spikes.size_max >= spikes.size_min)) return 'forecast_error.spikes.size_max must be at least size_min';
    if (!(finite(spikes.decay) && spikes.decay >= 0 && spikes.decay < 1)) return 'forecast_error.spikes.decay must be 0 or more and below 1';
    if (!(finite(spikes.decay_forecast_sigma) && spikes.decay_forecast_sigma >= 0)) return 'forecast_error.spikes.decay_forecast_sigma must be 0 or more';
  }
  return undefined;
}

function addCount(counts: CellCounts, cell: number, value: number): void {
  if (value > 0) counts.set(cell, (counts.get(cell) ?? 0) + value);
}

function roundCounts(raw: Map<number, number>): CellCounts {
  const rounded: CellCounts = new Map();
  for (const [cell, value] of raw) {
    const whole = Math.round(value);
    if (whole > 0) rounded.set(cell, whole);
  }
  return rounded;
}

export class SeasonTraffic {
  readonly calendar: SeasonCalendar;
  readonly slots: CellSlot[] = [];
  readonly installments: Installment[] = [];
  readonly regions: Map<string, SeedRegion>;
  readonly profiles: Map<string, SeedAudienceProfile>;
  readonly placements: Map<string, SeedPlacement>;
  readonly products: Map<string, SeedProduct>;
  readonly collections: Map<string, SeedCollection>;
  private readonly segmentProfiles = new Map<string, Set<string>>();
  private readonly installmentsById = new Map<string, Installment>();
  private readonly tables: DayTables[] = [];
  private readonly slotKeys: string[] = [];
  private trafficEvents: Array<{ event: TrafficMultiplierEvent; first: number; last: number }> = [];
  /** Unrounded expected impressions `B` per cell-day; a cell with none is absent. */
  private readonly expectedRaw: Array<Map<number, number>> = [];
  /** Per cell-day, the log error that depends on the cell alone and not on the as-of day. */
  private readonly cellNoise: Array<Map<number, number>> = [];
  /** `e^u` per placement-day: the daily truth variation that forecasts also carry. */
  private readonly dailyFactor: Float64Array[] = [];
  private readonly placementIndex = new Map<string, number>();
  private readonly slotPlacement: number[] = [];
  private readonly publisherIds: string[] = [];
  private readonly slotPublisher: number[] = [];
  /** `σ_ch · k_p` per slot. */
  private slotErrorScale!: Float64Array;
  /** `b_p` per publisher: the log offset of its forecasts. */
  private publisherBias!: Float64Array;
  /** `x_p` per publisher per day: the AR(1) error, unit variance. */
  private readonly publisherDraw: Float64Array[] = [];
  private readonly publisherSpikes: Spike[][] = [];

  constructor(readonly seed: SeasonSeed) {
    if (!seed.seed_id) invalid('seed_id is required');
    if (!Number.isInteger(seed.days) || seed.days <= 0 || seed.days > 1100) invalid('days must be an integer from 1 to 1100');
    this.calendar = new SeasonCalendar(seed.start_date, seed.days);
    checkShares(seed.regions, 'region');
    checkShares(seed.audience_profiles, 'audience profile');
    if (seed.traffic.weekday_shape.length !== 7) invalid('traffic.weekday_shape needs seven values, Sunday first');
    const forecastProblem = forecastErrorProblem(seed);
    if (forecastProblem) invalid(forecastProblem);
    this.regions = indexBy(seed.regions, 'region_id', 'region');
    this.profiles = indexBy(seed.audience_profiles, 'profile_id', 'audience profile');
    const properties = indexBy(seed.properties, 'property_id', 'property');
    this.placements = indexBy(seed.placements, 'placement_id', 'placement');
    this.products = indexBy(seed.products, 'product_id', 'product');
    this.collections = indexBy(seed.collections ?? [], 'collection_id', 'collection');
    for (const segment of seed.segments) {
      if (this.segmentProfiles.has(segment.segment_id)) invalid(`Duplicate segment ${segment.segment_id}`);
      for (const id of segment.profile_ids) if (!this.profiles.has(id)) invalid(`Segment ${segment.segment_id} names unknown profile ${id}`);
      this.segmentProfiles.set(segment.segment_id, new Set(segment.profile_ids));
    }
    for (const placement of seed.placements) {
      for (const id of placement.property_ids) if (!properties.has(id)) invalid(`Placement ${placement.placement_id} names unknown property ${id}`);
      const installment = isInstallmentChannel(placement.channel);
      if (!installment && !(placement.base_daily_impressions! > 0)) {
        invalid(`Placement ${placement.placement_id} needs base_daily_impressions`);
      }
    }
    for (const product of seed.products) {
      if (product.placement_ids.length === 0) invalid(`Product ${product.product_id} has no placements`);
      for (const id of product.placement_ids) if (!this.placements.has(id)) invalid(`Product ${product.product_id} names unknown placement ${id}`);
      if (product.segment_id && !this.segmentProfiles.has(product.segment_id)) invalid(`Product ${product.product_id} names unknown segment ${product.segment_id}`);
      for (const id of product.collection_ids ?? []) if (!this.collections.has(id)) invalid(`Product ${product.product_id} names unknown collection ${id}`);
      if (product.delivery_type === 'guaranteed' && !(product.rate_cpm! > 0)) invalid(`Guaranteed product ${product.product_id} needs rate_cpm`);
    }
    this.buildInstallments();
    this.buildSlots();
    this.buildPublishers(properties);
    this.buildTables(properties);
  }

  private buildInstallments(): void {
    for (const collection of this.collections.values()) {
      const placement = this.placements.get(collection.placement_id);
      if (!placement) invalid(`Collection ${collection.collection_id} names unknown placement ${collection.placement_id}`);
      if (placement.channel !== collection.channel) invalid(`Collection ${collection.collection_id} is ${collection.channel} but its placement is ${placement.channel}`);
      if (!placement.property_ids.includes(collection.property_id)) invalid(`Collection ${collection.collection_id} property is not on its placement`);
      const first = this.calendar.dayOfDate(collection.first_date);
      for (let n = 1; n <= collection.count; n++) {
        const day = first + (n - 1) * collection.cadence_days;
        const installment: Installment = {
          installment_id: `${collection.collection_id}_${String(n).padStart(3, '0')}`,
          collection_id: collection.collection_id,
          sequence: n,
          scheduled_date: this.calendar.dateOf(day),
          scheduled_day: day,
          actual_day: day,
          outcome: 'runs',
          expected_audience: collection.expected_audience,
        };
        this.installments.push(installment);
        this.installmentsById.set(installment.installment_id, installment);
      }
    }
    for (const event of this.seed.events ?? []) {
      if (event.type === 'traffic_multiplier') {
        if (!(event.multiplier >= 0)) invalid(`Event ${event.event_id} needs a non-negative multiplier`);
        const first = this.calendar.dayOfDate(event.start_date);
        const last = this.calendar.dayOfDate(event.end_date);
        if (last < first) invalid(`Event ${event.event_id} ends before it starts`);
        this.trafficEvents.push({ event, first, last });
        continue;
      }
      const installment = this.installmentsById.get(event.installment_id);
      if (!installment) invalid(`Event ${event.event_id} names unknown installment ${event.installment_id}`);
      if (installment.outcome !== 'runs') invalid(`Installment ${event.installment_id} has two scripted events`);
      if (event.type === 'installment_cancelled') {
        installment.outcome = 'cancelled';
        delete installment.actual_day;
      } else {
        const to = this.calendar.dayOfDate(event.to_date);
        if (to <= installment.scheduled_day) invalid(`Event ${event.event_id} must postpone to a later date`);
        installment.outcome = 'postponed';
        installment.actual_day = to;
      }
    }
  }

  private buildSlots(): void {
    const push = (slot: Omit<CellSlot, 'index'>) => {
      this.slots.push({ index: this.slots.length, ...slot });
      this.slotKeys.push(`${slot.region_id}/${slot.property_id}/${slot.placement_id}/${slot.profile_id}/${slot.installment_id ?? ''}`);
    };
    for (const placement of this.placements.values()) {
      if (isInstallmentChannel(placement.channel)) {
        for (const installment of this.installments) {
          const collection = this.collections.get(installment.collection_id)!;
          if (collection.placement_id !== placement.placement_id) continue;
          for (const region of this.regions.values()) {
            for (const profile of this.profiles.values()) {
              push({
                region_id: region.region_id,
                property_id: collection.property_id,
                placement_id: placement.placement_id,
                profile_id: profile.profile_id,
                installment_id: installment.installment_id,
              });
            }
          }
        }
        continue;
      }
      for (const propertyId of placement.property_ids) {
        for (const region of this.regions.values()) {
          for (const profile of this.profiles.values()) {
            push({ region_id: region.region_id, property_id: propertyId, placement_id: placement.placement_id, profile_id: profile.profile_id });
          }
        }
      }
    }
  }

  private slotKey(slot: CellSlot): string {
    return this.slotKeys[slot.index];
  }

  private eventMultiplier(day: number, slot: CellSlot): number {
    let multiplier = 1;
    for (const { event, first, last } of this.trafficEvents) {
      if (day < first || day > last || !this.eventCovers(event, slot)) continue;
      multiplier *= event.multiplier;
    }
    return multiplier;
  }

  private eventCovers(event: TrafficMultiplierEvent, slot: CellSlot): boolean {
    return (!event.placement_ids || event.placement_ids.includes(slot.placement_id))
      && (!event.property_ids || event.property_ids.includes(slot.property_id))
      && (!event.region_ids || event.region_ids.includes(slot.region_id));
  }

  /** Publishers, their draws and spikes, and the error scale of every slot. */
  private buildPublishers(properties: Map<string, SeedProperty>): void {
    const { forecast_error: error } = this.seed;
    const rng = this.seed.rng_seed;
    const days = this.calendar.days;
    const configured = new Map((error.publishers ?? []).map(entry => [entry.publisher_id, entry]));
    const publisherOf = (propertyId: string) => {
      const property = properties.get(propertyId)!;
      return property.publisher_id ?? property.property_id;
    };
    const index = new Map<string, number>();
    for (const slot of this.slots) {
      const id = publisherOf(slot.property_id);
      if (!index.has(id)) {
        index.set(id, this.publisherIds.length);
        this.publisherIds.push(id);
      }
      this.slotPublisher.push(index.get(id)!);
    }
    this.publisherBias = new Float64Array(this.publisherIds.length);
    this.slotErrorScale = new Float64Array(this.slots.length);
    const phi = error.ar1_phi;
    const innovation = Math.sqrt(1 - phi * phi);
    this.publisherIds.forEach((id, p) => {
      const entry = configured.get(id);
      this.publisherBias[p] = -((entry?.bias ?? 0) + (entry?.bias_sigma ?? 0) * standardNormal(`${rng}|fe-bias|${id}`));
      const draws = new Float64Array(days);
      for (let day = 0; day < days; day++) {
        const fresh = standardNormal(`${rng}|fe-x|${id}|${day}`);
        draws[day] = day === 0 ? fresh : phi * draws[day - 1] + innovation * fresh;
      }
      this.publisherDraw.push(draws);
      this.publisherSpikes.push(this.drawSpikes(id));
    });
    for (const slot of this.slots) {
      const placement = this.placements.get(slot.placement_id)!;
      const entry = configured.get(publisherOf(slot.property_id));
      this.slotErrorScale[slot.index] = error.channel_sigma[placement.channel]! * (entry?.error_scale ?? 1);
    }
  }

  private drawSpikes(publisherId: string): Spike[] {
    const spikes = this.seed.forecast_error.spikes;
    if (!spikes) return [];
    const rng = this.seed.rng_seed;
    const span = spikes.per_publisher_max - spikes.per_publisher_min + 1;
    const count = spikes.per_publisher_min + Math.floor(uniform(`${rng}|spike-count|${publisherId}`) * span);
    const drawn: Spike[] = [];
    for (let i = 0; i < count; i++) {
      drawn.push({
        day: Math.floor(uniform(`${rng}|spike-day|${publisherId}|${i}`) * this.calendar.days),
        size: spikes.size_min + uniform(`${rng}|spike-size|${publisherId}|${i}`) * (spikes.size_max - spikes.size_min),
        decay: spikes.decay,
        estimatedDecay: spikes.decay ** Math.exp(spikes.decay_forecast_sigma * standardNormal(`${rng}|spike-decay-est|${publisherId}|${i}`)),
      });
    }
    return drawn;
  }

  /** The true multiplier of a publisher's spikes on a day. */
  private trueTail(publisher: number, day: number): number {
    let tail = 1;
    for (const spike of this.publisherSpikes[publisher]) {
      if (spike.day <= day) tail *= 1 + spike.size * spike.decay ** (day - spike.day);
    }
    return tail;
  }

  /** The multiplier a forecast made on `asOfDay` expects of a publisher's spikes on `day`:
   * only spikes already started by both days, each with its estimated decay. */
  private knownTail(publisher: number, day: number, asOfDay: number, decaySigma?: number): number {
    let tail = 1;
    for (const spike of this.publisherSpikes[publisher]) {
      if (spike.day > asOfDay || spike.day > day) continue;
      const decay = decaySigma === undefined ? spike.estimatedDecay : spike.estimatedDecay ** Math.exp(decaySigma);
      tail *= 1 + spike.size * decay ** (day - spike.day);
    }
    return tail;
  }

  private buildTables(properties: Map<string, SeedProperty>): void {
    const days = this.calendar.days;
    const rng = this.seed.rng_seed;
    const actualRaw: Array<Map<number, number>> = [];
    for (let day = 0; day < days; day++) {
      this.expectedRaw.push(new Map());
      this.cellNoise.push(new Map());
      actualRaw.push(new Map());
    }
    const { weekday_shape: weekdayShape, trend_per_day: trend = 0, month_shape: monthShape, truth_daily_sigma: dailySigma = 0 } = this.seed.traffic;
    const monthFactors = Array.from({ length: days }, (_, day) => monthShape?.[this.calendar.month(day)] ?? 1);
    const inSeason = (day: number) => day >= 0 && day < days;
    [...this.placements.keys()].forEach((id, index) => {
      this.placementIndex.set(id, index);
      const factors = new Float64Array(days);
      for (let day = 0; day < days; day++) factors[day] = Math.exp(dailySigma * standardNormal(`${rng}|truth-day|${day}|${id}`));
      this.dailyFactor.push(factors);
    });
    for (const slot of this.slots) this.slotPlacement.push(this.placementIndex.get(slot.placement_id)!);
    const truthTail = this.publisherSpikes.map((_, p) => {
      const tail = new Float64Array(days);
      for (let day = 0; day < days; day++) tail[day] = this.trueTail(p, day);
      return tail;
    });

    for (const slot of this.slots) {
      const region = this.regions.get(slot.region_id)!;
      const profile = this.profiles.get(slot.profile_id)!;
      const shares = region.share * profile.share;
      const factors = this.dailyFactor[this.slotPlacement[slot.index]];
      const tail = truthTail[this.slotPublisher[slot.index]];
      if (slot.installment_id) {
        const installment = this.installmentsById.get(slot.installment_id)!;
        const collection = this.collections.get(installment.collection_id)!;
        const curve = audienceCurve(collection.channel);
        const audienceFactor = Math.max(
          0,
          1 + collection.audience_sigma * standardNormal(`${rng}|installment|${installment.installment_id}`),
        );
        curve.forEach((weight, k) => {
          const expectedDay = installment.scheduled_day + k;
          if (inSeason(expectedDay)) addCount(this.expectedRaw[expectedDay], slot.index, installment.expected_audience * weight * shares * monthFactors[expectedDay]);
          if (installment.actual_day === undefined) return;
          const actualDay = installment.actual_day + k;
          if (!inSeason(actualDay)) return;
          addCount(
            actualRaw[actualDay],
            slot.index,
            installment.expected_audience * audienceFactor * weight * shares * monthFactors[actualDay]
              * factors[actualDay] * this.eventMultiplier(actualDay, slot) * tail[actualDay],
          );
        });
        continue;
      }
      const placement = this.placements.get(slot.placement_id)!;
      const base = placement.base_daily_impressions! * (properties.get(slot.property_id)?.traffic_multiplier ?? 1) * shares;
      for (let day = 0; day < days; day++) {
        const expected = base * weekdayShape[this.calendar.weekday(day)] * (1 + trend * day) * monthFactors[day];
        addCount(this.expectedRaw[day], slot.index, expected);
        addCount(actualRaw[day], slot.index, expected * factors[day] * this.eventMultiplier(day, slot) * tail[day]);
      }
    }

    const { cell_kappa: kappa, cell_min_impressions: floor } = this.seed.forecast_error;
    for (let day = 0; day < days; day++) {
      for (const [cell, expected] of this.expectedRaw[day]) {
        const noise = kappa === 0 ? 0 : (kappa / Math.sqrt(Math.max(expected, floor))) * standardNormal(`${rng}|fe-cell|${day}|${this.slotKey(this.slots[cell])}`);
        this.cellNoise[day].set(cell, noise);
      }
      this.tables.push({ expected: roundCounts(this.expectedRaw[day]), actual: roundCounts(actualRaw[day]) });
    }
  }

  /** `g(h)`: how much the publisher error has widened at the lead time `day - asOfDay`. */
  private leadTimeGrowth(day: number, asOfDay: number): number {
    const { lead_time_horizon_days: horizon, lead_time_tau_days: tau } = this.seed.forecast_error;
    return Math.sqrt(1 + Math.min(Math.max(0, day - asOfDay), horizon) / tau);
  }

  /** Expected and actual impressions on a day. Empty outside the season. */
  day(day: number): DayTables {
    return this.tables[day] ?? { expected: new Map(), actual: new Map() };
  }

  /**
   * The forecast of a cell's impressions on `day`, as known on `asOfDay`,
   * unrounded: the expected traffic and daily variation, times the spikes
   * already started, times the lead-time error clipped to the season's
   * `log_clip`. Zero outside the season or where nothing is expected.
   * Scripted events never appear in it.
   */
  forecastValue(day: number, cell: number, asOfDay: number): number {
    const expected = this.expectedRaw[day]?.get(cell);
    if (expected === undefined) return 0;
    const { forecast_error: error } = this.seed;
    const publisher = this.slotPublisher[cell];
    const growth = this.leadTimeGrowth(day, asOfDay);
    const logError = this.publisherBias[publisher]
      + this.slotErrorScale[cell] * growth * this.publisherDraw[publisher][day]
      + this.cellNoise[day].get(cell)!;
    const clipped = Math.max(-error.log_clip, Math.min(error.log_clip, logError));
    const spikes = this.publisherSpikes[publisher];
    const tail = spikes.length === 0 ? 1 : this.knownTail(publisher, day, asOfDay);
    return expected * this.dailyFactor[this.slotPlacement[cell]][day] * tail * Math.exp(clipped);
  }

  /**
   * The log standard deviations behind a forecast's error: `publisher` is the
   * lead-time term `σ_ch · k_p · g(h)`, shared by every cell of a publisher
   * and channel; `cell` is the term that shrinks with the cell's size.
   */
  errorSigmas(day: number, cell: number, asOfDay: number): { publisher: number; cell: number } {
    const { forecast_error: error } = this.seed;
    const expected = this.expectedRaw[day]?.get(cell) ?? 0;
    return {
      publisher: this.slotErrorScale[cell] * this.leadTimeGrowth(day, asOfDay),
      cell: error.cell_kappa / Math.sqrt(Math.max(expected, error.cell_min_impressions)),
    };
  }

  /** The publisher a cell's forecast error is shared with: its property's `publisher_id`, else the property. */
  publisherOf(cell: number): string {
    return this.publisherIds[this.slotPublisher[cell]];
  }

  /**
   * Half the log spread of the spike tail a forecast made on `asOfDay`
   * expects on `day`, between the fastest and slowest decay around each
   * spike's own estimate that `decay_forecast_sigma` allows. Zero when no
   * spike of the cell's publisher has started.
   */
  spikeTailSpread(day: number, cell: number, asOfDay: number): number {
    const publisher = this.slotPublisher[cell];
    if (this.publisherSpikes[publisher].length === 0) return 0;
    const sigma = this.seed.forecast_error.spikes!.decay_forecast_sigma;
    const slow = this.knownTail(publisher, day, asOfDay, -sigma);
    const fast = this.knownTail(publisher, day, asOfDay, sigma);
    return Math.abs(Math.log(slow) - Math.log(fast)) / 2;
  }

  installment(installmentId: string): Installment | undefined {
    return this.installmentsById.get(installmentId);
  }

  floorOf(cell: number): number {
    return this.placements.get(this.slots[cell].placement_id)!.floor_cpm;
  }

  /** Rates for a cell: the placement's base, scaled by the profile, with a
   * small seeded spread. */
  rates(day: number, cell: number): CellRates {
    const slot = this.slots[cell];
    const base = this.placements.get(slot.placement_id)!.rates;
    const multipliers = this.profiles.get(slot.profile_id)!.rate_multipliers ?? {};
    const draw = (metric: keyof CellRates) => {
      const value = base[metric] * (multipliers[metric] ?? 1)
        * (1 + RATE_SIGMA * standardNormal(`${this.seed.rng_seed}|rate|${metric}|${day}|${this.slotKey(slot)}`));
      return Math.min(1, Math.max(0, value));
    };
    return {
      viewability: draw('viewability'),
      click_rate: draw('click_rate'),
      completion_rate: draw('completion_rate'),
      conversion_rate: draw('conversion_rate'),
    };
  }

  /** Reporting delay of a product: its slowest placement. */
  reportingLagDays(productId: string): number {
    const product = this.requireProduct(productId);
    return Math.max(...product.placement_ids.map(id => this.placements.get(id)!.reporting_lag_days));
  }

  requireProduct(productId: string): SeedProduct {
    const product = this.products.get(productId);
    if (!product) throw new EngineError('UNKNOWN_PRODUCT', `Unknown product ${productId}`);
    return product;
  }

  /** Installments a product sells: those of its collections. */
  productInstallments(productId: string): Installment[] {
    const collections = new Set(this.requireProduct(productId).collection_ids ?? []);
    return this.installments.filter(installment => collections.has(installment.collection_id));
  }

  /**
   * Cells a product covers under the given targeting, as sorted slot
   * indices. Segments are unions of profiles, so a profile in two targeted
   * segments is counted once.
   */
  eligibleCells(productId: string, targeting: BookingTargeting = {}): number[] {
    const product = this.requireProduct(productId);
    for (const id of targeting.region_ids ?? []) {
      if (!this.regions.has(id)) throw new EngineError('INVALID_TARGETING', `Unknown region ${id}`);
    }
    let profiles: Set<string> | undefined;
    if (targeting.segment_ids && targeting.segment_ids.length > 0) {
      profiles = new Set();
      for (const id of targeting.segment_ids) {
        const members = this.segmentProfiles.get(id);
        if (!members) throw new EngineError('INVALID_TARGETING', `Unknown segment ${id}`);
        for (const profile of members) profiles.add(profile);
      }
    }
    const productProfiles = product.segment_id ? this.segmentProfiles.get(product.segment_id)! : undefined;
    const productInstallments = new Set(this.productInstallments(productId).map(installment => installment.installment_id));
    for (const id of targeting.installment_ids ?? []) {
      if (!productInstallments.has(id)) throw new EngineError('INVALID_TARGETING', `Product ${productId} does not sell installment ${id}`);
    }
    const regions = targeting.region_ids && targeting.region_ids.length > 0 ? new Set(targeting.region_ids) : undefined;
    const installments = targeting.installment_ids && targeting.installment_ids.length > 0 ? new Set(targeting.installment_ids) : undefined;
    const placements = new Set(product.placement_ids);
    const propertyIds = product.property_ids ? new Set(product.property_ids) : undefined;
    return this.slots
      .filter(slot => placements.has(slot.placement_id)
        && (!propertyIds || propertyIds.has(slot.property_id))
        && (!productProfiles || productProfiles.has(slot.profile_id))
        && (!profiles || profiles.has(slot.profile_id))
        && (!regions || regions.has(slot.region_id))
        && (!slot.installment_id || (productInstallments.has(slot.installment_id)
          && (!installments || installments.has(slot.installment_id)))))
      .map(slot => slot.index);
  }
}

const trafficCache = new WeakMap<SeasonSeed, SeasonTraffic>();

/** The traffic for a seed, generated on first use and reused after. */
export function trafficFor(seed: SeasonSeed): SeasonTraffic {
  let traffic = trafficCache.get(seed);
  if (!traffic) {
    traffic = new SeasonTraffic(seed);
    trafficCache.set(seed, traffic);
  }
  return traffic;
}
