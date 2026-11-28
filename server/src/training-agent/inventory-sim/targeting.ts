/**
 * AdCP targeting for the simulated inventory tenants: turns a package's
 * `targeting_overlay` into engine targeting (region and audience-profile
 * ids), and describes what each product accepts. Pure over a `SeasonSeed`
 * and a `SeedProduct`; no dependency on the platform.
 *
 * Geography is the seed's sales regions, reachable by country and by place.
 * Audiences are the seed's segments, exposed as product-scoped binary
 * signals. Whatever a dimension resolves to is sent to the engine as ids; a
 * dimension that resolves to nothing is refused, never sent as an empty list.
 */
import type { BookingTargeting, SeasonSeed, SeedProduct } from './engine/index.js';
import { PLACE_CATALOG_VERSION, placeSystem, placeType } from './geo-places.js';

export { PLACE_CATALOG_VERSION, placeSystem, placeType };

export type TargetingErrorCode = 'INVALID_REQUEST' | 'UNSUPPORTED_FEATURE' | 'PRODUCT_UNAVAILABLE';

/** A targeting refusal, carrying the AdCP error code and the offending field path. */
export class TargetingError extends Error {
  constructor(
    readonly code: TargetingErrorCode,
    message: string,
    readonly field: string,
  ) {
    super(message);
    this.name = 'TargetingError';
  }
}

export interface ParsedOverlay {
  targeting: BookingTargeting;
  /** The overlay as the package echoes it; undefined when none was sent. */
  echo: Record<string, unknown> | undefined;
}

type Json = Record<string, unknown>;

const COUNTRY_KEYS = ['geo_countries', 'geo_countries_exclude'] as const;
const PLACE_KEYS = ['geo_places', 'geo_places_exclude'] as const;
const SIGNAL_KEY = 'signal_targeting_groups';
const LABEL_KEY = 'geo_value_labels';
const FULL_ONLY_KEYS: readonly string[] = [...PLACE_KEYS, SIGNAL_KEY];
const COUNTRY_PATTERN = /^[A-Z]{2}$/;
const LABEL_KEY_PATTERN = /^[A-Z]{2}(-[A-Z0-9]{1,3})?$/;

const invalid = (field: string, message: string) => new TargetingError('INVALID_REQUEST', message, field);
const unsupported = (field: string, message: string) => new TargetingError('UNSUPPORTED_FEATURE', message, field);

function isObject(value: unknown): value is Json {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** `full` products take places and signals as well as countries. */
export function targetableBy(product: SeedProduct): 'full' | 'countries' {
  return product.share_of_voice === undefined && !product.collection_ids?.length ? 'full' : 'countries';
}

function seedCountries(seed: SeasonSeed): string[] {
  return [...new Set(seed.regions.map(region => region.country))];
}

function profilesOfSegment(seed: SeasonSeed, segmentId: string): string[] {
  return seed.segments.find(segment => segment.segment_id === segmentId)?.profile_ids ?? [];
}

function productProfiles(seed: SeasonSeed, product: SeedProduct): Set<string> {
  return new Set(product.segment_id
    ? profilesOfSegment(seed, product.segment_id)
    : seed.audience_profiles.map(profile => profile.profile_id));
}

function stringArray(value: unknown, field: string): string[] {
  if (!Array.isArray(value) || value.length === 0) throw invalid(field, `${field} must be a non-empty array`);
  value.forEach((item, index) => {
    if (typeof item !== 'string' || item.length === 0) throw invalid(`${field}[${index}]`, `${field}[${index}] must be a non-empty string`);
  });
  return value as string[];
}

function countryList(value: unknown, field: string): string[] {
  const codes = stringArray(value, field).map(code => code.toUpperCase());
  codes.forEach((code, index) => {
    if (!COUNTRY_PATTERN.test(code)) throw invalid(`${field}[${index}]`, `${field}[${index}] must be an ISO 3166-1 alpha-2 country code`);
  });
  return codes;
}

/** Whether an entry names this catalogue's place system and region type. */
function isOurCatalog(entry: Json): boolean {
  const system = placeSystem();
  return entry.system === system && entry.place_type === placeType(system);
}

function checkLabelKeys(labels: unknown, field: string, allowed: Set<string>, keyPattern?: RegExp): void {
  if (!isObject(labels)) throw invalid(field, `${field} must be an object`);
  for (const [key, label] of Object.entries(labels)) {
    if (keyPattern && !keyPattern.test(key)) throw invalid(`${field}.${key}`, `${field} key ${key} is not a valid code`);
    if (typeof label !== 'string' || label.length === 0) throw invalid(`${field}.${key}`, `${field}.${key} must be a non-empty string`);
    if (!allowed.has(key)) throw invalid(`${field}.${key}`, `${field} key ${key} does not appear in the targeted values`);
  }
}

function checkPlaces(value: unknown, key: string, seed: SeasonSeed, prefix: string): void {
  const field = `${prefix}.${key}`;
  if (!Array.isArray(value) || value.length === 0) throw invalid(field, `${key} must be a non-empty array`);
  const regions = new Map(seed.regions.map(region => [region.region_id, region]));
  value.forEach((entry, index) => {
    const at = `${field}[${index}]`;
    if (!isObject(entry)) throw invalid(at, `${key} entries must be objects`);
    for (const required of ['country', 'system', 'place_type'] as const) {
      if (typeof entry[required] !== 'string' || entry[required] === '') throw invalid(`${at}.${required}`, `${required} is required`);
    }
    if (!COUNTRY_PATTERN.test((entry.country as string).toUpperCase())) {
      throw invalid(`${at}.country`, 'country must be an ISO 3166-1 alpha-2 code');
    }
    const values = stringArray(entry.values, `${at}.values`);
    if (new Set(values).size !== values.length) throw invalid(`${at}.values`, 'values must be unique');
    if (entry.system_version !== undefined && (typeof entry.system_version !== 'string' || entry.system_version === '')) {
      throw invalid(`${at}.system_version`, 'system_version must be a non-empty string');
    }
    if (entry.value_labels !== undefined) checkLabelKeys(entry.value_labels, `${at}.value_labels`, new Set(values));
    // Values of a foreign system or place type cannot be judged here; the
    // product-level check refuses them as unsupported.
    if (!isOurCatalog(entry)) return;
    if (entry.system_version !== undefined && entry.system_version !== PLACE_CATALOG_VERSION) {
      throw invalid(`${at}.system_version`, `Unsupported system_version ${String(entry.system_version)}; the catalogue serves ${PLACE_CATALOG_VERSION}`);
    }
    const country = (entry.country as string).toUpperCase();
    if (!seedCountries(seed).includes(country)) return;
    values.forEach((id, valueIndex) => {
      const region = regions.get(id);
      if (!region) throw invalid(`${at}.values[${valueIndex}]`, `Unknown place ${id}`);
      if (region.country !== country) {
        throw invalid(`${at}.values[${valueIndex}]`, `Place ${id} is in ${region.country}, not ${country}`);
      }
    });
  });
}

function checkSignals(value: unknown, seed: SeasonSeed, prefix: string): void {
  const field = `${prefix}.${SIGNAL_KEY}`;
  if (!isObject(value)) throw invalid(field, `${SIGNAL_KEY} must be an object`);
  if (value.operator !== 'all') throw invalid(`${field}.operator`, "operator must be 'all'");
  if (!Array.isArray(value.groups) || value.groups.length === 0) throw invalid(`${field}.groups`, 'groups must be a non-empty array');
  const segments = new Set(seed.segments.map(segment => segment.segment_id));
  value.groups.forEach((group, groupIndex) => {
    const at = `${field}.groups[${groupIndex}]`;
    if (!isObject(group)) throw invalid(at, 'groups entries must be objects');
    if (group.operator !== 'any' && group.operator !== 'none') throw invalid(`${at}.operator`, "operator must be 'any' or 'none'");
    if (!Array.isArray(group.signals) || group.signals.length === 0) throw invalid(`${at}.signals`, 'signals must be a non-empty array');
    group.signals.forEach((signal, signalIndex) => {
      const path = `${at}.signals[${signalIndex}]`;
      if (!isObject(signal)) throw invalid(path, 'signals entries must be objects');
      const ref = signal.signal_ref;
      if (!isObject(ref) || typeof ref.scope !== 'string' || typeof ref.signal_id !== 'string' || ref.signal_id === '') {
        throw invalid(`${path}.signal_ref`, 'signal_ref needs a scope and a signal_id');
      }
      if (signal.value_type !== 'binary') throw invalid(`${path}.value_type`, "value_type must be 'binary'");
      if (signal.value !== true) throw invalid(`${path}.value`, 'value must be true; exclude a signal through a none group');
      if (ref.scope === 'product' && !segments.has(ref.signal_id)) {
        throw invalid(`${path}.signal_ref.signal_id`, `Unknown signal ${ref.signal_id}`);
      }
    });
  });
}

/**
 * Checks everything about an overlay that can be judged without a product:
 * shapes, region and segment ids, places' countries against their regions,
 * `system_version`, include/exclude overlap and label keys. Dimensions this
 * module does not know are left to `parseOverlay`.
 */
export function validateOverlayGlobally(overlay: unknown, seed: SeasonSeed, fieldPrefix: string): void {
  if (overlay === undefined || overlay === null) return;
  if (!isObject(overlay)) throw invalid(fieldPrefix, 'targeting_overlay must be an object');
  const present = (key: string) => overlay[key] !== undefined && overlay[key] !== null;

  const countries = new Map<string, string[]>();
  for (const key of COUNTRY_KEYS) {
    if (present(key)) countries.set(key, countryList(overlay[key], `${fieldPrefix}.${key}`));
  }
  const included = new Set(countries.get('geo_countries') ?? []);
  (countries.get('geo_countries_exclude') ?? []).forEach((code, index) => {
    if (included.has(code)) {
      throw invalid(`${fieldPrefix}.geo_countries_exclude[${index}]`, `${code} is both included and excluded`);
    }
  });

  for (const key of PLACE_KEYS) {
    if (present(key)) checkPlaces(overlay[key], key, seed, fieldPrefix);
  }
  const includedPlaces = new Set<string>();
  if (present('geo_places')) {
    for (const entry of overlay.geo_places as Json[]) {
      for (const id of entry.values as string[]) includedPlaces.add(`${String(entry.system)}\u0000${id}`);
    }
  }
  if (present('geo_places_exclude')) {
    (overlay.geo_places_exclude as Json[]).forEach((entry, entryIndex) => {
      (entry.values as string[]).forEach((id, valueIndex) => {
        if (includedPlaces.has(`${String(entry.system)}\u0000${id}`)) {
          throw invalid(`${fieldPrefix}.geo_places_exclude[${entryIndex}].values[${valueIndex}]`, `Place ${id} is both included and excluded`);
        }
      });
    });
  }

  if (present(LABEL_KEY)) {
    const labelled = new Set<string>();
    for (const list of countries.values()) list.forEach(code => labelled.add(code));
    checkLabelKeys(overlay[LABEL_KEY], `${fieldPrefix}.${LABEL_KEY}`, labelled, LABEL_KEY_PATTERN);
  }

  if (present(SIGNAL_KEY)) checkSignals(overlay[SIGNAL_KEY], seed, fieldPrefix);
}

function checkPlaceSupport(overlay: Json, seed: SeasonSeed, prefix: string): void {
  const system = placeSystem();
  const countriesOfSeed = seedCountries(seed);
  for (const key of PLACE_KEYS) {
    const entries = overlay[key];
    if (!Array.isArray(entries)) continue;
    entries.forEach((entry: Json, index) => {
      const at = `${prefix}.${key}[${index}]`;
      if (entry.system !== system) throw unsupported(`${at}.system`, `Place system ${String(entry.system)} is not supported; use ${system}`);
      if (entry.place_type !== placeType(system)) {
        throw unsupported(`${at}.place_type`, `Place type ${String(entry.place_type)} is not supported; use ${placeType(system)}`);
      }
      if (!countriesOfSeed.includes((entry.country as string).toUpperCase())) {
        throw unsupported(`${at}.country`, `Places in ${String(entry.country)} are not supported`);
      }
    });
  }
}

/** Segments this product offers as signal options: the others that overlap its audience. */
function offeredSegments(seed: SeasonSeed, product: SeedProduct): SeasonSeed['segments'] {
  const profiles = productProfiles(seed, product);
  return seed.segments.filter(segment => segment.segment_id !== product.segment_id
    && segment.profile_ids.some(id => profiles.has(id)));
}

function resolveRegions(overlay: Json, seed: SeasonSeed): string[] {
  const all = seed.regions.map(region => region.region_id);
  const inCountries = (codes: unknown) => {
    const wanted = new Set((codes as string[]).map(code => code.toUpperCase()));
    return seed.regions.filter(region => wanted.has(region.country)).map(region => region.region_id);
  };
  const placeValues = (entries: unknown) => (entries as Json[]).flatMap(entry => entry.values as string[]);

  let regions = new Set(all);
  if (overlay.geo_countries) regions = new Set(inCountries(overlay.geo_countries).filter(id => regions.has(id)));
  if (overlay.geo_places) {
    const places = new Set(placeValues(overlay.geo_places));
    regions = new Set([...regions].filter(id => places.has(id)));
  }
  if (overlay.geo_countries_exclude) inCountries(overlay.geo_countries_exclude).forEach(id => regions.delete(id));
  if (overlay.geo_places_exclude) placeValues(overlay.geo_places_exclude).forEach(id => regions.delete(id));
  return all.filter(id => regions.has(id)).sort();
}

function resolveProfiles(overlay: Json, seed: SeasonSeed, product: SeedProduct, prefix: string): string[] {
  const field = `${prefix}.${SIGNAL_KEY}`;
  const offered = new Set(offeredSegments(seed, product).map(segment => segment.segment_id));
  const groups = (overlay[SIGNAL_KEY] as Json).groups as Json[];
  let profiles = productProfiles(seed, product);

  groups.forEach((group, groupIndex) => {
    const members = new Set<string>();
    (group.signals as Json[]).forEach((signal, signalIndex) => {
      const ref = signal.signal_ref as Json;
      const path = `${field}.groups[${groupIndex}].signals[${signalIndex}].signal_ref`;
      if (ref.scope !== 'product' || !offered.has(ref.signal_id as string)) {
        throw unsupported(path, `Signal ${String(ref.signal_id)} is not offered on this product`);
      }
      profilesOfSegment(seed, ref.signal_id as string).forEach(id => members.add(id));
    });
    profiles = new Set([...profiles].filter(id => (group.operator === 'any' ? members.has(id) : !members.has(id))));
  });

  if (profiles.size === 0) {
    throw new TargetingError('PRODUCT_UNAVAILABLE', 'The targeting leaves no inventory in this product', field);
  }
  return [...profiles].sort();
}

function echoOf(overlay: Json): Json {
  const echo: Json = {};
  for (const [key, value] of Object.entries(overlay)) {
    if (value === null || value === undefined || key === LABEL_KEY) continue;
    echo[key] = structuredClone(value);
  }
  for (const key of COUNTRY_KEYS) {
    if (Array.isArray(echo[key])) echo[key] = (echo[key] as string[]).map(code => code.toUpperCase());
  }
  for (const key of PLACE_KEYS) {
    if (!Array.isArray(echo[key])) continue;
    echo[key] = (echo[key] as Json[]).map(entry => {
      const { value_labels: _labels, ...rest } = entry;
      return { ...rest, country: (rest.country as string).toUpperCase(), system_version: rest.system_version ?? PLACE_CATALOG_VERSION };
    });
  }
  return echo;
}

/**
 * Parses a package's `targeting_overlay` for one product into engine
 * targeting and the overlay to echo back. Throws `TargetingError`:
 * `INVALID_REQUEST` for malformed input, `UNSUPPORTED_FEATURE` for a
 * dimension or value the product does not take, `PRODUCT_UNAVAILABLE` when
 * the targeting leaves no inventory.
 */
export function parseOverlay(overlay: unknown, seed: SeasonSeed, product: SeedProduct, fieldPrefix: string): ParsedOverlay {
  if (overlay === undefined || overlay === null) return { targeting: {}, echo: undefined };
  validateOverlayGlobally(overlay, seed, fieldPrefix);
  const given = overlay as Json;
  const full = targetableBy(product) === 'full';

  for (const [key, value] of Object.entries(given)) {
    if (value === null || value === undefined) continue;
    const allowed = (COUNTRY_KEYS as readonly string[]).includes(key) || key === LABEL_KEY || (full && FULL_ONLY_KEYS.includes(key));
    if (allowed) continue;
    const hint = key === 'signal_targeting' ? `; use ${SIGNAL_KEY}` : '';
    throw unsupported(`${fieldPrefix}.${key}`, `Targeting dimension ${key} is not supported by this product${hint}`);
  }
  checkPlaceSupport(given, seed, fieldPrefix);

  const targeting: BookingTargeting = {};
  const geoKey = [...COUNTRY_KEYS, ...PLACE_KEYS].find(key => given[key] !== undefined && given[key] !== null);
  if (geoKey) {
    const regionIds = resolveRegions(given, seed);
    if (regionIds.length === 0) {
      throw new TargetingError('PRODUCT_UNAVAILABLE', 'The targeting leaves no inventory in this product', `${fieldPrefix}.${geoKey}`);
    }
    targeting.region_ids = regionIds;
  }
  if (given[SIGNAL_KEY] !== undefined && given[SIGNAL_KEY] !== null) {
    targeting.profile_ids = resolveProfiles(given, seed, product, fieldPrefix);
  }
  return { targeting, echo: echoOf(given) };
}

/** Product fields that declare what targeting the product takes. */
export function productTargetingFields(seed: SeasonSeed, product: SeedProduct): Record<string, unknown> {
  const fields: Record<string, unknown> = {};
  const overlaySupport: Record<string, unknown> = { geo_countries: true, geo_countries_exclude: true };

  if (targetableBy(product) === 'full') {
    const system = placeSystem();
    const placeSupport = () => ({
      systems: {
        [system]: {
          countries: Object.fromEntries(seedCountries(seed).map(country => [country, [placeType(system)]])),
          current_version: PLACE_CATALOG_VERSION,
          system_versions: [PLACE_CATALOG_VERSION],
        },
      },
    });
    overlaySupport.geo_places = placeSupport();
    overlaySupport.geo_places_exclude = placeSupport();
    // The schema takes no empty options list: a product with no segment to offer declares no signal targeting.
    const offered = offeredSegments(seed, product);
    if (offered.length > 0) {
      overlaySupport[SIGNAL_KEY] = true;
      fields.signal_targeting_allowed = true;
      fields.signal_targeting_options = offered.map(segment => ({
        signal_ref: { scope: 'product', signal_id: segment.segment_id },
        name: segment.name,
        value_type: 'binary',
        allowed_targeting_modes: ['include', 'exclude'],
      }));
      fields.signal_targeting_rules = { resolution_model: 'direct_targeting', selection_mode: 'optional' };
    }
  }
  fields.overlay_support = overlaySupport;

  const own = product.segment_id ? seed.segments.find(segment => segment.segment_id === product.segment_id) : undefined;
  if (own) {
    fields.included_signals = [{
      signal_ref: { scope: 'product', signal_id: own.segment_id },
      name: own.name,
      value_type: 'binary',
    }];
  }
  return fields;
}

function matchesRequirement(support: unknown, required: unknown): boolean {
  if (Array.isArray(required)) {
    return Array.isArray(support) && required.every(item => support.includes(item));
  }
  if (isObject(required)) {
    if (support === true) return true;
    if (!isObject(support)) return false;
    return Object.entries(required).every(([key, value]) => Object.hasOwn(support, key) && matchesRequirement(support[key], value));
  }
  if (required === true) return support === true || isObject(support);
  return support === required;
}

/**
 * Whether a product's `overlay_support` covers a buyer's
 * `required_overlay_support`. A `true` requirement matches `true` or any
 * object; an object requirement matches `true` or an object holding every
 * requested key; arrays must be subsets. Missing fields never match.
 */
export function supportsOverlayRequirements(overlaySupport: unknown, required: unknown): boolean {
  if (required === undefined || required === null) return true;
  return matchesRequirement(overlaySupport, required);
}
