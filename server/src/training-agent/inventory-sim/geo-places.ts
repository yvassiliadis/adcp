/**
 * Place catalogue of the simulated inventory tenants: the seed's sales
 * regions, published as AdCP geo places. Pure over a `SeasonSeed`; no
 * dependency on the platform.
 */
import type { SeasonSeed } from './engine/index.js';

/** Catalogue version of the place system. */
export const PLACE_CATALOG_VERSION = '1';

const DEFAULT_PLACE_SYSTEM = 'https://adcontextprotocol.org/training-agent/places';

/** Place system URI, from `TRAINING_AGENT_INVENTORY_PLACE_SYSTEM`. */
export function placeSystem(): string {
  const configured = process.env.TRAINING_AGENT_INVENTORY_PLACE_SYSTEM?.trim();
  return configured || DEFAULT_PLACE_SYSTEM;
}

/** Place type of the seed's sales regions within a system. */
export function placeType(system: string): string {
  return `${system}/types/sales-region`;
}

export interface GeoPlaceMatch {
  value: string;
  country: string;
  subdivision?: string;
  place_type: string;
  label: string;
  canonical_name: string;
  parent_labels: string[];
  status: 'active';
}

export interface GeoPlaceResolution {
  request: {
    q?: string;
    value?: string;
    country: string;
    subdivision?: string;
    place_type?: string;
    system_version?: string;
    limit: number;
  };
  system: string;
  system_version: string;
  matches: GeoPlaceMatch[];
}

export type GeoPlaceResolverResult =
  | { status: 200; body: GeoPlaceResolution }
  | { status: 400; body: { error: string } };

const DEFAULT_LIMIT = 20;
const MAX_LIMIT = 100;

function badRequest(error: string): GeoPlaceResolverResult {
  return { status: 400, body: { error } };
}

function optionalString(query: Record<string, unknown>, key: string): string | undefined | Error {
  const raw = query[key];
  if (raw === undefined || raw === null) return undefined;
  if (typeof raw !== 'string' || raw.length === 0) return new Error(`${key} must be a non-empty string`);
  return raw;
}

const REGION_NAMES = new Intl.DisplayNames(['en'], { type: 'region' });

function countryName(code: string): string {
  try {
    return REGION_NAMES.of(code) ?? code;
  } catch {
    return code;
  }
}

/**
 * Answers a `get_geo_place_resolution` query over the seed's regions.
 * `q` matches region names and ids case-insensitively as substrings; `value`
 * is an exact region id. Results are always limited to the requested country.
 * An optional `subdivision` filters on the region's ISO 3166-2 code. Matches
 * come back in one page: `cursor` is ignored and `next_cursor` is never
 * emitted.
 */
export function resolveGeoPlaces(seed: SeasonSeed, query: Record<string, unknown>): GeoPlaceResolverResult {
  const q = optionalString(query, 'q');
  if (q instanceof Error) return badRequest(q.message);
  const value = optionalString(query, 'value');
  if (value instanceof Error) return badRequest(value.message);
  if (q === undefined && value === undefined) return badRequest('Supply exactly one of q or value');
  if (q !== undefined && value !== undefined) return badRequest('q and value are mutually exclusive');

  const rawCountry = query.country;
  if (typeof rawCountry !== 'string') return badRequest('country is required');
  const country = rawCountry.toUpperCase();
  if (!/^[A-Z]{2}$/.test(country)) return badRequest('country must be an ISO 3166-1 alpha-2 code');

  const subdivision = optionalString(query, 'subdivision');
  if (subdivision instanceof Error) return badRequest(subdivision.message);
  if (subdivision !== undefined && !/^[A-Z]{2}-[A-Z0-9]{1,3}$/.test(subdivision)) {
    return badRequest('subdivision must be an ISO 3166-2 code');
  }

  const system = placeSystem();
  const type = placeType(system);
  const requestedType = optionalString(query, 'place_type');
  if (requestedType instanceof Error) return badRequest(requestedType.message);
  const requestedVersion = optionalString(query, 'system_version');
  if (requestedVersion instanceof Error) return badRequest(requestedVersion.message);
  if (requestedVersion !== undefined && requestedVersion !== PLACE_CATALOG_VERSION) {
    return badRequest(`Unsupported system_version ${requestedVersion}; this catalogue serves ${PLACE_CATALOG_VERSION}`);
  }

  let limit = DEFAULT_LIMIT;
  const rawLimit = query.limit;
  if (rawLimit !== undefined && rawLimit !== null) {
    const parsed = typeof rawLimit === 'string' && /^\d+$/.test(rawLimit) ? Number(rawLimit) : rawLimit;
    if (typeof parsed !== 'number' || !Number.isInteger(parsed) || parsed < 1 || parsed > MAX_LIMIT) {
      return badRequest(`limit must be an integer between 1 and ${MAX_LIMIT}`);
    }
    limit = parsed;
  }

  const needle = q?.toLowerCase();
  const matches: GeoPlaceMatch[] = [];
  if (requestedType === undefined || requestedType === type) {
    for (const region of seed.regions) {
      if (region.country !== country) continue;
      if (subdivision !== undefined && region.subdivision !== subdivision) continue;
      const hit = value !== undefined
        ? region.region_id === value
        : region.name.toLowerCase().includes(needle!) || region.region_id.toLowerCase().includes(needle!);
      if (!hit) continue;
      matches.push({
        value: region.region_id,
        country: region.country,
        ...(region.subdivision ? { subdivision: region.subdivision } : {}),
        place_type: type,
        label: region.name,
        canonical_name: region.name,
        parent_labels: [countryName(region.country)],
        status: 'active',
      });
      if (matches.length === limit) break;
    }
  }

  return {
    status: 200,
    body: {
      request: {
        ...(q !== undefined ? { q } : {}),
        ...(value !== undefined ? { value } : {}),
        country,
        ...(subdivision !== undefined ? { subdivision } : {}),
        ...(requestedType !== undefined ? { place_type: requestedType } : {}),
        ...(requestedVersion !== undefined ? { system_version: requestedVersion } : {}),
        limit,
      },
      system,
      system_version: PLACE_CATALOG_VERSION,
      matches,
    },
  };
}
