/**
 * Pure AdCP targeting for the inventory tenants: overlay parsing, per-product
 * targeting declarations, the overlay-support matcher and the place resolver.
 */
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import Ajv from 'ajv';
import addFormats from 'ajv-formats';
import { afterEach, describe, expect, it } from 'vitest';
import {
  ADSERVER_SEASON,
  CONTENT_SEASON,
  type SeasonSeed,
  type SeedProduct,
} from '../../src/training-agent/inventory-sim/engine/index.js';
import {
  PLACE_CATALOG_VERSION,
  TargetingError,
  parseOverlay,
  placeSystem,
  placeType,
  productTargetingFields,
  supportsOverlayRequirements,
  targetableBy,
  validateOverlayGlobally,
} from '../../src/training-agent/inventory-sim/targeting.js';
import { resolveGeoPlaces } from '../../src/training-agent/inventory-sim/geo-places.js';

const PREFIX = 'packages[0].targeting_overlay';
const seed: SeasonSeed = ADSERVER_SEASON;
const product = (id: string): SeedProduct => {
  const found = [...ADSERVER_SEASON.products, ...CONTENT_SEASON.products].find(candidate => candidate.product_id === id);
  if (!found) throw new Error(`no product ${id}`);
  return found;
};
const full = product('sports_fans_display_guaranteed');
const unsegmented = ADSERVER_SEASON.products.find(candidate => candidate.share_of_voice === undefined && !candidate.segment_id)!;
const takeover = product('homepage_takeover_daily');
const content = CONTENT_SEASON.products[0];

function refusal(fn: () => unknown): TargetingError {
  try {
    fn();
  } catch (error) {
    expect(error).toBeInstanceOf(TargetingError);
    return error as TargetingError;
  }
  throw new Error('expected a TargetingError');
}

const place = (values: string[], overrides: Record<string, unknown> = {}) => ({
  country: 'US',
  system: placeSystem(),
  place_type: placeType(placeSystem()),
  values,
  ...overrides,
});
const signal = (id: string) => ({ signal_ref: { scope: 'product', signal_id: id }, value_type: 'binary', value: true });
const groups = (...entries: Array<['any' | 'none', string[]]>) => ({
  operator: 'all',
  groups: entries.map(([operator, ids]) => ({ operator, signals: ids.map(signal) })),
});

describe('place system', () => {
  const original = process.env.TRAINING_AGENT_INVENTORY_PLACE_SYSTEM;
  afterEach(() => {
    if (original === undefined) delete process.env.TRAINING_AGENT_INVENTORY_PLACE_SYSTEM;
    else process.env.TRAINING_AGENT_INVENTORY_PLACE_SYSTEM = original;
  });

  it('defaults, follows the environment and derives the region place type', () => {
    delete process.env.TRAINING_AGENT_INVENTORY_PLACE_SYSTEM;
    expect(placeSystem()).toBe('https://adcontextprotocol.org/training-agent/places');
    expect(placeType(placeSystem())).toBe('https://adcontextprotocol.org/training-agent/places/types/sales-region');
    process.env.TRAINING_AGENT_INVENTORY_PLACE_SYSTEM = 'https://seller.example/places';
    expect(placeSystem()).toBe('https://seller.example/places');
    expect(placeType('https://seller.example/places')).toBe('https://seller.example/places/types/sales-region');
    expect(PLACE_CATALOG_VERSION).toBe('1');
  });
});

describe('targetableBy', () => {
  it('is full without share of voice or collections, countries otherwise', () => {
    expect(targetableBy(full)).toBe('full');
    expect(targetableBy(unsegmented)).toBe('full');
    expect(targetableBy(takeover)).toBe('countries');
    expect(targetableBy(content)).toBe('countries');
  });
});

describe('parseOverlay: absent input', () => {
  it('returns empty targeting and no echo for an absent or null overlay', () => {
    expect(parseOverlay(undefined, seed, full, PREFIX)).toEqual({ targeting: {}, echo: undefined });
    expect(parseOverlay(null, seed, full, PREFIX)).toEqual({ targeting: {}, echo: undefined });
  });

  it('treats a null dimension as absent', () => {
    const parsed = parseOverlay({ geo_countries: null, geo_places: null, signal_targeting_groups: null, audience_include: null }, seed, full, PREFIX);
    expect(parsed.targeting).toEqual({});
    expect(parsed.echo).toEqual({});
  });
});

describe('parseOverlay: countries and places', () => {
  it('maps countries to their regions', () => {
    expect(parseOverlay({ geo_countries: ['US'] }, seed, full, PREFIX).targeting)
      .toEqual({ region_ids: ['us_northeast', 'us_south', 'us_west'] });
    expect(parseOverlay({ geo_countries: ['CA'] }, seed, full, PREFIX).targeting).toEqual({ region_ids: ['ca'] });
  });

  it('accepts lowercase countries, normalised to uppercase in the echo', () => {
    const parsed = parseOverlay({ geo_countries: ['us'] }, seed, full, PREFIX);
    expect(parsed.targeting.region_ids).toHaveLength(3);
    expect(parsed.echo).toEqual({ geo_countries: ['US'] });
  });

  it('unions values within a dimension', () => {
    expect(parseOverlay({ geo_countries: ['US', 'CA'] }, seed, full, PREFIX).targeting.region_ids).toHaveLength(4);
    const places = parseOverlay({ geo_places: [place(['us_west']), place(['us_south'])] }, seed, full, PREFIX);
    expect(places.targeting.region_ids).toEqual(['us_south', 'us_west']);
  });

  it('intersects across dimensions', () => {
    const parsed = parseOverlay({ geo_countries: ['US'], geo_places: [place(['us_west', 'us_south'])] }, seed, full, PREFIX);
    expect(parsed.targeting.region_ids).toEqual(['us_south', 'us_west']);
  });

  it('subtracts excluded countries and places', () => {
    expect(parseOverlay({ geo_countries_exclude: ['CA'] }, seed, full, PREFIX).targeting.region_ids)
      .toEqual(['us_northeast', 'us_south', 'us_west']);
    expect(parseOverlay({ geo_countries: ['US'], geo_places_exclude: [place(['us_west'])] }, seed, full, PREFIX).targeting.region_ids)
      .toEqual(['us_northeast', 'us_south']);
    expect(parseOverlay({ geo_countries_exclude: ['CA'], geo_places_exclude: [place(['us_west'])] }, seed, full, PREFIX).targeting.region_ids)
      .toEqual(['us_northeast', 'us_south']);
  });

  it('keeps a valid country with no seed region out of the result', () => {
    expect(parseOverlay({ geo_countries: ['US', 'GB'] }, seed, full, PREFIX).targeting.region_ids).toHaveLength(3);
    expect(parseOverlay({ geo_countries_exclude: ['GB'] }, seed, full, PREFIX).targeting.region_ids).toHaveLength(4);
    expect(() => validateOverlayGlobally({ geo_countries: ['GB'] }, seed, PREFIX)).not.toThrow();
  });

  it('is available to countries products too', () => {
    expect(parseOverlay({ geo_countries: ['CA'] }, seed, takeover, PREFIX).targeting).toEqual({ region_ids: ['ca'] });
    expect(parseOverlay({ geo_countries_exclude: ['CA'] }, CONTENT_SEASON, content, PREFIX).targeting.region_ids)
      .toEqual(['us_northeast', 'us_south', 'us_west']);
  });

  it('echoes the overlay with system_version added and labels dropped', () => {
    const parsed = parseOverlay({
      geo_countries: ['US'],
      geo_value_labels: { US: 'United States' },
      geo_places: [place(['us_west'], { value_labels: { us_west: 'US West' } }), place(['us_south'], { system_version: '1' })],
      geo_places_exclude: [place(['us_northeast'])],
    }, seed, full, PREFIX);
    expect(parsed.echo).toEqual({
      geo_countries: ['US'],
      geo_places: [place(['us_west'], { system_version: '1' }), place(['us_south'], { system_version: '1' })],
      geo_places_exclude: [place(['us_northeast'], { system_version: '1' })],
    });
  });

  it('does not mutate the overlay it was given', () => {
    const overlay = { geo_places: [place(['us_west'], { value_labels: { us_west: 'West' } })] };
    const snapshot = structuredClone(overlay);
    parseOverlay(overlay, seed, full, PREFIX);
    expect(overlay).toEqual(snapshot);
  });
});

describe('parseOverlay: refusals', () => {
  const cases: Array<[string, unknown, SeedProduct, string, string]> = [
    ['a non-object overlay', 'US', full, 'INVALID_REQUEST', PREFIX],
    ['an array overlay', [], full, 'INVALID_REQUEST', PREFIX],
    ['an empty country list', { geo_countries: [] }, full, 'INVALID_REQUEST', `${PREFIX}.geo_countries`],
    ['a non-array country list', { geo_countries: 'US' }, full, 'INVALID_REQUEST', `${PREFIX}.geo_countries`],
    ['a non-ISO country', { geo_countries: ['US', 'USA'] }, full, 'INVALID_REQUEST', `${PREFIX}.geo_countries[1]`],
    ['a non-ISO excluded country', { geo_countries_exclude: ['1'] }, full, 'INVALID_REQUEST', `${PREFIX}.geo_countries_exclude[0]`],
    ['a country both included and excluded', { geo_countries: ['US'], geo_countries_exclude: ['us'] }, full, 'INVALID_REQUEST', `${PREFIX}.geo_countries_exclude[0]`],
    ['a place entry that is not an object', { geo_places: ['us_west'] }, full, 'INVALID_REQUEST', `${PREFIX}.geo_places[0]`],
    ['an empty geo_places', { geo_places: [] }, full, 'INVALID_REQUEST', `${PREFIX}.geo_places`],
    ['a place without values', { geo_places: [place([])] }, full, 'INVALID_REQUEST', `${PREFIX}.geo_places[0].values`],
    ['a place without a system', { geo_places: [{ country: 'US', place_type: placeType(placeSystem()), values: ['us_west'] }] }, full, 'INVALID_REQUEST', `${PREFIX}.geo_places[0].system`],
    ['an unknown region id', { geo_places: [place(['us_west', 'nowhere'])] }, full, 'INVALID_REQUEST', `${PREFIX}.geo_places[0].values[1]`],
    ['a region in another country', { geo_places: [place(['ca'])] }, full, 'INVALID_REQUEST', `${PREFIX}.geo_places[0].values[0]`],
    ['a region in another country on exclude', { geo_places_exclude: [place(['us_west'], { country: 'CA' })] }, full, 'INVALID_REQUEST', `${PREFIX}.geo_places_exclude[0].values[0]`],
    ['an unsupported system_version', { geo_places: [place(['us_west'], { system_version: '2' })] }, full, 'INVALID_REQUEST', `${PREFIX}.geo_places[0].system_version`],
    ['a place in both include and exclude', { geo_places: [place(['us_west', 'us_south'])], geo_places_exclude: [place(['us_south'])] }, full, 'INVALID_REQUEST', `${PREFIX}.geo_places_exclude[0].values[0]`],
    ['value_labels keys outside values', { geo_places: [place(['us_west'], { value_labels: { us_south: 'South' } })] }, full, 'INVALID_REQUEST', `${PREFIX}.geo_places[0].value_labels.us_south`],
    ['geo_value_labels keys outside the countries', { geo_countries: ['US'], geo_value_labels: { CA: 'Canada' } }, full, 'INVALID_REQUEST', `${PREFIX}.geo_value_labels.CA`],
    ['a signal group operator other than all', { signal_targeting_groups: { operator: 'any', groups: [] } }, full, 'INVALID_REQUEST', `${PREFIX}.signal_targeting_groups.operator`],
    ['no signal groups', { signal_targeting_groups: { operator: 'all', groups: [] } }, full, 'INVALID_REQUEST', `${PREFIX}.signal_targeting_groups.groups`],
    ['a group operator that is not any or none', { signal_targeting_groups: { operator: 'all', groups: [{ operator: 'all', signals: [signal('news_loyalists')] }] } }, full, 'INVALID_REQUEST', `${PREFIX}.signal_targeting_groups.groups[0].operator`],
    ['a group without signals', { signal_targeting_groups: { operator: 'all', groups: [{ operator: 'any', signals: [] }] } }, full, 'INVALID_REQUEST', `${PREFIX}.signal_targeting_groups.groups[0].signals`],
    ['a signal without a ref', { signal_targeting_groups: { operator: 'all', groups: [{ operator: 'any', signals: [{ value_type: 'binary', value: true }] }] } }, full, 'INVALID_REQUEST', `${PREFIX}.signal_targeting_groups.groups[0].signals[0].signal_ref`],
    ['a non-binary signal', { signal_targeting_groups: { operator: 'all', groups: [{ operator: 'any', signals: [{ ...signal('news_loyalists'), value_type: 'categorical' }] }] } }, full, 'INVALID_REQUEST', `${PREFIX}.signal_targeting_groups.groups[0].signals[0].value_type`],
    ['a signal with value false', { signal_targeting_groups: { operator: 'all', groups: [{ operator: 'any', signals: [{ ...signal('news_loyalists'), value: false }] }] } }, full, 'INVALID_REQUEST', `${PREFIX}.signal_targeting_groups.groups[0].signals[0].value`],
    ['an unknown segment', { signal_targeting_groups: groups(['any', ['news_loyalists']], ['none', ['no_such_segment']]) }, full, 'INVALID_REQUEST', `${PREFIX}.signal_targeting_groups.groups[1].signals[0].signal_ref.signal_id`],

    ['a dimension the product does not support', { device_platform: ['ios'] }, full, 'UNSUPPORTED_FEATURE', `${PREFIX}.device_platform`],
    ['audience_include', { audience_include: [{ audience_id: 'a' }] }, full, 'UNSUPPORTED_FEATURE', `${PREFIX}.audience_include`],
    ['the deprecated signal_targeting', { signal_targeting: [] }, full, 'UNSUPPORTED_FEATURE', `${PREFIX}.signal_targeting`],
    ['places on a countries product', { geo_places: [place(['us_west'])] }, takeover, 'UNSUPPORTED_FEATURE', `${PREFIX}.geo_places`],
    ['place exclusion on a countries product', { geo_places_exclude: [place(['us_west'])] }, takeover, 'UNSUPPORTED_FEATURE', `${PREFIX}.geo_places_exclude`],
    ['signals on a countries product', { signal_targeting_groups: groups(['any', ['news_loyalists']]) }, takeover, 'UNSUPPORTED_FEATURE', `${PREFIX}.signal_targeting_groups`],
    ['a place system outside the product support', { geo_places: [place(['us_west'], { system: 'https://other.example/places' })] }, full, 'UNSUPPORTED_FEATURE', `${PREFIX}.geo_places[0].system`],
    ['a place type outside the product support', { geo_places: [place(['us_west'], { place_type: 'city' })] }, full, 'UNSUPPORTED_FEATURE', `${PREFIX}.geo_places[0].place_type`],
    ['a place country outside the product support', { geo_places_exclude: [place(['us_west'], { country: 'GB' })] }, full, 'UNSUPPORTED_FEATURE', `${PREFIX}.geo_places_exclude[0].country`],
    ['a signal that is the product segment itself', { signal_targeting_groups: groups(['any', ['sports_fans']]) }, full, 'UNSUPPORTED_FEATURE', `${PREFIX}.signal_targeting_groups.groups[0].signals[0].signal_ref`],
    ['a signal with a non-product scope', { signal_targeting_groups: { operator: 'all', groups: [{ operator: 'any', signals: [{ signal_ref: { scope: 'data_provider', data_provider_domain: 'x.example', signal_id: 'x' }, value_type: 'binary', value: true }] }] } }, full, 'UNSUPPORTED_FEATURE', `${PREFIX}.signal_targeting_groups.groups[0].signals[0].signal_ref`],

    ['a country with no seed region', { geo_countries: ['GB'] }, full, 'PRODUCT_UNAVAILABLE', `${PREFIX}.geo_countries`],
    ['countries disjoint from places', { geo_countries: ['CA'], geo_places: [place(['us_west'])] }, full, 'PRODUCT_UNAVAILABLE', `${PREFIX}.geo_countries`],
    ['every region excluded', { geo_countries_exclude: ['US', 'CA'] }, full, 'PRODUCT_UNAVAILABLE', `${PREFIX}.geo_countries_exclude`],
    ['every country excluded on a countries product', { geo_countries_exclude: ['US', 'CA'] }, takeover, 'PRODUCT_UNAVAILABLE', `${PREFIX}.geo_countries_exclude`],
    ['signals that leave no profiles', { signal_targeting_groups: groups(['any', ['household_decision_makers']], ['none', ['news_loyalists']]) }, product('sports_fans_display_guaranteed'), 'PRODUCT_UNAVAILABLE', `${PREFIX}.signal_targeting_groups`],
  ];

  it.each(cases)('refuses %s', (_name, overlay, target, code, field) => {
    const error = refusal(() => parseOverlay(overlay, seed, target, PREFIX));
    expect({ code: error.code, field: error.field }).toEqual({ code, field });
    expect(error.message.length).toBeGreaterThan(0);
  });

  it('refuses signals on the content tenant as unsupported, though they are globally valid', () => {
    const overlay = { signal_targeting_groups: groups(['any', ['commuters']]) };
    expect(() => validateOverlayGlobally(overlay, CONTENT_SEASON, PREFIX)).not.toThrow();
    const error = refusal(() => parseOverlay(overlay, CONTENT_SEASON, content, PREFIX));
    expect({ code: error.code, field: error.field }).toEqual({ code: 'UNSUPPORTED_FEATURE', field: `${PREFIX}.signal_targeting_groups` });
  });

  it('points the deprecated signal_targeting at its replacement', () => {
    expect(refusal(() => parseOverlay({ signal_targeting: [] }, seed, full, PREFIX)).message).toContain('signal_targeting_groups');
  });

  it('judges shapes before support for dimensions the product does take', () => {
    const error = refusal(() => parseOverlay({ geo_places: [place(['nowhere'], { system: 'https://other.example/p' })], geo_countries: ['USA'] }, seed, full, PREFIX));
    expect(error.code).toBe('INVALID_REQUEST');
  });

  it('judges globally malformed or unknown values before product support', () => {
    const unknownPlace = refusal(() => parseOverlay({ geo_places: [place(['nowhere'])] }, seed, takeover, PREFIX));
    expect({ code: unknownPlace.code, field: unknownPlace.field }).toEqual({ code: 'INVALID_REQUEST', field: `${PREFIX}.geo_places[0].values[0]` });
    const unknownSignal = refusal(() => parseOverlay({ signal_targeting_groups: groups(['any', ['no_such_segment']]) }, seed, takeover, PREFIX));
    expect({ code: unknownSignal.code, field: unknownSignal.field }).toEqual({ code: 'INVALID_REQUEST', field: `${PREFIX}.signal_targeting_groups.groups[0].signals[0].signal_ref.signal_id` });
    const malformed = refusal(() => parseOverlay({ signal_targeting_groups: { operator: 'any', groups: [] } }, seed, takeover, PREFIX));
    expect(malformed.code).toBe('INVALID_REQUEST');
  });

  it('refuses a repeated value inside one place entry', () => {
    for (const key of ['geo_places', 'geo_places_exclude']) {
      const error = refusal(() => parseOverlay({ [key]: [place(['us_west', 'us_west'])] }, seed, full, PREFIX));
      expect({ code: error.code, field: error.field }).toEqual({ code: 'INVALID_REQUEST', field: `${PREFIX}.${key}[0].values` });
    }
  });

  it('accepts a value repeated across entries and duplicate countries as a union', () => {
    const parsed = parseOverlay({ geo_countries: ['US', 'US'], geo_places: [place(['us_west']), place(['us_west'])] }, seed, full, PREFIX);
    expect(parsed.targeting.region_ids).toEqual(['us_west']);
  });

  it('never returns an empty region or profile list', () => {
    for (const overlay of [{ geo_countries: ['GB'] }, { signal_targeting_groups: groups(['none', ['sports_fans', 'news_loyalists', 'household_decision_makers']]) }]) {
      const error = refusal(() => parseOverlay(overlay, seed, unsegmented, PREFIX));
      expect(error.code).toBe('PRODUCT_UNAVAILABLE');
    }
  });
});

describe('validateOverlayGlobally', () => {
  it('passes absent, null and valid overlays', () => {
    expect(() => validateOverlayGlobally(undefined, seed, PREFIX)).not.toThrow();
    expect(() => validateOverlayGlobally(null, seed, PREFIX)).not.toThrow();
    expect(() => validateOverlayGlobally({ geo_places: [place(['us_west'])], signal_targeting_groups: groups(['any', ['sports_fans']]) }, seed, PREFIX)).not.toThrow();
  });

  it('does not judge dimensions that depend on the product', () => {
    expect(() => validateOverlayGlobally({ device_platform: ['ios'], geo_places: [place(['us_west'], { system: 'https://other.example/p' })] }, seed, PREFIX)).not.toThrow();
  });

  it('throws INVALID_REQUEST with the field for malformed input', () => {
    const error = refusal(() => validateOverlayGlobally({ geo_places: [place(['nowhere'])] }, seed, 'overlay'));
    expect({ code: error.code, field: error.field }).toEqual({ code: 'INVALID_REQUEST', field: 'overlay.geo_places[0].values[0]' });
  });
});

describe('parseOverlay: signals', () => {
  it('starts from all profiles on an unsegmented product', () => {
    const parsed = parseOverlay({ signal_targeting_groups: groups(['any', ['news_loyalists']]) }, seed, unsegmented, PREFIX);
    expect(parsed.targeting.profile_ids).toEqual(['professionals', 'retirees']);
    expect(parsed.targeting.region_ids).toBeUndefined();
  });

  it('unions the signals of an any group', () => {
    const parsed = parseOverlay({ signal_targeting_groups: groups(['any', ['news_loyalists', 'household_decision_makers']]) }, seed, unsegmented, PREFIX);
    expect(parsed.targeting.profile_ids).toEqual(['families', 'professionals', 'retirees']);
  });

  it('subtracts a none group', () => {
    const parsed = parseOverlay({ signal_targeting_groups: groups(['none', ['news_loyalists']]) }, seed, unsegmented, PREFIX);
    expect(parsed.targeting.profile_ids).toEqual(['families', 'young_urban']);
  });

  it('intersects groups across the all operator', () => {
    const parsed = parseOverlay({ signal_targeting_groups: groups(['any', ['news_loyalists']], ['any', ['household_decision_makers']]) }, seed, unsegmented, PREFIX);
    expect(parsed.targeting.profile_ids).toEqual(['professionals']);
  });

  it("intersects with the product's own segment", () => {
    const parsed = parseOverlay({ signal_targeting_groups: groups(['any', ['news_loyalists']]) }, seed, full, PREFIX);
    expect(parsed.targeting.profile_ids).toEqual(['professionals']);
    const excluded = parseOverlay({ signal_targeting_groups: groups(['none', ['news_loyalists']]) }, seed, full, PREFIX);
    expect(excluded.targeting.profile_ids).toEqual(['young_urban']);
  });

  it('combines geography and signals and echoes the groups as given', () => {
    const overlay = { geo_countries: ['CA'], signal_targeting_groups: groups(['none', ['news_loyalists']]) };
    const parsed = parseOverlay(overlay, seed, unsegmented, PREFIX);
    expect(parsed.targeting).toEqual({ region_ids: ['ca'], profile_ids: ['families', 'young_urban'] });
    expect(parsed.echo).toEqual(overlay);
  });
});

describe('productTargetingFields', () => {
  const system = placeSystem();
  const placeSupport = {
    systems: {
      [system]: {
        countries: { US: [placeType(system)], CA: [placeType(system)] },
        current_version: '1',
        system_versions: ['1'],
      },
    },
  };

  it('declares countries, places and signals on a full product', () => {
    const fields = productTargetingFields(seed, full);
    expect(fields.overlay_support).toEqual({
      geo_countries: true,
      geo_countries_exclude: true,
      geo_places: placeSupport,
      geo_places_exclude: placeSupport,
      signal_targeting_groups: true,
    });
    expect(fields.signal_targeting_allowed).toBe(true);
    expect(fields.signal_targeting_rules).toEqual({ resolution_model: 'direct_targeting', selection_mode: 'optional' });
  });

  it("offers the segments that share profiles, never the product's own", () => {
    const fields = productTargetingFields(seed, full);
    const options = fields.signal_targeting_options as Array<Record<string, unknown>>;
    expect(options.map(option => (option.signal_ref as { signal_id: string }).signal_id)).toEqual(['household_decision_makers', 'news_loyalists']);
    expect(options[0]).toEqual({
      signal_ref: { scope: 'product', signal_id: 'household_decision_makers' },
      name: 'Household decision makers',
      value_type: 'binary',
      allowed_targeting_modes: ['include', 'exclude'],
    });
    expect(fields.included_signals).toEqual([{ signal_ref: { scope: 'product', signal_id: 'sports_fans' }, name: 'Sports fans', value_type: 'binary' }]);
  });

  it('offers every segment on an unsegmented product and no included signals', () => {
    const fields = productTargetingFields(seed, unsegmented);
    expect((fields.signal_targeting_options as unknown[]).length).toBe(3);
    expect(fields.included_signals).toBeUndefined();
  });

  it('declares countries only on a countries product', () => {
    expect(productTargetingFields(seed, takeover)).toEqual({ overlay_support: { geo_countries: true, geo_countries_exclude: true } });
    expect(productTargetingFields(CONTENT_SEASON, content)).toEqual({ overlay_support: { geo_countries: true, geo_countries_exclude: true } });
  });

  it('declares no signal targeting when the seed has no segment to offer', () => {
    const fields = productTargetingFields({ ...seed, segments: [] }, unsegmented);
    expect(fields.signal_targeting_options).toBeUndefined();
    expect(fields.signal_targeting_allowed).toBeUndefined();
    expect(fields.signal_targeting_rules).toBeUndefined();
    expect(Object.keys(fields.overlay_support as Record<string, unknown>).sort()).toEqual(['geo_countries', 'geo_countries_exclude', 'geo_places', 'geo_places_exclude']);
  });

  it('declares exactly what parseOverlay accepts', () => {
    const support = productTargetingFields(seed, full).overlay_support as Record<string, unknown>;
    expect(Object.keys(support).sort()).toEqual(['geo_countries', 'geo_countries_exclude', 'geo_places', 'geo_places_exclude', 'signal_targeting_groups']);
  });
});

describe('supportsOverlayRequirements', () => {
  const support = productTargetingFields(seed, full).overlay_support;
  const system = placeSystem();
  const table: Array<[string, unknown, unknown, boolean]> = [
    ['true against true', { geo_countries: true }, { geo_countries: true }, true],
    ['true against an object', { geo_places: { systems: {} } }, { geo_places: true }, true],
    ['true against missing', { geo_countries: true }, { geo_places: true }, false],
    ['true against false', { geo_countries: false }, { geo_countries: true }, false],
    ['an object requirement against true', { geo_places: true }, { geo_places: { systems: { [system]: { countries: { US: [placeType(system)] } } } } }, true],
    ['a nested requirement against the declared support', support, { geo_places: { systems: { [system]: { countries: { US: [placeType(system)] } } } } }, true],
    ['an array subset', support, { geo_places: { systems: { [system]: { system_versions: ['1'] } } } }, true],
    ['an array that is not a subset', support, { geo_places: { systems: { [system]: { system_versions: ['1', '2'] } } } }, false],
    ['a country the support lacks', support, { geo_places: { systems: { [system]: { countries: { GB: [placeType(system)] } } } } }, false],
    ['a system the support lacks', support, { geo_places: { systems: { 'https://other.example/p': { countries: { US: ['city'] } } } } }, false],
    ['a place type the support lacks', support, { geo_places: { systems: { [system]: { countries: { US: ['city'] } } } } }, false],
    ['an unknown dimension', support, { geo_regions: true }, false],
    ['an inherited property name', support, { toString: true }, false],
    ['a scalar requirement equal to support', { max_values: 5 }, { max_values: 5 }, true],
    ['a scalar requirement that differs', { max_values: 5 }, { max_values: 6 }, false],
    ['several dimensions, all met', support, { geo_countries: true, signal_targeting_groups: true }, true],
    ['several dimensions, one unmet', support, { geo_countries: true, daypart_targets: true }, false],
    ['an object requirement against missing support', undefined, { geo_places: true }, false],
    ['no requirement', support, undefined, true],
    ['an empty requirement', support, {}, true],
    ['an object against a boolean false', { geo_places: false }, { geo_places: { systems: {} } }, false],
  ];
  it.each(table)('%s', (_name, given, required, expected) => {
    expect(supportsOverlayRequirements(given, required)).toBe(expected);
  });

  it('matches a countries product only on countries', () => {
    const countries = productTargetingFields(seed, takeover).overlay_support;
    expect(supportsOverlayRequirements(countries, { geo_countries: true })).toBe(true);
    expect(supportsOverlayRequirements(countries, { geo_places: true })).toBe(false);
  });
});

describe('resolveGeoPlaces', () => {
  const ok = (query: Record<string, unknown>) => {
    const result = resolveGeoPlaces(seed, query);
    if (result.status !== 200) throw new Error(`expected 200, got ${JSON.stringify(result)}`);
    return result.body;
  };
  const type = placeType(placeSystem());

  it('finds regions by name, case-insensitively', () => {
    const body = ok({ q: 'us north', country: 'US' });
    expect(body.matches.map(match => match.value)).toEqual(['us_northeast']);
    expect(body.system).toBe(placeSystem());
    expect(body.system_version).toBe('1');
    expect(body.matches[0]).toEqual({
      value: 'us_northeast',
      country: 'US',
      place_type: type,
      label: 'US Northeast',
      canonical_name: 'US Northeast',
      parent_labels: [expect.any(String)],
      status: 'active',
    });
  });

  it('finds regions by id and lists them all for a shared stem', () => {
    expect(ok({ q: 'us_', country: 'US' }).matches.map(match => match.value)).toEqual(['us_northeast', 'us_south', 'us_west']);
  });

  it('looks up an exact value', () => {
    expect(ok({ value: 'us_west', country: 'US' }).matches.map(match => match.value)).toEqual(['us_west']);
    expect(ok({ value: 'us_wes', country: 'US' }).matches).toEqual([]);
  });

  it('filters by country', () => {
    expect(ok({ q: 'us', country: 'CA' }).matches).toEqual([]);
    expect(ok({ value: 'ca', country: 'US' }).matches).toEqual([]);
    expect(ok({ q: 'canada', country: 'CA' }).matches.map(match => match.value)).toEqual(['ca']);
    expect(ok({ q: 'x', country: 'GB' }).matches).toEqual([]);
  });

  it('applies limit and normalises the request echo', () => {
    const body = ok({ q: 'us', country: 'us', limit: '2', cursor: 'abc', extra: 'ignored' });
    expect(body.matches).toHaveLength(2);
    expect(body.request).toEqual({ q: 'us', country: 'US', limit: 2 });
    expect('next_cursor' in body).toBe(false);
    expect(ok({ q: 'us', country: 'US' }).request.limit).toBe(20);
  });

  it('echoes place_type and system_version when sent', () => {
    expect(ok({ q: 'us', country: 'US', place_type: type, system_version: '1' }).request).toEqual({ q: 'us', country: 'US', place_type: type, system_version: '1', limit: 20 });
  });

  it('returns 200 with no matches for a foreign place type', () => {
    const body = ok({ q: 'us', country: 'US', place_type: 'city' });
    expect(body.matches).toEqual([]);
    expect(body.request.place_type).toBe('city');
  });

  it('filters on subdivision and echoes it', () => {
    const body = ok({ q: 'us', country: 'US', subdivision: 'US-CA' });
    expect(body.matches).toEqual([]);
    expect(body.request.subdivision).toBe('US-CA');
    const subdivided: SeasonSeed = { ...seed, regions: seed.regions.map(region => (region.region_id === 'us_west' ? { ...region, subdivision: 'US-CA' } : region)) };
    const result = resolveGeoPlaces(subdivided, { q: 'us', country: 'US', subdivision: 'US-CA' });
    expect(result.status === 200 && result.body.matches.map(match => match.value)).toEqual(['us_west']);
  });

  it.each([
    ['neither q nor value', { country: 'US' }],
    ['both q and value', { q: 'us', value: 'us_west', country: 'US' }],
    ['an empty q', { q: '', country: 'US' }],
    ['no country', { q: 'us' }],
    ['a non-ISO country', { q: 'us', country: 'USA' }],
    ['a limit of zero', { q: 'us', country: 'US', limit: 0 }],
    ['a limit above 100', { q: 'us', country: 'US', limit: 101 }],
    ['a fractional limit', { q: 'us', country: 'US', limit: 1.5 }],
    ['a non-numeric limit', { q: 'us', country: 'US', limit: 'many' }],
    ['an unsupported system_version', { q: 'us', country: 'US', system_version: '2' }],
    ['a malformed subdivision', { q: 'us', country: 'US', subdivision: 'california' }],
  ])('returns 400 for %s', (_name, query) => {
    const result = resolveGeoPlaces(seed, query);
    expect(result.status).toBe(400);
    expect((result.body as { error: string }).error.length).toBeGreaterThan(0);
  });

  describe.each(['3.2', '3.3.0-rc.3'])('against the %s response schema', version => {
    const root = join(process.cwd(), 'node_modules/@adcp/sdk/dist/lib/schemas-data', version);
    const ajv = new Ajv({ strict: false, allErrors: true });
    addFormats(ajv);
    for (const dir of ['core', 'enums']) {
      for (const file of readdirSync(join(root, dir)).filter(name => name.endsWith('.json'))) {
        ajv.addSchema(JSON.parse(readFileSync(join(root, dir, file), 'utf8')));
      }
    }
    const validate = ajv.getSchema(JSON.parse(readFileSync(join(root, 'core/get-geo-place-resolution-response.json'), 'utf8')).$id)!;

    it.each([
      ['a name search', { q: 'us', country: 'US' }],
      ['a value lookup', { value: 'ca', country: 'CA' }],
      ['an empty result', { q: 'zzz', country: 'US', limit: 5, place_type: type, system_version: '1' }],
      ['a subdivision filter', { q: 'us', country: 'US', subdivision: 'US-CA' }],
    ])('validates %s', (_name, query) => {
      const body = ok(query);
      expect(validate(body), JSON.stringify(validate.errors)).toBe(true);
    });
  });
});
