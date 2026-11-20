/**
 * Season files: the YAML/JSON loader that lets a season be supplied as data
 * instead of a compiled-in constant.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { parse as parseYaml } from 'yaml';
import {
  ADSERVER_SEASON,
  CONTENT_SEASON,
  EngineError,
  InventorySimulation,
  SeasonTraffic,
  type SeasonSeed,
} from '../../src/training-agent/inventory-sim/engine/index.js';
import {
  parseSeasonSeed,
  readSeasonFile,
  seasonToYaml,
} from '../../src/training-agent/inventory-sim/season-file.js';
import { TRAINING_FORMAT_LABELS } from '../../src/training-agent/product-factory.js';
import {
  INVENTORY_ADSERVER_TENANT_ID,
  INVENTORY_CONTENT_TENANT_ID,
  INVENTORY_SEASON_FILE_ENV,
  resolveInventoryTenantProfiles,
} from '../../src/training-agent/tenants/inventory.js';

const at = (iso: string) => Date.parse(iso);

let dir: string;
beforeAll(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'season-file-'));
});
afterAll(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

function write(name: string, content: string): string {
  const file = path.join(dir, name);
  fs.writeFileSync(file, content);
  return file;
}

function invalidSeedMessage(fn: () => unknown): string {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(EngineError);
  expect((caught as EngineError).code).toBe('INVALID_SEED');
  return (caught as EngineError).message;
}

const SEEDS: Array<[string, SeasonSeed]> = [
  ['ADSERVER_SEASON', ADSERVER_SEASON],
  ['CONTENT_SEASON', CONTENT_SEASON],
];

/** Smallest valid season, as YAML text. */
const MINIMAL_YAML = `
seed_id: tiny
rng_seed: tiny-v1
start_date: 2027-01-01
days: 30
currency: USD
regions:
  - { region_id: r1, country: US, name: Region one, share: 1 }
audience_profiles:
  - { profile_id: p1, name: Profile one, share: 1 }
segments: []
properties:
  - { property_id: site, name: Site, domain: site.example }
placements:
  - placement_id: banner
    name: Banner
    channel: display
    property_ids: [site]
    formats: [display_300x250]
    base_daily_impressions: 10000
    floor_cpm: 2
    rates: { viewability: 0.6, click_rate: 0.01, completion_rate: 0, conversion_rate: 0.02 }
    reporting_lag_days: 0
products:
  - product_id: banner_open
    name: Banner
    description: Open auction banner
    delivery_type: non_guaranteed
    placement_ids: [banner]
traffic:
  weekday_shape: [1, 1, 1, 1, 1, 1, 1]
forecast_error:
  channel_sigma: { display: 0.05 }
  ar1_phi: 0.5
  lead_time_tau_days: 14
  lead_time_horizon_days: 60
  log_clip: 0.6931471805599453
  cell_kappa: 0.5
  cell_min_impressions: 100
`;

describe('season files', () => {
  it('loads every shipped season through YAML and JSON unchanged', () => {
    for (const [name, seed] of SEEDS) {
      const yamlFile = write(`${name}.yaml`, seasonToYaml(seed));
      const jsonFile = write(`${name}.json`, JSON.stringify(seed));
      expect(readSeasonFile(yamlFile)).toEqual(seed);
      expect(readSeasonFile(jsonFile)).toEqual(seed);
    }
  });

  it('runs the same market from a loaded season as from the constant', () => {
    const now = at('2026-12-31T12:00:00Z');
    const flight = { flight_start_ms: at('2027-01-01'), flight_end_ms: at('2027-03-01') };
    const requests = [
      {
        seed: ADSERVER_SEASON,
        book: { booking_id: 'b', product_id: 'sports_hero_guaranteed', ...flight, budget: 10_000, impression_goal: 180_000 },
        query: { product_id: 'sports_hero_guaranteed', ...flight },
      },
      {
        seed: CONTENT_SEASON,
        book: { booking_id: 'b', product_id: 'morning_brief_midroll', targeting: { installment_ids: ['morning_brief_011'] }, ...flight, budget: 5000 },
        query: { product_id: 'morning_brief_midroll', targeting: { installment_ids: ['morning_brief_013'] }, ...flight },
      },
    ];
    for (const { seed, book, query } of requests) {
      const loaded = readSeasonFile(write(`${seed.seed_id}.yaml`, seasonToYaml(seed)));
      const run = (s: SeasonSeed) => {
        const sim = InventorySimulation.create(s, now);
        sim.book(book, now);
        sim.advanceTo(at('2027-02-01T00:00:00Z'));
        return { sim, forecast: sim.forecast(query, at('2027-02-01T00:00:00Z')) };
      };
      const fromConstant = run(seed);
      const fromFile = run(loaded);
      expect(fromFile.sim.state).toEqual(fromConstant.sim.state);
      expect(fromFile.forecast).toEqual(fromConstant.forecast);
    }
  });

  it('reads unquoted dates as strings', () => {
    const seed = readSeasonFile(write('dates.yaml', MINIMAL_YAML));
    expect(seed.start_date).toBe('2027-01-01');
  });

  it('names the path of each problem', () => {
    const badShape = write('bad-shape.yaml', MINIMAL_YAML.replace('days: 30', 'days: "30"') + 'forecast_eror: {}\n');
    const message = invalidSeedMessage(() => readSeasonFile(badShape));
    expect(message).toContain(`Season ${badShape} is invalid:`);
    expect(message).toContain('  days: ');
    expect(message).toContain('forecast_eror');

    const badShares = write('bad-shares.yaml', MINIMAL_YAML.replace('share: 1 }\naudience', 'share: 1.2 }\naudience'));
    expect(invalidSeedMessage(() => readSeasonFile(badShares))).toMatch(/region shares sum to 1\.2/);
  });

  it('indexes array paths and reports every zod issue', () => {
    const seed = JSON.parse(JSON.stringify(ADSERVER_SEASON));
    seed.placements[0].floor_cpm = 'high';
    seed.regions[1].share = 'half';
    const message = invalidSeedMessage(() => parseSeasonSeed(seed, 'inline'));
    expect(message).toContain('  placements[0].floor_cpm: ');
    expect(message).toContain('  regions[1].share: ');
  });

  it('names the field of a date that is not YYYY-MM-DD', () => {
    const file = write('bad-date.yaml', MINIMAL_YAML.replace('start_date: 2027-01-01', 'start_date: 2027-1-5'));
    expect(invalidSeedMessage(() => readSeasonFile(file))).toContain('  start_date: ');
  });

  it.each(['2027/01/05', 'abc', '2027-1-5'])('reports only the format problem for the date %s', date => {
    const seed = parseYaml(MINIMAL_YAML);
    seed.start_date = date;
    const message = invalidSeedMessage(() => parseSeasonSeed(seed, 'dates'));
    expect(message).toContain('start_date: must be a YYYY-MM-DD date');
    expect(message).not.toContain('not a calendar date');
  });

  it('refuses digit separators', () => {
    const file = write('separators.yaml', MINIMAL_YAML.replace('base_daily_impressions: 10000', 'base_daily_impressions: 2_200_000'));
    expect(invalidSeedMessage(() => readSeasonFile(file))).toContain('placements[0].base_daily_impressions');
  });

  it('refuses a product property that is unknown', () => {
    const file = write(
      'product-unknown-property.yaml',
      MINIMAL_YAML.replace('    placement_ids: [banner]', '    placement_ids: [banner]\n    property_ids: [missing]'),
    );
    const message = invalidSeedMessage(() => readSeasonFile(file));
    expect(message).toContain('products[0].property_ids[0]');
    expect(message).toContain('missing');
  });

  it('refuses a product property its placements do not serve', () => {
    const file = write(
      'product-unserved-property.yaml',
      MINIMAL_YAML
        .replace('properties:\n', 'properties:\n  - { property_id: harbor_weekly, name: Harbor, domain: harbor.example }\n')
        .replace('    placement_ids: [banner]', '    placement_ids: [banner]\n    property_ids: [harbor_weekly]'),
    );
    const message = invalidSeedMessage(() => readSeasonFile(file));
    expect(message).toContain('products[0].property_ids[0]');
    expect(message).toContain('harbor_weekly');
  });

  it('refuses a traffic event naming an unknown region, property or placement', () => {
    const file = write(
      'event-unknown-references.yaml',
      `${MINIMAL_YAML}events:
  - type: traffic_multiplier
    event_id: surge
    description: Surge
    start_date: 2027-01-05
    end_date: 2027-01-07
    multiplier: 2
    placement_ids: [no_placement]
    property_ids: [no_property]
    region_ids: [no_region]
`,
    );
    const message = invalidSeedMessage(() => readSeasonFile(file));
    expect(message).toContain('events[0].placement_ids[0]');
    expect(message).toContain('no_placement');
    expect(message).toContain('events[0].property_ids[0]');
    expect(message).toContain('no_property');
    expect(message).toContain('events[0].region_ids[0]');
    expect(message).toContain('no_region');
  });

  it('refuses a date that is not on the calendar', () => {
    const start = write('feb30.yaml', MINIMAL_YAML.replace('start_date: 2027-01-01', 'start_date: 2027-02-30'));
    const startMessage = invalidSeedMessage(() => readSeasonFile(start));
    expect(startMessage).toContain(`Season ${start} is invalid`);
    expect(startMessage).toContain('start_date');

    const event = write(
      'apr31.yaml',
      `${MINIMAL_YAML}events:
  - type: traffic_multiplier
    event_id: surge
    description: Surge
    start_date: 2027-04-01
    end_date: 2027-04-31
    multiplier: 2
`,
    );
    const eventMessage = invalidSeedMessage(() => readSeasonFile(event));
    expect(eventMessage).toContain(`Season ${event} is invalid`);
    expect(eventMessage).toContain('events[0].end_date');
  });

  it('refuses a product that sells a collection on another placement', () => {
    const seed = structuredClone(CONTENT_SEASON);
    seed.products[0].collection_ids = ['weekly_dispatch'];
    const message = invalidSeedMessage(() => parseSeasonSeed(seed, 'content-copy'));
    expect(message).toContain('products[0]');
    expect(message).toContain('weekly_dispatch');
  });

  it('refuses a sponsorship product with no collections', () => {
    const seed = structuredClone(CONTENT_SEASON);
    delete seed.products[1].collection_ids;
    const message = invalidSeedMessage(() => parseSeasonSeed(seed, 'content-copy'));
    expect(message).toContain('products[1]');
    expect(message).toContain('covers no inventory');
  });

  it('refuses empty and multi-document files', () => {
    expect(invalidSeedMessage(() => readSeasonFile(write('empty.yaml', '')))).toContain('empty');
    expect(invalidSeedMessage(() => readSeasonFile(write('comments.yaml', '# nothing here\n')))).toContain('empty');
    expect(invalidSeedMessage(() => readSeasonFile(write('multi.yaml', `${MINIMAL_YAML}\n---\n${MINIMAL_YAML}`)))).toContain('one document');
  });

  it('reports YAML syntax errors with the line', () => {
    const file = write('syntax.yaml', 'seed_id: tiny\nregions: [unclosed\ndays: 3\n');
    expect(invalidSeedMessage(() => readSeasonFile(file))).toMatch(/syntax\.yaml:\d+:\d+/);
  });

  it('refuses duplicate keys', () => {
    const file = write('dupes.yaml', MINIMAL_YAML.replace('days: 30', 'days: 30\ndays: 60'));
    expect(invalidSeedMessage(() => readSeasonFile(file))).toMatch(/dupes\.yaml:\d+:\d+/);
  });
});

describe('season values the tenants put on the wire', () => {
  const refuses: Array<[string, (yaml: string) => string, string]> = [
    ['a lowercase currency', y => y.replace('currency: USD', 'currency: usd'), 'currency:'],
    ['a currency that is not three letters', y => y.replace('currency: USD', 'currency: DOLLARS'), 'currency:'],
    ['a property domain that is not a hostname', y => y.replace('domain: site.example', 'domain: https://site.example/'), 'properties[0].domain:'],
    ['an upper-case property domain', y => y.replace('domain: site.example', 'domain: Site.Example'), 'properties[0].domain:'],
    ['a placement with no formats', y => y.replace('formats: [display_300x250]', 'formats: []'), 'placements[0].formats: needs at least one format'],
    ['a negative floor', y => y.replace('floor_cpm: 2', 'floor_cpm: -1'), 'placements[0].floor_cpm: must be 0 or more'],
  ];

  it.each(refuses)('refuses %s', (_name, edit, where) => {
    expect(invalidSeedMessage(() => parseSeasonSeed(parseYaml(edit(MINIMAL_YAML)), 'wire'))).toContain(where);
  });

  it('accepts a zero floor and a hyphenated multi-label domain', () => {
    const yaml = MINIMAL_YAML.replace('floor_cpm: 2', 'floor_cpm: 0').replace('domain: site.example', 'domain: news-desk.example.co');
    expect(parseSeasonSeed(parseYaml(yaml), 'wire').placements[0].floor_cpm).toBe(0);
  });
});

describe('season number ranges', () => {
  const starterPath = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../fixtures/inventory-sim/small-season.yaml');
  const collection = {
    collection_id: 'show',
    name: 'Show',
    channel: 'podcast',
    property_id: 'harbor_daily',
    placement_id: 'banner',
    first_date: '2026-11-01',
    cadence_days: 7,
    count: 10,
    expected_audience: 1000,
    audience_sigma: 0.1,
  };
  const SPIKES = { per_publisher_min: 1, per_publisher_max: 3, size_min: 0.2, size_max: 0.8, decay: 0.7, decay_forecast_sigma: 0.1 };
  /* eslint-disable @typescript-eslint/no-explicit-any */
  const cases: Array<[string, (seed: any) => void, string]> = [
    ['a negative reporting lag', s => { s.placements[0].reporting_lag_days = -1; }, 'placements[0].reporting_lag_days:'],
    ['a fractional reporting lag', s => { s.placements[0].reporting_lag_days = 1.5; }, 'placements[0].reporting_lag_days:'],
    ['a zero cadence', s => { s.collections = [{ ...collection, cadence_days: 0 }]; }, 'collections[0].cadence_days:'],
    ['a fractional cadence', s => { s.collections = [{ ...collection, cadence_days: 1.5 }]; }, 'collections[0].cadence_days:'],
    ['a zero count', s => { s.collections = [{ ...collection, count: 0 }]; }, 'collections[0].count:'],
    ['a fractional count', s => { s.collections = [{ ...collection, count: 2.5 }]; }, 'collections[0].count:'],
    ['a negative expected audience', s => { s.collections = [{ ...collection, expected_audience: -1 }]; }, 'collections[0].expected_audience:'],
    ['a negative audience sigma', s => { s.collections = [{ ...collection, audience_sigma: -0.1 }]; }, 'collections[0].audience_sigma:'],
    ['a zero share of voice', s => { s.products[0].share_of_voice = 0; }, 'products[0].share_of_voice:'],
    ['a share of voice above 1', s => { s.products[0].share_of_voice = 1.5; }, 'products[0].share_of_voice:'],
    ['a placement rate below 0', s => { s.placements[0].rates.click_rate = -0.1; }, 'placements[0].rates.click_rate:'],
    ['a placement rate above 1', s => { s.placements[0].rates.viewability = 1.1; }, 'placements[0].rates.viewability:'],
    ['a negative rate multiplier', s => { s.audience_profiles[0].rate_multipliers.click_rate = -1; }, 'audience_profiles[0].rate_multipliers.click_rate:'],
    ['a negative weekday shape', s => { s.traffic.weekday_shape[3] = -1; }, 'traffic.weekday_shape[3]:'],
    ['a negative traffic multiplier', s => { s.properties[0].traffic_multiplier = -1; }, 'properties[0].traffic_multiplier:'],
    ['a negative channel sigma', s => { s.forecast_error.channel_sigma.display = -0.1; }, 'forecast_error.channel_sigma.display:'],
    ['an ar1_phi of 1', s => { s.forecast_error.ar1_phi = 1; }, 'forecast_error.ar1_phi:'],
    ['a negative ar1_phi', s => { s.forecast_error.ar1_phi = -0.1; }, 'forecast_error.ar1_phi:'],
    ['a zero lead_time_tau_days', s => { s.forecast_error.lead_time_tau_days = 0; }, 'forecast_error.lead_time_tau_days:'],
    ['a negative lead_time_horizon_days', s => { s.forecast_error.lead_time_horizon_days = -1; }, 'forecast_error.lead_time_horizon_days:'],
    ['a fractional lead_time_horizon_days', s => { s.forecast_error.lead_time_horizon_days = 1.5; }, 'forecast_error.lead_time_horizon_days:'],
    ['a zero log_clip', s => { s.forecast_error.log_clip = 0; }, 'forecast_error.log_clip:'],
    ['a negative cell_kappa', s => { s.forecast_error.cell_kappa = -1; }, 'forecast_error.cell_kappa:'],
    ['a cell_min_impressions below 1', s => { s.forecast_error.cell_min_impressions = 0.5; }, 'forecast_error.cell_min_impressions:'],
    ['a negative truth_daily_sigma', s => { s.traffic.truth_daily_sigma = -0.1; }, 'traffic.truth_daily_sigma:'],
    ['a month_shape of eleven values', s => { s.traffic.month_shape = Array(11).fill(1); }, 'traffic.month_shape'],
    ['a month_shape with a zero', s => { s.traffic.month_shape = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]; }, 'traffic.month_shape[2]'],
    ['a negative bias_sigma', s => { s.forecast_error.publishers = [{ publisher_id: 'harbor_daily', bias: 0, bias_sigma: -1 }]; }, 'forecast_error.publishers[0].bias_sigma:'],
    ['a negative error_scale', s => { s.forecast_error.publishers = [{ publisher_id: 'harbor_daily', bias: 0, error_scale: -1 }]; }, 'forecast_error.publishers[0].error_scale:'],
    ['a negative per_publisher_min', s => { s.forecast_error.spikes = { ...SPIKES, per_publisher_min: -1 }; }, 'forecast_error.spikes.per_publisher_min:'],
    ['a size_min of -1', s => { s.forecast_error.spikes = { ...SPIKES, size_min: -1 }; }, 'forecast_error.spikes.size_min:'],
    ['a per_publisher_min above per_publisher_max', s => { s.forecast_error.spikes = { ...SPIKES, per_publisher_min: 4, per_publisher_max: 3 }; }, 'forecast_error.spikes.per_publisher_max:'],
    ['a size_min above size_max', s => { s.forecast_error.spikes = { ...SPIKES, size_min: 1, size_max: 0.5 }; }, 'forecast_error.spikes.size_max:'],
    ['a per_publisher_max above the days', s => { s.forecast_error.spikes = { ...SPIKES, per_publisher_max: 366 }; }, 'forecast_error.spikes.per_publisher_max:'],
    ['a spike decay of 1', s => { s.forecast_error.spikes = { ...SPIKES, decay: 1 }; }, 'forecast_error.spikes.decay:'],
    ['a negative decay_forecast_sigma', s => { s.forecast_error.spikes = { ...SPIKES, decay_forecast_sigma: -1 }; }, 'forecast_error.spikes.decay_forecast_sigma:'],
    ['a publisher no property has', s => { s.forecast_error.publishers = [{ publisher_id: 'nobody', bias: 0 }]; }, 'forecast_error.publishers[0].publisher_id: unknown publisher nobody'],
    ['a publisher listed twice', s => { s.forecast_error.publishers = [{ publisher_id: 'harbor_daily', bias: 0 }, { publisher_id: 'harbor_daily', bias: 0.1 }]; }, 'forecast_error.publishers[1].publisher_id'],
  ];
  /* eslint-enable @typescript-eslint/no-explicit-any */

  it('refuses out-of-range numbers', () => {
    for (const [name, mutate, where] of cases) {
      const seed = parseYaml(fs.readFileSync(starterPath, 'utf8'));
      mutate(seed);
      const message = invalidSeedMessage(() => parseSeasonSeed(seed, 'ranges'));
      expect(message, name).toContain(where);
    }
  });

  it('names what is wrong after the path', () => {
    const seed = parseYaml(fs.readFileSync(starterPath, 'utf8'));
    seed.placements[0].reporting_lag_days = -1;
    expect(invalidSeedMessage(() => parseSeasonSeed(seed, 'ranges'))).toContain(
      'placements[0].reporting_lag_days: must be a whole number, 0 or more',
    );
  });

  it('accepts the bounds themselves', () => {
    const seed = parseYaml(fs.readFileSync(starterPath, 'utf8'));
    seed.placements[0].reporting_lag_days = 0;
    seed.placements[0].rates.click_rate = 1;
    seed.products[0].share_of_voice = 1;
    seed.forecast_error.publishers = [{ publisher_id: 'harbor_daily', bias: -5, bias_sigma: 0, error_scale: 0 }];
    seed.forecast_error.ar1_phi = 0;
    seed.forecast_error.lead_time_horizon_days = 0;
    seed.forecast_error.cell_min_impressions = 1;
    seed.forecast_error.channel_sigma.display = 0;
    seed.traffic.truth_daily_sigma = 0;
    seed.traffic.trend_per_day = -0.01;
    expect(() => parseSeasonSeed(seed, 'ranges')).not.toThrow();
  });

  it('accepts spikes at their bounds', () => {
    const seed = parseYaml(fs.readFileSync(starterPath, 'utf8'));
    seed.forecast_error.spikes = { per_publisher_min: 0, per_publisher_max: 365, size_min: -0.99, size_max: -0.99, decay: 0, decay_forecast_sigma: 0 };
    expect(() => parseSeasonSeed(seed, 'ranges')).not.toThrow();
  });
});

describe('forecast error values', () => {
  it('names the old keys and the missing channel_sigma', () => {
    const seed = parseYaml(MINIMAL_YAML);
    seed.forecast_error = { day_sigma: 0.05, ar1_phi: 0.5, lead_time_tau_days: 14, lead_time_horizon_days: 60, log_clip: 0.7, cell_kappa: 0, cell_min_impressions: 100 };
    const message = invalidSeedMessage(() => parseSeasonSeed(seed, 'old'));
    expect(message).toContain('forecast_error.day_sigma');
    expect(message).toContain('forecast_error.channel_sigma');
  });

  it('refuses a placement channel with no channel_sigma, in the loader and in the engine', () => {
    const seed = parseYaml(MINIMAL_YAML);
    seed.placements.push({ ...seed.placements[0], placement_id: 'show_spot', channel: 'podcast' });
    const message = invalidSeedMessage(() => parseSeasonSeed(seed, 'podcast'));
    expect(message).toContain('forecast_error.channel_sigma.podcast');
    expect(message).toContain('show_spot');
    expect(invalidSeedMessage(() => new SeasonTraffic(seed))).toContain('forecast_error.channel_sigma.podcast');
  });

  it('refuses a channel_sigma key that is not a channel, naming the path', () => {
    const seed = parseYaml(MINIMAL_YAML);
    seed.forecast_error.channel_sigma.radio = 0.1;
    const message = invalidSeedMessage(() => parseSeasonSeed(seed, 'unknown channel'));
    expect(message).toContain('forecast_error.channel_sigma');
    expect(message).toContain('forecast_error.channel_sigma.radio');
  });

  it('keeps the supported-formats line last with the forecast error problems', () => {
    const seed = parseYaml(MINIMAL_YAML);
    seed.forecast_error.publishers = [{ publisher_id: 'nobody', bias: 0 }];
    seed.placements[0].formats = ['custom_banner'];
    const lines = invalidSeedMessage(() => parseSeasonSeed(seed, 'both', { formats: new Set(['display_300x250']) })).trim().split('\n');
    expect(lines.some(line => line.includes('unknown publisher nobody'))).toBe(true);
    expect(lines[lines.length - 1]).toContain('supported formats:');
  });

  it('groups properties into a publisher by publisher_id', () => {
    const seed = parseYaml(MINIMAL_YAML);
    seed.properties.push({ property_id: 'site_b', name: 'Site B', domain: 'site-b.example', publisher_id: 'house' });
    seed.forecast_error.publishers = [{ publisher_id: 'house', bias: 0.1 }, { publisher_id: 'site', bias: 0.2 }];
    expect(() => parseSeasonSeed(seed, 'groups')).not.toThrow();
  });

  /* eslint-disable @typescript-eslint/no-explicit-any */
  const engineCases: Array<[string, (seed: any) => void, string]> = [
    ['a missing ar1_phi', s => { delete s.forecast_error.ar1_phi; }, 'forecast_error.ar1_phi'],
    ['a missing channel_sigma', s => { delete s.forecast_error.channel_sigma; }, 'forecast_error.channel_sigma'],
    ['a NaN sigma', s => { s.forecast_error.channel_sigma.display = NaN; }, 'forecast_error.channel_sigma.display'],
    ['a missing lead_time_tau_days', s => { delete s.forecast_error.lead_time_tau_days; }, 'forecast_error.lead_time_tau_days'],
    ['a missing lead_time_horizon_days', s => { delete s.forecast_error.lead_time_horizon_days; }, 'forecast_error.lead_time_horizon_days'],
    ['a missing log_clip', s => { delete s.forecast_error.log_clip; }, 'forecast_error.log_clip'],
    ['a missing cell_kappa', s => { delete s.forecast_error.cell_kappa; }, 'forecast_error.cell_kappa'],
    ['a missing cell_min_impressions', s => { delete s.forecast_error.cell_min_impressions; }, 'forecast_error.cell_min_impressions'],
    ['a month_shape of eleven values', s => { s.traffic.month_shape = Array(11).fill(1); }, 'traffic.month_shape'],
    ['a negative truth_daily_sigma', s => { s.traffic.truth_daily_sigma = -1; }, 'traffic.truth_daily_sigma'],
    ['an unknown publisher', s => { s.forecast_error.publishers = [{ publisher_id: 'nobody', bias: 0 }]; }, 'forecast_error.publishers[0]'],
    ['a publisher listed twice', s => { s.forecast_error.publishers = [{ publisher_id: 'site', bias: 0 }, { publisher_id: 'site', bias: 0 }]; }, 'forecast_error.publishers[1]'],
    ['an invalid spikes block', s => { s.forecast_error.spikes = { per_publisher_min: 2, per_publisher_max: 1, size_min: 0, size_max: 1, decay: 0.5, decay_forecast_sigma: 0 }; }, 'forecast_error.spikes.per_publisher_max'],
  ];
  /* eslint-enable @typescript-eslint/no-explicit-any */

  it.each(engineCases)('makes the engine refuse %s instead of producing NaN', (_name, mutate, where) => {
    const seed = parseYaml(MINIMAL_YAML);
    mutate(seed);
    expect(invalidSeedMessage(() => new SeasonTraffic(seed))).toContain(where);
  });
});

describe('season file formats', () => {
  const CUSTOM = MINIMAL_YAML.replace('formats: [display_300x250]', 'formats: [custom_banner, display_300x250]');
  const FORMATS = new Set(['display_300x250', 'video_preroll']);

  it('refuses a format the host does not support', () => {
    const message = invalidSeedMessage(() => parseSeasonSeed(parseYaml(CUSTOM), 'custom', { formats: FORMATS }));
    expect(message).toContain('placements[0].formats[0]: unsupported format custom_banner');
    expect(message).not.toContain('formats[1]');
    expect(message).toContain('supported formats: display_300x250, video_preroll');
    const file = write('custom-format.yaml', CUSTOM);
    expect(invalidSeedMessage(() => readSeasonFile(file, { formats: FORMATS }))).toContain('custom_banner');
  });

  it('leaves formats to the host when none are given', () => {
    expect(parseSeasonSeed(parseYaml(CUSTOM), 'custom').placements[0].formats).toEqual(['custom_banner', 'display_300x250']);
    expect(readSeasonFile(write('custom-format-open.yaml', CUSTOM)).placements[0].formats[0]).toBe('custom_banner');
  });

  it('reports a format problem together with a reference problem', () => {
    const seed = parseYaml(CUSTOM);
    seed.products[0].property_ids = ['missing'];
    const message = invalidSeedMessage(() => parseSeasonSeed(seed, 'both', { formats: FORMATS }));
    expect(message).toContain('products[0].property_ids[0]: unknown property missing');
    expect(message).toContain('placements[0].formats[0]: unsupported format custom_banner');
    const lines = message.trim().split('\n');
    expect(lines[lines.length - 1]).toContain('supported formats:');
  });

  it('reports an unknown product placement without blaming its properties', () => {
    const seed = parseYaml(MINIMAL_YAML);
    seed.products[0].placement_ids = ['nowhere'];
    seed.products[0].property_ids = ['site'];
    const message = invalidSeedMessage(() => parseSeasonSeed(seed, 'placement'));
    expect(message).not.toContain('is not on any of the product');
    expect(message).toContain('nowhere');
  });

  it('knows every format the shipped seasons and the starter fixture use', () => {
    const starter = readSeasonFile(path.resolve(path.dirname(new URL(import.meta.url).pathname), '../fixtures/inventory-sim/small-season.yaml'));
    for (const seed of [ADSERVER_SEASON, CONTENT_SEASON, starter]) {
      for (const placement of seed.placements) {
        for (const format of placement.formats) expect(TRAINING_FORMAT_LABELS.has(format), `${seed.seed_id}: ${format}`).toBe(true);
      }
    }
  });
});

describe('resolveInventoryTenantProfiles', () => {
  const FIXTURE = path.resolve(
    path.dirname(new URL(import.meta.url).pathname),
    '../fixtures/inventory-sim/small-season.yaml',
  );
  const ADSERVER_VAR = INVENTORY_SEASON_FILE_ENV[INVENTORY_ADSERVER_TENANT_ID];
  const CONTENT_VAR = INVENTORY_SEASON_FILE_ENV[INVENTORY_CONTENT_TENANT_ID];

  it('names one env var per tenant', () => {
    expect(ADSERVER_VAR).toBe('TRAINING_AGENT_INVENTORY_ADSERVER_SEASON_FILE');
    expect(CONTENT_VAR).toBe('TRAINING_AGENT_INVENTORY_CONTENT_SEASON_FILE');
  });

  it('uses the shipped seasons, by identity, when nothing is set', () => {
    for (const env of [{}, { [ADSERVER_VAR]: '', [CONTENT_VAR]: '   ' }]) {
      const profiles = resolveInventoryTenantProfiles(env);
      expect(profiles[INVENTORY_ADSERVER_TENANT_ID].seed).toBe(ADSERVER_SEASON);
      expect(profiles[INVENTORY_CONTENT_TENANT_ID].seed).toBe(CONTENT_SEASON);
    }
  });

  it('serves a tenant from its season file and leaves the other shipped', () => {
    const profiles = resolveInventoryTenantProfiles({ [ADSERVER_VAR]: FIXTURE });
    expect(profiles[INVENTORY_ADSERVER_TENANT_ID].seed).toEqual(readSeasonFile(FIXTURE));
    expect(profiles[INVENTORY_ADSERVER_TENANT_ID].seed.seed_id).toBe('small-season');
    expect(profiles[INVENTORY_CONTENT_TENANT_ID].seed).toBe(CONTENT_SEASON);
  });

  it('names the env var and the path when the file is missing', () => {
    const missing = path.join(dir, 'nope.yaml');
    expect(() => resolveInventoryTenantProfiles({ [CONTENT_VAR]: missing })).toThrow(
      new RegExp(`${CONTENT_VAR}.*${missing.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`),
    );
  });

  it('names the env var, the path and the format when a season uses an unknown format', () => {
    const fixture = fs.readFileSync(FIXTURE, 'utf8');
    const custom = write('custom-format-tenant-season.yaml', fixture.replace('formats: [display_300x250]', 'formats: [custom_banner]'));
    expect(() => resolveInventoryTenantProfiles({ [ADSERVER_VAR]: custom })).toThrow(
      new RegExp(`${ADSERVER_VAR}=${custom.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}.*custom_banner`, 's'),
    );
  });

  it('names the env var and the path when the file is invalid', () => {
    const bad = write('bad-tenant-season.yaml', 'seed_id: x\n');
    expect(() => resolveInventoryTenantProfiles({ [ADSERVER_VAR]: bad })).toThrow(
      new RegExp(`${ADSERVER_VAR}.*${bad.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`),
    );
  });
});
