/**
 * Latency of get_products over the ad-server catalogue under the dry-run goal
 * check. Skipped unless INVENTORY_LATENCY=1; it prints a table and asserts only
 * that the worst case was loaded as intended.
 *
 * Two scenarios, both with goals on ros_display_guaranteed and
 * sports_fans_display_guaranteed, which share cells:
 * - worst case: every flight runs from two days out to the season end;
 * - realistic: 30-day flights starting within the next 30 days.
 */
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import express from 'express';
import http from 'node:http';
import { writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

process.env.PUBLIC_TEST_AGENT_TOKEN = 'inventory-latency-test-public-token';

import { stopSessionCleanup } from '../../src/training-agent/state.js';
import { setSandboxWallClockForTesting } from '../../src/training-agent/account-sandbox.js';
import { ADSERVER_SEASON, InventorySimulation } from '../../src/training-agent/inventory-sim/engine/index.js';

const DAY_MS = 24 * 60 * 60 * 1000;
const WALL_CLOCK = '2026-11-20T09:00:00Z';
const ACCOUNT = { brand: { domain: 'inventory-buyer.example' }, operator: 'inventory-buyer.example', sandbox: true };
const BRAND = { domain: 'inventory-buyer.example' };
const LEVELS = [0, 10, 50];
const SCENARIOS = {
  'worst case': () => ({ start: startMs, end: seasonEndMs }),
  realistic: (i: number) => ({ start: startMs + (i % 30) * DAY_MS, end: startMs + ((i % 30) + 30) * DAY_MS }),
} as const;
const GOAL_PRODUCTS = ['ros_display_guaranteed', 'sports_fans_display_guaranteed'] as const;
const GOAL_IMPRESSIONS = 100_000;
const RUNS = 5;

const nowMs = Date.parse(WALL_CLOCK);
const startMs = Date.parse(WALL_CLOCK.slice(0, 10)) + 2 * DAY_MS;
const seasonEndMs = Date.parse(ADSERVER_SEASON.start_date) + ADSERVER_SEASON.days * DAY_MS;
const day = (ms: number) => new Date(ms).toISOString().slice(0, 10);

const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
async function timed(run: () => unknown | Promise<unknown>): Promise<number> {
  const begin = performance.now();
  await run();
  return performance.now() - begin;
}
async function measure(run: () => unknown | Promise<unknown>): Promise<number> {
  await run();
  const samples: number[] = [];
  for (let i = 0; i < RUNS; i++) samples.push(await timed(run));
  return median(samples);
}

describe.skipIf(process.env.INVENTORY_LATENCY !== '1')('inventory get_products latency', () => {
  let server: http.Server;
  let baseUrl: string;
  let rpcId = 0;

  const call = async (name: string, args: Record<string, unknown>, token: string): Promise<Record<string, any>> => {
    rpcId += 1;
    const response = await fetch(`${baseUrl}/inventory-adserver/mcp`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'application/json, text/event-stream', authorization: `Bearer ${token}` },
      body: JSON.stringify({ jsonrpc: '2.0', id: rpcId, method: 'tools/call', params: { name, arguments: args } }),
    });
    const envelope = await response.json() as Record<string, any>;
    return envelope.result?.structuredContent ?? { __envelope: envelope };
  };

  beforeAll(async () => {
    const { createTrainingAgentRouter } = await import('../../src/training-agent/index.js');
    const app = express();
    app.use(express.json({ limit: '5mb' }));
    app.use('/api/training-agent', createTrainingAgentRouter({ disableRateLimit: true }));
    server = http.createServer(app);
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
    const address = server.address();
    if (!address || typeof address === 'string') throw new Error('Missing test server address');
    baseUrl = `http://127.0.0.1:${address.port}/api/training-agent`;
    const offset = nowMs - Date.now();
    setSandboxWallClockForTesting(() => Date.now() + offset);
  }, 90_000);

  afterAll(async () => {
    setSandboxWallClockForTesting();
    stopSessionCleanup();
    if (!server) return;
    server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
  });

  it('times the tenant handler and the engine forecasts at 0, 10 and 50 bookings', async () => {
    const rows: string[] = [];
    for (const [scenario, flightOf] of Object.entries(SCENARIOS)) for (const bookings of LEVELS) {
      const tag = `${scenario.replace(' ', '-')}-${bookings}`;
      const token = `demo-inventory-latency-${tag}-v1`;
      // The window get_products and the forecasts ask about: the whole season for the worst case, a 30-day flight in the middle of the starts otherwise.
      const queryStart = scenario === 'worst case' ? startMs : startMs + 15 * DAY_MS;
      const queryEnd = scenario === 'worst case' ? seasonEndMs : queryStart + 30 * DAY_MS;
      const synced = await call('sync_accounts', {
        idempotency_key: `inventory-latency-provision-${tag}`, accounts: [{ ...ACCOUNT, billing: 'operator' }],
      }, token);
      expect(synced.adcp_error ?? synced.__envelope).toBeUndefined();

      // The same bookings go through the tenant (for the handler) and straight
      // into an engine (for the forecast calls alone).
      const sim = InventorySimulation.create(ADSERVER_SEASON, nowMs);
      for (let i = 0; i < bookings; i++) {
        const product_id = GOAL_PRODUCTS[i % GOAL_PRODUCTS.length];
        const flight = flightOf(i);
        const created = await call('create_media_buy', {
          idempotency_key: `inventory-latency-book-${tag}-${i}`,
          account: ACCOUNT,
          brand: BRAND,
          start_time: new Date(flight.start).toISOString(),
          end_time: new Date(flight.end).toISOString(),
          packages: [{ product_id, pricing_option_id: `${product_id}_cpm`, budget: 10_000, impressions: GOAL_IMPRESSIONS }],
        }, token);
        expect(created.media_buy_id, `booking ${i} refused: ${JSON.stringify(created).slice(0, 300)}`).toBeDefined();
        sim.book({
          booking_id: `latency-${i}`, product_id, flight_start_ms: flight.start, flight_end_ms: flight.end, budget: 10_000, impression_goal: GOAL_IMPRESSIONS,
        }, nowMs);
      }

      const getProducts = async () => {
        const result = await call('get_products', {
          buying_mode: 'wholesale', account: ACCOUNT, brand: BRAND,
          filters: { start_date: day(queryStart), end_date: day(queryEnd - DAY_MS) },
        }, token);
        if (!Array.isArray(result.products)) throw new Error(`get_products failed: ${JSON.stringify(result).slice(0, 300)}`);
        return result.products.length as number;
      };
      const products = [...sim.traffic.products.values()];
      const forecasts = () => {
        for (const product of products) {
          sim.forecast({
            product_id: product.product_id,
            flight_start_ms: queryStart,
            flight_end_ms: queryEnd,
            ...(product.delivery_type === 'non_guaranteed' && { bid_cpm: 1 }),
          }, nowMs);
        }
      };

      const count = await getProducts();
      expect(count).toBe(products.length);
      const handlerMs = await measure(getProducts);
      const engineMs = await measure(forecasts);
      const before = sim.dryRuns;
      forecasts();
      const dryRuns = ((sim.dryRuns - before) / products.length).toFixed(1);
      rows.push(`| ${scenario} | ${bookings} | ${handlerMs.toFixed(1)} | ${engineMs.toFixed(1)} | ${dryRuns} |`);
    }
    const table = `\nget_products latency, ${ADSERVER_SEASON.seed_id} (${ADSERVER_SEASON.days} days), median of ${RUNS} after one warm-up\n| scenario | bookings | handler ms | engine ms | dry runs per product |\n|---|---|---|---|---|\n${rows.join('\n')}\n`;
    const out = process.env.INVENTORY_LATENCY_OUT ?? join(tmpdir(), 'inventory-latency.md');
    writeFileSync(out, table);
    process.stderr.write(`${table}(also written to ${out})\n`);
  }, 1_800_000);
});
