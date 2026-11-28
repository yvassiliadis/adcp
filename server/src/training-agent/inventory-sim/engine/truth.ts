/**
 * Sandbox readout of what the simulated world actually did: the seed's
 * regions, profiles and segments, each booking's delivery by cell, and
 * optionally the day-by-day actual traffic. It reads only processed days
 * and never forecasts, so it reveals nothing about days still to come.
 */
import type { InventorySimulation } from './simulation.js';
import type { BookingKind, BookingStatus, BookingTargeting, CellRates, DeliveryMetrics } from './types.js';

export interface SimulationTruthOptions {
  /** Only these bookings. Default: all of them. */
  bookingIds?: string[];
  /** Season day window `[startDay, endDay)` of actual impressions per cell. */
  dailyActuals?: { startDay: number; endDay: number };
}

export interface TruthCell {
  region_id: string;
  property_id: string;
  placement_id: string;
  profile_id: string;
  installment_id?: string;
  impressions: number;
  /** Unrounded sums over the days served of count × that day's rate. */
  clicks: number;
  viewable_impressions: number;
  completed_views: number;
  conversions: number;
  /** The rates the cell is drawn around. */
  rates: CellRates;
  /** Rates the booking actually saw: viewability, click and completion are
   * impression-weighted; conversion is conversions over clicks. */
  delivered_rates: CellRates;
}

export interface TruthBooking {
  booking_id: string;
  product_id: string;
  kind: BookingKind;
  status: BookingStatus;
  impression_goal?: number;
  cpm: number;
  targeting: BookingTargeting;
  delivered: DeliveryMetrics;
  cells: TruthCell[];
}

export interface SimulationTruth {
  seed_id: string;
  processed_through_day: number;
  world: {
    regions: Array<{ region_id: string; country: string; share: number }>;
    audience_profiles: Array<{ profile_id: string; share: number; rate_multipliers: Partial<CellRates> }>;
    segments: Array<{ segment_id: string; profile_ids: string[] }>;
  };
  bookings: TruthBooking[];
  dailyActuals?: {
    cell_index: Array<{
      cell_id: number;
      region_id: string;
      property_id: string;
      placement_id: string;
      profile_id: string;
      installment_id?: string;
    }>;
    cell_daily_actuals: Array<{ date: string; actuals: Array<[number, number]> }>;
  };
}

interface CellSums {
  impressions: number;
  clicks: number;
  viewable: number;
  completed: number;
  conversions: number;
}

export function simulationTruth(sim: InventorySimulation, options: SimulationTruthOptions = {}): SimulationTruth {
  const { traffic, state } = sim;
  const slotLabels = (cell: number) => {
    const { region_id, property_id, placement_id, profile_id, installment_id } = traffic.slots[cell];
    return { region_id, property_id, placement_id, profile_id, ...(installment_id !== undefined && { installment_id }) };
  };

  const wanted = options.bookingIds ? new Set(options.bookingIds) : undefined;
  const bookings: TruthBooking[] = state.bookings
    .filter(booking => !wanted || wanted.has(booking.booking_id))
    .map(booking => {
      const sums = new Map<number, CellSums>();
      for (const day of booking.days) {
        for (const [cell, count] of day.cells) {
          if (count <= 0) continue;
          const rates = traffic.rates(day.day, cell);
          const sum = sums.get(cell) ?? { impressions: 0, clicks: 0, viewable: 0, completed: 0, conversions: 0 };
          sum.impressions += count;
          sum.clicks += count * rates.click_rate;
          sum.viewable += count * rates.viewability;
          sum.completed += count * rates.completion_rate;
          sum.conversions += count * rates.click_rate * rates.conversion_rate;
          sums.set(cell, sum);
        }
      }
      const cells: TruthCell[] = [...sums]
        .filter(([, sum]) => sum.impressions > 0)
        .sort((a, b) => a[0] - b[0])
        .map(([cell, sum]) => ({
          ...slotLabels(cell),
          impressions: sum.impressions,
          clicks: sum.clicks,
          viewable_impressions: sum.viewable,
          completed_views: sum.completed,
          conversions: sum.conversions,
          rates: traffic.expectedRates(cell),
          delivered_rates: {
            viewability: sum.viewable / sum.impressions,
            click_rate: sum.clicks / sum.impressions,
            completion_rate: sum.completed / sum.impressions,
            conversion_rate: sum.clicks > 0 ? sum.conversions / sum.clicks : 0,
          },
        }));
      return {
        booking_id: booking.booking_id,
        product_id: booking.product_id,
        kind: booking.kind,
        status: booking.status,
        ...(booking.impression_goal !== undefined && { impression_goal: booking.impression_goal }),
        cpm: booking.cpm,
        targeting: structuredClone(booking.targeting),
        delivered: { ...booking.delivered },
        cells,
      };
    });

  const truth: SimulationTruth = {
    seed_id: state.seed_id,
    processed_through_day: state.processed_through_day,
    world: {
      regions: [...traffic.regions.values()].map(({ region_id, country, share }) => ({ region_id, country, share })),
      audience_profiles: [...traffic.profiles.values()].map(profile => ({
        profile_id: profile.profile_id,
        share: profile.share,
        rate_multipliers: { ...profile.rate_multipliers },
      })),
      segments: sim.seed.segments.map(segment => ({ segment_id: segment.segment_id, profile_ids: [...segment.profile_ids] })),
    },
    bookings,
  };

  if (options.dailyActuals) {
    const first = Math.max(0, options.dailyActuals.startDay);
    const last = Math.min(options.dailyActuals.endDay, state.processed_through_day, traffic.calendar.days);
    const used = new Set<number>();
    const cell_daily_actuals: Array<{ date: string; actuals: Array<[number, number]> }> = [];
    for (let day = first; day < last; day++) {
      const actuals: Array<[number, number]> = [];
      for (const [cell, count] of [...traffic.day(day).actual].sort((a, b) => a[0] - b[0])) {
        if (count <= 0) continue;
        actuals.push([cell, count]);
        used.add(cell);
      }
      cell_daily_actuals.push({ date: traffic.calendar.dateOf(day), actuals });
    }
    truth.dailyActuals = {
      cell_index: [...used].sort((a, b) => a - b).map(cell => ({ cell_id: cell, ...slotLabels(cell) })),
      cell_daily_actuals,
    };
  }
  return truth;
}
