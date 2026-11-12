// pumpfun-top5-forecast-01 entry (Node 22, standard library only).
//
// Forecast for the 5 largest pump.fun coins by market cap at the house's
// seven answer reads. Reads /input/task.json and /input/history.ndjson and
// writes /output/forecast.json: {probabilities, top5, sizes}.
//
// Model (horizon-adaptive, deterministic, no randomness, no network):
//   * Every coin in the history is tracked across the 10-minute marks:
//     log10 market cap in SOL, venue (curve / migrating / pool), age.
//   * For each coin we extrapolate its log-cap to the mean answer-read
//     instant. Recent momentum (log-cap slope over its whole observed
//     window, shrunk toward 0 and clamped) persists for about 1.5 h; after
//     that the coin decays at a venue-dependent rate (pool coins hold up
//     longest, curve coins fade fastest). The horizon H is read from task.json (mean of reads minus
//     the last mark), so the same code adapts from the 2-hour practice
//     (H ~ 0.7 h) to the final (H ~ 10 h), where most of today's chart
//     has faded and newcomers take places.
//   * Place probabilities are a softmax over the candidates' expected
//     log-caps, the same distribution at every place: the order churns
//     so much (in the practice, of the last mark's top 5 only one coin
//     was still in the answer's top 5 at the last read, 70 minutes later,
//     while a coin ranked #7 rose to #2) that a coin's rank by expected
//     cap says little about which place it will hold, only about whether
//     it is in the running at all. The distribution is scaled down to
//     leave room for coins born after the cut (in the practice, 3 of the
//     top 5 at the last read were born after the history ended, and 3 of
//     the 9 coins that ever held a top-5 place were): the unnamed share
//     is what scores for them.
//   * Sizes: place k's cap is taken as normal in log10 SOL around the
//     k-th ranked coin's expected log-cap, with sd widening with the
//     horizon (0.35 + 0.05*H), spread over the 82 bins of task.json.
//   * top5 is the top 5 coins by expected log-cap at the answer reads.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const MICRO = 1_000_000;

// --- tuned constants (fixed in advance, not fitted to the practice answer) ---
const MOMENTUM_HOURS = 1.5; // momentum persists this long after the cut
const SLOPE_SHRINK = 0.6; // whole-window slopes are shrunk toward 0: the last
// two marks alone are too noisy (a coin can jump 1813 -> 1303 -> 2012 SOL
// in 20 minutes without any lasting trend), so the slope is taken over the
// coin's whole observed window and damped.
const SLOPE_MIN = -1.0; // dex per hour, clamped
const SLOPE_MAX = 0.7; // dex per hour, clamped
const DECAY_POOL = 0.1; // dex per hour after the momentum window
const DECAY_MIGRATING = 0.15;
const DECAY_CURVE = 0.25;
const MU_FLOOR = 1.7; // log10 SOL; below this a coin cannot hold a top place
const SIGMA_TAU = 0.35; // softmax temperature in dex: a coin 0.35 dex below
// the best keeps e^-1 ~= 0.37 of its weight. The top ~15 coins by expected
// log-cap therefore share each place's named mass, as the churn demands.
const CANDIDATES = 15; // coins carried into the place distributions
const SIZE_SD_BASE = 0.35;
const SIZE_SD_PER_HOUR = 0.05;
const NEWCOMER_BASE = 0.45; // share of a place left to coins born after the cut
const NEWCOMER_PER_HOUR = 0.03; // newcomers dominate more as the horizon grows:
// in the practice, 3 of the top 5 at the last read were born after the
// history ended; in the final the reads are 7-13 h after the cut, so the
// unnamed share must be the larger part of every place.
const NEWCOMER_MAX = 0.65;
const P_MAX = 0.3; // no single coin gets more than this at a place (the
// baseline gives 0.30 to the coin then at #k for place k)
const P_MIN = 0.002; // rows below this are dropped (they score ~nothing)

interface MarkRow {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  venue: string;
  created_at: string;
  complete: boolean;
  mayhem: boolean;
}

interface Mark {
  at: string;
  ok: boolean;
  rows: MarkRow[] | null;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
  size: { edges: number[] };
  reads: string[];
};
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '');
const marks = lines
  .map((l) => JSON.parse(l) as Mark)
  .filter((m) => m.ok === true && Array.isArray(m.rows) && m.rows.length > 0);
if (marks.length === 0) throw new Error('the history holds no ok mark with rows');
const lastMark = marks[marks.length - 1]!;
const lastAt = Date.parse(lastMark.at);
const readTimes = (task.reads ?? []).map((r) => Date.parse(r)).filter((t) => !Number.isNaN(t));
const horizonHours =
  readTimes.length > 0
    ? readTimes.reduce((a, b) => a + b, 0) / readTimes.length / 3_600_000 - lastAt / 3_600_000
    : 1;
const H = Math.max(0, horizonHours);

// log10 of a cap in SOL, as the scorer takes it: log10(lamports) - 9.
const sizeOf = (lamports: string | number): number => Math.log10(Number(BigInt(lamports))) - 9;

// erfc(x), Numerical Recipes' erfcc (fractional error under 1.2e-7 everywhere).
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))
    );
  return x >= 0 ? r : 2 - r;
}
/** P(X >= x) for X normal with mean `mean` and sd `sd`. */
const above = (x: number, mean: number, sd: number): number =>
  x === -Infinity ? 1 : x === Infinity ? 0 : 0.5 * erfc((x - mean) / (sd * Math.SQRT2));

// --- per-coin history ---
interface Coin {
  mint: string;
  venue: string;
  complete: boolean;
  logcap: number; // at the last mark
  slope: number; // dex per hour over its last two appearances
  mu: number; // expected log10 cap (SOL) at the mean answer read
  createdAt: number;
}
const seen = new Map<string, { times: number[]; caps: number[]; venue: string; complete: boolean; createdAt: number }>();
for (const m of marks) {
  const t = Date.parse(m.at);
  for (const r of m.rows!) {
    const e = seen.get(r.mint);
    const cap = sizeOf(r.market_cap_lamports);
    if (e) {
      e.times.push(t);
      e.caps.push(cap);
      e.venue = r.venue;
      e.complete = r.complete;
    } else {
      seen.set(r.mint, {
        times: [t],
        caps: [cap],
        venue: r.venue,
        complete: r.complete,
        createdAt: Date.parse(r.created_at),
      });
    }
  }
}

const decayOf = (venue: string): number =>
  venue === 'pool' ? DECAY_POOL : venue === 'migrating' ? DECAY_MIGRATING : DECAY_CURVE;

const coins: Coin[] = [];
for (const [mint, e] of seen) {
  const n = e.times.length;
  const logcap = e.caps[n - 1]!;
  let slope = 0;
  if (n >= 2) {
    const dtHours = (e.times[n - 1]! - e.times[0]!) / 3_600_000;
    if (dtHours > 0) {
      slope = ((e.caps[n - 1]! - e.caps[0]!) / dtHours) * SLOPE_SHRINK;
      slope = Math.max(SLOPE_MIN, Math.min(SLOPE_MAX, slope));
    }
  }
  const mu = Math.max(
    MU_FLOOR,
    logcap + slope * Math.min(H, MOMENTUM_HOURS) - decayOf(e.venue) * Math.max(0, H - MOMENTUM_HOURS),
  );
  coins.push({ mint, venue: e.venue, complete: e.complete, logcap, slope, mu, createdAt: e.createdAt });
}
// Highest expected cap first; ties broken by mint for determinism.
coins.sort((a, b) => b.mu - a.mu || (a.mint < b.mint ? -1 : a.mint > b.mint ? 1 : 0));
if (coins.length < task.places) throw new Error(`the history ranks ${coins.length} coins, fewer than the ${task.places} places`);

const cands = coins.slice(0, CANDIDATES);

// --- place probabilities: softmax over expected log-caps ---
// The same distribution at every place: weight exp((mu - mu_max) / tau).
// Scaled so each place's named mass is `namedMass`; the rest is left to
// newcomers and movers via the scorer's unnamed share.
const newcomerShare = Math.min(NEWCOMER_BASE + NEWCOMER_PER_HOUR * H, NEWCOMER_MAX);
const namedMass = 1 - newcomerShare;
const topMu = cands[0]!.mu;
const weights = cands.map((c) => Math.exp((c.mu - topMu) / SIGMA_TAU));
const weightSum = weights.reduce((a, b) => a + b, 0);

interface ProbRow {
  mint: string;
  place: number;
  p: number;
}
const probabilities: ProbRow[] = [];
for (let place = 1; place <= task.places; place++) {
  for (let i = 0; i < cands.length; i++) {
    const p = Math.floor(Math.min((weights[i]! / weightSum) * namedMass, P_MAX) * MICRO) / MICRO;
    if (p >= P_MIN) probabilities.push({ mint: cands[i]!.mint, place, p });
  }
}

// --- sizes: normal in log10(cap in SOL) around the k-th ranked coin's mu ---
const sizes: Array<{ place: number; bin: number; p: number }> = [];
const edges = task.size.edges;
const sizeSd = SIZE_SD_BASE + SIZE_SD_PER_HOUR * H;
for (let place = 1; place <= task.places; place++) {
  const mean = cands[place - 1]!.mu;
  for (let bin = 0; bin <= edges.length; bin++) {
    const lo = bin === 0 ? -Infinity : edges[bin - 1]!;
    const hi = bin === edges.length ? Infinity : edges[bin]!;
    const u = Math.floor((above(lo, mean, sizeSd) - above(hi, mean, sizeSd)) * MICRO);
    if (u > 0) sizes.push({ place, bin, p: u / MICRO });
  }
}

const top5 = coins.slice(0, task.places).map((c) => c.mint);
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top5, sizes })}\n`);
