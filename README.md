# Pump.fun Top-5 Forecast — @maelaozera

Entry for make.fun's `pumpfun-top5-forecast-01` ($200): predict the 5 largest
pump.fun coins by market cap, 24 hours out.

`main.ts` (Node 22, standard library only) reads `/input/task.json` and
`/input/history.ndjson` and writes `/output/forecast.json`
(`{probabilities, top5, sizes}`).

How it forecasts:

- Every coin is tracked across the 10-minute marks (log10 market cap in SOL,
  venue, age). Its log-cap is extrapolated to the mean answer-read instant:
  recent momentum (whole-window slope, shrunk and clamped) persists ~1.5 h,
  then the coin decays at a venue-dependent rate (pool coins hold up longest,
  curve coins fade fastest). The horizon is read from `task.json`, so the
  same code adapts from the 2-hour practice (~0.7 h) to the final (~10 h).
- Place probabilities are a softmax over the candidates' expected log-caps
  (temperature 0.35 dex), the same distribution at every place, because the
  order churns so much that a coin's rank says little about which place it
  will hold. The named mass per place leaves room for coins born after the
  cut (in the practice, 3 of the top 5 at the last read were born after the
  history ended): the unnamed share is what scores for them.
- Sizes are normal in log10(cap in SOL) around each place's expected cap,
  with the standard deviation widening with the horizon.

Local check on the practice history: the program runs in ~0.2 s,
deterministically (two runs byte-identical), and the forecast is valid
(exactly the three required keys, per-place sums ≤ 1). Scored on the
practice's last answer read — the only read whose top 5 the kit publishes —
its order score is −21.50, against the house baseline's −23.20.
