# SolarCast NZ: 3-minute spoken script

**English** | [繁體中文](SCRIPT_3MIN.zh-TW.md)

342 words, about 2:26 at 140 words per minute. That leaves about 34 seconds of the 3 minutes for pauses or a short live demo. The same text is in the speaker notes of `SolarCast_NZ_Finals_3min.pptx`. Rehearse with a timer. If you run long, cut the sentence "We fixed our rules in advance and looked at the test data once" first.

## Slide 1: Title (about 17 s, 39 words)

Hello, we are [team names], and this is SolarCast NZ. It tells anyone with solar panels in New Zealand how much power they will give next hour, so they use their own sunshine instead of buying from the grid.

## Slide 2: The problem (about 22 s, 52 words)

Solar output changes by the hour, and clouds make the next hour hard to guess. But an EV charger, a hot-water cylinder or a heat pump uses a lot of power. Run it in sunshine and it is almost free. So we asked: how many kilowatts will my panels give next hour?

## Slide 3: What we built (about 18 s, 43 words)

We use free Open-Meteo weather for fourteen regions, from 2022 to 2026. Our model is two gradient-boosted tree models averaged: one predicts sunlight, one predicts cloudiness. It also reads the weather service's day-ahead forecast, and converts sunlight into kilowatts for any panel size.

## Slide 4: Results (about 29 s, 67 words)

On months the model never saw, the average error is 1.12 kilowatts on a 50 kilowatt system, about 2.2 percent of capacity. That is 32.6 percent lower than the strongest simple baseline, and we win in all fourteen regions. We fixed our rules in advance and looked at the test data once. One caveat: we tested on modelled weather, so real roofs will likely see larger errors.

## Slide 5: Live example (about 18 s, 42 words)

This is our dashboard. Each circle is a region, coloured by what a five kilowatt home would make next hour. Click one to see our forecast beside what really happened, and its value in dollars. [If time allows, press Replay today's sun.]

## Slide 6: EV decision (about 23 s, 53 words)

Here is why it matters. We replayed 6,818 region-days with the same EV charging rule. Charging on arrival takes eighteen kilowatt-hours a day from the grid. A good habit, charging from ten to two, takes 8.5. Our forecast takes 7.0, about one and a half kilowatt-hours better. A perfect forecast would reach 5.3.

## Slide 7: Limits (about 13 s, 31 words)

Our limits: we used modelled weather, not real roofs; the car and price numbers are assumptions; and fast-changing clouds are hard to predict. Next, we would compare with real rooftop data.

## Slide 8: Close (about 6 s, 15 words)

Thank you. It is all open source, and we are happy to take your questions.
