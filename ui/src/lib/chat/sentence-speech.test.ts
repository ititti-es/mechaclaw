import { describe, expect, it } from "vitest";
import {
  playSentenceSpeech,
  speakableText,
  splitSpeechSentences,
  type SpeechAudioBackend,
} from "./sentence-speech.ts";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function fakeBackend(log: string[]): SpeechAudioBackend<string> {
  return {
    decode: async ({ audioBase64 }) => audioBase64,
    play: async (clip) => {
      log.push(`play:${clip}`);
    },
    splice: (clips) => clips.join("+"),
  };
}

const flush = () =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
  });

describe("speakableText", () => {
  it("drops code blocks and markdown syntax", () => {
    expect(
      speakableText("## Done\n**Build** passed. See [logs](http://x).\n```sh\nrm -rf\n```\n- `ok`"),
    ).toBe("Done. Build passed. See logs. ok.");
  });
});

describe("splitSpeechSentences", () => {
  it("keeps the first sentence alone, glues fragments and merges short neighbours", () => {
    expect(
      splitSpeechSentences(
        "Alright, here's the deal. Eh... Ma would tell me to be nicer. Fine. Get some rest, boss.",
      ),
    ).toEqual([
      "Alright, here's the deal.",
      "Eh... Ma would tell me to be nicer. Fine. Get some rest, boss.",
    ]);
  });

  it("does not split inside versions or lowercase continuations", () => {
    expect(splitSpeechSentences("We shipped v2.1 today. e.g. this stays.")).toEqual([
      "We shipped v2.1 today. e.g. this stays.",
    ]);
  });
});

describe("playSentenceSpeech", () => {
  it("plays parts in order as soon as each is ready, then splices them", async () => {
    const log: string[] = [];
    const replies = [deferred<string>(), deferred<string>(), deferred<string>()];
    const parts = ["one", "two", "three"];
    const run = playSentenceSpeech({
      parts,
      synthesize: async (text) => ({ audioBase64: await replies[parts.indexOf(text)].promise }),
      backend: fakeBackend(log),
      signal: new AbortController().signal,
    });
    replies[2].resolve("c");
    await flush();
    expect(log).toEqual([]);
    replies[0].resolve("a");
    await flush();
    expect(log).toEqual(["play:a"]);
    replies[1].resolve("b");
    expect(await run).toBe("a+b+c");
    expect(log).toEqual(["play:a", "play:b", "play:c"]);
  });

  it("skips a failed part and still splices the rest", async () => {
    const log: string[] = [];
    const states: string[] = [];
    const spliced = await playSentenceSpeech({
      parts: ["one", "two"],
      synthesize: async (text) => {
        if (text === "one") {
          throw new Error("boom");
        }
        return { audioBase64: "b" };
      },
      backend: fakeBackend(log),
      signal: new AbortController().signal,
      onState: (state) => states.push(state.status),
    });
    expect(spliced).toBe("b");
    expect(log).toEqual(["play:b"]);
    expect(states.at(-1)).toBe("done");
  });

  it("limits requests in flight", async () => {
    let inFlight = 0;
    let peak = 0;
    await playSentenceSpeech({
      parts: ["1", "2", "3", "4", "5"],
      synthesize: async (text) => {
        peak = Math.max(peak, ++inFlight);
        await flush();
        inFlight--;
        return { audioBase64: text };
      },
      backend: fakeBackend([]),
      signal: new AbortController().signal,
      concurrency: 2,
    });
    expect(peak).toBe(2);
  });

  it("stops without waiting for a slow part", async () => {
    const stop = new AbortController();
    const states: string[] = [];
    const run = playSentenceSpeech({
      parts: ["slow"],
      synthesize: () => new Promise(() => {}),
      backend: fakeBackend([]),
      signal: stop.signal,
      onState: (state) => states.push(state.status),
    });
    await flush();
    stop.abort();
    expect(await run).toBeNull();
    expect(states).toEqual(["loading", "stopped"]);
  });
});
