// Sentence-by-sentence Listen: each sentence is synthesized on its own, played in order as
// soon as it is ready, and the finished parts are spliced into one clip for replay.

/** Fragments shorter than this ("Eh...", "Fine.") join the next sentence. */
const MIN_SENTENCE_CHARS = 12;
/** Neighbouring sentences merge until a request reaches this many characters. */
const TARGET_REQUEST_CHARS = 160;
/** Requests in flight at once; later sentences wait so the first ones return soonest. */
export const SENTENCE_SPEECH_CONCURRENCY = 3;

const SENTENCE_END = /[.!?]+["')\]]*(?=\s+["'([]?[A-Z0-9])/g;

/** Markdown reduced to what should be read aloud. */
export function speakableText(markdown: string): string {
  return (
    markdown
      .replace(/```[\s\S]*?```/g, " ")
      .replace(/`([^`]*)`/g, "$1")
      .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
      .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
      .replace(/^\s{0,3}(#{1,6}|>|[-*+]|\d+[.)])\s+/gm, "")
      .replace(/(\*\*|__|\*|_|~~)(\S(?:[\s\S]*?\S)?)\1/g, "$2")
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      // A heading or list item is its own sentence even without closing punctuation.
      .map((line) => (/[.!?:;]["')\]]*$/.test(line) ? line : `${line}.`))
      .join(" ")
      .replace(/\s+/g, " ")
      .trim()
  );
}

/** Split speakable text into request-sized runs of whole sentences, in order. */
export function splitSpeechSentences(text: string): string[] {
  const sentences: string[] = [];
  let start = 0;
  for (const match of text.matchAll(SENTENCE_END)) {
    const end = (match.index ?? 0) + match[0].length;
    sentences.push(text.slice(start, end).trim());
    start = end;
  }
  sentences.push(text.slice(start).trim());

  const glued: string[] = [];
  for (const sentence of sentences.filter(Boolean)) {
    const previous = glued.at(-1);
    if (previous !== undefined && previous.length < MIN_SENTENCE_CHARS) {
      glued[glued.length - 1] = `${previous} ${sentence}`;
    } else {
      glued.push(sentence);
    }
  }

  // Short neighbours share a request: fewer round trips. The first run never grows, so the
  // first audio comes back as early as possible.
  const runs: string[] = [];
  for (const sentence of glued) {
    const previous = runs.at(-1);
    if (
      runs.length > 1 &&
      previous !== undefined &&
      previous.length + sentence.length + 1 <= TARGET_REQUEST_CHARS
    ) {
      runs[runs.length - 1] = `${previous} ${sentence}`;
    } else {
      runs.push(sentence);
    }
  }
  return runs;
}

export type SpeechAudio = { audioBase64: string; mimeType?: string };

/** Decoding and playback, injected so tests can run without Web Audio. */
export type SpeechAudioBackend<Clip> = {
  decode(audio: SpeechAudio): Promise<Clip>;
  /** Resolves when the clip finished playing or `signal` aborted it. */
  play(clip: Clip, signal: AbortSignal): Promise<void>;
  splice(clips: Clip[]): Clip;
};

export type SentenceSpeechState =
  | { status: "loading" }
  | { status: "playing"; part: number; parts: number }
  | { status: "done"; failedParts: number }
  | { status: "stopped" }
  | { status: "failed"; error: string };

/**
 * Synthesize `parts` with bounded concurrency and play them strictly in order, each as soon
 * as it and every earlier part are ready. A failed part is skipped; when every part has
 * settled the decoded parts are spliced and returned for replay (null if none succeeded).
 */
export async function playSentenceSpeech<Clip>(params: {
  parts: string[];
  synthesize: (text: string, signal: AbortSignal) => Promise<SpeechAudio>;
  backend: SpeechAudioBackend<Clip>;
  signal: AbortSignal;
  onState?: (state: SentenceSpeechState) => void;
  concurrency?: number;
}): Promise<Clip | null> {
  const { parts, synthesize, backend, signal, onState } = params;
  const clips: Array<Promise<Clip | null>> = [];
  const slots = Math.max(1, params.concurrency ?? SENTENCE_SPEECH_CONCURRENCY);
  let active = 0;
  const waiting: Array<() => void> = [];
  const acquire = async () => {
    if (active < slots) {
      active++;
      return;
    }
    await new Promise<void>((resolve) => {
      waiting.push(() => {
        active++;
        resolve();
      });
    });
  };
  const release = () => {
    active--;
    waiting.shift()?.();
  };

  for (const text of parts) {
    clips.push(
      acquire().then(async () => {
        try {
          if (signal.aborted) {
            return null;
          }
          return await backend.decode(await synthesize(text, signal));
        } catch {
          return null;
        } finally {
          release();
        }
      }),
    );
  }

  onState?.({ status: "loading" });
  const aborted = new Promise<null>((resolve) => {
    if (signal.aborted) {
      resolve(null);
    } else {
      signal.addEventListener("abort", () => resolve(null), { once: true });
    }
  });
  const ready: Clip[] = [];
  for (const [index, pending] of clips.entries()) {
    // Stop must not wait for a slow part to come back.
    const clip = await Promise.race([pending, aborted]);
    if (signal.aborted) {
      onState?.({ status: "stopped" });
      return null;
    }
    if (clip === null) {
      continue;
    }
    ready.push(clip);
    onState?.({ status: "playing", part: index + 1, parts: parts.length });
    await backend.play(clip, signal);
    if (signal.aborted) {
      onState?.({ status: "stopped" });
      return null;
    }
  }

  if (ready.length === 0) {
    onState?.({ status: "failed", error: "speech unavailable" });
    return null;
  }
  onState?.({ status: "done", failedParts: parts.length - ready.length });
  const [only] = ready;
  return ready.length === 1 && only !== undefined ? only : backend.splice(ready);
}

const SPLICE_GAP_SECONDS = 0.12;

/** Web Audio backend: decoded parts share the context's sample rate, so splicing is a copy. */
export function webAudioSpeechBackend(context: AudioContext): SpeechAudioBackend<AudioBuffer> {
  return {
    async decode({ audioBase64 }) {
      const bytes = Uint8Array.from(atob(audioBase64), (char) => char.charCodeAt(0));
      return await context.decodeAudioData(bytes.buffer);
    },
    play(clip, signal) {
      return new Promise<void>((resolve) => {
        if (signal.aborted) {
          resolve();
          return;
        }
        const source = context.createBufferSource();
        source.buffer = clip;
        source.connect(context.destination);
        const stop = () => {
          try {
            source.stop();
          } catch {
            // Already stopped.
          }
        };
        signal.addEventListener("abort", stop, { once: true });
        source.addEventListener(
          "ended",
          () => {
            signal.removeEventListener("abort", stop);
            resolve();
          },
          { once: true },
        );
        void context.resume().then(() => source.start());
      });
    },
    splice(clips) {
      const gap = Math.round(SPLICE_GAP_SECONDS * context.sampleRate);
      const channels = Math.max(...clips.map((clip) => clip.numberOfChannels));
      const length = clips.reduce((sum, clip) => sum + clip.length, 0) + gap * (clips.length - 1);
      const out = context.createBuffer(channels, length, context.sampleRate);
      for (let channel = 0; channel < channels; channel++) {
        const target = out.getChannelData(channel);
        let offset = 0;
        for (const clip of clips) {
          target.set(clip.getChannelData(Math.min(channel, clip.numberOfChannels - 1)), offset);
          offset += clip.length + gap;
        }
      }
      return out;
    },
  };
}
