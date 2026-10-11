// Control UI Listen action: reads a message aloud sentence by sentence through `tts.speak`.
import { consume } from "@lit/context";
import { html, nothing } from "lit";
import { property, state } from "lit/decorators.js";
import type { TtsSpeakResult } from "../../../packages/gateway-protocol/src/schema/channels.js";
import { applicationContext, type ApplicationContext } from "../app/context.ts";
import { t } from "../i18n/index.ts";
import {
  playSentenceSpeech,
  speakableText,
  splitSpeechSentences,
  webAudioSpeechBackend,
  type SentenceSpeechState,
} from "../lib/chat/sentence-speech.ts";
import { OpenClawLightDomElement } from "../lit/openclaw-element.ts";
import { icons } from "./icons.ts";
import "./tooltip.ts";

const SPLICED_CACHE_LIMIT = 8;

// One voice at a time across the page; spliced clips are kept per text for instant replay.
let activeStop: AbortController | null = null;
let sharedContext: AudioContext | null = null;
const splicedByText = new Map<string, AudioBuffer>();

function audioContext(): AudioContext {
  sharedContext ??= new AudioContext();
  return sharedContext;
}

function rememberSpliced(text: string, clip: AudioBuffer) {
  splicedByText.delete(text);
  splicedByText.set(text, clip);
  while (splicedByText.size > SPLICED_CACHE_LIMIT) {
    splicedByText.delete(splicedByText.keys().next().value as string);
  }
}

export function renderListenButton(markdown: string) {
  return html`<openclaw-listen-button .markdown=${markdown}></openclaw-listen-button>`;
}

class ListenButton extends OpenClawLightDomElement {
  @consume({ context: applicationContext, subscribe: true })
  @property({ attribute: false })
  private context: Pick<ApplicationContext, "gateway"> | undefined;
  @property({ attribute: false }) markdown = "";
  @state() private speech: SentenceSpeechState | null = null;
  private stop: AbortController | null = null;

  override disconnectedCallback() {
    super.disconnectedCallback();
    this.stop?.abort();
  }

  private async toggle() {
    if (this.stop) {
      this.stop.abort();
      return;
    }
    const text = speakableText(this.markdown);
    const client = this.context?.gateway.snapshot.client;
    if (!text || !client) {
      return;
    }
    activeStop?.abort();
    const stop = new AbortController();
    this.stop = activeStop = stop;
    const backend = webAudioSpeechBackend(audioContext());
    try {
      const cached = splicedByText.get(text);
      if (cached) {
        this.speech = { status: "playing", part: 1, parts: 1 };
        await backend.play(cached, stop.signal);
        this.speech = stop.signal.aborted
          ? { status: "stopped" }
          : { status: "done", failedParts: 0 };
        return;
      }
      const spliced = await playSentenceSpeech({
        parts: splitSpeechSentences(text),
        // The gateway request cannot be cancelled; a stopped Listen ignores its result.
        synthesize: (part) => client.request<TtsSpeakResult>("tts.speak", { text: part }),
        backend,
        signal: stop.signal,
        onState: (next) => {
          if (this.stop === stop) {
            this.speech = next;
          }
        },
      });
      if (spliced) {
        rememberSpliced(text, spliced);
      }
    } finally {
      if (this.stop === stop) {
        this.stop = null;
        // `stop` is not reactive; the button must leave its busy look once playback ends.
        this.requestUpdate();
      }
      if (activeStop === stop) {
        activeStop = null;
      }
    }
  }

  override render() {
    const busy = this.stop !== null;
    const label = busy
      ? t("chat.messages.stopListening")
      : this.speech?.status === "failed"
        ? t("chat.messages.listenFailed")
        : t("chat.messages.listen");
    const progress =
      this.speech?.status === "playing" && this.speech.parts > 1
        ? html`<span class="chat-listen-btn__progress"
            >${this.speech.part}/${this.speech.parts}</span
          >`
        : nothing;
    return html`
      <openclaw-tooltip .content=${label}>
        <button
          class="chat-reply-btn chat-listen-btn"
          type="button"
          aria-label=${label}
          aria-pressed=${busy ? "true" : "false"}
          data-listen-state=${this.speech?.status ?? "idle"}
          @click=${() => void this.toggle()}
        >
          ${busy ? (this.speech?.status === "loading" ? icons.loader : icons.square) : icons.volume2}
          ${progress}
        </button>
      </openclaw-tooltip>
    `;
  }
}

if (!customElements.get("openclaw-listen-button")) {
  customElements.define("openclaw-listen-button", ListenButton);
}

declare global {
  interface HTMLElementTagNameMap {
    "openclaw-listen-button": ListenButton;
  }
}
