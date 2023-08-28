<script setup lang="ts">
import { debounce } from 'lodash-es/';
import { useRoute, useRouter } from 'vue-router';
import { useSpeechRecognition } from '@vueuse/core';
import type { Rhyme, Completion } from './types';

const COMPLETIONS_DEBOUNCE = 200;

// const { data, pending } = await useFetch('/api/rhymes', {
//   query: { q: searchQuery },
//   immediate: true,
// });
// const { rhymes, completions, loading } = storeToRefs(useRhymesStore());

const completions = ref<Completion[]>([]);
const rhymes = ref<Rhyme[]>([]);
const loading = ref(false);

function fetchRhymes(q: string) {}
function fetchCompletions(q: string) {}
function abortFetch() {}

const route = useRoute();
const router = useRouter();
const q = ref('');
const searchInput = ref();
const debouncedFetchCompletions = debounce(fetchCompletions, COMPLETIONS_DEBOUNCE);
const showListenTip = ref(false);
const speech = useSpeechRecognition({
  lang: 'en-US',
  interimResults: false,
  continuous: true,
});
const { isListening, isSupported: listenSupported } = speech;

// override timeout
if (speech?.recognition) {
  speech.recognition.onend = () => {
    if (isListening.value) {
      speech.recognition!.start();
    } else {
      speech.stop();
    }
  };
}

const counts = computed(() => ({
  rhyme: rhymes.value?.filter(r => r.type === 'rhyme').length || 0,
  l2: rhymes.value?.filter(r => r.type === 'rhyme-l2').length || 0,
  sug: rhymes.value?.filter(r => r.type === 'suggestion').length || 0,
}));

const label = computed(() => {
  return [
    ct2str(counts.value.rhyme, 'rhyme'),
    counts.value.l2 > 0 && ct2str(counts.value.l2, 'maybe', 'maybe'),
    counts.value.sug > 0 && ct2str(counts.value.sug, 'suggestion'),
  ]
    .filter(Boolean)
    .join(', ');
});

watchEffect(() => {
  if (searchInput.value) {
    searchInput.value.$data.input = route?.query.q ?? '';
  }
});

watch([q], () => {
  abortFetch?.();
  track('engagement', 'search', q.value);
  const query = {} as any;
  if (q.value) query.q = q.value;
  router.push({ query });
  fetchRhymes(q.value);
});

watch(
  () => [route?.query.q],
  () => {
    q.value = (route?.query.q ?? '') as string;
  },
);

watch(speech.result, onSpeechResult);

function onSpeechResult() {
  let val = speech.result.value?.toLowerCase().trim();
  const words = val.split(' ');

  if (speech.isFinal) {
    showListenTip.value = false;
    if (!val) return;
    if (val === 'stop listening') {
      speech.toggle();
      return;
    } else if (val === 'clear search') {
      val = '';
    } else if (words.length > 2 && words.every(w => w.length === 1)) {
      val = words.join('');
    }

    q.value = val;
    if (isListening.value) speech.recognition!.stop();
    setTimeout(() => {
      if (!isListening.value) speech.recognition!.start();
    }, 100);
  }
}

function onSelectItem(val: string) {
  q.value = val;
}

function onEnter(e: KeyboardEvent) {
  q.value = ((e.target as HTMLInputElement)?.value ?? '').trim();
  searchInput.value.selectItem(q.value);
}

function onClickSearch(e: MouseEvent) {
  q.value = searchInput.value.$data.input;
}

function onInput(e: any) {
  searchInput.value.$data.currentSelectionIndex = -1;
  if (e.input.trim()) debouncedFetchCompletions(e.input);
}

function onLink(val: string) {
  q.value = val;
}

function onFocus(e: FocusEvent) {
  window.oncontextmenu = () => false;
  (document.getElementById(searchInput.value.$data.inputId) as any).select();
}

function track(category: string, action: string, label: string) {
  const gtag = (window as any).gtag;
  if (gtag) {
    gtag('event', action, {
      event_category: category,
      event_label: label,
    });
  }
}

function ct2str(ct: number, singularWord: string, pluralWord?: string) {
  const plWord = pluralWord ?? `${singularWord}s`;
  if (ct === 0) return `No ${plWord} found`;
  if (ct === 1) return `1 ${singularWord}`;
  return `${ct} ${plWord}`;
}

function formatText(text: string) {
  return text?.replace(/\bi\b/g, 'I');
}

fetchRhymes(q.value);
</script>

<template>
  <div id="app">
  <nav>
    <h1><router-link to="/">Song Rhymes</router-link></h1>
  </nav>

  <main>
    <fieldset>
      <vue3-simple-typeahead
        ref="searchInput"
        placeholder="Find rhymes in songs..."
        :items="completions"
        :min-input-length="1"
        @onInput="onInput"
        @selectItem="onSelectItem"
        @keyup.enter="onEnter"
        @onFocus="onFocus"
      >
        <template #list-item-text="slot">
          <span v-html="slot.boldMatchText(slot.itemProjection(slot.item))"></span>
        </template>
      </vue3-simple-typeahead>

      <button @click.prevent="onClickSearch"><i class="fa fa-search" /></button>

      <button v-if="listenSupported" @click.prevent="() => {
        speech.toggle();
        showListenTip = isListening;
      }" :class="{ listen: true, 'is-listening': isListening }">
        <i :class="{ fa: true, 'fa-lg': true, 'fa-microphone': !isListening, 'fa-stop': isListening }" />
      </button>
    </fieldset>

    <section class="output" ref="outputEl">
      <label v-if="loading">Searching...</label>
      <label v-else-if="showListenTip">
        Say words to search. Try also: "stop listening", "clear search",
        or spelling out a word
      </label>
      <label v-else-if="!q">Top {{ counts.rhyme }} most rhymed words</label>
      <label v-else-if="q">{{ label }}</label>

      <ul v-if="rhymes && !loading">
        <li v-for="r of rhymes" :key="r.text" :class="`hit ${r.type}`">
          <a @click="() => onLink(r.text)">{{ formatText(r.text) }}</a>
          <span v-if="!!r.frequency && r.type === 'rhyme'" class="freq"> ({{ r.frequency }}) </span>
        </li>
      </ul>
    </section>
  </main>

  <footer>
    SongRhymes by
    <a target="_blank" rel="noopener noreferer" href="https://linkedin.com/in/david-kellerman">&nbsp;David Kellerman</a>
    <div class="links">
      &nbsp;&mdash;
      <a target="_blank" rel="noopener noreferrer" href="https://github.com/dkellerman/songisms">&nbsp;Source code</a>
      &nbsp;&#183;
      <a target="_blank" rel="noopener noreferrer" href="https://bipium.com">&nbsp;Metronome</a>
      &nbsp;&#183;
      <a target="_blank" rel="noopener noreferrer" href="https://open.spotify.com/artist/2fxGUIL1BUCzWwKqP1ykUi">&nbsp;Music</a>
    </div>
  </footer>
  </div>
</template>

<style lang="scss">
.simple-typeahead {
  width: initial !important;
  background: red !important;

  input[type='text'] {
    border-radius: 0;
    position: sticky;
    top: 0;
    background: white;
    z-index: 100;
    width: 80vw;
    min-width: 190px;
    max-width: 610px;
    font-size: 17px;
  }
}
</style>
