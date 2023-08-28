<script setup lang="ts">
import { debounce } from 'lodash-es';
import type { Rhyme, CompletionsResponse, RhymesResponse } from './types';
import ListenButton from './ListenButton.vue';

const route = useRoute();
const router = useRouter();
const rtConfig = useRuntimeConfig();
const apiBaseUrl = (rtConfig.public.apiBaseUrl ?? 'http://localhost:8000')
  // workaround node 18 ofetch bug for SSR by using 127.0.0.1 for dev
  .replaceAll('localhost', '127.0.0.1');

const q = ref((route.query.q ?? '') as string);
const completionsQuery = ref('');
const searchInput = ref();
const showListenTip = ref(false);

// completions fetch
const { data: completionsData } = await useFetch<CompletionsResponse>(`${apiBaseUrl}/rhymes/completions/`, {
  query: { q: completionsQuery, limit: 20 },
  immediate: false,
});
const completions = computed<string[]>(() => completionsData.value?.hits.map(h => h.text) ?? []);
const fetchCompletions = debounce((q: string) => completionsQuery.value = q, 200);

// rhymes fetch
const { data: rhymesData, pending } = await useFetch<RhymesResponse>(`${apiBaseUrl}/rhymes/`, {
  query: { q, limit: 100 },
  immediate: true,
});
const rhymes = computed<Rhyme[]>(() => rhymesData.value?.hits ?? []);

// computed
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
  track('engagement', 'search', q.value);
  const query = {} as any;
  if (q.value) query.q = q.value;
  router.push({ query });
});

watch(() => [route?.query.q], () => {
  q.value = (route?.query.q ?? '') as string;
});

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
  if (e.input.trim()) fetchCompletions(e.input);
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
</script>

<template>
  <div id="app">
    <Head>
      <Title v-if="q">Rhymes for {{ q }} | Song Rhymes</Title>
      <Title v-else>Top 100 rhymes | Song Rhymes</Title>
    </Head>

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

        <button class="search" @click.prevent="onClickSearch"><i class="fa fa-search" /></button>

        <ClientOnly>
          <ListenButton
            @on-query="(val: string) => { q = val; showListenTip = false; }"
            @on-started="showListenTip = true"
            @on-stopped="showListenTip = false"
          />
        </ClientOnly>
      </fieldset>

      <section class="output" ref="outputEl">
        <label v-if="pending">
          <i class="fa fa-spinner" />
          Searching...
        </label>
        <label v-else-if="showListenTip">
          <strong>Say words to search. Try also: "stop listening", "clear search",
          or spelling out a word</strong>
        </label>
        <label v-else-if="!q">Top {{ counts.rhyme }} most rhymed words</label>
        <label v-else-if="q">{{ label }}</label>

        <ul v-if="rhymes">
          <li v-for="r of rhymes" :key="r.text" :class="`hit ${r.type}`">
            <a @click="() => onLink(r.text)">{{ formatText(r.text) }}</a>
            <span v-if="!!r.frequency && r.type === 'rhyme'" class="freq"> ({{ r.frequency }}) </span>
          </li>
        </ul>
      </section>
    </main>

    <footer>
      Song Rhymes by
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

  input[type='text'] {
    border-radius: 0;
    position: sticky;
    top: 0;
    background: white;
    z-index: 100;
    width: 50vw;
    min-width: 190px;
    max-width: 610px;
    font-size: 17px;
  }
}
</style>
