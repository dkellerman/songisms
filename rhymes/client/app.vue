<script setup lang="ts">
import { debounce } from 'lodash-es';
import type { Rhyme, CompletionsResponse, RhymesResponse } from './types';
import ListenButton from './ListenButton.vue';

const route = useRoute();
const router = useRouter();
const config = useRuntimeConfig();
// workaround node 18 ofetch bug for SSR by using 127.0.0.1 for dev
const apiBaseUrl = config.public.apiBaseUrl.replaceAll('localhost', '127.0.0.1');
const searchQuery = ref((route.query?.q ?? '') as string);
const completionsQuery = ref('');
const searchInput = ref();
const showListenTip = ref(false);
const partialSpeechResult = ref<string>();
const voteMode = ref(false);
const voterUid = ref<string>();

// completions fetch
const { data: completionsData } = await useFetch<CompletionsResponse>(`${apiBaseUrl}/rhymes/completions/`, {
  query: { q: completionsQuery, limit: 20 },
  immediate: false,
});
const completions = computed<string[]>(() => completionsData.value?.hits.map(h => h.text) ?? []);
const fetchCompletions = debounce((q: string) => completionsQuery.value = q, 200);

// rhymes fetch
const rhymesQuery = computed(() => ({
  q: searchQuery.value,
  limit: 100,
  voter_uid: voterUid.value ?? undefined,
}));
const { data: rhymesData, pending } = await useFetch<RhymesResponse>(`${apiBaseUrl}/rhymes/`, {
  query: rhymesQuery,
  immediate: true,
});
const rhymes = computed<Rhyme[]>(() => rhymesData.value?.hits ?? []);

// vote fetch
const voteQuery = ref();
await useFetch(`${apiBaseUrl}/rhymes/vote/`, {
  method: 'POST',
  body: voteQuery,
  immediate: false,
});

// computed
const counts = computed(() => ({
  rhyme: rhymes.value?.filter(r => r.type === 'rhyme').length || 0,
  maybe: rhymes.value?.filter(r => r.type === 'rhyme-l2').length || 0,
  sug: rhymes.value?.filter(r => r.type === 'suggestion').length || 0,
}));

const searchInfoLabel = computed(() => {
  return [
    `<span class="info-rhymes">${ct2str(counts.value.rhyme, 'rhyme')}</span>`,
    counts.value.maybe > 0 && `<span class="info-maybe">${ct2str(counts.value.maybe, 'maybe', 'maybe')}</span>`,
    counts.value.sug > 0 && `<span class="info-sug">${ct2str(counts.value.sug, 'suggestion')}</span>`,
  ]
    .filter(Boolean)
    .join(', ');
});

// fill the search input with the query param
watchEffect(() => {
  if (searchInput.value) {
    searchInput.value.$data.input = route?.query.q ?? '';
  }
});

// watch for URL query param changes
watch(() => [route?.query.q], () => {
  searchQuery.value = (route?.query.q ?? '') as string;
});

// additional things to do on search
watch([searchQuery], () => {
  track('engagement', 'search', searchQuery.value);

  const query = {} as any;
  if (searchQuery.value) query.q = searchQuery.value;
  window.scrollTo(0, 0);
  router.push({ query });
});

// activate vote mode
watch(() => voteMode.value, (val: boolean) => {
  if (val) {
    import('get-browser-fingerprint').then(({ default: getBrowserFingerprint }) => {
      voterUid.value = String(getBrowserFingerprint());
    });
  } else {
    voterUid.value = undefined;
  }
  window.scrollTo(0, 0);
});

function onSelectItem(val: string) {
  searchQuery.value = val;
}

function onEnter(e: KeyboardEvent) {
  searchQuery.value = ((e.target as HTMLInputElement)?.value ?? '').trim();
  searchInput.value.selectItem(searchQuery.value);
}

function onClickSearch(e: MouseEvent) {
  searchQuery.value = searchInput.value.$data.input;
}

function onInput(e: any) {
  searchInput.value.$data.currentSelectionIndex = -1;
  if (e.input.trim()) fetchCompletions(e.input);
}

function onLink(val: string) {
  searchQuery.value = val;
}

function onFocus(e: FocusEvent) {
  window.oncontextmenu = () => false;
  const el = document.getElementById(searchInput.value?.$el?.id);
  if (!el) return;
  el.querySelector('input')?.select();
}

function onVoiceQuery(val: string) {
  clearListeningText();
  searchQuery.value = val;
}

function clearListeningText() {
  showListenTip.value = false;
  partialSpeechResult.value = '';
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
  return text.replace(/\bi\b/g, 'I');
}

function castVote(rhyme: Rhyme, vote: 'good' | 'bad') {
  voteQuery.value = {
    voter_uid: voterUid.value,
    anchor: searchQuery.value,
    alt1: rhyme.text,
    label: vote,
  };
  rhyme.vote = vote;
}

function uncastVote(rhyme: Rhyme) {
  voteQuery.value = {
    voter_uid: voterUid.value,
    anchor: searchQuery.value,
    alt1: rhyme.text,
    remove: 'all',
  };
  rhyme.vote = undefined;
}

</script>

<template>
  <div id="app">
    <Head>
      <Title v-if="searchQuery">Rhymes for {{ searchQuery }} | Song Rhymes</Title>
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

        <button class="search" @click.prevent="onClickSearch" title="Search">
          <i class="fa fa-search fa-lg" />
        </button>

        <ClientOnly>
          <ListenButton
            @on-query="onVoiceQuery"
            @on-partial-result="partialSpeechResult = $event"
            @on-started="() => { clearListeningText(); showListenTip = true; }"
            @on-stopped="clearListeningText"
          />
        </ClientOnly>
      </fieldset>

      <section class="output">
        <label v-if="partialSpeechResult">
          <i class="fa fa-spinner" />&nbsp;
          <em>{{ partialSpeechResult }}</em>
        </label>

        <label v-else-if="showListenTip">
          <strong>
            Say words to search. Try also: "stop listening", "clear search", or spelling out a word
          </strong>
        </label>

        <label v-else-if="pending">
          <i class="fa fa-spinner" /> Searching...
        </label>

        <label v-else-if="!searchQuery">
          Top {{ counts.rhyme }} most rhymed words
        </label>

        <label v-else-if="searchQuery" v-html="searchInfoLabel" />

        <ul v-if="rhymes">
          <li v-for="rhyme of rhymes" :key="`${rhyme.text}`" :class="`hit ${rhyme.type}`">
            <a @click="() => onLink(rhyme.text)">
              {{ formatText(rhyme.text) }}
            </a>

            <span v-if="voteMode && searchQuery">
              <ClientOnly>
                <div v-if="!rhyme.vote" class="vote">
                  <i class="fa fa-thumbs-up" @click="() => castVote(rhyme, 'good')" />
                  <i class="fa fa-thumbs-down" @click="() => castVote(rhyme, 'bad')" />
                </div>
                <div v-else class="unvote">
                  <i :class="{
                    fa: true,
                    'fa-thumbs-up': rhyme.vote === 'good',
                    'fa-thumbs-down': rhyme.vote === 'bad' }"
                  />
                  <i class="fa fa-remove" @click="() => uncastVote(rhyme)" />
                </div>
              </ClientOnly>
            </span>

            <span v-else-if="!!rhyme.frequency && rhyme.type === 'rhyme'" class="freq">
              ({{ rhyme.frequency }})
            </span>
          </li>
        </ul>
      </section>
    </main>

    <footer>
      Song Rhymes by&nbsp;
      <a target="_blank" rel="noopener noreferer" href="https://linkedin.com/in/david-kellerman">
        David Kellerman
      </a>
      <div class="links">
        &nbsp;&mdash;&nbsp;
        <a target="_blank" rel="noopener noreferrer" href="https://github.com/dkellerman/songisms">
          Source code
        </a>
        &nbsp;&#183;&nbsp;
        <a target="_blank" rel="noopener noreferrer" href="https://bipium.com">
          Metronome
        </a>
        &nbsp;&#183;&nbsp;
        <a target="_blank" rel="noopener noreferrer" href="https://open.spotify.com/artist/2fxGUIL1BUCzWwKqP1ykUi">
          Music
        </a>
      </div>
      <span class="vote">[
        <a v-if="voteMode" @click.prevent="voteMode=false">Exit vote mode</a>
        <a v-else @click.prevent="voteMode=true">Vote mode</a>
      ]</span>
    </footer>
  </div>
</template>

<style lang="scss" scoped>
section {
  display: initial;
}
</style>
