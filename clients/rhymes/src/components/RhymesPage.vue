<script>
export default {
  name: 'RhymesPage',
};
</script>

<script setup>
import axios from 'axios';
import { debounce, some } from 'lodash-es';
import { isMobile } from 'mobile-device-detect';
import { ref, watch, computed } from 'vue';
import router from '@/router';

const PER_PAGE = 100;
const DEBOUNCE_TIME = isMobile ? 1000 : 500;
const FETCH_RHYMES = `
    query Rhymes($q: String, $offset: Int, $limit: Int) {
      rhymes(q: $q, offset: $offset, limit: $limit) {
        ngram
        frequency
        type
      }
    }
  `;

const q = computed(() => router.currentRoute.value.query.q ?? '');
const rhymes = ref();
const page = ref(1);
const hasNextPage = ref(false);
const loading = ref(false);
const abortController = ref();
const outputEl = ref();

const counts = computed(() => ({
  rhyme: rhymes.value?.filter(r => r.type === 'rhyme').length || 0,
  l2: rhymes.value?.filter(r => r.type === 'rhyme-l2').length || 0,
  sug: rhymes.value?.filter(r => r.type === 'suggestion').length || 0,
}));

const label = computed(() => {
  return [
    ct2str(counts.value.rhyme, 'rhyme'),
    counts.value.l2 > 0 && ct2str(counts.value.l2, 'rhyme-of-rhyme', 'rhymes-of-rhymes'),
    counts.value.sug > 0 && ct2str(counts.value.sug, 'suggestion'),
  ]
    .filter(Boolean)
    .join(', ');
});

async function fetchRhymes() {
  loading.value = true;
  const url = `${process.env.VUE_APP_SISM_API_BASE_URL}/graphql/`;
  abortController.value = new AbortController();

  const resp = await axios.post(
    url,
    {
      query: FETCH_RHYMES,
      variables: { q: q.value, offset: (page.value - 1) * PER_PAGE, limit: PER_PAGE },
    },
    {
      signal: abortController.value.signal,
    },
  );

  let newRhymes = resp.data.data.rhymes;
  console.log('* rhymes', page.value, newRhymes);

  if (page.value > 1) newRhymes = [...rhymes.value, ...newRhymes];

  if (outputEl.value && isMobile && some(newRhymes, r => r.ngram.length >= 15)) {
    outputEl.value.style.setProperty('--cols', 1);
  } else {
    outputEl.value.style.removeProperty('--cols');
  }

  rhymes.value = newRhymes;
  hasNextPage.value = newRhymes?.length === page.value * PER_PAGE && newRhymes.length < 200;
  loading.value = false;
}

function abort() {
  try {
    abortController.value?.abort();
  } catch (e) {
    console.error(e);
  }
}

function track(category, action, label) {
  if (window.gtag) {
    window.gtag('event', action, {
      event_category: category,
      event_label: label,
    });
  }
}

function ct2str(ct, singularWord, pluralWord) {
  const plWord = pluralWord ?? `${singularWord}s`;
  if (ct === 0) return `No ${plWord}`;
  if (ct === 1) return `1 ${singularWord}`;
  return `${ct} ${plWord}`;
}

async function search() {
  abort();
  return fetchRhymes();
}

const debouncedSearch = debounce(val => router.push({ query: { q: val } }), DEBOUNCE_TIME);

function onInput(e) {
  abort();
  debouncedSearch(e.target.value);
}

watch(
  () => router.currentRoute.value.query.q,
  () => {
    track('engagement', 'search', router.currentRoute.value.query.q);
    page.value = 1;
    search();
  },
);

watch(page, () => {
  track('engagement', 'more', router.currentRoute.value.query.q);
  search();
});

search();
</script>

<template>
  <fieldset>
    <input type="text" :value="q" @input="onInput" placeholder="Find rhymes in songs..." />
  </fieldset>

  <section class="output" ref="outputEl">
    <label v-if="loading">Searching...</label>
    <label v-else-if="!q">Top {{ counts.rhyme }} rhymes</label>
    <label v-else-if="q">{{ label }}</label>

    <ul v-if="rhymes && (!loading || page > 1)">
      <li v-for="r of rhymes" :key="r.ngram" :class="`hit ${r.type}`">
        <router-link :to="{ path: '/', query: { q: r.ngram } }">{{ r.ngram }}</router-link>
        <span v-if="!!r.frequency && r.type === 'rhyme'" class="freq"> ({{ r.frequency }}) </span>
      </li>
    </ul>

    <button v-if="!loading && hasNextPage" class="more" @click="page++">More...</button>
  </section>
</template>

<style scoped lang="scss">
@import './rhymes.scss';
</style>
