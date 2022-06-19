<script>
export default {
  name: 'RhymesPage',
};
</script>

<script setup>
import { debounce } from 'lodash-es';
import { ref, computed, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { storeToRefs } from 'pinia';
import { useRhymesStore } from '@/store';

const SUGGEST_DEBOUNCE = 200;

const route = useRoute();
const router = useRouter();
const q = ref(route.query.q ?? '');
const page = ref(route.query.page ?? 1);
const searchInput = ref(null);
const outputEl = ref();
const { rhymes, hasNextPage, suggestions, loading } = storeToRefs(useRhymesStore());
const { fetchRhymes, fetchSuggestions, abort } = useRhymesStore();
const debouncedFetchSuggestions = debounce(fetchSuggestions, SUGGEST_DEBOUNCE);

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

async function search(newQ) {
  abort?.();
  q.value = newQ;
}

watch([q, page], () => {
  track('engagement', page.value === 1 ? 'search' : 'more', q.value);
  router.push({ query: { q: q.value } });
  fetchRhymes(q.value, page.value);
});

watch(
  () => [route.query.q, route.query.page],
  () => {
    q.value = route.query.q ?? '';
    page.value = route.query.page ?? 1;
  },
);

function onEnter(e) {
  q.value = (e.target.value ?? '').trim();
}

function onClickSearch(e) {
  q.value = searchInput.value;
}

function onInput(e) {
  searchInput.value = e.input;
  debouncedFetchSuggestions(e.input);
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

fetchRhymes();
</script>

<template>
  <fieldset>
    <vue3-simple-typeahead
      placeholder="Find rhymes in songs..."
      @onInput="onInput"
      @selectItem="e => search(e)"
      @keyup.enter="onEnter"
      :items="suggestions"
      :min-input-length="1"
    >
      <template #list-item-text="slot">
        <span v-html="slot.boldMatchText(slot.itemProjection(slot.item))"></span>
      </template>
    </vue3-simple-typeahead>
    <button @click.prevent="onClickSearch"><i class="fa fa-search" /></button>
  </fieldset>

  <section class="output" ref="outputEl">
    <label v-if="loading">Searching...</label>
    <label v-else-if="!q">Top {{ counts.rhyme }} rhymes</label>
    <label v-else-if="q">{{ label }}</label>

    <ul v-if="rhymes && (!loading || page > 1)">
      <li v-for="r of rhymes" :key="r.ngram" :class="`hit ${r.type}`">
        <a
          @click="
            () => {
              q = r.ngram;
              page = 1;
            }
          "
          >{{ r.ngram }}</a
        >
        <span v-if="!!r.frequency && r.type === 'rhyme'" class="freq"> ({{ r.frequency }}) </span>
      </li>
    </ul>

    <button v-if="!loading && hasNextPage" class="more" @click="page++">More...</button>
  </section>
</template>

<style lang="scss">
@import 'vue3-simple-typeahead/dist/vue3-simple-typeahead.css';

input[type='text'] {
  width: 100%;
  &::-webkit-search-cancel-button {
    -webkit-appearance: searchfield-cancel-button;
  }
}
</style>

<style scoped lang="scss">
@import '@/rhymes.scss';
</style>
