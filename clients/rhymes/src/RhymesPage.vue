<script>
export default {
  name: 'RhymesPage',
};
</script>

<script setup>
import { debounce } from 'lodash-es';
import {ref, computed, watch, watchEffect} from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { storeToRefs } from 'pinia';
import { useRhymesStore } from '@/store';

const SUGGEST_DEBOUNCE = 200;

const route = useRoute();
const router = useRouter();
const q = ref(route.query.q ?? '');
const page = ref(route.query.page ?? 1);
const outputEl = ref();
const searchInput = ref();
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

watchEffect(() => {
  if (searchInput.value) searchInput.value.selectItem(route.query.q ?? '');
});

watch([q, page], () => {
  abort?.();
  track('engagement', page.value === 1 ? 'search' : 'more', q.value);
  const query = {};
  if (q.value) query.q = q.value;
  router.push({ query });
  fetchRhymes(q.value, page.value);
});

watch(
  () => [route.query.q, route.query.page],
  () => {
    q.value = route.query.q ?? '';
    page.value = route.query.page ?? 1;
  },
);

function onSelectItem(val) {
  q.value = val;
  q.page = 1;
}

function onEnter(e) {
  q.value = (e.target.value ?? '').trim();
}

function onClickSearch(e) {
  q.value = searchInput.value.$data.input;
}

function onInput(e) {
  if (e.input.trim()) debouncedFetchSuggestions(e.input);
}

function onLink(val) {
  q.value = val;
  page.value = 1;
}

function onNextPage() {
  page.value++;
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
      ref="searchInput"
      placeholder="Find rhymes in songs..."
      :items="suggestions"
      :min-input-length="1"
      @onInput="onInput"
      @selectItem="onSelectItem"
      @keyup.enter="onEnter"
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
        <a @click="() => onLink(r.ngram)">{{ r.ngram }}</a
        >
        <span v-if="!!r.frequency && r.type === 'rhyme'" class="freq"> ({{ r.frequency }}) </span>
      </li>
    </ul>

    <button v-if="!loading && hasNextPage" class="more" @click="onNextPage">More...</button>
  </section>
</template>

<style lang="scss">
@import 'vue3-simple-typeahead/dist/vue3-simple-typeahead.css';

input[type='text'] {
  width: 100%;
  border-radius: 0;
}
</style>

<style scoped lang="scss">
@import '@/rhymes.scss';
</style>
