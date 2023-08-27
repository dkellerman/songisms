<script>
export default {
  name: 'RhymesPage',
};
</script>

<script setup>
import { debounce } from 'lodash-es';
import { ref, computed, watch, watchEffect } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { storeToRefs } from 'pinia';
import { useRhymesStore } from '@/store';
import { useSpeechRecognition } from '@vueuse/core';

const COMPLETIONS_DEBOUNCE = 200;

const route = useRoute();
const router = useRouter();
const q = ref(route.query.q ?? '');
const page = ref(route.query.page ?? 1);
const outputEl = ref();
const searchInput = ref();
const { rhymes, hasNextPage, completions, loading } = storeToRefs(useRhymesStore());
const { fetchRhymes, fetchCompletions, abort } = useRhymesStore();
const debouncedFetchCompletions = debounce(fetchCompletions, COMPLETIONS_DEBOUNCE);
const showListenTip = ref(false);

const speech = useSpeechRecognition({
  lang: 'en-US',
  interimResults: false,
  continuous: true,
});
const { isListening, isSupported: listenSupported } = speech;

// override timeout
speech.recognition.onend = () => {
  if (isListening.value) {
    speech.recognition.start();
  } else {
    speech.stop();
  }
};

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
    searchInput.value.$data.input = route.query.q ?? '';
  }
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
    page.value = 1;
    if (isListening.value) speech.recognition.stop();
    setTimeout(() => {
      if (!isListening.value) speech.recognition.start();
    }, 100);
  }
}

function onSelectItem(val) {
  q.value = val;
  q.page = 1;
}

function onEnter(e) {
  q.value = (e.target.value ?? '').trim();
  searchInput.value.selectItem(q.value);
}

function onClickSearch(e) {
  q.value = searchInput.value.$data.input;
}

function onInput(e) {
  searchInput.value.$data.currentSelectionIndex = -1;
  if (e.input.trim()) debouncedFetchCompletions(e.input);
}

function onLink(val) {
  q.value = val;
  page.value = 1;
}

function onNextPage() {
  page.value++;
}

function onFocus(e) {
  window.oncontextmenu = () => false;
  document.getElementById(searchInput.value.$data.inputId).select();
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
  if (ct === 0) return `No ${plWord} found`;
  if (ct === 1) return `1 ${singularWord}`;
  return `${ct} ${plWord}`;
}

function formatNgram(ngram) {
  return ngram?.replace(/\bi\b/g, 'I');
}

fetchRhymes(q.value, page.value);
</script>

<template>
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

    <ul v-if="rhymes && (!loading || page > 1)">
      <li v-for="r of rhymes" :key="r.ngram" :class="`hit ${r.type}`">
        <a @click="() => onLink(r.ngram)">{{ formatNgram(r.ngram) }}</a>
        <span v-if="!!r.frequency && r.type === 'rhyme'" class="freq"> ({{ r.frequency }}) </span>
      </li>
    </ul>

    <button v-if="!loading && hasNextPage" class="more" @click="onNextPage">More...</button>
  </section>
</template>

<style lang="scss" scoped>
@import '@/rhymes.scss';
@import 'vue3-simple-typeahead/dist/vue3-simple-typeahead.css';
</style>
