<script>
export default {
  name: 'RhymesPage',
};
</script>

<script setup>
import axios from 'axios';
import { some, debounce } from 'lodash-es';
import { isMobile } from 'mobile-device-detect';
import { ref, computed, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';

const PER_PAGE = 50;
const SUGGEST_DEBOUNCE = 100;
const FETCH_RHYMES = `
    query Rhymes($q: String, $offset: Int, $limit: Int) {
      rhymes(q: $q, offset: $offset, limit: $limit) {
        ngram
        frequency
        type
      }
    }
  `;
const FETCH_SUGGESTIONS = `
    query Suggestions($q: String) {
      rhymesSuggest(q: $q) {
        text
      }
    }
  `;

const route = useRoute();
const router = useRouter();
const q = ref(route.query.q ?? '');
const page = ref(route.query.page ?? 1);
const rhymes = ref();
const searchInput = ref(null);
const suggestions = ref([]);
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
    counts.value.l2 > 0 && ct2str(counts.value.l2, 'maybe-rhyme'),
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

async function fetchSuggestions(val) {
  const url = `${process.env.VUE_APP_SISM_API_BASE_URL}/graphql/`;
  const resp = await axios.post(url, {
    query: FETCH_SUGGESTIONS,
    variables: { q: val },
  });
  let data = resp.data.data.rhymesSuggest;
  console.log('*suggest', data);
  suggestions.value = data.map(item => item.text);
}

const debouncedFetchSuggestions = debounce(fetchSuggestions, SUGGEST_DEBOUNCE);

function abort() {
  try {
    abortController.value?.abort();
  } catch (e) {
    console.log('search canceled');
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

async function search(newQ) {
  abort();
  q.value = newQ;
}

watch([q, page], () => {
  track('engagement', 'more', q.value);
  router.push({ query: { q: q.value } });
  fetchRhymes();
});

watch(() => [route.query.q, route.query.page], () => {
  q.value = route.query.q ?? '';
  page.value = route.query.page ?? 1;
});

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

fetchRhymes();
</script>

<template>
  <fieldset>
    <vue3-simple-typeahead
      placeholder="Find rhymes in songs..."
      @onInput="onInput"
      @selectItem="(e) => search(e)"
      @keyup.enter="onEnter"
      :items="suggestions"
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
        <a @click="() => { q = r.ngram; page = 1; }">{{ r.ngram }}</a>
        <span v-if="!!r.frequency && r.type === 'rhyme'" class="freq"> ({{ r.frequency }}) </span>
      </li>
    </ul>

    <button v-if="!loading && hasNextPage" class="more" @click="page++">More...</button>
  </section>
</template>

<style lang="scss">
  input[type='text'].simple-typeahead-input {
    border-radius: 0;
    width: 100%;
    &::-webkit-search-cancel-button {
      -webkit-appearance: searchfield-cancel-button;
    }
  }
</style>

<style scoped lang="scss">
  fieldset {
    display: flex;
    flex-flow: row nowrap;
    align-items: center;
    margin: 20px 0 12px 0;
    padding-right: 5px;

    width: 80vw;
    min-width: 180px;
    max-width: 600px;

    button {
      padding: 12px;
      margin: 0;
      background: #eee;
      border: 1px dotted #aaa;
    }
  }

  .output {
    label {
      font-size: 18px;
    }

    ul {
      --gap: 20;
      margin-top: 30px;
      list-style: none;
      padding-left: 0;
      display: flex;
      flex-flow: row wrap;
      max-width: 768px;
      gap: var(--gap)px;

      li {
        text-indent: 0;
        font-size: larger;
        margin-bottom: 12px;
        a {
          cursor: pointer;
        }

        &:before {
          display: none;
        }

        &.hit {
          &.rhyme a {
            opacity: 1.0;
            font-style: normal;
          }

          &.rhyme-l2 a {
            opacity: 0.6;
            font-style: normal;
          }

          &.suggestion a {
            opacity: 0.6;
            font-style: italic;
            overflow: visible;
            padding: 3px 3px 3px 0;
          }

          .freq {
            font-size: medium;
            color: #666;
          }
        }
      }
    }
  }

  @function colWidth($defaultCols) {
    @return calc(
      (100% - (var(--gap) * var(--cols, #{$defaultCols - 1}) * 1px)) /
      var(--cols, #{$defaultCols})
    );
  }

  @media screen and (max-width: 374px) {
    ul li {
      width: colWidth(1);
    }
  }
  @media screen and (min-width: 375px) and (max-width: 479px) {
    ul li {
      width: colWidth(2);
    }
  }
  @media screen and (min-width: 480px) {
    ul li {
      width: colWidth(3);
    }
  }
</style>
