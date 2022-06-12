<script>
  export default {
    name: 'RhymesPage'
  }
</script>

<script setup>
  import axios from 'axios';
  import debounce from 'lodash/debounce';
  import { ref, watch, computed } from 'vue';

  const PER_PAGE = 50;

  const FETCH_RHYMES = `
    query Rhymes($q: String, $offset: Int, $limit: Int) {
      rhymes(q: $q, offset: $offset, limit: $limit) {
        ngram
        frequency
        type
      }
    }
  `;

  const result = ref();
  const page = ref(null);
  const q = ref('');
  const loading = ref();
  const abortController = ref();

  const counts = computed(() => ({
    rhyme: result.value?.items?.filter(r => r.type === 'rhyme').length || 0,
    l2: result.value?.items?.filter(r => r.type === 'rhyme-l2').length || 0,
    sug: result.value?.items?.filter(r => r.type === 'suggestion').length || 0,
  }));

  const label = computed(() => {
    return [
      `${ct2str(counts.value.rhyme, 'rhyme')} found`,
      counts.value.l2 > 0 && ct2str(counts.value.l2, 'rhyme-of-rhyme', 'rhymes-of-rhymes'),
      counts.value.sug > 0 && ct2str(counts.value.sug, 'suggestion'),
    ].filter(Boolean).join(', ')
  });

  async function fetchRhymes() {
    loading.value = true;
    const url = `${process.env.VUE_APP_SISM_API_BASE_URL}/graphql/`;
    abortController.value = new AbortController();
    const resp = await axios.post(url, {
      query: FETCH_RHYMES,
      variables: { q, offset: (page.value - 1) * PER_PAGE, limit: PER_PAGE },
    });

    abortController.value = null;
    let rhymes = resp.data.data.rhymes;
    console.log("***", rhymes);

    if (page.value > 1)
      rhymes = [...result.value.items, ...rhymes];

    result.value = {
      items: rhymes,
      hasNext: rhymes.length === page.value * PER_PAGE
    };

    loading.value = false;
  }

  function abort() {
    try {
      abortController.value?.abort();
    } catch (e) {
      console.error(e);
    }
  }

  // function track(category, action, label) {
  //   if (window.gtag) {
  //     window.gtag('event', action, {
  //       event_category: category,
  //       event_label: label,
  //     });
  //   }
  // }

  function ct2str(ct, singularWord, pluralWord) {
    const plWord = pluralWord ?? `${singularWord}s`;
    if (ct === 0) return `No ${plWord}`;
    if (ct === 1) return `1 ${singularWord}`;
    return `${ct} ${plWord}`;
  }

  const debouncedSearch = debounce(() => {
    page.value = 1;
    fetchRhymes();
  }, 500);

  watch(page, fetchRhymes);

  watch(q, () => {
    abort();
    debouncedSearch();
  });

  page.value = 1;
</script>

<template>
  <article>
    <fieldset>
      <input
        type="text"
        v-model="q"
        placeholder="Find rhymes in songs..."
      />
    </fieldset>

    <output>
      <label v-if="loading">Searching...</label>

      <div v-if="!loading">
        <label v-if="!q">Top {{counts.rhyme}} rhymes</label>

        <label v-if="q">{{ label }}</label>

        <ul>
          <li v-for="r of result.items" :key="r.ngram">
            <span class="`hit ${r.type}`" @click="() => q.value = r.ngram">
              {{r.ngram}}
            </span>
            <span v-if="!!r.frequency && r.type === 'rhyme'" class="'freq'">({{r.frequency}})</span>
          </li>
        </ul>
      </div>
    </output>
  </article>
</template>

<style scoped lang="scss">
  article {
    fieldset {
      display: flex;
      flex-flow: row wrap;
      align-items: center;
      width: 100%;
      margin: 20px 0 12px 0;
      padding-right: 5px;
      gap: 20px;

      input[type='text'] {
        border-radius: 0;
        width: 50vw;
        min-width: 180px;
        max-width: 500px;

        &::-webkit-search-cancel-button {
          -webkit-appearance: searchfield-cancel-button;
        }
      }
    }

    output label {
      font-size: large;
    }
  }

  ul {
    list-style: none;
    padding-left: 0;
    display: flex;
    flex-flow: row wrap;
    max-width: 768px;
    gap: 20px;

    li {
      text-indent: 0;
      font-size: larger;

      &:before {
        display: none;
      }

      .freq {
        font-size: medium;
        color: #666;
      }

      .hit {
        text-decoration: underline;
        color: blue;
        cursor: pointer;

        &.rhyme-l2 {
          opacity: 0.6;
        }

        &.suggestion {
          opacity: 0.6;
          font-style: italic;
        }
      }
    }

    @media screen and (max-width: 374px) {
      width: 100%;
    }
    @media screen and (min-width: 375px) and (max-width: 479px) {
      width: calc((100% - 20px) / 2);
    }
    @media screen and (min-width: 480px) {
      width: calc((100% - 40px) / 3);
    }
  }
</style>

