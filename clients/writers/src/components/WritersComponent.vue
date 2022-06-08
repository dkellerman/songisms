<script setup>
  import axios from 'axios';
  import debounce from 'lodash/debounce';
  import { ref, watch } from 'vue'

  const FETCH_WRITERS = `
    query Writers($q: String, $page: Int) {
      writers(q: $q, page: $page) {
        items {
          name
          songCt
        }
        pages
        total
        hasNext
      }
    }
  `;

  const writers = ref(null);
  const page = ref(null);
  const hasNext = ref(false);
  const q = ref('');

  async function fetchWriters() {
    const url = `${process.env.VUE_APP_SISM_API_BASE_URL}/graphql/`;
    const resp = await axios.post(url, {
      query: FETCH_WRITERS,
      variables: { q: q.value ?? null, page: page.value },
    });
    const newWriters = resp.data.data.writers.items;
    console.log('* writers', newWriters);
    if (page.value === 1)
      writers.value = newWriters;
    else
      writers.value.push(...newWriters);
    hasNext.value = resp.data.data.writers.hasNext;
  }

  watch(page, fetchWriters);
  watch(q, debounce(() => {
    page.value = 1;
    fetchWriters();
  }, 500));

  page.value = 1;
</script>

<template>
  <h2>Writers</h2>

  <input v-model.trim="q" placeholder="Search by name..." />

  <ul class="none">
    <li v-for="writer in writers" :key="writer.name">
      <a href="javascript:void(0)">{{ writer.name }}</a>
      <small>({{ writer.songCt }})</small>
    </li>
  </ul>
  <button class="more compact" v-if="hasNext" @click="page++">More...</button>
</template>

<script>
  export default {
    name: 'WritersComponent'
  }
</script>

<style scoped lang="scss">
  h2 {
    margin: 5px 0;
  }
  input {
    width: 100%;
    max-width: 500px;
  }
  li {
    margin-bottom: 10px;
    a {
      margin-right: 5px;
    }
  }
</style>
