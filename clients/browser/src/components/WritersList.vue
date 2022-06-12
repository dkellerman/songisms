<script>
export default {
  name: 'WritersList',
};
</script>

<script setup>
import axios from 'axios';
import { debounce } from 'lodash-es';
import { ref, watch } from 'vue';

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

const result = ref();
const page = ref(null);
const q = ref('');

async function fetchWriters() {
  const url = `${process.env.VUE_APP_SISM_API_BASE_URL}/graphql/`;
  const resp = await axios.post(url, {
    query: FETCH_WRITERS,
    variables: { q: q.value ?? null, page: page.value },
  });
  const newResult = resp.data.data.writers;
  console.log('* writers', newResult);
  if (page.value === 1) result.value = newResult;
  else
    result.value = {
      ...newResult,
      items: [...result.value.items, ...newResult.items],
    };
}

watch(page, fetchWriters);
watch(
  q,
  debounce(() => {
    page.value = 1;
    fetchWriters();
  }, 500),
);

page.value = 1;
</script>

<template>
  <h2>Writers</h2>

  <input v-model.trim="q" placeholder="Search by name..." />
  <label v-if="result?.total">{{ result.total }} writers found</label>

  <ul class="none">
    <li v-for="writer in result?.items" :key="writer.name">
      <a href="javascript:void(0)">{{ writer.name }}</a>
      <small>({{ writer.songCt }})</small>
    </li>
  </ul>
  <button class="more compact" v-if="result?.hasNext" @click="page++">More...</button>
</template>

<style scoped lang="scss">
h2 {
  margin: 5px 0;
}
input {
  width: 100%;
  max-width: 500px;
  margin-bottom: 10px;
}
label {
  font-size: large;
}
li {
  margin-bottom: 10px;
  a {
    margin-right: 5px;
  }
}
</style>
