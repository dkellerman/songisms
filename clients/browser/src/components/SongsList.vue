<script>
export default {
  name: 'SongsList',
};
</script>

<script setup>
import axios from 'axios';
import { debounce } from 'lodash-es';
import { ref, watch } from 'vue';

const LIST_SONGS = `
    query Songs($q: String, $page: Int, $ordering: [String]) {
      songs(q: $q, page: $page, ordering: $ordering) {
        q
        total
        page
        hasNext
        items {
          title
          artists {
            name
          }
          spotifyId
        }
      }
    }
  `;

const result = ref();
const page = ref(null);
const q = ref('');

async function fetchSongs() {
  const url = `${process.env.VUE_APP_SISM_API_BASE_URL}/graphql/`;
  const resp = await axios.post(url, {
    query: LIST_SONGS,
    variables: { q: q.value ?? null, page: page.value },
  });

  const newResult = resp.data.data.songs;
  console.log('* songs', newResult);
  if (page.value === 1) result.value = newResult;
  else
    result.value = {
      ...newResult,
      items: [...result.value.items, ...newResult.items],
    };
}

watch(page, fetchSongs);
watch(
  q,
  debounce(() => {
    page.value = 1;
    fetchSongs();
  }, 500),
);

page.value = 1;
</script>

<template>
  <h2>Songs</h2>

  <input v-model.trim="q" placeholder="Search by name..." />
  <label v-if="result?.total">{{ result.total }} songs found</label>

  <ul class="none">
    <li v-for="song in result?.items" :key="song.spotifyId">
      <router-link :to="`/songs/${song.spotifyId}`">{{ song.title }}</router-link>
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
