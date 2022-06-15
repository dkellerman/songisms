<script>
export default {
  name: 'SongsList',
};
</script>

<script setup>
import { debounce } from 'lodash-es';
import { ref, watch } from 'vue';
import { storeToRefs } from 'pinia';
import { useSongsStore } from '@/stores/songs';

const { songs, hasNext, total, curQuery, curPage } = storeToRefs(useSongsStore());
const { fetchSongs } = useSongsStore();
const page = ref(curPage.value);
const q = ref(curQuery.value);

const newSearch = () => {
  page.value = 1;
  fetchSongs(q.value, page.value);
};

watch(page, () => fetchSongs(q.value, page.value));
watch(q, debounce(newSearch, 500));

if (!songs.value) {
  fetchSongs(q.value, 1);
}
</script>

<template>
  <h2>Songs</h2>

  <input v-model.trim="q" placeholder="Search by name..." />
  <label v-if="total !== undefined">{{ total }} songs found</label>

  <ul class="none" v-if="songs">
    <li v-for="song in songs" :key="song.spotifyId">
      <router-link :to="`/songs/${song.spotifyId}`">{{ song.title }}</router-link>
      &mdash; {{ song.artists.map(a => a.name).join(', ') }}
    </li>
  </ul>
  <button class="more compact" v-if="hasNext" @click="page++">More...</button>
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
