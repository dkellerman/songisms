<script>
export default {
  name: 'SongLinks',
};
</script>

<script setup>
import { useSongsStore } from '@/stores/songs';
import { computed } from 'vue';
import { useRoute } from 'vue-router';

const route = useRoute();
const { getNextSong, getRandomSong } = useSongsStore();
const nextSong = computed(() => (route.name === 'SongDetail' ? getNextSong(route.params.id) : null));
const randomSong = computed(() => getRandomSong());
</script>

<template class="song-links">
  <router-link v-if="nextSong" :to="{ name: 'SongDetail', params: { id: nextSong.spotifyId } }">Next Song</router-link>
  <router-link v-if="randomSong" :to="{ name: 'SongDetail', params: { id: randomSong.spotifyId } }"
    >Random Song</router-link
  >
</template>
