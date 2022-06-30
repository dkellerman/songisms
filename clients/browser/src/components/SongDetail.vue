<script>
import LyricsComponent from "@/components/LyricsComponent";

export default {
  name: 'SongDetail',
  components: [
    LyricsComponent,
  ],
};
</script>

<script setup>
import { computed, watchEffect } from 'vue';
import { useRoute } from 'vue-router';
import { storeToRefs } from 'pinia';
import { useSongsStore } from '@/stores/songs';
import Lyrics from "@/components/LyricsComponent";

const route = useRoute();
const spotifyId = computed(() => route.params.id);
const { song } = storeToRefs(useSongsStore());
const { fetchSong } = useSongsStore();
const adminLink = computed(() => `https://songisms.herokuapp.com/admin/api/song/${song.value?.id}/change`);
const smLink = computed(() => (song.value ? song.value.metadata.songMeanings?.href : null));

watchEffect(() => {
  if (spotifyId.value) fetchSong(spotifyId.value);
});
</script>

<template>
  <nav aria-label="breadcrumbs">
    <router-link :to="{ name: 'Songs' }">&lt; All Songs</router-link>
  </nav>

  <section v-if="song">
    <h2>{{ song.title }} {{ song.isNew ? '[New]' : ''}}</h2>
    <div v-html="song.spotifyPlayer" />

    <dl>
      <dt>Artists</dt>
      <dd>{{ song.artists?.map(a => a.name).join(', ') }}</dd>

      <dt>Writers</dt>
      <dd>
        <span v-for="(w, index) in song.writers" :key="w.id">
          <span v-if="index != 0">, </span>
          <router-link :to="`/writers/${w.id}`">{{ w.name }}</router-link>
        </span>
      </dd>

      <dt>Links</dt>
      <dd>
        <ul class="none links">
          <li v-if="adminLink">
            <a :href="adminLink">Admin</a>
          </li>
          <li v-if="song.youtubeUrl">
            <a :href="song.youtubeUrl">Youtube</a>
          </li>
          <li v-if="song.jaxstaUrl">
            <a :href="song.jaxstaUrl">Jaxsta</a>
          </li>
          <li v-if="song.spotifyUrl">
            <a :href="song.spotifyUrl">Spotify</a>
          </li>
          <li v-if="song.audioFileUrl">
            <a :href="song.audioFileUrl">Audio</a>
          </li>
          <li v-if="smLink">
            <a :href="smLink">Song Meanings</a>
          </li>
        </ul>
      </dd>

      <dt>Lyrics</dt>
      <dd><LyricsComponent :lyrics="song.lyrics" /></dd>


      <dt>Rhymes</dt>
      <dd v-html="song.rhymesRaw?.replace(/\n/g, '<br>').replace(/;/g, ' / ')"></dd>
    </dl>
  </section>
</template>

<style scoped lang="scss">
section {
  min-width: 320px;
  max-width: 800px;

  ul.links {
    display: flex;
    flex-flow: row wrap;
    gap: 10px;
    li:not(:last-of-type) {
      &:after {
        content: ' | ';
      }
    }
  }

  dl {
    dt {
      font-weight: bold;
      background: #eee;
      margin-bottom: 10px;
      padding: 5px 5px 3px 5px;
    }

    dd {
      margin: 5px 0 30px 0;
    }
  }
}
</style>
