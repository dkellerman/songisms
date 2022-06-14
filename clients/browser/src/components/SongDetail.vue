<script>
export default {
  name: 'SongDetail',
};
</script>

<script setup>
import { ref } from 'vue';
import router from '@/router';
import axios from 'axios';

const GET_SONG = `
    query ($id: String!) {
      song(spotifyId: $id) {
        title
        spotifyId
        spotifyPlayer
        spotifyUrl
        jaxstaUrl
        youtubeUrl
        audioFileUrl
        lyrics
        rhymesRaw

        artists {
          name
        }
        writers {
          name
        }
      }
    }
  `;

const id = router.currentRoute.value.params.id;
const song = ref();
const adminLink = song.value
  ? `https://songisms.herokuapp.com/admin/api/song/?q=${encodeURIComponent(song.value.title)}`
  : null;

async function fetchSong() {
  const url = `${process.env.VUE_APP_SISM_API_BASE_URL}/graphql/`;
  const resp = await axios.post(url, {
    query: GET_SONG,
    variables: { id },
  });
  song.value = resp.data.data.song;
  console.log('* song', song.value);
}

fetchSong();
</script>

<template>
  <nav aria-label="breadcrumbs">
    <router-link to="/songs">&lt; All Songs</router-link>
  </nav>

  <article>
    <h2>{{ song.title }}</h2>
    <div v-html="song.spotifyPlayer" />

    <small> [ <a :href="adminLink" target="_blank" rel="noreferrer">Admin</a> ] </small>

    <dl>
      <dt>Artists</dt>
      <dd>{{ song.artists?.map(a => a.name).join(', ') }}</dd>

      <dt>Writers</dt>
      <dd>{{ song.writers?.map(w => w.name).join(', ') }}</dd>

      <dt>Links</dt>
      <dd>
        <ul>
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
        </ul>
      </dd>

      <dt>Lyrics</dt>
      <dd v-html="song.lyrics?.replace(/\n/g, '<br>')"></dd>

      <dt>Rhymes</dt>
      <dd v-html="song.rhymesRaw?.replace(/\n/g, '<br>').replace(/;/g, ' / ')"></dd>
    </dl>
  </article>
</template>

<style scoped lang="scss">
article {
  min-width: 320px;
  max-width: 800px;

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
