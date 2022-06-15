<script>
export default {
  name: 'WriterDetail',
};
</script>

<script setup>
import axios from 'axios';
import { ref, watchEffect } from 'vue';
import router from '@/router';

const GET_WRITER = `
    query ($id: Int!) {
      writer(id: $id) {
        name
        altNames
        songs {
          spotifyId
          title
          artists {
            name
          }
        }
      }
    }
  `;

const id = ref(router.currentRoute.value.params.id);
const writer = ref();
const adminLink = writer.value ? `https://songisms.herokuapp.com/admin/api/writer/${id}` : null;

async function fetchWriter() {
  const url = `${process.env.VUE_APP_SISM_API_BASE_URL}/graphql/`;
  const resp = await axios.post(url, {
    query: GET_WRITER,
    variables: { id: id.value },
  });
  writer.value = resp.data.data.writer;
  console.log('* writer', writer.value);
}

watchEffect(() => {
  fetchWriter();
});
</script>

<template>
  <nav aria-label="breadcrumbs">
    <router-link to="/writers">&lt; All Writers</router-link>
  </nav>

  <div v-if="writer">
    <h2>{{ writer.name }}</h2>

    <section class="altnames">
      <label>{{ writer.altNames?.length || 'No' }} alternate names</label>
      <ul v-if="writer.altNames" class="none">
        <li v-for="altName in writer.altNames" :key="altName">
          {{ altName }}
        </li>
      </ul>
    </section>

    <section class="songs">
      <label>{{ writer.songs?.length || 'No' }} songs</label>
      <ul v-if="writer.songs" class="none">
        <li v-for="song in writer.songs" :key="song.spotifyId">
          <router-link :to="`/songs/${song.spotifyId}`">{{ song.title }}</router-link>
          &mdash;
          <span class="artist">{{ song.artists.map(a => a.name).join(', ') }}</span>
        </li>
      </ul>
    </section>
  </div>
</template>

<style scoped lang="scss">
ul {
  li {
    margin: 10px 0;
    .artist {
      font-size: medium;
    }
  }
}
</style>
