import axios from 'axios';
import { defineStore } from 'pinia';

const SONGS_INDEX = `
  query SongsIndex {
    songsIndex {
      spotifyId
      title
    }
  }
`;

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

const url = `${process.env.VUE_APP_SISM_API_BASE_URL}/graphql/`;

export const useSongsStore = defineStore('songs', {
  state: () => ({
    songsIndex: undefined,
    songs: undefined,
    total: undefined,
    hasNext: undefined,
    curQuery: undefined,
    curPage: undefined,
  }),
  actions: {
    async fetchSongsIndex() {
      const resp = await axios.post(url, {
        query: SONGS_INDEX,
      });
      if (resp.data.errors) {
        console.error('Fetch song index errors', resp.data.errors);
        throw new Error(resp.data.errors[0].message);
      }
      this.songsIndex = resp.data.data.songsIndex;
    },

    async fetchSongs(q, page = 1) {
      const resp = await axios.post(url, {
        query: LIST_SONGS,
        variables: { q: q ?? null, page },
      });

      if (resp.data.errors) {
        console.error('Fetch songs errors', resp.data.errors);
        throw new Error(resp.data.errors[0].message);
      }

      const result = resp.data.data.songs;
      console.log('* songs', result);
      this.total = result.total;
      this.hasNext = result.hasNext;
      if (page === 1) {
        this.songs = result.items;
      } else {
        this.songs = [...(this.songs ?? []), ...result.items];
      }
      this.curQuery = q;
      this.curPage = page;
    },

    getNextSong(curSongId) {
      const songsList = this.songs?.length ? this.songs : this.songsIndex;
      if (!songsList?.length) return;

      const curIdx = songsList.findIndex(s => s.spotifyId === curSongId) ?? 0;
      const idx = curIdx >= songsList.length - 1 ? 0 : curIdx + 1;
      return songsList[idx];
    },

    getRandomSong() {
      const songsList = this.songsIndex;
      if (!songsList?.length) return;
      const idx = Math.floor(Math.random() * songsList.length);
      return songsList[idx];
    },
  },
});
