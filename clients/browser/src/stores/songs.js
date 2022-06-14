import axios from "axios";
import { defineStore } from 'pinia';

const SONGS_INDEX = `
  query SongsIndex {
    songsIndex {
      spotifyId
      title
      artists {
        name
      }
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
  }),
  actions: {
    async fetchSongsIndex() {
      const resp = await axios.post(url, {
        query: SONGS_INDEX,
      });
      this.songsIndex = resp.data.data.songsIndex;
    },

    async fetchSongs(q, page=1) {
      const resp = await axios.post(url, {
        query: LIST_SONGS,
        variables: { q: q ?? null, page },
      });

      const result = resp.data.data.songs;
      console.log('* songs', result);
      this.total = result.total;
      this.hasNext = result.hasNext;
      if (page === 1) {
        this.songs = result.items;
      }  else {
        this.songs = [...(this.songs ?? []), ...result.items];
      }
    }
  }
});
