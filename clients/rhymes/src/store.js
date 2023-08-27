import axios from 'axios';
import { defineStore } from 'pinia';

const PER_PAGE = 100;
const COMPLETIONS_COUNT = 20;

const FETCH_RHYMES = `
    query Rhymes($q: String, $offset: Int, $limit: Int) {
      rhymes(q: $q, offset: $offset, limit: $limit) {
        ngram
        frequency
        type
      }
    }
  `;

const FETCH_COMPLETIONS = `
    query Completions($q: String, $ct: Int) {
      completions(q: $q, ct: $ct) {
        text
      }
    }
  `;

const url = `${process.env.VUE_APP_SISM_API_BASE_URL}/graphql/`;

export const useRhymesStore = defineStore('rhymes', {
  state: () => ({
    rhymes: [],
    completions: [],
    hasNextPage: false,
    loading: false,
  }),
  actions: {
    async fetchCompletions(q) {
      const resp = await axios.post(url, {
        query: FETCH_COMPLETIONS,
        variables: { q, ct: COMPLETIONS_COUNT },
      });
      if (resp.data.errors) {
        console.error('Completions errors', resp.data.errors);
        throw new Error(resp.data.errors[0].message);
      }
      let data = resp.data.data.completions;
      this.completions = data.map(item => item.text);
    },

    async fetchRhymes(q, page = 1) {
      this.loading = true;
      this.abortController = new AbortController();
      const resp = await axios.post(
        url,
        {
          query: FETCH_RHYMES,
          variables: {
            q,
            offset: (page - 1) * PER_PAGE,
            limit: PER_PAGE,
          },
        },
        {
          signal: this.abortController.signal,
        },
      );

      if (resp.data.errors) {
        console.error('Fetch rhymes errors', resp.data.errors);
        throw new Error(resp.data.errors[0].message);
      }

      let newRhymes = resp.data.data.rhymes;

      if (page > 1) newRhymes = [...(this.rhymes ?? []), ...newRhymes];

      this.rhymes = newRhymes;
      this.hasNextPage = newRhymes?.length === page * PER_PAGE && newRhymes.length < 100;
      this.loading = false;
    },

    abort() {
      try {
        this.abortController?.abort();
      } catch (e) {
        console.error(e);
      }
    },
  },
});
