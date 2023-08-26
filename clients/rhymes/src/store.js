import axios from 'axios';
import { defineStore } from 'pinia';

const PER_PAGE = 100;
const SUGGESTION_COUNT = 20;

const FETCH_RHYMES = `
    query Rhymes($q: String, $offset: Int, $limit: Int) {
      rhymes(q: $q, offset: $offset, limit: $limit) {
        ngram
        frequency
        type
      }
    }
  `;

const FETCH_SUGGESTIONS = `
    query Suggestions($q: String, $ct: Int) {
      rhymesSuggest(q: $q, ct: $ct) {
        text
      }
    }
  `;

const url = `${process.env.VUE_APP_SISM_API_BASE_URL}/graphql/`;

export const useRhymesStore = defineStore('rhymes', {
  state: () => ({
    rhymes: [],
    suggestions: [],
    hasNextPage: false,
    loading: false,
  }),
  actions: {
    async fetchSuggestions(q) {
      const resp = await axios.post(url, {
        query: FETCH_SUGGESTIONS,
        variables: { q, ct: SUGGESTION_COUNT },
      });
      if (resp.data.errors) {
        console.error('Suggest errors', resp.data.errors);
        throw new Error(resp.data.errors[0].message);
      }
      let data = resp.data.data.rhymesSuggest;
      console.log('*suggest', data);
      this.suggestions = data.map(item => item.text);
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
      console.log('* rhymes', page, newRhymes);

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
