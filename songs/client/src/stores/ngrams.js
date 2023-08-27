import axios from 'axios';
import { defineStore } from 'pinia';

const LIST_NGRAMS = `
  query NGrams($q: String, $page: Int, $tags: [String], $ordering: [String]) {
    ngrams(q: $q, page: $page, tags: $tags, ordering: $ordering) {
      q
      total
      page
      hasNext
      items {
        text
        n
        count
        songCount
        pct
        adjPct
        songPct
        titlePct
      }
    }
  }
`;

const url = `${process.env.VUE_APP_SISM_API_BASE_URL}/graphql/`;

export const useNGramsStore = defineStore('ngrams', {
  state: () => ({
    ngrams: undefined,
    total: undefined,
    hasNext: undefined,
    curQuery: undefined,
    curPage: undefined,
  }),
  actions: {
    async fetchNGrams(q, page = 1, tags = [], ordering = []) {
      const resp = await axios.post(url, {
        query: LIST_NGRAMS,
        variables: {
          q: q ?? null,
          page,
          tags,
          ordering,
        },
      });

      if (resp.data.errors) {
        console.error('Fetch ngrams errors', resp.data.errors);
        throw new Error(resp.data.errors[0].message);
      }

      const result = resp.data.data.ngrams;
      console.log('* ngrams', result);
      this.total = result.total;
      this.hasNext = result.hasNext;
      if (page === 1) {
        this.ngrams = result.items;
      } else {
        this.ngrams = [...(this.ngrams ?? []), ...result.items];
      }
      this.curQuery = q;
      this.curPage = page;
    },
  },
});
