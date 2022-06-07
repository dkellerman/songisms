import { useState, useEffect, useRef } from 'react';
import { gql } from '@apollo/client';
import { useAPIClient } from './useAPIClient';

export const FETCH_RHYMES = gql`
  query Rhymes($q: String, $searchType: String, $offset: Int, $limit: Int) {
    rhymes(q: $q, searchType: $searchType, offset: $offset, limit: $limit) {
      ngram
      frequency
      type
    }
  }
`;

export function useRhymes(q, searchType, page = 1, pageSize = 50) {
  const [rhymes, setRhymes] = useState();
  const [loading, setLoading] = useState();
  const client = useAPIClient();
  const abortController = useRef();

  useEffect(() => {
    if (!client) return;

    (async function () {
      if (page === 1) setLoading(true);
      abortController.current = new AbortController();

      const qstr = (q ?? '').toLowerCase().trim();
      const offset = (page - 1) * pageSize;
      const resp = await client.query({
        query: FETCH_RHYMES,
        variables: { q: qstr, offset, limit: pageSize, searchType },
        context: {
          fetchOptions: {
            signal: abortController.current.signal,
          },
        },
      });
      abortController.current = null;
      console.log('* rhymes', page, resp.data.rhymes);
      if (page === 1)
        setRhymes(resp.data.rhymes);
      else
        setRhymes(cur => [...cur, ...resp.data.rhymes]);
      setLoading(false);
    })();
  }, [q, client, searchType, page, pageSize]);

  return {
    rhymes,
    loading,
    hasNextPage: rhymes?.length === page * pageSize,
    abort: () => {
      try {
        abortController.current?.abort();
      } catch (e) {
        console.error(e);
      }
    },
  };
}
