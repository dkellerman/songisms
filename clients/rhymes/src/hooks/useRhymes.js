import { useState, useEffect, useRef } from 'react';
import { gql } from '@apollo/client';
import { useAPIClient } from './useAPIClient';

export const FETCH_RHYMES = gql`
  query Rhymes($q: String, $limit: Int, $searchType: String) {
    rhymes(q: $q, limit: $limit, searchType: $searchType) {
      ngram
      frequency
      type
    }
  }
`;

export function useRhymes(q, searchType, limit) {
  const [rhymes, setRhymes] = useState();
  const [loading, setLoading] = useState();
  const client = useAPIClient();
  const abortController = useRef();

  useEffect(() => {
    if (!client) return;

    (async function () {
      setLoading(true);
      abortController.current = new AbortController();

      const qstr = (q ?? '').toLowerCase().trim();
      const resp = await client.query({
        query: FETCH_RHYMES,
        variables: { q: qstr, limit, searchType },
        context: {
          fetchOptions: {
            signal: abortController.current.signal,
          },
        },
      });
      abortController.current = null;
      console.log('* rhymes', resp.data.rhymes);
      setRhymes(resp.data.rhymes);
      setLoading(false);
    })();
  }, [q, client, searchType, limit]);

  return { rhymes, loading, abort: abortController.current?.abort };
}
