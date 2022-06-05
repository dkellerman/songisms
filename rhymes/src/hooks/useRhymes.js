import { useState, useEffect } from 'react';
import { gql } from '@apollo/client';
import { useClient } from './useClient';

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
  const client = useClient();

  useEffect(() => {
    if (!client) return;

    (async function () {
      if (!rhymes?.length) setLoading(true);
      const resp = await client.query({
        query: FETCH_RHYMES,
        variables: { q: q ?? '', limit, searchType },
      });
      console.log('* rhymes', resp.data.rhymes);
      setRhymes(resp.data.rhymes);
      setLoading(false);
    })();
  }, [q, client, searchType, limit]);

  return { rhymes, loading };
}
