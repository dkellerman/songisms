import { useState, useEffect } from 'react';
import { gql } from '@apollo/client';
import { useClient } from '../hooks/useClient';

export const LIST_SONGS = gql`
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

export function useSongs(initialFilters) {
  const [songs, setSongs] = useState();
  const [filters, setFilters] = useState(initialFilters);
  const client = useClient();

  useEffect(() => {
    if (!client) return;

    (async function () {
      const resp = await client.query({
        query: LIST_SONGS,
        variables: filters,
      });

      const newSongs = resp.data.songs;
      if (filters.append)
        setSongs({
          ...newSongs,
          items: [...(songs?.items ?? []), ...newSongs.items],
        });
      else setSongs(newSongs);
    })();
  }, [filters, client]);

  return [songs, filters, setFilters];
}
