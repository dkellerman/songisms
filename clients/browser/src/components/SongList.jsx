import { useCallback } from 'react';
import debounce from 'lodash/debounce';
import styled from 'styled-components';
import Link from 'next/link';
import { useSongs } from '../hooks/useSongs';

const StyledSongList = styled.article`
  input {
    margin-bottom: 10px;
    min-width: 320px;
    max-width: 90vw;
  }

  table {
    display: grid;
    grid-template-columns: repeat(2, minmax(100px, 300px));
    column-gap: 15px;
    border-collapse: collapse;
    margin: 20px 0;

    thead,
    tbody,
    tr {
      display: contents;
    }
    th {
      text-align: left;
    }
    td {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
  }

  button {
    margin: 10px;
  }
`;

const initialFilters = { ordering: ['-updated', '-created'] };

export default function SongList() {
  const [songs, filters, setFilters] = useSongs(initialFilters);
  const { total, hasNext, items, page, q } = songs ?? {};

  const search = useCallback(
    debounce(async e => {
      const q = e.target.value?.trim();
      q ? setFilters({ q }) : setFilters(initialFilters);
    }, 500),
    [],
  );

  const loadMore = useCallback(async () => {
    setFilters({ ...filters, page: page + 1, append: true });
  }, [page, q]);

  return (
    <StyledSongList>
      <h2>Songs</h2>

      <fieldset>
        <input type="text" onChange={search} defaultValue={q || ''} placeholder="Search for songs..." />
      </fieldset>

      {items && <label>{(!items.length && 'No songs found') || `${total} songs found`}</label>}

      {!!items?.length && (
        <table>
          <thead>
            <tr>
              <th>Title</th>
              <th>Artists</th>
            </tr>
          </thead>
          <tbody>
            {items?.map(song => (
              <tr key={song.spotifyId}>
                <td>
                  <Link href={`/songs/${song.spotifyId}`}>{song.title}</Link>
                </td>
                <td>{song.artists.map(a => a.name).join(', ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {hasNext && <button className="more compact" onClick={loadMore}>Show more</button>}
    </StyledSongList>
  );
}
