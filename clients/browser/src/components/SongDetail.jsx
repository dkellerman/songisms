import Link from 'next/link';
import styled from 'styled-components';

const StyledSongDetail = styled.article`
  min-width: 320px;
  max-width: 800px;
  dl {
    dt {
      font-weight: bold;
      background: #eee;
      margin-bottom: 10px;
      padding: 5px 5px 3px 5px;
    }
    dd {
      margin: 5px 0 30px 0;
    }
  }
`;

export default function SongDetail({ song }) {
  const adminLink = `https://songisms.herokuapp.com/admin/api/song/?q=${encodeURIComponent(song.title)}`;

  return (
    <StyledSongDetail>
      <nav aria-label="breadcrumbs">
        <Link href="/songs">&lt; All Songs</Link>
      </nav>

      <h2>{song.title}</h2>

      <div dangerouslySetInnerHTML={{ __html: song.spotifyPlayer }} />

      <small>[ <a  href={adminLink} target="_blank" rel="noreferrer">Admin</a> ]</small>

      <dl>
        <dt>Artists</dt>
        <dd>{song.artists?.map(a => a.name).join(', ')}</dd>

        <dt>Writers</dt>
        <dd>{song.writers?.map(w => w.name).join(', ')}</dd>

        <dt>Links</dt>
        <dd>
          <ul>
            {song.youtubeUrl && <li><a href={song.youtubeUrl}>Youtube</a></li>}
            {song.jaxstaUrl && <li><a href={song.jaxstaUrl}>Jaxsta</a></li>}
            {song.spotifyUrl && <li><a href={song.spotifyUrl}>Spotify</a></li>}
            {song.audioFileUrl && <li><a href={song.audioFileUrl}>Audio</a></li>}
          </ul>
        </dd>

        <dt>Lyrics</dt>
        <dd
          dangerouslySetInnerHTML={{
            __html: song.lyrics?.replace(/\n/g, '<br>'),
          }}
        ></dd>

        <dt>Rhymes</dt>
        <dd
          dangerouslySetInnerHTML={{
            __html: song.rhymesRaw?.replace(/\n/g, '<br>').replace(/;/g, ' / '),
          }}
        ></dd>
      </dl>
    </StyledSongDetail>
  );
}
