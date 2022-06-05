import Link from 'next/link';
import styled from 'styled-components';

const StyledSongDetail = styled.article`
  min-width: 320px;
  max-width: 800px;
  dl {
    dt {
      font-weight: bold;
      background: #eee;
      margin-bottom: 5px;
      padding: 3px;
    }
    dd {
      margin: 5px 0 30px 0;
    }
  }
`;

export default function SongDetail({ song }) {
  return (
    <StyledSongDetail>
      <nav aria-label="breadcrumbs">
        <Link href="/songs">&lt; All Songs</Link>
      </nav>

      <h2>{song.title}</h2>

      <div dangerouslySetInnerHTML={{ __html: song.spotifyPlayer }} />

      <dl>
        <dt>Artists</dt>
        <dd>{song.artists?.map(a => a.name).join(', ')}</dd>

        <dt>Writers</dt>
        <dd>{song.writers?.map(w => w.name).join(', ')}</dd>

        <dt>Links</dt>
        <dd>
          <ul>
            <li>{song.youtubeUrl && <a href={song.youtubeUrl}>Youtube</a>}</li>
            <li>{song.jaxstaUrl && <a href={song.jaxstaUrl}>Jaxsta</a>}</li>
            <li>{song.spotifyUrl && <a href={song.spotifyUrl}>Spotify</a>}</li>
            <li>{song.audioFileUrl && <a href={song.audioFileUrl}>Audio</a>}</li>
          </ul>
        </dd>

        <dt>Lyrics</dt>
        <dd
          dangerouslySetInnerHTML={{
            __html: song.lyrics?.replace(/\n/g, '<br>'),
          }}
        ></dd>
      </dl>
    </StyledSongDetail>
  );
}