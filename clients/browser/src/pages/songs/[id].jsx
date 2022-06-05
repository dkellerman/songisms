import { gql } from '@apollo/client';
import { getAPIClient } from '../../hooks/useAPIClient';
import SongDetail from '../../components/SongDetail';

const GET_SONG = gql`
  query ($id: String!) {
    song(spotifyId: $id) {
      title
      spotifyId
      spotifyPlayer
      spotifyUrl
      jaxstaUrl
      youtubeUrl
      audioFileUrl
      lyrics

      artists {
        name
      }
      writers {
        name
      }
    }
  }
`;

export default function SongDetailPage({ song }) {
  return <SongDetail song={song} />;
}

export async function getServerSideProps(ctx) {
  const { id } = ctx.query;
  const client = await getAPIClient(ctx);

  const resp = await client.query({
    query: GET_SONG,
    variables: { id },
  });

  const song = resp.data.song;

  return {
    props: {
      song,
    },
  };
}
