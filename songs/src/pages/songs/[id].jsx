import { gql } from '@apollo/client';
import { getClient } from '../../hooks/useClient';
import SongDetail from '../../components/SongDetail';
import Layout from "../../components/Layout";

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
  return (
    <Layout site="songs">
      <SongDetail song={song} />
    </Layout>
  );
}

export async function getServerSideProps(ctx) {
  const { id } = ctx.query;
  const client = await getClient(ctx);

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
