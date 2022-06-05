import SongList from '../../components/SongList';
import Layout from "../../components/Layout";

export default function SongListPage({ songs }) {
  return (
    <Layout site="songs">
      <SongList songs={songs} />
    </Layout>
  );
}
