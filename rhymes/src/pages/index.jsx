import dynamic from 'next/dynamic';
import Layout from "../components/Layout";

const Rhymes = dynamic(() => import('../components/Rhymes'), {
  ssr: false,
});

export default function RhymesPage() {
  return (
    <Layout site="rhymes">
      <Rhymes />;
    </Layout>
  );
}
