import dynamic from 'next/dynamic';

const Rhymes = dynamic(() => import('../components/Rhymes'), {
  ssr: false,
});

export default function RhymesPage() {
  return <Rhymes />;
}
