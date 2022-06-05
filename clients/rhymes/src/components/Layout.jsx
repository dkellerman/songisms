import { useMemo } from 'react';
import Link from 'next/link';
import Head from 'next/head';

export default function Layout({ children }) {
  return (
    <>
      <Head>
        <title>Rhymium</title>
      </Head>
      <nav>
        <h1>
          <Link href="/">Rhymium</Link>
        </h1>
        <div className="links">
          <a href="https://bipium.com">Metronome</a>
        </div>
      </nav>

      <main>{children}</main>
    </>
  );
}
