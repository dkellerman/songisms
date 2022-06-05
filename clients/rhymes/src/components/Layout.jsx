import { useMemo } from 'react';
import Link from 'next/link';
import Head from 'next/head';

export default function Layout({ children }) {
  return (
    <>
      <Head>
        <title>Rhymium</title>
        <link rel="icon" href="/favicon.ico?v=1" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
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
