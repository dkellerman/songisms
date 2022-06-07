import { useMemo } from 'react';
import Link from 'next/link';
import Head from 'next/head';
import { useAuth } from '../hooks/useAuth';

export default function Layout({ site = 'default', children }) {
  const { user, logout } = useAuth();

  async function onLogout() {
    await logout();
    window.location.href = '/login';
  }

  return (
    <>
      <Head>
        <title>Songisms</title>
        <link rel="icon" href="/favicon.png?v=1" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      <nav>
        <h1>
          <Link href="/">Songisms</Link>
        </h1>
        <div className="links">
          {user && (
            <button className="logout compact" onClick={onLogout}>
              Log out
            </button>
          )}
        </div>
      </nav>

      <main>{children}</main>
    </>
  );
}