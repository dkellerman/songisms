import { useMemo } from 'react';
import Link from 'next/link';
import Head from 'next/head';
import styled from 'styled-components';
import merge from 'lodash/merge';
import { useAuth } from '../hooks/useAuth';
import { StyledLayout, Nav, Main } from './StyledLayout';

export default function Layout({ site = 'default', children }) {
  const { user, logout } = useAuth();

  async function onLogout() {
    await logout();
    window.location.href = '/login';
  }

  return (
    <StyledLayout>
      <Head>
        <title>Songisms</title>
      </Head>
      <Nav>
        <h1>
          <Link href="/">Songisms</Link>
        </h1>
        <div className="links">{user && <button onClick={onLogout}>Log out</button>}</div>
      </Nav>

      <Main>{children}</Main>
    </StyledLayout>
  );
}
