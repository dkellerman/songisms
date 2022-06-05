import {useMemo} from "react";
import Link from 'next/link';
import Head from "next/head";
import { StyledLayout, Nav, Main } from './StyledLayout';


export default function Layout({ children }) {
  return (
    <StyledLayout>
      <Head>
        <title>Rhymium</title>
      </Head>
      <Nav>
        <h1>
          <Link href="/">Rhymium</Link>
        </h1>
        <div className="links">
          <a href="https://bipium.com">Metronome</a>
        </div>
      </Nav>

      <Main>{children}</Main>
    </StyledLayout>
  );
}
