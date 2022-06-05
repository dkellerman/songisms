import { Html, Head, Main, NextScript } from 'next/document';

export default function Document(ctx) {
  return (
    <Html>
      <Head>
        <link href="https://unpkg.com/papercss@1.8.3/dist/paper.min.css" rel="stylesheet" key="stylelib" />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
