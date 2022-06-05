import { Html, Head, Main, NextScript } from 'next/document';

export default function Document(ctx) {
  return (
    <Html>
      <Head>
        <title>Rhymium</title>
        <link href="https://unpkg.com/papercss@1.8.3/dist/paper.min.css" rel="stylesheet" key="papercss" />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
