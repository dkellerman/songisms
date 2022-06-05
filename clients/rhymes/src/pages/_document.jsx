import { Html, Head, Main, NextScript } from 'next/document';

export default function Document(ctx) {
  return (
    <Html>
      <Head>
        <link href="https://unpkg.com/papercss@1.8.3/dist/paper.min.css" rel="stylesheet" key="stylelib" />
        <script src="https://www.googletagmanager.com/gtag/js?id=UA-158752156-1"></script>
        <script>
          {`
            window.dataLayer = window.dataLayer || [];
            function gtag() { window.dataLayer.push(arguments); }
            gtag('js', new Date());
            gtag('config', 'UA-158752156-1');
          `}
        </script>
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
