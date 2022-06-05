import { Html, Head, Main, NextScript } from 'next/document';
import Script from 'next/script';

export default function Document(ctx) {
  return (
    <Html>
      <Head>
        <link href="https://unpkg.com/papercss@1.8.3/dist/paper.min.css" rel="stylesheet" key="stylelib" />
        <Script src="https://www.googletagmanager.com/gtag/js?id=UA-158752156-1" />
        <Script id="google-analytics">
          {`
            window.dataLayer = window.dataLayer || [];
            function gtag() { window.dataLayer.push(arguments); }
            gtag('js', new Date());
            gtag('config', 'UA-158752156-1');
          `}
        </Script>
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
