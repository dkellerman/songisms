import Layout from '../components/Layout';
import Script from 'next/script';
import 'papercss/dist/paper.min.css';
import '../../../shared/layout.scss';

function RhymesApp({ Component, pageProps }) {
  return (
    <>
      <Script src="https://www.googletagmanager.com/gtag/js?id=UA-158752156-1" strategy="afterInteractive"></Script>
      <Script
        id="google-analytics"
        strategy="afterInteractive"
        dangerouslySetInnerHTML={{
          __html: `
        window.dataLayer = window.dataLayer || [];
        function gtag() { window.dataLayer.push(arguments); }
        gtag('js', new Date());
        gtag('config', 'UA-158752156-1');
      `,
        }}
      ></Script>

      <Layout>
        <Component {...pageProps} />
      </Layout>
    </>
  );
}

export default RhymesApp;
