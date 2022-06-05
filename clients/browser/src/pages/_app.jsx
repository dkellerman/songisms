import Layout from '../components/Layout';
import 'papercss/dist/paper.min.css';
import '../../../shared/layout.scss';

function BrowserApp({ Component, pageProps }) {
  return (
    <Layout>
      <Component {...pageProps} />
    </Layout>
  );
}

export default BrowserApp;
