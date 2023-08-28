// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  devtools: { enabled: true },
  nitro: { preset: 'vercel'},
  runtimeConfig: {
    public: {
      apiBaseUrl: process.env.SISM_RHYMES_API_BASE_URL,
    },
  },
  routeRules: {},
  app: {
    head: {
      title: 'Song Rhymes',
      link: [
        { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css?family=Roboto:300,300italic,700,700italic' },
        { rel: 'stylesheet', href: 'https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.css' },
        { rel: 'stylesheet', href: 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css' },
        { rel: 'icon', href: '/favicon.ico' }
      ],
      script: [
        { src: 'https://www.googletagmanager.com/gtag/js?id=UA-158752156-1' },
        { children: `
            window.dataLayer = window.dataLayer || [];
            function gtag() { window.dataLayer.push(arguments); }
            gtag('js', new Date());
            gtag('config', 'UA-158752156-1');
        `}
      ],
    }
  },
  css: [
    "vue3-simple-typeahead/dist/vue3-simple-typeahead.css",
    "milligram/dist/milligram.min.css",
    "@/app.scss",
  ],
});
