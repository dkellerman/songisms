import SimpleTypeahead from 'vue3-simple-typeahead';

export default defineNuxtPlugin(nuxtApp => {
  nuxtApp.vueApp.use(SimpleTypeahead);
});
