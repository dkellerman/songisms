<script>
export default {
  name: 'NGrams',
};
</script>

<script setup>
import { debounce } from 'lodash-es';
import { ref, watch } from 'vue';
import { storeToRefs } from 'pinia';
import { useNGramsStore } from '@/stores/ngrams';

const { ngrams, hasNext, total, curQuery, curPage } = storeToRefs(useNGramsStore());
const { fetchNGrams } = useNGramsStore();
const page = ref(curPage.value);
const q = ref(curQuery.value);

const newSearch = () => {
  page.value = 1;
  fetchNGrams(q.value, page.value);
};

watch(page, () => fetchNGrams(q.value, page.value));
watch(q, debounce(newSearch, 500));

if (!ngrams.value) {
  fetchNGrams(q.value, 1);
}
</script>

<template>
  <h2>NGrams</h2>

  <input v-model.trim="q" placeholder="Search ngrams..." />
  <label v-if="total !== undefined">{{ total.toLocaleString() }} ngrams</label>

  <ul class="none" v-if="ngrams">
    <li v-for="ngram in ngrams" :key="ngram.text">
      {{ ngram.text }} <small v-if="ngram.songCount">({{ ngram.songCount }})</small>
    </li>
  </ul>

  <button class="more compact" v-if="hasNext" @click="page++">More...</button>
</template>

<style scoped lang="scss">
h2 {
  margin: 5px 0;
}
input {
  width: 100%;
  max-width: 500px;
  margin-bottom: 10px;
}
label {
  font-size: large;
}
li {
  margin-bottom: 10px;
  a {
    margin-right: 5px;
  }
  &.new {
    font-style: italic;
    opacity: .6;
  }
}
</style>
