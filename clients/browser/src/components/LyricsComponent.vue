<script>
export default {
  name: 'LyricsComponent',
};
</script>

<script setup>
import { computed, defineProps } from 'vue';

const props = defineProps(['lyrics']);
const lines = computed(() => props.lyrics.split('\n'));
</script>

<template>
  <table>
    <tbody>
      <tr v-for="(line, index) in lines" :key="index">
        <td v-if="!line.trim()">&nbsp;</td>
        <td v-else v-html="line" />
      </tr>
    </tbody>
  </table>
</template>

<style scoped lang="scss">
  td {
    padding: 0;
    border: 0;
    margin: 0;
  }
  br {
    height: 20px;
  }
</style>
