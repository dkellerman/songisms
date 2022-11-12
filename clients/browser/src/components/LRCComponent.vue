<script>
export default {
  name: 'LRCComponent',
};
</script>

<script setup>
import { computed, defineProps, ref } from 'vue';

const props = defineProps(['lyrics', 'audio']);
const lines = computed(() => props.lyrics.split('\n'));
const lrc = ref(lines.value.map(l => [l, null, null]));
const lrcIndex = ref(0);
const lrcPos = ref(1);

function onLRCUp() {
  if (lrcIndex.value > 0) {
    do {
      lrcIndex.value--;
    } while (!lines.value[lrcIndex.value].trim());
    lrcPos.value = 1;
    document.getElementById(`l${lrcIndex.value}`).scrollIntoView({block: "center", inline: "nearest"});
  }
}

function onLRCDown() {
  if (lrcIndex.value < lines.value.length - 1) {
    do {
      lrcIndex.value++;
    } while (!lines.value[lrcIndex.value].trim());
    lrcPos.value = 1;
    document.getElementById(`l${lrcIndex.value}`).scrollIntoView({block: "center", inline: "nearest"});
  }
}

function onMark() {
  lrc.value[lrcIndex.value][lrcPos.value] = Math.floor(Math.random() * 1000);
  if (lrcPos.value === 1) lrcPos.value = 2;
  else onLRCDown();
}

function onUnmark() {
  lrc.value[lrcIndex.value][1] = null;
  lrc.value[lrcIndex.value][2] = null;
  lrcPos.value = 1;
}

window.addEventListener('keyup', e => {
  if (e.key === '.') onLRCDown();
  else if (e.key === ',') onLRCUp();
  else if (e.key === 'm') onMark();
  else if (e.key === 'M') onUnmark();
});

</script>

<template>
  <table>
    <tbody>
      <tr :id="`l${index}`" v-for="([line, start, end], index) in lrc" :key="index">
        <td v-if="start">[{{start}}]</td>
        <td v-if="!line.trim()">&nbsp;</td>
        <td v-else v-html="line" :class="lrcIndex === index ? 'curline' : ''" />
        <td v-if="end">[{{end}}]</td>
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
  .curline {
    font-weight: bold;
    background: beige;
  }
</style>
