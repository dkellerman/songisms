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

const rawLRC = computed(() => {
  const all = [];
  for (const [line, start, end] of lrc.value) {
    if (start || end)
      all.push(`[${start || ''}-${end || ''}] ${line.trim()}`);
    else
      all.push(line.trim());
  }
  return all.join('\n');
});

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
  <div class="actions">
    <button @click="onLRCUp">Up (,)</button>
    <button @click="onLRCDown">Down (.)</button>
    <button @click="onMark">Mark (m)</button>
    <button @click="onUnmark">Unmark (M)</button>
  </div>
  <table>
    <tbody>
      <tr :id="`l${index}`" v-for="([line, start, end], index) in lrc" :key="index">
        <td v-if="!line.trim()" colspan="2">&nbsp;</td>
        <td v-if="line.trim()" class="stamp" contenteditable="true">
          <span v-if="line.trim() && (start || end)">[{{start}}-{{end}}]</span>
        </td>
        <td v-if="line.trim()" v-html="line" :class="lrcIndex === index ? 'curline line' : 'line'" />
      </tr>
    </tbody>
  </table>
  <strong>Raw LRC</strong>
  <textarea v-model="rawLRC" rows="10" />
</template>

<style scoped lang="scss">
  .actions {
    margin: 20px 0;
    button {
      padding: 3px;
      margin-right: 5px;
    }
  }
  td {
    padding: 0;
    border: 0;
    margin: 0;
    &.stamp {
      white-space: nowrap;
      min-width: 100px;
      font-size: smaller;
      background: #eee;
      text-align: center;
    }
    &.curline {
      font-weight: bold;
      background: beige;
    }
  }
  br {
    height: 20px;
  }
  textarea {
    width: 100%;
  }
</style>
