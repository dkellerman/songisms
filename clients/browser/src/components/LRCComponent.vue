<script>
export default {
  name: 'LRCComponent',
};
</script>

<script setup>
import {computed, defineProps, onMounted, onUnmounted, ref} from 'vue';

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
  lrc.value[lrcIndex.value][lrcPos.value] = curTimeStamp.value ?? '';
  if (lrcPos.value === 1) lrcPos.value = 2;
  else onLRCDown();
}

function onUnmark() {
  lrc.value[lrcIndex.value][1] = null;
  lrc.value[lrcIndex.value][2] = null;
  lrcPos.value = 1;
}

function onNext() {}
function onPrev() {}

const curTimeStamp = ref(':00');

function onTimeUpdate(e) {
  curTimeStamp.value = secs2ts(e.target.currentTime);
}

function togglePlay() {
  if (props.audio.value.paused) props.audio.value.play();
  else props.audio.value.pause();
}

function onKey(e) {
  if (e.key === '<') onLRCDown();
  else if (e.key === '>') onLRCUp();
  else if (e.key === 'm') onMark();
  else if (e.key === 'M') onUnmark();
  else if (e.key === '[') onPrev();
  else if (e.key === ']') onNext();
  else if (e.key === 'p') togglePlay();
}

function secs2ts(seconds) {
  if (seconds === null) return null;
  const mm = Math.floor(seconds / 60);
  const ss = seconds % 60;
  const ts = (mm === 0 ? '' :
    String(mm).padStart(2, '0')) + ":" +
    ss.toFixed(2).padStart(2, '0');
  return ts.replace(/\.0+$/, '');
}

function ts2secs(ts) {
  if (ts === null) return null;
  if (ts.indexOf(':') === -1) return ts;
  const [mm, ss] = ts.split(':').map(s => parseFloat(s));
  return mm * 60 + ss;
}

onMounted(() => {
  window.addEventListener('keyup', onKey);
  props.audio.addEventListener('timeupdate', onTimeUpdate);
});

onUnmounted(() => {
  window.removeEventListener('keyup', onKey);
  props.audio.removeEventListener('timeupdate', onTimeUpdate);
});

</script>

<template>
  <div class="actions">
    <span class="lrc-controls">
      <button @click="onLRCUp">Up (&gt;)</button>
      <button @click="onLRCDown">Down (&lt;)</button>
      <button @click="onMark">Mark (m)</button>
      <button @click="onUnmark">Unmark (M)</button>
    </span>

    <span v-if="props.audio" class="audio-controls">
      Audio:
      <button @click="onNext">Next (])</button>
      <button @click="onPrev">Prev ([)</button>
      <button @click="togglePlay">Toggle (p)</button>
      <strong>[{{ curTimeStamp }}]</strong>
    </span>
    <span v-else>[No audio playing]</span>
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

  <br>
  <strong>Raw LRC</strong>
  <textarea v-model="rawLRC" rows="10" />
</template>

<style scoped lang="scss">
  .actions {
    padding: 10px;
    margin: 10px 0;
    background: #cefad0;
    button {
      padding: 5px;
      margin-right: 5px;
      font-size: medium;
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
