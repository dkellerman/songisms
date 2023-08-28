<script setup lang="ts">
import { useSpeechRecognition } from '@vueuse/core';

const emit = defineEmits([
  'onQuery',
  'onStarted',
  'onStopped',
]);

const speech = useSpeechRecognition({
  lang: 'en-US',
  interimResults: false,
  continuous: true,
});

const { isListening, isSupported: listenSupported } = speech;

const TIMEOUT_MS = 60 * 1000;
let timeout: ReturnType<typeof setTimeout> | undefined;

// override timeout
if (speech?.recognition) {
  const onend = speech.recognition.onend;
  const onstart = speech.recognition.onstart;

  speech.recognition.onstart = (event: Event) => {
    timeout = setTimeout(() => {
      speech.stop();
      timeout = undefined;
    }, TIMEOUT_MS);

    if (onstart) onstart.call(speech.recognition!, event);
  };

  speech.recognition.onend = (event: Event) => {
    if (timeout) {
      clearTimeout(timeout);
      timeout = undefined;
    }

    if (isListening.value) {
      speech.recognition!.start();
    } else {
      speech.stop();
      if (onend) onend.call(speech.recognition!, event);
    }
  };
}

function onSpeechResult() {
  let val = speech.result.value?.toLowerCase().trim();
  console.log("*", val);

  if (speech.isFinal) {
    console.log("FINAL");
    if (!val) return;
    const words = val.split(' ');

    if (val === 'stop listening') {
      speech.toggle();
      return;
    } else if (val === 'clear search') {
      val = '';
    } else if (words.length > 2 && words.every(w => w.length === 1)) {
      val = words.join('');
    }

    emit('onQuery', val);

    // start and stop to reset transcript
    if (isListening.value) speech.recognition!.stop();
    setTimeout(() => {
      if (!isListening.value) speech.recognition!.start();
    }, 100);
  }
}

function toggle() {
  speech.toggle();
  if (isListening.value)
    emit('onStarted');
  else
    emit('onStopped');
}

watch(speech.result, onSpeechResult);
</script>

<template>
    <button v-if="listenSupported" @click.prevent="toggle" :class="{ listen: true, 'is-listening': isListening }">
      <i :class="{ fa: true, 'fa-lg': true, 'fa-microphone': !isListening, 'fa-stop': isListening }" />
    </button>
</template>
