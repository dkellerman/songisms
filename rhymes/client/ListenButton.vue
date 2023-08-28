<script setup lang="ts">
import { useSpeechRecognition } from '@vueuse/core';

const emit = defineEmits([
  'onQuery',
  'onStarted',
]);

const speech = useSpeechRecognition({
  lang: 'en-US',
  interimResults: false,
  continuous: true,
});

const { isListening, isSupported: listenSupported } = speech;

// override timeout
if (speech?.recognition) {
  speech.recognition.onend = () => {
    if (isListening.value) {
      speech.recognition!.start();
    } else {
      speech.stop();
    }
  };
}

function onSpeechResult() {
  let val = speech.result.value?.toLowerCase().trim();
  const words = val.split(' ');

  if (speech.isFinal) {
    if (!val) return;
    if (val === 'stop listening') {
      speech.toggle();
      return;
    } else if (val === 'clear search') {
      val = '';
    } else if (words.length > 2 && words.every(w => w.length === 1)) {
      val = words.join('');
    }

    emit('onQuery', val);
    if (isListening.value) speech.recognition!.stop();
    setTimeout(() => {
      if (!isListening.value) speech.recognition!.start();
    }, 100);
  }
}

watch(speech.result, onSpeechResult);
</script>

<template>
    <button v-if="listenSupported" @click.prevent="() => {
      speech.toggle();
      emit('onStarted');
    }" :class="{ listen: true, 'is-listening': isListening }">
      <i :class="{ fa: true, 'fa-lg': true, 'fa-microphone': !isListening, 'fa-stop': isListening }" />
    </button>
</template>
