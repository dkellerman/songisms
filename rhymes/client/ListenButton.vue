<script setup lang="ts">
const emit = defineEmits([
  'onQuery',
  'onStarted',
  'onStopped',
  'onPartialResult',
]);

const SR = window && ((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition);
const sr = ref<typeof SR>();
const isSupported = ref(!!SR);
const isListening = ref(false);
const transcript = ref('');
const lastPartialResult = ref('');
const TIMEOUT_MS = 120 * 1000;
let timeout: ReturnType<typeof setTimeout> | undefined;

function createSpeechRecognition() {
  const srObj = new SR();
  srObj.continuous = true;
  srObj.interimResults = true;
  srObj.lang = 'en-US';
  srObj.onstart = onSpeechStarted;
  srObj.onend = onSpeechEnded;
  srObj.onresult = onSpeechResult;
  srObj.onerror = onSpeechError;
  return srObj;
}

function startTimer() {
  clearTimer();
  timeout = setTimeout(() => {
    stop();
    timeout = undefined;
  }, TIMEOUT_MS);
}

function clearTimer() {
  if (timeout) {
    clearTimeout(timeout);
    timeout = undefined;
  }
}

function onSpeechStarted(event: any) {
}

function onSpeechEnded(event: any) {
  if (isListening.value) {
    // override the natural timeout of the speech recognition object
    sr.value.start();
  } else {
    clearTimer();
  }
}

function onSpeechResult(event: SpeechRecognitionEvent) {
  let hasFinal = false;
  let partialResult;

  for (let i = event.resultIndex; i < event.results.length; i++) {
    if (event.results[i].isFinal) {
      hasFinal = true;
      transcript.value += event.results[i][0].transcript.toLowerCase().trim();
      switch (transcript.value) {
        case 'stop listening':
        case 'stop search':
        case 'end search':
        case 'stop recording':
        case 'stop mic':
          stop();
          break;
        case 'clear search':
        case 'reset search':
        case 'clear query':
        case 'top rhymes':
          transcript.value = '';
        default:
          emit('onQuery', transcript.value);
          startTimer(); // restart listening for timeout
          transcript.value = '';
      }
    } else {
      const item = event.results[i].item(0);
      if (!partialResult || (
        item.confidence > partialResult.confidence &&
        item.confidence > .90
      )) partialResult = item;
    }
  }

  if (!hasFinal) {
    if (partialResult?.transcript?.trim() &&
        partialResult.transcript.length > lastPartialResult.value.length
    ) {
      const val = partialResult.transcript.toLowerCase().trim();
      lastPartialResult.value = val;
      emit('onPartialResult', val);
    }
  } else {
    lastPartialResult.value = '';
  }
}

function onSpeechError(event: any) {
  if (event.error === 'no-speech') return;
  console.error('speech recognition error', event);
  stop();
}

function start() {
  sr.value = createSpeechRecognition();
  sr.value.start();
  isListening.value = true;
  startTimer(); // override listen timeout
  emit('onStarted');
}

function stop() {
  if (sr.value) sr.value.stop();
  isListening.value = false;
  transcript.value = '';
  lastPartialResult.value = '';
  emit('onStopped');
}

function toggle() {
  if (isListening.value) stop(); else start();
}
</script>

<template>
    <button
      v-if="isSupported"
      @click.prevent="toggle"
      :title="isListening ? 'Stop listening' : 'Voice search'"
      :class="{ listen: true, 'is-listening': isListening }"
    >
      <i v-if="isListening" class="fa fa-stop" :style="{fontSize: '14px'}" />
      <i v-else class="fa fa-microphone" :style="{fontSize: '18px'}" />
    </button>
</template>
