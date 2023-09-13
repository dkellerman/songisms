<script setup lang="ts">
import type { RLHF, RLHFResponse } from '../types';

const config = useRuntimeConfig();
const apiBaseUrl = config.public.apiBaseUrl.replaceAll('localhost', '127.0.0.1');

const { data, refresh } = await useFetch<RLHFResponse>(`${apiBaseUrl}/rhymes/rlhf/`, {
  query: { limit: 10 },
  immediate: true,
});

const queue: Ref<RLHF[]> = computed(() => [
  ...(queue.value ?? []),
  ...(data.value?.hits ?? [])
]);

const cur = ref();

// vote fetch
const voteQuery = ref();
const voterUid = ref<string>();
await useFetch(`${apiBaseUrl}/rhymes/vote/`, {
  method: 'POST',
  body: voteQuery,
  immediate: false,
});

function next() {
  if (queue.value.length) cur.value = queue.value.shift();
  if (queue.value.length < 3) refresh();
}

function pick(label: string) {
  voteQuery.value = {
    anchor: cur.value.anchor,
    alt1: cur.value.alt1,
    alt2: cur.value.alt2,
    voter_uid: voterUid.value,
    label,
  };
  next();
}

onMounted(() => {
  import('get-browser-fingerprint').then(({ default: getBrowserFingerprint }) => {
    voterUid.value = String(getBrowserFingerprint());
    if (!voterUid.value) {
      alert('Hmmm, something went wrong, your votes will not be registered currently');
    }
  });
});
</script>

<template>
  <div id="app">
    <Head>
      <Title>PickEm Rhymes</Title>
    </Head>
    <nav>
      <h1>Pick ’em</h1>
    </nav>
    <main v-if="cur">
      <div class="anchor">
        <label>
          <div class="option">Word</div>
          {{ cur.anchor }}
        </label>
      </div>
      <div class="alts">
        <label @click="pick('alt1')">
          <div class="option">Option 1</div>
          {{ cur.alt1 }}
        </label>
        <label @click="pick('alt2')">
          <div class="option">Option 2</div>
          {{ cur.alt2 }}
        </label>
      </div>
      <div class="actions">
        <button class="button-clear" @click="pick('neither')">Equally bad</button>
        <button class="button-clear" @click="pick('both')">Equally good</button>
        <button class="button-clear" @click="pick('flagged')">WTF?!</button>
        <button class="button-clear" @click="next">Skip</button>
      </div>
    </main>

    <main v-else class="instructions">
      <div>
        A word will be shown, along with two possible rhymes.
        <strong>
          Pick the one that feels like the <em>closer</em> rhyme
          by clicking/tapping on the word.
        </strong>
      </div>
      <div>
        Don't think too much about it, just go with your gut - there are no
        right or wrong answers.
      </div>
      <div>
        <p>
          You may also pick one of the options below the rhymes.
          <strong>
            If it's confusing or too hard to answer, just click Skip and go the next one.
          </strong>
        </p>
      </div>
      <button @click="next">Start</button>
    </main>
  </div>
</template>

<style lang="scss" scoped>
nav {
  background: aliceblue;
  text-align: center;
  h1 {
    font-size: 30px;
  }
}
main {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  gap: 20px;
  &.instructions {
    text-align: center;
    font-size: 18px;
    button {
      zoom: 1.5;
    }
  }
  .anchor {
    label {
      font-size: 36px;
      text-align: center;
    }
  }
  .alts {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    gap: 100px;
    label {
      font-size: 28px;
      text-align: center;
      color: blue;
      &:hover, &:active, &:focus {
        cursor: pointer;
        color: blue;
      }
    }
  }

  .actions {
    margin-top: 60px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    button {
      font-size: 18px;
      color: #aaa;
      &:hover {
        cursor: pointer;
        color: blue;
        text-decoration: underline;
      }
    }
  }

  .option {
    font-size: 16px;
    text-align: center;
    text-decoration: underline;
    color: #999;
  }
}

@media screen and (min-width: 640px) {
  .actions {
    flex-direction: row !important;
  }
}
</style>
