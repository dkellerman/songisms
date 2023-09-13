<script setup lang="ts">
const route = useRoute();
const router = useRouter();
const config = useRuntimeConfig();
const apiBaseUrl = config.public.apiBaseUrl.replaceAll('localhost', '127.0.0.1');

const { data, refresh } = await useFetch<any>(`${apiBaseUrl}/rhymes/rlhf/`, {
  query: { limit: 10 },
  immediate: true,
});
const queue = computed<string[]>(() => data.value?.hits ?? []);
const cur = ref();

function next() {
  if (queue.value.length) cur.value = queue.value.shift();
  if (queue.value.length === 0) refresh();
}

function pick(label: string) {
  alert(label);
}
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

    <main v-else>
      <h1>Pick the best rhyme</h1>
      <div>
        A word will be shown, along with two possible rhymes.
        <strong>Pick the best one by clicking/tapping on it.</strong>
      </div>
      <div>
        Don't think too much about it, just go with your gut - there are no
        right or wrong answers.
      </div>
      <div>
        <p>If necessary you may also pick one of the options below the rhymes:</p>
        <ul>
          <li><strong>Equally bad</strong> - Neither word rhymes at all</li>
          <li><strong>Equally good</strong> - Both words rhyme about the same amount</li>
          <li><strong>WTF?!</strong> - This is stupid</li>
          <li><strong>Skip</strong> - If unsure for any reason, just skip and move on</li>
        </ul>
      </div>
      <button @click="next">Start</button>
    </main>
  </div>
</template>

<style lang="scss" scoped>
nav {
  background: aliceblue;
  text-align: center;
}
main {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
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
      &:hover {
        &:hover {
         cursor: pointer;
          color: blue;
        }
      }
    }
  }

  .actions {
    margin-top: 60px;
    display: flex;
    flex-direction: row;
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
    font-size: 14px;
    text-align: center;
    text-decoration: underline;
    color: #999;
  }
}
</style>
