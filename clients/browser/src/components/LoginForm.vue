<script>
export default {
  name: 'LoginForm',
};
</script>

<script setup>
import { ref } from 'vue';
import auth from '../auth';
import router from '../router';

const username = ref('');
const password = ref('');
const error = ref('');

async function doLogin() {
  try {
    await auth.login(username.value, password.value);
    await router.push('/');
  } catch (e) {
    console.error(e);
    error.value = e.message;
  }
}
</script>

<template>
  <h2>Log in</h2>

  <div>
    <div v-if="error" role="alert">{{ error }}</div>
    <fieldset>
      <label>Username:</label>
      <input type="text" v-model="username" />
      <label>Password:</label>
      <input type="password" v-model="password" />
    </fieldset>
    <button @click="doLogin">Log in</button>
  </div>
</template>

<style scoped lang="scss"></style>
