<script>
export default {
  name: 'LoginForm',
};
</script>

<script setup>
import { ref } from 'vue';
import router from '@/router';
import { useAuth } from '@/stores/auth';

const username = ref('');
const password = ref('');
const error = ref('');
const { login } = useAuth();

async function doLogin() {
  try {
    await login(username.value, password.value);
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

<style scoped lang="scss">
input,
label,
button {
  margin: 10px 0;
}
</style>
