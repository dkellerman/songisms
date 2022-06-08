<script>
  export default {
    name: 'App',
  };
</script>

<script setup>
  import auth from './auth';
  import router from './router';

  async function doLogout() {
    await auth.logout();
    window.location.href = '/login';
  }
</script>

<template>
  <nav>
    <h1><router-link to="/">Songisms</router-link></h1>
    <div class="links">
      <router-link to="/songs" v-if="auth.user.value">Songs</router-link>
      <router-link to="/writers">Writers</router-link>
      <router-link to="/login" v-if="!auth.user.value && router.currentRoute.value.path !== '/login'">Login</router-link>
      <button v-if="auth.user.value" class="logout compact" @click="doLogout">Logout</button>
    </div>
  </nav>
  <main>
    <router-view />
  </main>
</template>

<style lang="scss">
  @import "../node_modules/papercss/dist/paper.min.css";
  @import "../../shared/layout.scss";
  html, body, #app {
    padding: 0;
    margin: 0;
  }
  main {
    padding: 20px;
  }
</style>
