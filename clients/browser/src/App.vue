<script>
export default {
  name: 'App',
};
</script>

<script setup>
import router from '@/router';
import { useAuth } from '@/stores/auth';
import { useSongsStore } from '@/stores/songs';
import { storeToRefs } from 'pinia';
import { watchEffect } from 'vue';

const { isLoggedIn } = storeToRefs(useAuth());
const { logout } = useAuth();
const { songsIndex } = storeToRefs(useSongsStore());
const { fetchSongsIndex } = useSongsStore();

watchEffect(() => {
  if (isLoggedIn.value && !songsIndex.value) {
    fetchSongsIndex();
  }
});

async function doLogout() {
  await logout();
  window.location.href = '/login';
}
</script>

<template>
  <nav>
    <h1><router-link to="/">Songisms</router-link></h1>
    <div class="links">
      <router-link to="/songs" v-if="isLoggedIn">Songs</router-link>
      <router-link to="/writers">Writers</router-link>
      <router-link to="/login" v-if="!isLoggedIn && router.currentRoute.value.path !== '/login'">Login</router-link>
      <button v-if="isLoggedIn" class="logout compact" @click="doLogout">Logout</button>
    </div>
  </nav>
  <main>
    <router-view :key="$route.path" />
  </main>
</template>

<style lang="scss">
@import '../node_modules/papercss/dist/paper.min.css';
@import '../../shared/layout.scss';
html,
body,
#app {
  padding: 0;
  margin: 0;
}
main {
  padding: 20px;
}
</style>
