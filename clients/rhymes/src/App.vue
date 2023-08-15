<script>
export default {
  name: 'App',
};
</script>

<template>
  <nav>
    <h1><router-link to="/">Song Rhymes</router-link></h1>
    <div class="links">
      <a href="https://bipium.com">Metronome</a>
    </div>
  </nav>
  <main>
    <router-view />
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
