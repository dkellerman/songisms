<script>
export default {
  name: 'App',
};
</script>

<template>
  <nav>
    <h1><router-link to="/">Song Rhymes</router-link></h1>
  </nav>
  <main>
    <router-view />
  </main>
  <footer>
    SongRhymes by
    <a target="_blank" rel="noopener noreferer" href="https://linkedin.com/in/david-kellerman">&nbsp;David Kellerman</a>
    <div class="links">
      &nbsp;&mdash;
      <a target="_blank" rel="noopener noreferrer" href="https://github.com/dkellerman/songisms">&nbsp;Source code</a>
      &nbsp;&#183;
      <a target="_blank" rel="noopener noreferrer" href="https://bipium.com">&nbsp;Metronome</a>
      &nbsp;&#183;
      <a target="_blank" rel="noopener noreferrer" href="https://open.spotify.com/artist/2fxGUIL1BUCzWwKqP1ykUi">&nbsp;Music</a>
    </div>
</footer>
</template>

<style lang="scss">
@import '../node_modules/milligram/dist/milligram.min.css';
@import './rhymes.scss';
</style>
